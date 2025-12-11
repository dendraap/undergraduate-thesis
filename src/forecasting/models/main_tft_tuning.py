from src.forecasting.utils.libraries_data_handling import pd, np
from src.forecasting.utils.libraries_others import re, os, time, Enum, Optional, gc, pickle, json
from src.forecasting.utils.libraries_plotting import plt
from src.forecasting.utils.libraries_modelling import torch, concatenate, TimeSeries, Scaler, TFTModel, Callback, EarlyStopping, ModelCheckpoint, optuna, PyTorchLightningPruningCallback, Trial, plot_contour, plot_optimization_history, plot_param_importances, GaussianLikelihood, ParameterSampler, MeanAbsolutePercentageError, mean_absolute_percentage_error
from src.forecasting.constants.columns import col_decode, col_encode
from src.forecasting.utils.memory import cleanup
from src.forecasting.constants.enums import ColumnGroup, PeriodList
from src.forecasting.utils.data_split import dataframe_train_test_split, dataframe_train_test_split, timeseries_train_test_split
from src.forecasting.utils.extract_best_epochs import extract_best_epoch_from_checkpoint
from src.forecasting.models.nbeats_tuning_w_optuna import nbeats_tuning_w_optuna
from src.forecasting.models.empty_worst_model import empty_worst_model

# ========================= TODO ========================= #
# TODO: Change this NBEATS code to TFT Model
# ========================= TODO ========================= #

def print_callback(study, trial):
    print(f'\nTrial {trial.number} done ✅')
    print(f'Value: {trial.value}')
    print(f'Params: {trial.params}')
    print(f'✅ Best so far: {study.best_value} with: \n{study.best_trial.params}\n')


if __name__ == "__main__":
    # ========================= SET UP ========================= #

    # Initialize internal precision of matrix multiplication
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    
    # Make dir to store results
    os.makedirs('models/checkpoint_tuning_tft/', exist_ok=True)

    # Setting number after coma to max 5 digits
    np.set_printoptions(suppress=True, precision=5)


    # ========================= LOAD DATASET ========================= #
    # Load xlsx dataset
    df_past     = pd.read_csv('data/processed/past_covariates_nonoutliers_with_pre_normalization.csv')
    df_category = pd.read_csv('data/processed/future_covariates_one_hot_encoding.csv')
    # df_category = pd.read_csv('data/processed/future_covariates.csv')


    # ========================= DATA PREPROCESSING ========================= #
    # Convert timestamp to datatime
    df_past['t'] = pd.to_datetime(df_past['t'], format='%Y-%m-%d %H:%M:%S')

    # Set index
    df_past = df_past.set_index('t').asfreq('h')

    # Convert timestamp to datatime
    df_category['t'] = pd.to_datetime(df_category['t'], format='%Y-%m-%d %H:%M:%S')

    # Set index
    df_category = df_category.set_index('t').asfreq('h')

    # Cut categorical data end time to match with df_past
    df_category = df_category.iloc[:len(df_past)]


    ## ========================= LOAD CORRELATION RESULTS ========================= ##
    # Load correlation results
    results_r = pd.read_csv('data/processed/correlation_scores.csv')

    # Preparing feature selection input
    X_num = df_past[df_past.columns[ColumnGroup.TARGET:]]

    # Take very low correlation level (0.00 - 0.199) to drop
    X_num_drop = results_r[results_r['Correlation'] <= 0.2]['Feature'].to_list()

    # Encode drop colomns name
    X_num_drop = [col_encode[feature] for feature in X_num_drop]

    # Drop columns
    X_num = X_num.drop(columns=X_num_drop)

    ## ========================= DATA SPLIT ========================= ##
    # Split dataset into Y and X
    Y = df_past[df_past.columns[:ColumnGroup.TARGET]].astype('float32')
    X = pd.concat([X_num, df_category], axis=1).astype('float32')

    # Split to data train 80% and test 20%
    Y_train, Y_test = dataframe_train_test_split(Y, test_size=0.1)
    X_train, X_test = dataframe_train_test_split(X, test_size=0.1)

    # Change to TimeSeries Dataset
    Y_train = TimeSeries.from_dataframe(Y_train, value_cols=Y_train.columns.tolist(), freq='h').astype('float32')
    X_train = TimeSeries.from_dataframe(X_train, value_cols=X_train.columns.tolist(), freq='h').astype('float32')
    Y_test  = TimeSeries.from_dataframe(Y_test, value_cols=Y_test.columns.tolist(), freq='h').astype('float32')
    X_test  = TimeSeries.from_dataframe(X_test, value_cols=X_test.columns.tolist(), freq='h').astype('float32')

    # Change unsplitted feature for inference
    Y_series = TimeSeries.from_dataframe(Y, value_cols=Y.columns.tolist(), freq='h').astype('float32')
    X_series = TimeSeries.from_dataframe(X, value_cols=X.columns.tolist(), freq='h').astype('float32')


    ## ========================= NORMALIZATION ========================= ##
    # Preparing the Scalers
    Y_scaler = Scaler()
    X_scaler = Scaler()

    # Normalize data
    Y_train_transformed  = Y_scaler.fit_transform(Y_train).astype('float32')
    X_train_transformed  = X_scaler.fit_transform(X_train).astype('float32')

    # Normalize data for inference
    Y_series_transformed = Y_scaler.fit_transform(Y_series).astype('float32')
    X_series_transformed = X_scaler.fit_transform(X_series).astype('float32')

    # Delete worst model first to save disk
    save_path = 'reports/nbeats_tuned_params.xlsx'
    empty_worst_model(
        work_dir   = 'models/checkpoint_tuning_nbeats',
        excel_path = save_path,
        print_all  = False,
        patience   = 0.0
    )


    # ========================= DATA MODELLING ========================= #
    # Initialize parameter grid possibilites

    # Initialize ParameterSampler combination
    # params_grid = {
    #     'input_chunk_length' : [int(PeriodList.W1), int(PeriodList.D1 * 10), int(PeriodList.W1 * 2)],
    #     'output_chunk_length': [int(PeriodList.D1), int(PeriodList.D1 * 2)],
    #     'batch_size'         : [32, 64, 96],
    #     'num_stacks'         : [10, 20, 30],
    #     'num_blocks'         : [1, 2, 4],
    #     'num_layers'         : [2, 4, 6],
    #     'layer_widths'       : [256, 512],
    #     'dropout'            : [0.2],
    #     'add_encoders'       : [True],
    #     'stride'             : [3]
    # }

    # This is for experiments 1
    # params_grid = {
    #     'input_chunk_length' : [264],
    #     'output_chunk_length': [24],
    #     'batch_size'         : [96],
    #     'num_stacks'         : [20],
    #     'num_blocks'         : [2],
    #     'num_layers'         : [4],
    #     'layer_widths'       : [512],
    #     'dropout'            : [0.2],
    #     'add_encoders'       : [True],
    #     'stride'             : [24]
    # }

    # This is for experiments 2
    # params_grid = {
    #     'input_chunk_length' : [
    #         int(PeriodList.D1 * 13), int(PeriodList.D1 * 12), int(PeriodList.D1 * 11), int(PeriodList.D1 * 10), 
    #         int(PeriodList.D1 * 9),  int(PeriodList.D1 * 8), int(PeriodList.W1)
    #     ],
    #     'output_chunk_length': [int(PeriodList.D1)],
    #     'batch_size'         : [96, 64, 32],
    #     'num_stacks'         : [30, 20, 10, 5],
    #     'num_blocks'         : [1, 2, 4],
    #     'num_layers'         : [2, 4],
    #     'layer_widths'       : [256, 512],
    #     'dropout'            : [0.2],
    #     'add_encoders'       : [True],
    #     'stride'             : [24]
    # }

    # Change this value for experimental purpose
    # n_iter = 1500

    # Tuning using ParameterSampler
    # tuning_results = nbeats_tuning(
    #     Y                = Y_train_transformed,
    #     X                = X_train_transformed,
    #     Y_actual         = Y,
    #     Y_scaler         = Y_scaler,
    #     pre_normalization= True,
    #     max_epochs       = 100,
    #     params_grid      = params_grid,
    #     n_iter           = n_iter,
    #     col_list         = X_num.columns.to_list(),
    #     col_is_one_hot   = True if len(df_category.columns) > 6 else False,
    #     custom_checkpoint= True,
    #     save_path        = 'reports/nbeats_params_results.xlsx'
    # )

    # Tuning using Optuna
    study_nbeats = optuna.create_study(direction='minimize')
    study_nbeats.optimize(
        lambda trial: nbeats_tuning_w_optuna(
            Y                = Y_train_transformed,
            X                = X_train_transformed,
            Y_actual         = Y,
            Y_scaler         = Y_scaler,
            pre_normalization= True,
            max_epochs       = 150,
            col_list         = X_num.columns.to_list(),
            col_is_one_hot   = True if len(df_category.columns) > 6 else False,
            custom_checkpoint= True,
            save_path        = save_path,
            trial            = trial
        ), 
        n_trials=4000, 
        callbacks=[print_callback]
    )
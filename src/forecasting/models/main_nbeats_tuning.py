from src.forecasting.utils.libraries_data_handling import pd, np
from src.forecasting.utils.libraries_others import re, os, time, Enum, Optional, gc, pickle, json
from src.forecasting.utils.libraries_plotting import plt
from src.forecasting.utils.libraries_modelling import torch, concatenate, TimeSeries, Scaler, NBEATSModel, Callback, EarlyStopping, ModelCheckpoint, optuna, PyTorchLightningPruningCallback, Trial, plot_contour, plot_optimization_history, plot_param_importances, GaussianLikelihood, ParameterSampler, MeanAbsolutePercentageError, mean_absolute_percentage_error
from src.forecasting.constants.columns import col_decode, col_encode
from src.forecasting.utils.memory import cleanup
from src.forecasting.constants.enums import ColumnGroup, PeriodList
from src.forecasting.utils.data_split import dataframe_train_valid_test_split
from src.forecasting.utils.extract_best_epochs import extract_best_epoch_from_checkpoint
from src.forecasting.models.nbeats_tuning_w_optuna import nbeats_tuning_w_optuna
from src.forecasting.models.empty_worst_model import empty_worst_model
from src.forecasting.utils.scale_timeseries_per_component import scale_Y_timeseries_per_component, scale_X_timeseries_per_component

def get_targets(df, target_cols=None):
    if target_cols is None:
        target_cols = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6']
    available = [col for col in target_cols if col in df.columns]
    return df[available].astype('float32')

def get_features(df, target_cols=None):
    if target_cols is None:
        target_cols = ['y1','y2','y3','y4','y5','y6']
    available_targets = [col for col in target_cols if col in df.columns]
    return df.drop(columns=available_targets).astype('float32')

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
    work_dir = 'models/checkpoint_tuning_nbeats2/'
    os.makedirs(work_dir, exist_ok=True)

    # Setting number after coma to max 5 digits
    np.set_printoptions(suppress=True, precision=5)


    # ========================= LOAD DATASET ========================= #
    # Load xlsx dataset
    ## CHANGE NUMBER BELOW FOR CHOOSE THE DATASET ##
    dataset_used = 1
    ## CHANGE NUMBER ABOVE FOR CHOOSE THE DATASET ## 

    df_past = None
    dataset_type = None
    prenorm_type = None
    if dataset_used == 1:
        dataset_type = 'sqrt'
        prenorm_type = 'sqrt'
        df_past = pd.read_csv('data/processed/past_covariates_sqrt_transform.csv')
    elif dataset_used == 2:
        dataset_type = 'sqrt_NoOzon'
        prenorm_type = 'sqrt'
        df_past = pd.read_csv('data/processed/past_covariates_sqrt_transform_NoOzon.csv')
    elif dataset_used == 3:
        dataset_type = 'log1p'
        prenorm_type = 'log1p'
        df_past = pd.read_csv('data/processed/past_covariates_log_transform.csv')
    elif dataset_used == 4:
        dataset_type = 'log1p_NoOzon'
        prenorm_type = 'log1p'
        df_past = pd.read_csv('data/processed/past_covariates_log_transform_NoOzon.csv')
    else:
        dataset_type = 'default'
        prenorm_type = None
        df_past = pd.read_csv('data/processed/past_covariates.csv')



    # ========================= DATA PREPROCESSING ========================= #
    # Convert timestamp to datatime
    df_past['t'] = pd.to_datetime(df_past['t'], format='%Y-%m-%d %H:%M:%S')

    # Set index
    df_past = df_past.set_index('t').asfreq('h')


    ## ========================= LOAD CORRELATION RESULTS ========================= ##
    # Load correlation results
    results_r = pd.read_csv('data/processed/correlation_scores.csv')

    # Take very low correlation level (0.00 - 0.199) to drop
    dropped_covariates = results_r[results_r['Correlation'] <= 0.2]['Feature'].to_list()

    # Encode drop colomns name
    dropped_covariates = [col_encode[feature] for feature in dropped_covariates]

    ## ========================= DATA SPLIT ========================= ##
    # Split dataset into Y and X
    # Drop low correlation columns.
    # df_past = df_past.drop(columns=[dropped_covariates]) # KEEP FOR DO DROP OR COMMEND IF WON'T DROP
    Y = get_targets(df_past)
    X = get_features(df_past)

    # Split to data train and test
    valid_size = 0.2
    test_size  = 0.1
    Y_train, Y_valid, Y_test = dataframe_train_valid_test_split(
        Y, valid_size=valid_size, test_size=test_size
    )

    X_train, X_valid, X_test = dataframe_train_valid_test_split(
        X, valid_size=valid_size, test_size=test_size
    )

    # Change to TimeSeries Dataset
    Y_train = TimeSeries.from_dataframe(Y_train, value_cols=Y_train.columns.tolist(), freq='h').astype('float32')
    X_train = TimeSeries.from_dataframe(X_train, value_cols=X_train.columns.tolist(), freq='h').astype('float32')
    Y_valid = TimeSeries.from_dataframe(Y_valid, value_cols=Y_valid.columns.tolist(), freq='h').astype('float32')
    X_valid = TimeSeries.from_dataframe(X_valid, value_cols=X_valid.columns.tolist(), freq='h').astype('float32')
    Y_test  = TimeSeries.from_dataframe(Y_test, value_cols=Y_test.columns.tolist(), freq='h').astype('float32')
    X_test  = TimeSeries.from_dataframe(X_test, value_cols=X_test.columns.tolist(), freq='h').astype('float32')


    ## ========================= NORMALIZATION ========================= ##
    # Initialize Y scalers
    Y_scalers = {}
    Y_train_transformed = Y_train.copy()

    # Normalize Y Train
    Y_train_transformed, Y_scalers = scale_Y_timeseries_per_component(
        Y_train, fit=True
    )

    # Transform VALID & TEST
    Y_valid_transformed = Y_valid.copy()
    Y_test_transformed  = Y_test.copy()

    # Normalize Y Validation
    Y_valid_transformed, _ = scale_Y_timeseries_per_component(
        Y_valid, scalers=Y_scalers, fit=False
    )

    # Normalize Y Test
    Y_test_transformed, _ = scale_Y_timeseries_per_component(
        Y_test, scalers=Y_scalers, fit=False
    )

    # Initialize X Columns to normalize
    x_normalize_cols = ['x1', 'x3', 'x5', 'x6']

    # Initialize X scalers
    X_scalers = {}
    X_train_transformed = X_train.copy()

    # Normalize X Train
    X_train_transformed, X_scalers = scale_X_timeseries_per_component(
        ts   = X_train,
        cols = x_normalize_cols,
        fit  = True
    )

    # Transform VALID & TEST
    X_valid_transformed = X_valid.copy()
    X_test_transformed  = X_test.copy()

    # Normalize X Validation
    X_valid_transformed, _ = scale_X_timeseries_per_component(
        ts      = X_valid,
        cols    = x_normalize_cols,
        scalers = X_scalers,
        fit     = False
    )

    # Normalize X Test
    X_test_transformed, _ = scale_X_timeseries_per_component(
        ts      = X_test,
        cols    = x_normalize_cols,
        scalers = X_scalers,
        fit     = False
    )


    # ========================= DATA MODELLING ========================= #
    # Excel save location
    save_path  = 'reports/nbeats_tuned_params_optimized.xlsx'

    # Tuning using Optuna
    study_nbeats = optuna.create_study(direction='minimize')
    study_nbeats.optimize(
        lambda trial: nbeats_tuning_w_optuna(
            dataset_type     = dataset_type,
            prenorm_type     = prenorm_type,
            Y_train          = Y_train_transformed,
            X_train          = X_train_transformed,
            Y_valid          = Y_valid_transformed,
            X_valid          = X_valid_transformed,
            Y_scalers        = Y_scalers,
            X_scalers        = X_scalers,
            Y_actual         = Y[:Y_valid.end_time()],
            validation_split = valid_size,
            max_epochs       = 150,
            Y_col_list       = Y.columns.to_list(),
            X_col_list       = X.columns.to_list(),
            custom_checkpoint= True,
            save_path        = save_path,
            work_dir         = work_dir,
            trial            = trial
        ), 
        n_trials=3000, 
        callbacks=[print_callback]
    )
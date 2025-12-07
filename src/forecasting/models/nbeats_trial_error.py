from src.forecasting.utils.libraries_data_handling import pd, np
from src.forecasting.utils.libraries_others import re, os, time, Enum, Optional, gc, pickle, json
from src.forecasting.utils.libraries_plotting import plt
from src.forecasting.utils.libraries_modelling import torch, TimeSeries, Scaler, NBEATSModel, EarlyStopping, ParameterSampler, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from src.forecasting.constants.columns import col_decode, col_encode
from src.forecasting.utils.memory import cleanup
from src.forecasting.constants.enums import ColumnGroup, PeriodList
from src.forecasting.utils.data_split import dataframe_train_test_split, dataframe_train_test_split, timeseries_train_test_split
from src.forecasting.utils.extract_best_epochs import extract_best_epoch_from_checkpoint
from src.forecasting.models.nbeats_functions import build_fit_nbeats_model, evaluate_cv


if __name__ == "__main__":
    # ========================= SET UP ========================= #
    # Make dir to store results
    os.makedirs('models/trial_error/', exist_ok=True)

    # Setting number after coma to max 5 digits
    np.set_printoptions(suppress=True, precision=5)


    # ========================= LOAD DATASET ========================= #
    # Load xlsx dataset
    df_past     = pd.read_csv('data/processed/past_covariates_nonoutliers_with_pre_normalization.csv')
    df_category = pd.read_csv('data/processed/future_covariates_one_hot_encoding.csv')


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
    Y = df_past[df_past.columns[:ColumnGroup.TARGET]]
    X = pd.concat([X_num, df_category], axis=1)

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


    # ========================= DATA MODELLING ========================= #
    model_name = 'trial_and_error'
    work_dir = 'models/checkpoint_tuning_nbeats/'

    # Fit model
    start_time = time.time()

    model, Y_fit_scaler, X_fit_scaler = build_fit_nbeats_model(
        Y                = Y,
        X                = X,
        max_epochs       = 30,
        batch_size       = 1024,
        num_stacks       = 20,
        num_blocks       = 2,
        num_layers       = 4,
        layer_widths     = 512,
        include_encoders = True,
        dropout          = 0.1,
        validation_split = 0.2,
        model_name       = model_name,
        work_dir         = work_dir,
        include_stopper  = True,
    )

    cost_time = time.time() - start_time
    print(f'N-BEATS Fit cost: {cost_time:.2f} seconds')

    # Cross Validation with Rolling Forecast
    cv_test = model.historical_forecasts(
        series           = Y,
        past_covariates  = X,
        start            = Y.start_time(),
        forecast_horizon = len(Y) // 7,
        stride           = len(Y) // 7,
        retrain          = False,
        last_points_only = False,
    )

    # Evaluate
    mape_cv = evaluate_cv(
        forecasts  = cv_test,
        scaler     = Y_fit_scaler,
        df_actual  = Y,
    )
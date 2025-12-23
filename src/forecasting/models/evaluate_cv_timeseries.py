from src.forecasting.utils.libraries_data_handling import np, pd, math
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import concatenate, TimeSeries, Scaler, mean_absolute_percentage_error
from src.forecasting.utils.memory import cleanup
from src.forecasting.tuils.pre_normalization import sqrt_transform_inverse, log1p_transform_inverse, 

def evaluate_cv_timeseries(
    forecasts    : list[TimeSeries],
    scalers      : dict[str, Scaler],
    df_actual    : pd.DataFrame,
    prenorm_type : str | None = None
) -> dict[str, float]:
    """
    Evaluating combined forecast results using MAPE per component.
    Args:
        forecasts (list[TimeSeries]) : List of forecasted TimeSeries from historical_forecast().
        scalers (dict[str, Scaler])  : List Scaler of each variables, used to inverse transform forecasted values.
        df_actual (pd.DataFrame)     : Dataset actual for comparison.
        pre_norm (str | None = None) : Type prenormalization used to inverse transform.

    Returns:
        dict[str, float]: Dictionary of MAPE scores per component.
    """

    # Merge all forecast results
    pred = concatenate(forecasts, axis=0)

    # Inverse scaling & prenorm transform per target
    inv_components = []
    for col in pred.components:

        # Inverse normalization
        ts_inv = scalers[col].inverse_transform(pred[col])

        # Inverse pre-normalization
        if prenorm_type is None:
            pass
        elif prenorm_type == 'sqrt':
            ts_inv = ts_inv ** 2
        elif prenorm_type == 'log1p':
            ts_inv = np.expm1(ts_inv)
            
        inv_components.append(ts_inv)

    pred = concatenate(inv_components, axis=1)
    # pred = scaler.inverse_transform(pred) ## LAMA

    # Extact actual and prediction
    start  = pred.start_time()
    end    = pred.end_time()
    actual = df_actual.loc[start:end]
    pred   = pred.to_dataframe()

    # Calculate MAPE per variables
    mape_results = {}
    for col in pred.columns:

        # Avoid NaN results
        try:
            val = mean_absolute_percentage_error(actual[col].values, pred[col].values)
            if isinstance(val, float) and math.isnan(val):
                print('!! MAPE is NAN. Change to 9999')
                mape_results[col] = 9999
            else:
                mape_results[col] = val
        except Exception as e:
            print(f'!! {e} MAPE is NAN. Change to 9999')
            mape_results[col] = 9999
            
    cleanup(pred, actual, inv_components)
    return mape_results
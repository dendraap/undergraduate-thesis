from src.forecasting.utils.libraries_data_handling import np, pd, math
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import concatenate, TimeSeries, Scaler, mean_absolute_percentage_error
from src.forecasting.utils.memory import cleanup

def evaluate_cv_timeseries(
    forecasts  : list[TimeSeries],
    scaler     : Scaler,
    df_actual  : pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluating combined forecast results using MAPE per component.
    Args:
        forecasts (list[TimeSeries]) : List of forecasted TimeSeries from historical_forecast().
        scaler (Scaler)              : Scaler used to inverse transform forecasted values.
        df_actual (pd.DataFrame)     : Dataset actual for comparison.

    Returns:
        dict[str, float]: Dictionary of MAPE scores per component.
    """

    # Merge all forecast results
    pred = concatenate(forecasts, axis=0)

    # Inverse transform
    pred = scaler.inverse_transform(pred)

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
            val = mean_absolute_percentage_error(actual[col], pred[col])
            if isinstance(val, float) and math.isnan(val):
                print('!! MAPE is NAN. Change to 9999')
                mape_results[col] = 9999
            else:
                mape_results[col] = val
        except Exception as e:
            print(f'!! {e} MAPE is NAN. Change to 9999')
            mape_results[col] = 9999
            
    cleanup(pred, actual)
    return mape_results
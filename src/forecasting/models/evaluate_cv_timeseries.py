from src.forecasting.utils.libraries_data_handling import np, pd, math
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import concatenate, TimeSeries, Scaler, mape
from src.forecasting.utils.memory import cleanup

def evaluate_cv_timeseries(
    forecasts    : list[TimeSeries],
    scaler       : Scaler,
    df_actual    : pd.DataFrame,
    prenorm_type : str | None = None
) -> dict[str, float]:
    """
    Evaluating combined forecast results using MAPE per component.
    Args:
        forecasts (list[TimeSeries]) : List of forecasted TimeSeries from historical_forecast().
        scaler (Scaler)              : Y Scaler to inverse.
        df_actual (pd.DataFrame)     : Dataset actual for comparison.
        pre_norm (str | None = None) : Type prenormalization used to inverse transform.

    Returns:
        dict[str, float]: Dictionary of MAPE scores per component.
    """

    # Merge all forecast results
    pred = concatenate(forecasts, axis=0)

    # Inverse scaling & prenorm transform per target
    pred = scaler.inverse_transform(pred)
    
    if prenorm_type is None:
        pass
    elif prenorm_type == 'sqrt':
        pred = pred ** 2
    elif prenorm_type == 'log1p':
        pred = pred.map(np.expm1)
    
    # inv_components = []
    # for col in pred.components:

    #     # Inverse normalization
    #     ts_inv = scalers[col].inverse_transform(pred[col])

    #     # Inverse pre-normalization
    #     if prenorm_type is None:
    #         pass
    #     elif prenorm_type == 'sqrt':
    #         ts_inv = ts_inv ** 2
    #     elif prenorm_type == 'log1p':
    #         # ts_inv = np.expm1(ts_inv)
    #         ts_inv = ts_inv.map(np.expm1)
            
    #     inv_components.append(ts_inv)

    # pred = concatenate(inv_components, axis=1)
    # print(f'{pred.components} - start: {pred.shape}')
    # pred = scaler.inverse_transform(pred) ## LAMA

    # Extact actual and prediction
    start  = pred.start_time()
    end    = pred.end_time()
    actual = df_actual.loc[start:end]

    actual = TimeSeries.from_dataframe(actual, value_cols=actual.columns.tolist(), freq='h').astype('float32')

    # Calculate MAPE per variables
    mape_results = {}
    for col in pred.components:

        # Avoid NaN results
        try:
            # val = mean_absolute_percentage_error(actual[col], pred[col])
            val = mape(actual[col], pred[col])
            print(val)
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
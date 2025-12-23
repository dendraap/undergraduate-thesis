from src.forecasting.utils.libraries_modelling import concatenate, TimeSeries, Scaler

def scale_Y_timeseries_per_component(
    ts      : TimeSeries,
    scalers : dict | None = None,
    fit     : bool = True,
):
    scaled = []
    out_scalers = scalers if scalers is not None else {}

    for col in ts.components:
        if fit:
            scaler = Scaler()
            out_scalers[col] = scaler
            scaled.append(scaler.fit_transform(ts[col]))
        else:
            scaled.append(out_scalers[col].transform(ts[col]))

    return concatenate(scaled, axis=1), out_scalers

def scale_X_timeseries_per_component(
    ts, 
    cols, 
    scalers=None, 
    fit=True
):
    scaled = []
    out_scalers = scalers if scalers is not None else {}

    for col in ts.components:
        if col in cols:
            if fit:
                scaler = Scaler()
                out_scalers[col] = scaler
                scaled.append(scaler.fit_transform(ts[col]))
            else:
                scaled.append(out_scalers[col].transform(ts[col]))
        else:
            scaled.append(ts[col])

    return concatenate(scaled, axis=1), out_scalers

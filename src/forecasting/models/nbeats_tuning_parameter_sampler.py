from src.forecasting.utils.libraries_data_handling import np, pd, math
from src.forecasting.utils.libraries_others import os, re, time, json, psutil, shutil, datetime
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import concatenate, TimeSeries, Scaler, ParameterSampler,torch, NBEATSModel, EarlyStopping, MeanAbsolutePercentageError, mean_absolute_percentage_error, ModelCheckpoint
from src.forecasting.utils.extract_best_epochs import extract_best_epoch_from_checkpoint
from src.forecasting.utils.memory import cleanup
from src.forecasting.constants.enums import PeriodList

def delete_worst_model(
    work_dir   : str,
    excel_path : str,
    print_all  : bool,
    patience   : float
) -> None:
    """
    Function to delete worst model to save disk.
    Args:
        work_dir (str)   : Path to main folder that contain many folders of model_name.
        excel_path (str) : Path to tuning results in excel.
        print_all (bool) : Whether to print all deleting progress or just 1 print.
        patience (float) : Patience of MAPE_sum maximum model to delete.

    Returns:
        None: This function does not return anything.
    """
    
    df = pd.read_excel(excel_path)
    valid_models = set(df['model_name'].astype(str))

    # Get MAPE_sum from excel
    mape_sum = dict(zip(df['model_name'].astype(str), df['MAPE_sum']))

    # Iterate through each folder in work_dir
    for model_name in os.listdir(work_dir):
        folder_path = os.path.join(work_dir, model_name)

        if not os.path.isdir(folder_path):
            continue

        # Model_name not found in excel -> model error is NaN, can't store to excel -> delete model file, keep folder model_name
        if model_name not in valid_models:
            if print_all:
                print(f'❌ Deleting (not in Excel) file/folder in : {model_name}')
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                else:
                    shutil.rmtree(item_path)

        # Model_name found in excel but MAPE_sum > {patience} -> delete model file, keep folder model_name
        elif mape_sum.get(model_name, float('inf')) > patience:
            if print_all:
                print(f'✅ Deleting (MAPE_sum > 1.4) file/folder in : {model_name}')
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                else:
                    shutil.rmtree(item_path)
    if not print_all:
        print(f'✅ Success deleting worst models in {work_dir}')
    return None

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


def build_nbeats(
    Y                   : TimeSeries,
    X                   : TimeSeries,
    input_chunk_length  : int,
    output_chunk_length : int,
    n_epochs            : int,
    batch_size          : int,
    num_stacks          : int,
    num_blocks          : int, 
    num_layers          : int,
    layer_widths        : int,
    dropout             : float,
    include_encoders    : bool,
    validation_split    : float,
    model_name          : str,
    work_dir            : str,
    include_stopper     : bool,
    custom_checkpoint   : bool
) -> NBEATSModel: 
    """
    Function to build Fit of N-BEATS Model.

        Args:
            Y (TimeSeries)            : Targeted variables to predict. 
            X (TimeSeries)            : Exogenous variables to predict Y.
            input_chunk_length (int)  : How many model look to predict.
            output_chunk_length (int) : How many model can produce prediction.
            batch_size (int)          : Number of data points before making update. Larger -> more robust but need more Memory
            num_stacks (int)          : Number of stacks in N-BEATS.
            num_blocks (int)          : Number of blocks of each stacks in N-BEATS.
            num_layers (int)          : Number of fully connected layers of each blocks in N-BEATS
            layer_widths (int)        : Number of neuron patterns. Larger -> need more resource.
            include_encoders (bool)   : Optionally, adding some cyclic covariates ex. (hour, dayofweek, week, etc)
            dropout (float)           : Dropout probability to be used in fully connected layers.
            validation_split (float)  : To split data input into train and validation to monitor val_loss.
            model_name (str)          : The model name to prevent error for same name.
            work_dir (str)            : Path location to save checkpoints best epochs model.
            include_stopper (bool)    : Whether to utilize EarlyStopping or not.
            custom_checkpoint (bool)  : Whether to load default checkpoint or custom checkpoint.

        Returns:
            NBEATSModel : This function return the model configuration.
    """

    # Split
    Y_fit, Y_val = timeseries_train_test_split(Y, test_size=validation_split)
    X_fit, X_val = timeseries_train_test_split(X, test_size=validation_split)

    # Initialize TorchMetrics, used as the monitor
    torch_metrics = MeanAbsolutePercentageError()

    # Check if include encoders or not
    if include_encoders:
        add_encoders = {
            'cyclic': {'past': ['hour', 'day','dayofweek', 'dayofyear', 'week', 'weekday', 'weekofyear', 'month', 'quarter']}
        }

    # pl_trainer_kwargs setup
    pl_trainer_kwargs = {}
    if torch.cuda.is_available():
        pl_trainer_kwargs['accelerator'] = 'gpu'
        pl_trainer_kwargs['devices'] = [0]

    callbacks = []
    if include_stopper:
        early_stopper = EarlyStopping(
            monitor   = 'val_MeanAbsolutePercentageError', #val_loss
            patience  = 8,
            min_delta = 0.01,
            mode      = 'min'
        )
        callbacks.append(early_stopper)
        
    # Custom model checkpoint setup
    if custom_checkpoint:
        checkpoints = 'checkpoints'
        checkpoint_callback = ModelCheckpoint(
            dirpath    = os.path.join(work_dir, model_name, checkpoints),
            filename   = 'MAPE-best-epoch={epoch}-val_MAPE={val_MeanAbsolutePercentageError:.4f}-val_loss={val_loss:.4f}',
            monitor    = 'val_MeanAbsolutePercentageError',
            save_top_k = 1,
            mode       = 'min',
            auto_insert_metric_name = False
        )
        callbacks.append(checkpoint_callback)

    pl_trainer_kwargs['callbacks'] = callbacks
    
    if custom_checkpoint:
        pl_trainer_kwargs['enable_checkpointing'] = True

    # Initialize model
    model = NBEATSModel(
        input_chunk_length  = input_chunk_length,
        output_chunk_length = output_chunk_length,
        n_epochs            = n_epochs,
        batch_size          = batch_size,
        num_stacks          = num_stacks,
        num_blocks          = num_blocks,
        num_layers          = num_layers,
        layer_widths        = layer_widths,
        dropout             = dropout,
        random_state        = 1502,
        torch_metrics       = torch_metrics,
        pl_trainer_kwargs   = pl_trainer_kwargs,
        model_name          = model_name,
        work_dir            = work_dir,
        log_tensorboard     = True,
        save_checkpoints    = True,   # Enable Darts default checkpoint (_model.pth.tar)
        # save_checkpoints    = False,  # Disable Darts default checkpoint (_model.pth.tar)
        add_encoders        = add_encoders if include_encoders else None
    )

    # Fit model
    model.fit(
        series              = Y_fit,
        past_covariates     = X_fit,
        val_series          = Y_val,
        val_past_covariates = X_val,
        load_best           = True,
        # Change this stride = 1 to somethingelse for experiment
        stride              = 1
    )

    # Get best model .ckpt file. Used for custom ModelCheckoint
    ## Scan first then take to best epoch
    ckpt_dir  = os.path.join(work_dir, model_name, checkpoints)
    ckpt_list = os.listdir(ckpt_dir)

    # Load best model file_name baeed on usecase
    if custom_checkpoint :
        best_ckpt = next(
            (f for f in ckpt_list if re.search(r'MAPE-best-epoch=\d+', f)),
            None
        )
        print('!! Model loaded from custom checkpoint')
    else :
        best_ckpt = next(
            (f for f in ckpt_list if re.search(r'best-epoch=\d+', f)),
            None
        )
        print('✅Model loaded from default checkpoint')

    # To avoid function errors when best_ckpt not found
    if best_ckpt:
        model = model.load_from_checkpoint(
            model_name = model_name,
            work_dir   = work_dir,
            # best       = True    # For default checkpoint Darts
            file_name  = best_ckpt
        )
    else:
        print('⚠️ No checkpoint file found, returning last trained model.')

    # Cleanup memory
    cleanup(ckpt_dir, ckpt_list, best_ckpt, callbacks, checkpoint_callback, early_stopper, add_encoders)
    return model


def nbeats_tuning(
    Y                 : TimeSeries,
    X                 : TimeSeries,
    Y_actual          : pd.DataFrame,
    Y_scaler          : Scaler,
    pre_normalization : bool,
    max_epochs        : int,
    params_grid       : (dict[str, np.ndarray]),
    n_iter            : int,
    col_list          : list,
    col_is_one_hot    : bool,
    custom_checkpoint : bool,
    save_path         : str

): 
    """
    Function hyperparameter tuning for N-BEATS using random search (parameter sampler) and rolling forecast evaluation.

    Args:
        Y (TimeSeries)                      : Target series.
        X (TimeSeries)                      : Past Covariates.
        Y_actual (pd.DataFrame)             : Actual targeted data to compare.
        Y_scaler (Scaler)                   : Targetted scaler to transform/inverse.
        pre_normalization (bool)            : To store in the results which data is used.
        max_epochs (int)                    : Max training epochs.
        params_grid (dict[str, np.ndarray]) : List of hyperparameter sample form.
        n_iter (int)                        : Number of random hyperparameter sample form to evaluate.
        col_list (list)                     : List of numeric covariates used to train.
        col_is_one_hot (bool)               : Whether use categoric covariates as ordinal or one hot encoding. 
        custom_checkpoint (bool)            : Whether to load default checkpoint or custom checkpoint.
        save_path (str)                     : Path location to save tuning results as xlsx or not.

    Returns:
        None : This function just to tuning. Tuning results already stored in excel format.
    """
    
    # Initializ parameter to fit using random search
    params_list = list(ParameterSampler(params_grid, n_iter=n_iter, random_state=1502))

    # Iterate through each parameter take
    for params in params_list:
        # print('###==============================================================================================###\n')
        print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
        print('↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓\n')
        print(
            f'🔃 Tuning with params:\n'
            f'\tinput_chunk_length : {params["input_chunk_length"]}\n'
            f'\toutput_chunk_length : {params["output_chunk_length"]}\n'
            f'\tbatch_size : {params["batch_size"]}\n'
            f'\tnum_stacks : {params["num_stacks"]}\n'
            f'\tnum_blocks : {params["num_blocks"]}\n'
            f'\tnum_layers : {params["num_layers"]}\n'
            f'\tlayer_widths : {params["layer_widths"]}\n'
            f'\tdropout : {params["dropout"]}\n'
            f'\tadd_encoders : {params["add_encoders"]}\n'
            f'\tstride : {params["stride"]}\n'
            f'\tColumns used : {col_list}\n'
            f'\tCategoric is one hot encoding: {col_is_one_hot}\n'
        )

        # Generate model name and work dir
        model_name = (
            f'nbeats_ic{params["input_chunk_length"]}_oc{params["output_chunk_length"]}_bs{params["batch_size"]}'
            f'_st{params["num_stacks"]}_bl{params["num_blocks"]}_ly{params["num_layers"]}'
            f'_wd{params["layer_widths"]}_dp{params["dropout"]}_encoders{params["add_encoders"]}_stride{params["stride"]}'
            f'_cov{len(col_list)}_onehot{col_is_one_hot}_monitorMAPE'
            # Uncomment code below for experiments
            # '_TRY_(2_retrain_True)'
        )
        work_dir = 'models/checkpoint_tuning_nbeats/'
        folder_path = os.path.join(work_dir, model_name)
 
        # Check if excel file is exist
        if save_path and os.path.exists(save_path):
            try:
                existing_df = pd.read_excel(save_path)

                # If model_name already trained, skip the model
                if (model_name in existing_df.get("model_name", [])) or os.path.exists(folder_path):
                    print(f'⚠️ Skipping {model_name} — already trained.\n')
                    print('↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                    continue
            
                # Fit model
                start_time = time.time()

                model = build_nbeats(
                    Y                   = Y,
                    X                   = X,
                    input_chunk_length  = params["input_chunk_length"],
                    output_chunk_length = params["output_chunk_length"],
                    n_epochs            = max_epochs,
                    batch_size          = params["batch_size"],
                    num_stacks          = params["num_stacks"],
                    num_blocks          = params["num_blocks"],
                    num_layers          = params["num_layers"],
                    layer_widths        = params["layer_widths"],
                    dropout             = params["dropout"],
                    include_encoders    = params["add_encoders"],
                    validation_split    = 0.2,
                    model_name          = model_name,
                    work_dir            = work_dir,
                    include_stopper     = True,
                    custom_checkpoint   = custom_checkpoint
                )

                cost_time = time.time() - start_time
                
                if torch.cuda.is_available():
                    gpu_id       = 0
                    ram_usage_mb = torch.cuda.memory_allocated(gpu_id) / (1024**2)
                else:
                    process      = psutil.Process(os.getpid())
                    ram_usage_mb = process.memory_info().rss / (1024 ** 2)

                print(f'✅ N-BEATS Fit cost: {cost_time:.2f} seconds')
                print(f'🧠 RAM used after training: {ram_usage_mb:.2f} MB')

                # Cross Validation with Rolling Forecast
                cv_test = model.historical_forecasts(
                    series           = Y,
                    past_covariates  = X,
                    start            = Y.start_time(),
                    forecast_horizon = params["stride"], #len(Y) // params["stride"]
                    stride           = params["stride"], #len(Y) // params["stride"]
                    retrain          = False,
                    last_points_only = False,
                )

                # Evaluate
                mape_cv = evaluate_cv_timeseries(
                    forecasts  = cv_test,
                    scaler     = Y_scaler,
                    df_actual  = Y_actual,
                )
                print(f'💹MAPE_sum : {sum(mape_cv.values())}')
                print(f'〽️MAPE CV: {mape_cv}')

                # Store params to xlsx
                params_record = {
                    'model_name'         : model_name,
                    'GPU'                : True if torch.cuda.is_available() else False,
                    'ram_usage_MB'       : round(ram_usage_mb, 2),
                    'fit_cost_seconds'   : round(cost_time, 2),
                    'pre-normalization'  : pre_normalization,
                    'input_chunk_length' : model.input_chunk_length,
                    'output_chunk_length': model.output_chunk_length,
                    'n_epochs'           : extract_best_epoch_from_checkpoint(
                        work_dir     = work_dir, 
                        model_name   = model_name, 
                        custom_model = custom_checkpoint
                    ),
                    'batch_size'         : model.batch_size,
                    'num_stacks'         : model.num_stacks,
                    'num_blocks'         : model.num_blocks,
                    'num_layers'         : model.num_layers,
                    'layer_widths'       : model.layer_widths[0],
                    'dropout'            : model.dropout,
                    'random_state'       : 1502,
                    'validation_split'   : 0.2,
                    'stride'             : params['stride'],
                    'covariates'         : json.dumps(col_list),
                    'one_hot_encoding'   : col_is_one_hot,
                    'add_encoders'       : json.dumps({'cyclic': {'past': ['hour', 'day','dayofweek', 'dayofyear', 'week', 'weekday', 'weekofyear', 'month', 'quarter']}}) if params['add_encoders'] else None
                }

                # EarlyStopping config to store in results
                early_stopping_config = {
                    'monitor'  : 'val_MeanAbsolutePercentageError',
                    'patience' : 8,
                    'min_delta': 0.01,
                    'mode'     : 'min'
                }

                # ModelCheckpoint config to store in results
                if custom_checkpoint:
                    checkpoints = 'checkpoints'
                    checkpoint_config = {
                        'dirpath'   : os.path.join(work_dir, model_name, checkpoints),
                        'filename'  : "MAPE-best-epoch={epoch}-val_MAPE={val_MeanAbsolutePercentageError:.4f}-val_loss={val_loss:.4f}",
                        'monitor'   : 'val_MeanAbsolutePercentageError',
                        'save_top_k': 1,
                        'mode'      : 'min',
                        'auto_insert_metric_name': False
                    }


                # Trainer config to store in results
                pl_trainer_kwargs = {
                    'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
                    'devices'    : [0] if torch.cuda.is_available() else None,
                    'callbacks'  : {
                        'early_stopping'  : '',
                        'model_checkpoint': '' if custom_checkpoint else None
                    }
                }
                
                # Save MAPE results
                mape_results = {**{f'MAPE_{k}': v for k, v in mape_cv.items()}}

                # Initialize result
                df_results = pd.DataFrame([{
                    'timestamp'         : datetime.now(),
                    'MAPE_sum'          : sum(mape_cv.values()),
                    **mape_results,
                    **params_record,
                    'early_stopping'    : json.dumps(early_stopping_config),
                    'checkpoint_config' : json.dumps(checkpoint_config) if custom_checkpoint else 'Default',
                    'trainer_config'    : json.dumps(pl_trainer_kwargs),
                }])

                # Store results to existing record
                df_results = pd.concat([existing_df, df_results], ignore_index=True)

                # Save path optionally
                if save_path:
                    df_results.to_excel(save_path, index=False)
                    print(f'✅ Results saved to {save_path}')

                # Clean up memory
                cleanup(model, cv_test, existing_df, df_results)

                # Clean up disk
                delete_worst_model(
                    work_dir   = 'models/checkpoint_tuning_nbeats',
                    excel_path = 'reports/nbeats_params_results.xlsx',
                    print_all  = False,
                    patience   = 1.1
                )
                
                print('\n↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')

            except Exception as e:
                # delete_worst_model(
                #     work_dir   = 'models/checkpoint_tuning_nbeats',
                #     excel_path = 'reports/nbeats_params_results.xlsx',
                #     print_all  = False,
                #     patience   = 1.4
                # )
                print(f'⚠️ Warning: Failed to read {save_path} — {e}')
                print('\n↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    return None
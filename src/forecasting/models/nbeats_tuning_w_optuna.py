from src.forecasting.utils.libraries_data_handling import np, pd, math
from src.forecasting.utils.libraries_others import os, re, time, json, psutil, shutil, datetime
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import torch, concatenate, TimeSeries, Scaler, NBEATSModel, Callback, EarlyStopping, ModelCheckpoint, optuna, PyTorchLightningPruningCallback, Trial, plot_contour, plot_optimization_history, plot_param_importances, GaussianLikelihood, TrialPruned, ParameterSampler, MeanAbsolutePercentageError, mean_absolute_percentage_error
from src.forecasting.utils.extract_checkpoint_result import extract_checkpoint_results
from src.forecasting.utils.memory import cleanup
from src.forecasting.constants.enums import PeriodList
from src.forecasting.models.empty_worst_model import empty_worst_model
from src.forecasting.models.evaluate_cv_timeseries import evaluate_cv_timeseries
from src.forecasting.models.nbeats_build_w_optuna import nbeats_build_w_optuna
from src.forecasting.models.nbeats_store_to_excel import nbeats_store_to_excel


def nbeats_tuning_w_optuna(
    dataset_type      : str,
    Y_train           : TimeSeries,
    X_train           : TimeSeries,
    Y_valid           : TimeSeries,
    X_valid           : TimeSeries,
    Y_scalers         : dict,
    X_scalers         : dict,
    Y_actual          : pd.DataFrame,
    validation_split  : float,
    max_epochs        : int,
    Y_col_list        : list,
    X_col_list        : list,
    custom_checkpoint : bool,
    save_path         : str,
    work_dir          : str,
    trial             : Trial
) -> float: 
    """
    Function hyperparameter tuning for N-BEATS using random search (parameter sampler) and rolling forecast evaluation.

    Args:
        dataset_type (str)                  : Type of dataset used (i.e. sqrt, sqrt_NoOzon, log1p, log1p_NoOzon, or without optimized dataset, default).
        Y_train (TimeSeries)                : Train targeted series.
        X_train (TimeSeries)                : Train covariates series.
        Y_valid (TimeSeries)                : Validation targeted series.
        X_valid (TimeSeries)                : Validation covariates series.
        Y_scalers (dict)                    : List of scaler of each targeted series.
        X_scalers (dict)                    : List of scaler of each covariates series.
        Y_actual (pd.DataFrame)             : Actual targeted data to compare.
        validation_split (float)            : Validation data size.
        max_epochs (int)                    : Max training epochs.
        Y_col_list (list)                   : List of targeted columns used.
        X_col_list (list)                   : List of covariates columns used.
        custom_checkpoint (bool)            : Whether to load default checkpoint or custom checkpoint.
        save_path (str)                     : Path location to save tuning results as xlsx.
        work_dir (str)                      : Working directory to store tuning folder.
        trial (Trial)                       : An Optuna class object.

    Returns:
        float: This function return MAPE_sum (sum MAPE of 6 target variables) score, used for Optuna optimization.
    """

    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print('↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓\n')

    # Setup for parameter tuning
    # Set encoder options #first
    encoder_options = {
        "enc0": None,
    
        # Add hour and day (future only)
        "enc1": {
            'cyclic': {'past': ['hour']},
            'datetime_attribute': {'past': ['hour']},
            'position': {'past': ['relative']},
            'tz': 'Asia/Jakarta'
        },
    
        # Add hour and day (past + future)
        "enc2": {
            'cyclic': {'past': ['hour', 'dayofweek']},
            'datetime_attribute': {'past': ['hour', 'dayofweek']},
            'position': {'past': ['relative']},
            'tz': 'Asia/Jakarta'
        },
    
        # Add month (future only)
        "enc3": {
            'cyclic': {'past': ['hour', 'dayofweek', 'weekday']},
            'datetime_attribute': {'past': ['hour', 'dayofweek', 'weekday']},
            'position': {'past': ['relative']},
            'tz': 'Asia/Jakarta'
        },
    
        # Add month (past + future)
        "enc4": {
            'cyclic': {'past': ['hour', 'dayofweek', 'weekday', 'month']},
            'datetime_attribute': {'past': ['hour', 'dayofweek', 'weekday', 'month']},
            'position': {'past': ['relative']},
            'tz': 'Asia/Jakarta'
        },
    }
    input_chunk_length  = trial.suggest_categorical('input_chunk_length', [
        int(PeriodList.D1), int(PeriodList.D1 * 2), int(PeriodList.D1 * 3), int(PeriodList.D1 * 4), int(PeriodList.D1 * 5), int(PeriodList.D1 * 6), int(PeriodList.W1),
        int(PeriodList.D1 * 8), int(PeriodList.D1 * 9), int(PeriodList.D1 * 10), int(PeriodList.D1 * 11), int(PeriodList.D1 * 12), int(PeriodList.D1 * 13), int(PeriodList.W1 * 2)
    ])
    output_chunk_length = trial.suggest_categorical('output_chunk_length', [24])  # CHANGE THIS FOR EXPERIMENTS 12 or 24
    batch_size          = trial.suggest_categorical('batch_size', [32, 64, 96])
    num_stacks          = trial.suggest_int('num_stacks', 5, 30)
    num_blocks          = trial.suggest_categorical('num_blocks', [1, 2, 3, 4])
    num_layers          = trial.suggest_categorical('num_layers', [2, 4])
    layer_widths        = trial.suggest_categorical('layer_widths', [64, 128, 256, 512])
    dropout             = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    lr                  = trial.suggest_categorical('lr', [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001])
    enc_key             = trial.suggest_categorical('add_encoders', list(encoder_options.keys()))
    add_encoders        = encoder_options[enc_key]

    
    print(
        f'🔃 Tuning N-BEATS with:\n'
        f'\dataset type : {dataset_type}\n'
        f'\tinput_chunk_length : {input_chunk_length}\n'
        f'\toutput_chunk_length : {output_chunk_length}\n'
        f'\tbatch_size : {batch_size}\n'
        f'\tnum_stacks : {num_stacks}\n'
        f'\tnum_blocks : {num_blocks}\n'
        f'\tnum_layers : {num_layers}\n'
        f'\tlayer_widths : {layer_widths}\n'
        f'\tdropout : {dropout}\n'
        f'\tlr : {lr}\n'
        f'\tstride : {output_chunk_length}\n'
        f'\validation_size : {validation_split}\n'
        f'\tY_columns_used : {Y_col_list}\n'
        f'\tX_columns_used : {X_col_list}\n'
        f'\tadd_encoders : {add_encoders}\n'
    )

    # Generate model name and work dir
    model_name = (
        f'optuna_nbeats_type{dataset_type}_ic{input_chunk_length}_oc{output_chunk_length}_bs{batch_size}'
        f'_st{num_stacks}_bl{num_blocks}_ly{num_layers}'
        f'_wd{layer_widths}_dp{dropout}_lr{lr}_encoders{add_encoders}_stride{output_chunk_length}'
        f'_vl{validation_split}_Ycol{len(Y_col_list)}_Xcol{len(X_col_list)}_monitorMAPE'
    )
    folder_path = os.path.join(work_dir, model_name)

    # Initialize some variable for storing to excel
    random_state = 1502
    GPU = False
    if torch.cuda.is_available():
        GPU = True

    # Check if excel file is exist
    if save_path and os.path.exists(save_path):
         print(f'ℹ️ Excel already exists at {save_path}, skipping creation.\n')
    else:
        # Create new excel file
        columns = [
            'timestamp', 'MAPE_sum', 'MAPE_y1','MAPE_y2', 'MAPE_y3', 'MAPE_y4', 'MAPE_y6', 
            'val_MAPE', 'val_loss', 'status', 'model_name', 'GPU', 'ram_usage_MB', 
            'fit_cost_seconds', 'dataset_type', 'input_chunk_length', 'output_chunk_length',
            'n_epochs', 'batch_size', 'num_stacks', 'num_blocks', 'num_layers',
            'layer_widths', 'dropout', 'lr', 'random_state', 'validation_split',
            'stride', 'Y_cols', 'X_cols', 'Y_scalers', 'X_scalers', 'add_encoders',
            'early_stopping', 'checkpoint_config', 'trainer_config'
        ]
        df_empty = pd.DataFrame(columns=columns)
        df_empty.to_excel(save_path, index=False)
        print(f'✅ Empty Excel file created with headers at {save_path}')

        # Clean up memory
        cleanup(df_empty)
    
    try:
        existing_df = pd.read_excel(save_path)

        # If model_name already trained, skip fit the model
        # if (model_name in existing_df.get("model_name", [])) or os.path.exists(folder_path):
        if "model_name" in existing_df.columns and model_name in existing_df["model_name"].values:
            print(f'⚠️ Skipping {model_name} — already trained.')

            # Take MAPE_sum from excel
            old_score = existing_df.loc[existing_df['model_name'] == model_name, 'MAPE_sum'].values

            # Check if MAPE_sum is exist
            if len(old_score) > 0 and not pd.isna(old_score[0]):
                print(f'✅ Old MAPE_sum: {old_score[0]}')
                
                # Clean up memory
                cleanup(existing_df)
                
                print('\n↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                return float(old_score[0])
            else:
                print(f'⚠️ Old MAPE_sum is empty')

                # Clean up memory
                cleanup(existing_df)
                
                print('\n↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                return float('inf')
        else:
            
            # Handling model that can be hungry of RAM
            estimate_trainable_params = (input_chunk_length * num_stacks * num_blocks * num_layers * layer_widths)

            # Avoid huge model
            if estimate_trainable_params > 22000000:
                print(f'⚠️ Skipping {model_name}. Model can be hungry of RAM.')
                print('!! Saving to excel instead ....')
                
                # Store BIG params combination that can trigger OOM to xlsx
                nbeats_store_to_excel(
                    model_name          = model_name,
                    work_dir            = work_dir,
                    GPU                 = GPU,
                    dataset_type        = dataset_type,
                    input_chunk_length  = input_chunk_length,
                    output_chunk_length = output_chunk_length,
                    batch_size          = batch_size,
                    num_stacks          = num_stacks,
                    num_blocks          = num_blocks,
                    num_layers          = num_layers,
                    layer_widths        = layer_widths,
                    dropout             = dropout,
                    lr                  = lr,
                    random_state        = random_state,
                    validation_split    = validation_split,
                    Y_cols              = Y_col_list,
                    X_cols              = X_col_list,
                    Y_scalers           = Y_scalers,
                    X_scalers           = X_scalers,
                    add_encoders        = add_encoders,
                    custom_checkpoint   = custom_checkpoint,
                    status              = 'OOM SKIPPED',
                    existing_df         = existing_df,
                    save_path           = save_path,
                )

                # Clean up disk
                empty_worst_model(
                    work_dir   = 'models/checkpoint_tuning_nbeats',
                    excel_path = save_path,
                    print_all  = False,
                    patience   = 0.0
                )

                # Clean up memory
                cleanup(existing_df)

                print('\n↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                return float('inf')

            # If the model is not huge, try to fit it.
            try:
                # Fit model
                start_time = time.time()
            
                model = nbeats_build_w_optuna(
                    Y_train             = Y_train,
                    X_train             = X_train,
                    Y_valid             = Y_valid,
                    X_valid             = X_valid,
                    input_chunk_length  = input_chunk_length,
                    output_chunk_length = output_chunk_length,
                    n_epochs            = max_epochs,
                    batch_size          = batch_size,
                    num_stacks          = num_stacks,
                    num_blocks          = num_blocks,
                    num_layers          = num_layers,
                    layer_widths        = layer_widths,
                    dropout             = dropout,
                    add_encoders        = add_encoders,
                    model_name          = model_name,
                    work_dir            = work_dir,
                    include_stopper     = True,
                    custom_checkpoint   = custom_checkpoint,
                    lr                  = lr,
                    use_pruner          = True,
                    trial               = trial
                )
            
                cost_time = time.time() - start_time
                
                if torch.cuda.is_available():
                    gpu_id       = 0
                    ram_usage_MB = torch.cuda.memory_allocated(gpu_id) / (1024**2)
                else:
                    process      = psutil.Process(os.getpid())
                    ram_usage_MB = process.memory_info().rss / (1024 ** 2)
            
                print(f'\n✅ N-BEATS Fit cost: {cost_time:.2f} seconds')
                print(f'🧠 RAM used after training: {ram_usage_MB:.2f} MB\n')
            
                # Cross Validation with Rolling Forecast
                cv_test = model.historical_forecasts(
                    series           = Y_train.append(Y_valid),
                    past_covariates  = X_train.append(X_valid),
                    start            = Y_train.get_timestamp_at_point(24),
                    forecast_horizon = output_chunk_length,
                    stride           = output_chunk_length,
                    retrain          = False,
                    last_points_only = False,
                )
            
                # Evaluate
                mape_cv = evaluate_cv_timeseries(
                    forecasts  = cv_test,
                    scaler     = Y_scalers,
                    df_actual  = Y_actual,
                )
                    
                # Save MAPE results
                MAPE_sum     = sum(mape_cv.values())
                mape_results = {**{f'MAPE_{k}': v for k, v in mape_cv.items()}}
            
                print(f'\n💹 MAPE_sum : {MAPE_sum}')
                print(f'🧠 MAPE CV: {mape_cv}\n')

                # Extract checkpoints model results
                best_epoch, best_val_mape, best_val_loss = extract_checkpoint_results(
                    work_dir     = work_dir,
                    model_name   = model_name,
                    custom_model = custom_checkpoint
                )
                print(f'✅ Best epoch: {best_epoch}. Best val_MAPE: {best_val_mape}. Best val_loss: {best_val_loss}')
                
                # Store params to xlsx
                nbeats_store_to_excel(
                    model_name          = model_name,
                    work_dir            = work_dir,
                    GPU                 = GPU,
                    dataset_type        = dataset_type,
                    input_chunk_length  = input_chunk_length,
                    output_chunk_length = output_chunk_length,
                    batch_size          = batch_size,
                    num_stacks          = num_stacks,
                    num_blocks          = num_blocks,
                    num_layers          = num_layers,
                    layer_widths        = layer_widths,
                    dropout             = dropout,
                    lr                  = lr,
                    random_state        = random_state,
                    validation_split    = validation_split,
                    Y_cols              = Y_col_list,
                    X_cols              = X_col_list,
                    Y_scalers           = Y_scalers,
                    X_scalers           = X_scalers,
                    add_encoders        = add_encoders,
                    custom_checkpoint   = custom_checkpoint,
                    status              = 'SUCCESS',
                    existing_df         = existing_df,
                    save_path           = save_path,
                    ram_usage_MB        = ram_usage_MB,
                    fit_cost_seconds    = cost_time,
                    best_epoch          = best_epoch,
                    best_val_mape       = best_val_mape,
                    best_val_loss       = best_val_loss,
                    MAPE_sum            = MAPE_sum,
                    mape_results        = mape_results
                )
            
                # Clean up disk
                empty_worst_model(
                    work_dir   = 'models/checkpoint_tuning_nbeats',
                    excel_path = save_path,
                    print_all  = False,
                    patience   = 0.0
                )
                print('\n↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                
                # Clean up memory
                cleanup(model, cv_test, existing_df)
                return MAPE_sum

            # Handling if model pruned by Optuna
            except TrialPruned:
                print(f'⚠️ Trial {trial.number} pruned.')
                print('!! Saving to excel....')
                
                # Store pruned params to xlsx
                nbeats_store_to_excel(
                    model_name          = model_name,
                    work_dir            = work_dir,
                    GPU                 = GPU,
                    dataset_type        = dataset_type,
                    input_chunk_length  = input_chunk_length,
                    output_chunk_length = output_chunk_length,
                    batch_size          = batch_size,
                    num_stacks          = num_stacks,
                    num_blocks          = num_blocks,
                    num_layers          = num_layers,
                    layer_widths        = layer_widths,
                    dropout             = dropout,
                    lr                  = lr,
                    random_state        = random_state,
                    validation_split    = validation_split,
                    Y_cols              = Y_col_list,
                    X_cols              = X_col_list,
                    Y_scalers           = Y_scalers,
                    X_scalers           = X_scalers,
                    add_encoders        = add_encoders,
                    custom_checkpoint   = custom_checkpoint,
                    status              = 'PRUNED',
                    existing_df         = existing_df,
                    save_path           = save_path,
                )

                # Clean up disk
                empty_worst_model(
                    work_dir   = 'models/checkpoint_tuning_nbeats',
                    excel_path = save_path,
                    print_all  = False,
                    patience   = 0.0
                )

                # Clean up memory
                cleanup(existing_df)

                print('↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                raise
            
    except Exception as e:
        print(f"⚠️ Error reading Excel: {e}\n")
        print('↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
        print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
        raise
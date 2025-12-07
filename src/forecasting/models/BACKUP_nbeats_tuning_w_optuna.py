from src.forecasting.utils.libraries_data_handling import np, pd, math
from src.forecasting.utils.libraries_others import os, re, time, json, psutil, shutil, datetime
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import torch, concatenate, TimeSeries, Scaler, NBEATSModel, Callback, EarlyStopping, ModelCheckpoint, optuna, PyTorchLightningPruningCallback, Trial, plot_contour, plot_optimization_history, plot_param_importances, GaussianLikelihood, TrialPruned, ParameterSampler, MeanAbsolutePercentageError, mean_absolute_percentage_error
from src.forecasting.utils.extract_best_epochs import extract_best_epoch_from_checkpoint
from src.forecasting.utils.extract_checkpoint_result import extract_checkpoint_results
from src.forecasting.utils.memory import cleanup
from src.forecasting.constants.enums import PeriodList
from src.forecasting.models.empty_worst_model import empty_worst_mdoel
from src.forecasting.models.evaluate_cv_timeseries import evaluate_cv_timeseries
from src.forecasting.models.built_nbeats_w_optuna import built_nbeats_w_optuna

def nbeats_tuning_w_optuna(
    Y                 : TimeSeries,
    X                 : TimeSeries,
    Y_actual          : pd.DataFrame,
    Y_scaler          : Scaler,
    pre_normalization : bool,
    max_epochs        : int,
    col_list          : list,
    col_is_one_hot    : bool,
    custom_checkpoint : bool,
    save_path         : str,
    trial             : Trial
) -> float: 
    """
    Function hyperparameter tuning for N-BEATS using random search (parameter sampler) and rolling forecast evaluation.

    Args:
        Y (TimeSeries)                      : Target series.
        X (TimeSeries)                      : Past Covariates.
        Y_actual (pd.DataFrame)             : Actual targeted data to compare.
        Y_scaler (Scaler)                   : Targetted scaler to transform/inverse.
        pre_normalization (bool)            : To store in the results which data is used.
        max_epochs (int)                    : Max training epochs.
        # params_grid (dict[str, np.ndarray]) : List of hyperparameter sample form.
        n_iter (int)                        : Number of random hyperparameter sample form to evaluate.
        col_list (list)                     : List of numeric covariates used to train.
        col_is_one_hot (bool)               : Whether use categoric covariates as ordinal or one hot encoding. 
        custom_checkpoint (bool)            : Whether to load default checkpoint or custom checkpoint.
        save_path (str)                     : Path location to save tuning results as xlsx or not.
        trial (Trial)                       : An Optuna class object.

    Returns:
        float: This function return MAPE_sum (sum MAPE of 6 target variables) score for Optuna optimization.
    """

    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print('↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓\n')
    
    # Setup for parameter tuning
    input_chunk_length  = trial.suggest_categorical('input_chunk_length', [
        int(PeriodList.D1), int(PeriodList.D1 * 2), int(PeriodList.D1 * 3), int(PeriodList.D1 * 4), int(PeriodList.D1 * 5), int(PeriodList.D1 * 6), int(PeriodList.W1),
        int(PeriodList.D1 * 8), int(PeriodList.D1 * 9), int(PeriodList.D1 * 10), int(PeriodList.D1 * 11), int(PeriodList.D1 * 12), int(PeriodList.D1 * 13), int(PeriodList.W1 * 2)
    ])
    output_chunk_length = trial.suggest_categorical('output_chunk_length', [12, 24])
    batch_size          = trial.suggest_categorical('batch_size', [32, 64, 96])
    num_stacks          = trial.suggest_int('num_stacks', 5, 30)
    num_blocks          = trial.suggest_categorical('num_blocks', [1, 2, 3, 4])
    num_layers          = trial.suggest_categorical('num_layers', [2, 4])
    layer_widths        = trial.suggest_categorical('layer_widths', [64, 128, 256, 512])
    dropout             = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    lr                  = trial.suggest_categorical('lr', [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5], log=True)
    add_encoders        = trial.suggest_categorical('add_encoders', [True, False]) 
    validation_split    = trial.suggest_categorical('validation_split', [0.1, 0.15, 0.2, 0.25, 0.3])
    
    print(
        f'🔃 Tuning N-BEATS using Optuna with params:\n'
        f'\tinput_chunk_length : {input_chunk_length}\n'
        f'\toutput_chunk_length : {output_chunk_length}\n'
        f'\tbatch_size : {batch_size}\n'
        f'\tnum_stacks : {num_stacks}\n'
        f'\tnum_blocks : {num_blocks}\n'
        f'\tnum_layers : {num_layers}\n'
        f'\tlayer_widths : {layer_widths}\n'
        f'\tdropout : {dropout}\n'
        f'\tlr : {lr}\n'
        f'\tadd_encoders : {add_encoders}\n'
        f'\tstride : {output_chunk_length}\n'
        f'\tColumns used : {col_list}\n'
        f'\tCategoric is one hot encoding: {col_is_one_hot}\n'
    )

    # Generate model name and work dir
    model_name = (
        f'optuna_nbeats_ic{input_chunk_length}_oc{output_chunk_length}_bs{batch_size}'
        f'_st{num_stacks}_bl{num_blocks}_ly{num_layers}'
        f'_wd{layer_widths}_dp{dropout}_lr{lr}_encoders{add_encoders}_stride{output_chunk_length}'
        f'_vl{validation_split}_cov{len(col_list)}_onehot{col_is_one_hot}_monitorMAPE'
    )
    work_dir = 'models/checkpoint_tuning_nbeats/'
    folder_path = os.path.join(work_dir, model_name)

    # Check if excel file is exist
    if save_path and os.path.exists(save_path):
        try:
            existing_df = pd.read_excel(save_path)
    
            # If model_name already trained, skip fit the model
            if (model_name in existing_df.get("model_name", [])) or os.path.exists(folder_path):
                print(f'⚠️ Skipping {model_name} — already trained.')

                # Take MAPE_sum from excel
                old_score = existing_df.loc[existing_df['model_name'] == model_name, 'MAPE_sum'].values

                # Check if MAPE_sum is exist
                if len(old_score) > 0:
                    print(f'✅ Old MAPE_sum: {old_score[0]}\n')
                    print('↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                    return float(old_score[0])
                else:
                    print(f'⚠️ Old MAPE_sum is empty\n')
                    print('↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                    return float('inf')
            else:
                
                # Handling model that can be hungry of RAM
                estimate_trainable_params = (input_chunk_length * num_stacks * num_blocks * num_layers * layer_widths)

                # Avoid huge model
                if estimate_trainable_params > 22000000:
                    print(f'⚠️ Skipping {model_name}. Model can be hungry of RAM.')
                    print('!! Saving to excel instead ....')
                    
                    # Store pruned params to xlsx
                    params_record = {
                        'model_name'         : model_name,
                        'GPU'                : True if torch.cuda.is_available() else False,
                        'pre-normalization'  : pre_normalization,
                        'input_chunk_length' : input_chunk_length,
                        'output_chunk_length': output_chunk_length,
                        'batch_size'         : batch_size,
                        'num_stacks'         : num_stacks,
                        'num_blocks'         : num_blocks,
                        'num_layers'         : num_layers,
                        'layer_widths'       : layer_widths,
                        'dropout'            : dropout,
                        'lr'                 : lr,
                        'random_state'       : 1502,
                        'validation_split'   : 0.2,
                        'stride'             : output_chunk_length,
                        'covariates'         : json.dumps(col_list),
                        'one_hot_encoding'   : col_is_one_hot,
                        'add_encoders'       : json.dumps({'cyclic': {'past': ['hour', 'day','dayofweek', 'dayofyear', 'week', 'weekday', 'weekofyear', 'month', 'quarter']}}) if add_encoders else None
                    }
                
                    # EarlyStopping config to store in results
                    early_stopping_config = {
                        'monitor'  : 'val_MeanAbsolutePercentageError',
                        'patience' : 8,
                        'min_delta': 0.01,
                        'mode'     : 'min'
                    }
                
                    # ModelCheckpoint config to store in results
                    checkpoints = 'checkpoints'
                    if custom_checkpoint:
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
                
                    # Initialize result
                    df_results = pd.DataFrame([{
                        'timestamp'         : datetime.now(),
                        'status'            : 'OOM SKIPPED',
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
                        print(f'✅ Results saved to {save_path}\n')

                    # Clean up disk
                    empty_worst_model(
                        work_dir   = 'models/checkpoint_tuning_nbeats',
                        excel_path = 'reports/nbeats_params_results.xlsx',
                        print_all  = False,
                        patience   = 1.0
                    )

                    # Clean up memory
                    cleanup(existing_df, df_results)

                    print('↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                    return float('inf')

                # If the model is not huge, try to fit it.
                try:
                    # Fit model
                    start_time = time.time()
                
                    model = build_nbeats_w_optuna(
                        Y                   = Y,
                        X                   = X,
                        input_chunk_length  = input_chunk_length,
                        output_chunk_length = output_chunk_length,
                        n_epochs            = max_epochs,
                        batch_size          = batch_size,
                        num_stacks          = num_stacks,
                        num_blocks          = num_blocks,
                        num_layers          = num_layers,
                        layer_widths        = layer_widths,
                        dropout             = dropout,
                        include_encoders    = add_encoders,
                        validation_split    = validation_split,
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
                        ram_usage_mb = torch.cuda.memory_allocated(gpu_id) / (1024**2)
                    else:
                        process      = psutil.Process(os.getpid())
                        ram_usage_mb = process.memory_info().rss / (1024 ** 2)
                
                    print(f'\n✅ N-BEATS Fit cost: {cost_time:.2f} seconds')
                    print(f'🧠 RAM used after training: {ram_usage_mb:.2f} MB\n')
                
                    # Cross Validation with Rolling Forecast
                    cv_test = model.historical_forecasts(
                        series           = Y,
                        past_covariates  = X,
                        start            = Y.start_time(),
                        forecast_horizon = output_chunk_length,
                        stride           = output_chunk_length,
                        retrain          = False,
                        last_points_only = False,
                    )
                
                    # Evaluate
                    mape_cv = evaluate_cv_timeseries(
                        forecasts  = cv_test,
                        scaler     = Y_scaler,
                        df_actual  = Y_actual,
                    )
                        
                    # Save MAPE results
                    MAPE_sum = sum(mape_cv.values())
                
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
                    params_record = {
                        'model_name'         : model_name,
                        'GPU'                : True if torch.cuda.is_available() else False,
                        'ram_usage_MB'       : round(ram_usage_mb, 2),
                        'fit_cost_seconds'   : round(cost_time, 2),
                        'pre-normalization'  : pre_normalization,
                        'input_chunk_length' : model.input_chunk_length,
                        'output_chunk_length': model.output_chunk_length,
                        'n_epochs'           : best_epoch,
                        'batch_size'         : model.batch_size,
                        'num_stacks'         : model.num_stacks,
                        'num_blocks'         : model.num_blocks,
                        'num_layers'         : model.num_layers,
                        'layer_widths'       : model.layer_widths[0],
                        'dropout'            : model.dropout,
                        'lr'                 : lr,
                        'random_state'       : 1502,
                        'validation_split'   : 0.2,
                        'stride'             : model.output_chunk_length,
                        'covariates'         : json.dumps(col_list),
                        'one_hot_encoding'   : col_is_one_hot,
                        'add_encoders'       : json.dumps({'cyclic': {'past': ['hour', 'day','dayofweek', 'dayofyear', 'week', 'weekday', 'weekofyear', 'month', 'quarter']}}) if add_encoders else None
                    }
                
                    # EarlyStopping config to store in results
                    early_stopping_config = {
                        'monitor'  : 'val_MeanAbsolutePercentageError',
                        'patience' : 8,
                        'min_delta': 0.01,
                        'mode'     : 'min'
                    }
                
                    # ModelCheckpoint config to store in results
                    checkpoints = 'checkpoints'
                    if custom_checkpoint:
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
                        'MAPE_sum'          : MAPE_sum,
                        **mape_results,
                        'val_MAPE'          : best_val_mape,
                        'val_loss'          : best_val_loss,
                        'status'            : 'SUCCESS',
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
                        print(f'✅ Results saved to {save_path}\n')
                
                    # Clean up disk
                    empty_worst_model(
                        work_dir   = 'models/checkpoint_tuning_nbeats',
                        excel_path = 'reports/nbeats_params_results.xlsx',
                        print_all  = False,
                        patience   = 1.0
                    )
                    print('↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                    
                    # Clean up memory
                    cleanup(model, cv_test, existing_df, df_results)
                    return MAPE_sum

                # Handling if model pruned by Optuna
                except TrialPruned:
                    print(f'⚠️ Trial {trial.number} pruned.')
                    print('!! Saving to excel....')
                    
                    # Store pruned params to xlsx
                    params_record = {
                        'model_name'         : model_name,
                        'GPU'                : True if torch.cuda.is_available() else False,
                        'pre-normalization'  : pre_normalization,
                        'input_chunk_length' : input_chunk_length,
                        'output_chunk_length': output_chunk_length,
                        'batch_size'         : batch_size,
                        'num_stacks'         : num_stacks,
                        'num_blocks'         : num_blocks,
                        'num_layers'         : num_layers,
                        'layer_widths'       : layer_widths,
                        'dropout'            : dropout,
                        'lr'                 : lr,
                        'random_state'       : 1502,
                        'validation_split'   : 0.2,
                        'stride'             : output_chunk_length,
                        'covariates'         : json.dumps(col_list),
                        'one_hot_encoding'   : col_is_one_hot,
                        'add_encoders'       : json.dumps({'cyclic': {'past': ['hour', 'day','dayofweek', 'dayofyear', 'week', 'weekday', 'weekofyear', 'month', 'quarter']}}) if add_encoders else None
                    }
                
                    # EarlyStopping config to store in results
                    early_stopping_config = {
                        'monitor'  : 'val_MeanAbsolutePercentageError',
                        'patience' : 8,
                        'min_delta': 0.01,
                        'mode'     : 'min'
                    }
                
                    # ModelCheckpoint config to store in results
                    checkpoints = 'checkpoints'
                    if custom_checkpoint:
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
                
                    # Initialize result
                    df_results = pd.DataFrame([{
                        'timestamp'         : datetime.now(),
                        'status'            : 'PRUNED',
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
                        print(f'✅ Results saved to {save_path}\n')

                    # Clean up disk
                    empty_worst_model(
                        work_dir   = 'models/checkpoint_tuning_nbeats',
                        excel_path = 'reports/nbeats_params_results.xlsx',
                        print_all  = False,
                        patience   = 1.0
                    )

                    # Clean up memory
                    cleanup(existing_df, df_results)

                    print('↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
                    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
                    raise
                
        except Exception as e:
            print(f"⚠️ Error reading Excel: {e}\n")
            print('↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑')
            print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
            existing_df = None
    
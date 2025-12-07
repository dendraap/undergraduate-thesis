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
        n_iter (int)                        : Number of random hyperparameter sample form to evaluate.
        col_list (list)                     : List of numeric covariates used to train.
        col_is_one_hot (bool)               : Whether use categoric covariates as ordinal or one hot encoding. 
        custom_checkpoint (bool)            : Whether to load default checkpoint or custom checkpoint.
        save_path (str)                     : Path location to save tuning results as xlsx or not.
        trial (Trial)                       : An Optuna class object.

    Returns:
        float: This function return MAPE_sum (sum MAPE of 6 target variables) score, used for Optuna optimization.
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
    lr                  = trial.suggest_categorical('lr', [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001])
    add_encoders        = trial.suggest_categorical('add_encoders', [True, False]) 
    validation_split    = trial.suggest_categorical('validation_split', [0.1, 0.15, 0.2, 0.25, 0.3])

    # Below is for DEBUGGING purpose
    # input_chunk_length  = trial.suggest_categorical('input_chunk_length', [
    #     int(PeriodList.D1), int(PeriodList.D1 * 2), int(PeriodList.D1 * 3), int(PeriodList.D1 * 4), int(PeriodList.D1 * 5), int(PeriodList.D1 * 6), int(PeriodList.W1),
    #     int(PeriodList.D1 * 8)
    # ])
    # output_chunk_length = trial.suggest_categorical('output_chunk_length', [12])
    # batch_size          = trial.suggest_categorical('batch_size', [96])
    # num_stacks          = trial.suggest_categorical('num_stacks', [5])
    # num_blocks          = trial.suggest_categorical('num_blocks', [1])
    # num_layers          = trial.suggest_categorical('num_layers', [2])
    # layer_widths        = trial.suggest_categorical('layer_widths', [256])
    # dropout             = trial.suggest_categorical('dropout', [0.1])
    # lr                  = trial.suggest_categorical('lr', [0.001])
    # add_encoders        = trial.suggest_categorical('add_encoders', [False]) 
    # validation_split    = trial.suggest_categorical('validation_split', [0.2])
    
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
        f'\tvalidation_split : {validation_split}\n'
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

    # Initialize some variable for storing to excel
    random_state = 1502
    GPU = False
    if torch.cuda.is_available():
        GPU = True

    # Check if excel file is exist
    if save_path and os.path.exists(save_path):
        try:
            existing_df = pd.read_excel(save_path)
    
            # If model_name already trained, skip fit the model
            # if (model_name in existing_df.get("model_name", [])) or os.path.exists(folder_path):
            if "model_name" in existing_df.columns and model_name in existing_df["model_name"].values:
                print(f'⚠️ Skipping {model_name} — already trained.')

                # Take MAPE_sum from excel
                old_score = existing_df.loc[existing_df['model_name'] == model_name, 'MAPE_sum'].values

                # Check if MAPE_sum is exist
                if len(old_score) > 0:
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
                        pre_normalization   = pre_normalization,
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
                        col_list            = col_list,
                        col_is_one_hot      = col_is_one_hot,
                        add_encoders        = add_encoders,
                        custom_checkpoint   = custom_checkpoint,
                        status              = 'OOM SKIPPED',
                        existing_df         = existing_df,
                        save_path           = save_path,
                    )

                    # Clean up disk
                    empty_worst_model(
                        work_dir   = 'models/checkpoint_tuning_nbeats',
                        excel_path = 'reports/nbeats_params_results.xlsx',
                        print_all  = False,
                        patience   = 0.95
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
                        ram_usage_MB = torch.cuda.memory_allocated(gpu_id) / (1024**2)
                    else:
                        process      = psutil.Process(os.getpid())
                        ram_usage_MB = process.memory_info().rss / (1024 ** 2)
                
                    print(f'\n✅ N-BEATS Fit cost: {cost_time:.2f} seconds')
                    print(f'🧠 RAM used after training: {ram_usage_MB:.2f} MB\n')
                
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
                        pre_normalization   = pre_normalization,
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
                        col_list            = col_list,
                        col_is_one_hot      = col_is_one_hot,
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
                        excel_path = 'reports/nbeats_params_results.xlsx',
                        print_all  = False,
                        patience   = 0.95
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
                        pre_normalization   = pre_normalization,
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
                        col_list            = col_list,
                        col_is_one_hot      = col_is_one_hot,
                        add_encoders        = add_encoders,
                        custom_checkpoint   = custom_checkpoint,
                        status              = 'PRUNED',
                        existing_df         = existing_df,
                        save_path           = save_path,
                    )

                    # Clean up disk
                    empty_worst_model(
                        work_dir   = 'models/checkpoint_tuning_nbeats',
                        excel_path = 'reports/nbeats_params_results.xlsx',
                        print_all  = False,
                        patience   = 0.95
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
    else:
        # Create new excel file
        columns = [
            'timestamp', 'MAPE_sum', 'MAPE_y1','MAPE_y2', 'MAPE_y3', 'MAPE_y4', 'MAPE_y6', 
            'val_MAPE', 'val_loss', 'status', 'model_name', 'GPU', 'ram_usage_MB', 
            'fit_cost_seconds', 'pre-normalization', 'input_chunk_length', 'output_chunk_length',
            'n_epochs', 'batch_size', 'num_stacks', 'num_blocks', 'num_layers',
            'layer_widths', 'dropout', 'lr', 'random_state', 'validation_split',
            'stride', 'covariates', 'one_hot_encoding', 'add_encoders',
            'early_stopping', 'checkpoint_config', 'trainer_config'
        ]
        df_empty = pd.DataFrame(columns=columns)
        df_empty.to_excel(save_path, index=False)
        print(f'✅ Empty Excel file created with headers at {save_path}')

        # Clean up memory
        cleanup(df_empty)
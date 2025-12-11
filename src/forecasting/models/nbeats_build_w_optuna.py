from src.forecasting.utils.libraries_others import os, re
from src.forecasting.utils.data_split import timeseries_train_test_split
from src.forecasting.utils.libraries_modelling import torch, TimeSeries, NBEATSModel, Callback, EarlyStopping, ModelCheckpoint, optuna, PyTorchLightningPruningCallback, Trial, GaussianLikelihood, MeanAbsolutePercentageError
from src.forecasting.utils.memory import cleanup

# to avoid import of both lightning and pytorch_lightning
class PatchedPruningCallback(optuna.integration.PyTorchLightningPruningCallback, Callback):
    pass

def nbeats_build_w_optuna(
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
    custom_checkpoint   : bool,
    lr                  : float,
    use_pruner          : bool,
    trial               : Trial
) -> NBEATSModel: 
    """
    Function to build Fit of N-BEATS Model with Optuna Tuning Optimization.

        Args:
            Y (TimeSeries)            : Targeted variables to predict. 
            X (TimeSeries)            : Exogenous variables to predict Y.
            input_chunk_length (int)  : How many model look to predict.
            output_chunk_length (int) : How many model can produce prediction.
            batch_size (int)          : Number of data points before making update.
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
            lr (float)                : Learning rate.
            use_pruner (bool)         : Whether to use pruner collbacks for optuna or not.

        Returns:
            NBEATSModel : This function return the model configuration.
    """

    # Split
    Y_fit, Y_val = timeseries_train_test_split(Y, test_size=validation_split)
    X_fit, X_val = timeseries_train_test_split(X, test_size=validation_split)

    # Initialize TorchMetrics, used as the monitor
    torch_metrics = MeanAbsolutePercentageError()

    # Check if include encoders or not
    add_encoders = None
    if include_encoders:
        add_encoders = {
            'cyclic': {'past': ['hour', 'day','dayofweek', 'dayofyear', 'week', 'weekday', 'weekofyear', 'month', 'quarter']}
        }

    # pl_trainer_kwargs setup
    pl_trainer_kwargs = {}
    if torch.cuda.is_available():
        pl_trainer_kwargs['accelerator'] = 'gpu'
        pl_trainer_kwargs['devices']     = [0]
        num_workers                      = 4
    else :
        pl_trainer_kwargs['accelerator'] = 'cpu'
        pl_trainer_kwargs['devices']     = 1
        num_workers                      = 0

    callbacks     = []
    early_stopper = None
    if include_stopper:
        early_stopper = EarlyStopping(
            monitor   = 'val_MeanAbsolutePercentageError', #val_loss
            patience  = 8,
            min_delta = 0.01,
            mode      = 'min',
            verbose   = True
        )
        callbacks.append(early_stopper)
        
    # Custom model checkpoint setup
    checkpoints         = 'checkpoints'
    checkpoint_callback = None
    if custom_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath    = os.path.join(work_dir, model_name, checkpoints),
            filename   = 'MAPE-best-epoch={epoch}-val_MAPE={val_MeanAbsolutePercentageError:.4f}-val_loss={val_loss:.4f}',
            monitor    = 'val_MeanAbsolutePercentageError',
            save_top_k = 1,
            mode       = 'min',
            auto_insert_metric_name = False
        )
        callbacks.append(checkpoint_callback)

    # Optuna pruning callback
    if use_pruner:
        pruner = PatchedPruningCallback(trial, monitor='val_MeanAbsolutePercentageError')
        callbacks.append(pruner)

    if custom_checkpoint:
        pl_trainer_kwargs['enable_checkpointing'] = True
        
    pl_trainer_kwargs['callbacks'] = callbacks

    # reproducibility
    torch.manual_seed(42)

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
        optimizer_kwargs    = {'lr': lr},
        likelihood          = GaussianLikelihood(),
        random_state        = 1502,
        torch_metrics       = torch_metrics,
        pl_trainer_kwargs   = pl_trainer_kwargs,
        model_name          = model_name,
        work_dir            = work_dir,
        log_tensorboard     = True,
        save_checkpoints    = True,   # Enable Darts default checkpoint (_model.pth.tar)
        add_encoders        = add_encoders if include_encoders else None
    )

    # Fit model
    model.fit(
        series              = Y_fit,
        past_covariates     = X_fit,
        val_series          = Y_val,
        val_past_covariates = X_val,
        load_best           = True,
        stride              = 1,
        dataloader_kwargs   = {'num_workers': num_workers},
    )

    # Get best model .ckpt file. Used for custom ModelCheckoint
    ## Scan first then take to best epoch
    ckpt_dir  = os.path.join(work_dir, model_name, checkpoints)
    best_ckpt = None

    # Check if directory is exist
    if os.path.exists(ckpt_dir):
        ckpt_list = os.listdir(ckpt_dir)
        print(f'\n📂 Files in checkpoint dir: {ckpt_list}')

        pattern_custom  = r"MAPE-best-epoch=(\d+)-val_MAPE=(-?\d+(?:\.\d+)?)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"
        pattern_default = r"best-epoch=(\d+)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"
        pattern_last    = r"last-epoch=(\d+)(?:\.ckpt)?"

        # Load best model file_name based on usecase
        if custom_checkpoint :
            # Load from custom checkpoint. Iterate through each files.
            for f in ckpt_list:
                if re.search(pattern_custom, f):
                    best_ckpt = f
                    print('!! Model loaded from custom checkpoint')
                    break
        else :
            # Load from default checkpoint. Iterate through each files.
            for f in ckpt_list:
                if re.search(pattern_default, f):
                    best_ckpt = f
                    print('✅ Model loaded from default checkpoint')
                    break
        
        # To avoid function errors when best_ckpt not found
        if best_ckpt:
            model = model.load_from_checkpoint(
                model_name = model_name,
                work_dir   = work_dir,
                file_name  = best_ckpt
            )
        
        else:
            for f in ckpt_list:
                if re.search(pattern_last, f):
                    best_ckpt = f
                    print('⚠️ No best model found. Using last checkpoint')
                    break
    else:
        print('⚠️ No directory found, canceling load best model.')
        
    # Cleanup memory
    cleanup(ckpt_dir, ckpt_list, best_ckpt, callbacks, checkpoint_callback, early_stopper, add_encoders)
    return model

def nbeats_build(
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
    custom_checkpoint   : bool,
    lr                  : float
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
            lr (float)                : Learning rate.

        Returns:
            NBEATSModel : This function return the model configuration.
    """

    # Split
    Y_fit, Y_val = timeseries_train_test_split(Y, test_size=validation_split)
    X_fit, X_val = timeseries_train_test_split(X, test_size=validation_split)

    # Initialize TorchMetrics, used as the monitor
    torch_metrics = MeanAbsolutePercentageError()

    # Check if include encoders or not
    add_encoders = None
    if include_encoders:
        add_encoders = {
            'cyclic': {'past': ['hour', 'day','dayofweek', 'dayofyear', 'week', 'weekday', 'weekofyear', 'month', 'quarter']}
        }

    # pl_trainer_kwargs setup
    pl_trainer_kwargs = {}
    if torch.cuda.is_available():
        pl_trainer_kwargs['accelerator'] = 'gpu'
        pl_trainer_kwargs['devices']     = [0]
        num_workers                      = 4
    else :
        pl_trainer_kwargs['accelerator'] = 'cpu'
        pl_trainer_kwargs['devices']     = 1
        num_workers                      = 0

    callbacks     = []
    early_stopper = None
    if include_stopper:
        early_stopper = EarlyStopping(
            monitor   = 'val_MeanAbsolutePercentageError', #val_loss
            patience  = 8,
            min_delta = 0.01,
            mode      = 'min',
            verbose   = True
        )
        callbacks.append(early_stopper)
        
    # Custom model checkpoint setup
    checkpoints         = 'checkpoints'
    checkpoint_callback = None
    if custom_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath    = os.path.join(work_dir, model_name, checkpoints),
            filename   = 'MAPE-best-epoch={epoch}-val_MAPE={val_MeanAbsolutePercentageError:.4f}-val_loss={val_loss:.4f}',
            monitor    = 'val_MeanAbsolutePercentageError',
            save_top_k = 1,
            mode       = 'min',
            auto_insert_metric_name = False
        )
        callbacks.append(checkpoint_callback)

    if custom_checkpoint:
        pl_trainer_kwargs['enable_checkpointing'] = True
        
    pl_trainer_kwargs['callbacks'] = callbacks

    # reproducibility
    torch.manual_seed(42)

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
        optimizer_kwargs    = {'lr': lr},
        likelihood          = GaussianLikelihood(),
        random_state        = 1502,
        torch_metrics       = torch_metrics,
        pl_trainer_kwargs   = pl_trainer_kwargs,
        model_name          = model_name,
        work_dir            = work_dir,
        log_tensorboard     = True,
        save_checkpoints    = True,   # Enable Darts default checkpoint (_model.pth.tar)
        add_encoders        = add_encoders if include_encoders else None
    )

    # Fit model
    model.fit(
        series              = Y_fit,
        past_covariates     = X_fit,
        val_series          = Y_val,
        val_past_covariates = X_val,
        load_best           = True,
        stride              = 1,
        dataloader_kwargs   = {'num_workers': num_workers},
    )

    # Get best model .ckpt file. Used for custom ModelCheckoint
    ## Scan first then take to best epoch
    ckpt_dir  = os.path.join(work_dir, model_name, checkpoints)
    best_ckpt = None

    # Check if directory is exist
    if os.path.exists(ckpt_dir):
        ckpt_list = os.listdir(ckpt_dir)
        print(f'\n📂 Files in checkpoint dir: {ckpt_list}')

        pattern_custom  = r"MAPE-best-epoch=(\d+)-val_MAPE=(-?\d+(?:\.\d+)?)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"
        pattern_default = r"best-epoch=(\d+)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"
        pattern_last    = r"last-epoch=(\d+)(?:\.ckpt)?"

        # Load best model file_name based on usecase
        if custom_checkpoint :
            # Load from custom checkpoint. Iterate through each files.
            for f in ckpt_list:
                if re.search(pattern_custom, f):
                    best_ckpt = f
                    print('!! Model loaded from custom checkpoint')
                    break
        else :
            # Load from default checkpoint. Iterate through each files.
            for f in ckpt_list:
                if re.search(pattern_default, f):
                    best_ckpt = f
                    print('✅ Model loaded from default checkpoint')
                    break
        
        # To avoid function errors when best_ckpt not found
        if best_ckpt:
            model = model.load_from_checkpoint(
                model_name = model_name,
                work_dir   = work_dir,
                file_name  = best_ckpt
            )
        
        else:
            for f in ckpt_list:
                if re.search(pattern_last, f):
                    best_ckpt = f
                    print('⚠️ No best model found. Using last checkpoint')
                    break
    else:
        print('⚠️ No directory found, canceling load best model.')
        
    # Cleanup memory
    cleanup(ckpt_dir, ckpt_list, best_ckpt, callbacks, checkpoint_callback, early_stopper, add_encoders)
    return model
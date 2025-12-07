from src.forecasting.utils.libraries_others import Optional, Tuple, os, re

def extract_checkpoint_results(
    work_dir     : str, 
    model_name   : str,
    custom_model : bool
) -> Optional[Tuple[int, float, float]]:
    """
    This function extract the best number of epochs from saved checkpoint.

    Args:
        work_dir (str)      : Path to the root where the model checkpoint is stored.
        model_name (str)    : Subdirectory of model name that contains checkpoints.
        custom_model (bool) : To identify which model will be extracted.

    Return:
        Optional[Tuple[int, float, float]] : This function return best epoch, val_mape, and val_loss or None if checkpoints not exist.
    """


    # Merge path
    checkpoint = 'checkpoints'
    ckpt_dir   = os.path.join(work_dir, model_name, checkpoint)

    # Check if dictionary is exist
    if not os.path.exists(ckpt_dir):
        print('⚠️ Extract best model fail. Dictionary not found.')
        return 0, 0.0, 0.0

    # Get all files in checkpoints folder
    ckpt_files      = os.listdir(ckpt_dir)
    pattern_custom  = r"MAPE-best-epoch=(\d+)-val_MAPE=(-?\d+(?:\.\d+)?)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"
    pattern_default = r"best-epoch=(\d+)-val_loss=(-?\d+(?:\.\d+)?)(?:\.ckpt)?"


    # Iterate through each files
    for f in ckpt_files:

        # Find file name if using custom checkpoint
        if custom_model:
            match = re.search(pattern_custom, f)

            # Get best epoch, val_mape, and val_loss
            if match:
                try:
                    epoch    = int(match.group(1))
                    val_mape = float(match.group(2))
                    val_loss = float(match.group(3))
                    print(f'✅ Success extracting best custom checkpoint. epoch: {epoch}. val_mape: {val_mape}. val_loss: {val_loss}')
                    return epoch, val_mape, val_loss
                    
                except ValueError as e:
                    print(f'⚠️ Failed to parse checkpoint {f}: {e}')
                    return 0, 0.0, 0.0
        
        # Find file name if using default checkpoints
        else:
            match = re.search(pattern_default, f)

            # Get best epoch
            if match:
                epoch    = int(match.group(1))
                val_loss = float(match.group(2))
                print(f'✅ Success extracting best default checkpoint. epoch: {epoch}. val_loss: {val_loss}')
                return epoch, 0.0, val_loss
    return 0.0, 0.0, 0.0

def extract_best_model_checkpoint(
    work_dir          : str,
    model_name        : str,
    custom_checkpoint : bool
) -> Optional[str]:
    """
    This function extract the best number of epochs from saved checkpoint.

    Args:
        work_dir (str)           : Path to the root where the model checkpoint is stored.
        model_name (str)         : Subdirectory of model name that contains checkpoints.
        custom_checkpoint (bool) : To identify which model will be extracted.

    Return:
        Optional[str] : This function return best checkpoint filename or None if dictionary is not exist.
    """

    # Merge path
    checkpoint = 'checkpoints'
    ckpt_dir   = os.path.join(work_dir, model_name, checkpoint)
    ckpt_files = os.listdir(ckpt_dir)

    # Check if dictionary is exist
    if not os.path.exists(ckpt_dir):
        print('⚠️ Extract best model fail. Dictionary not found.')
        return None

    pattern_custom  = r"MAPE-best-epoch=.*\.ckpt"
    pattern_default = r"best-epoch=.*\.ckpt"

    if custom_checkpoint:
        best_checkpoint  = [f for f in ckpt_files if re.match(pattern_custom, f)]
        return best_checkpoint
    else:
        best_checkpoint = [f for f in ckpt_files if re.match(pattern_default, f)]
        return best_checkpoint
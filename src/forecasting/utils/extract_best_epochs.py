from src.forecasting.utils.libraries_others import Optional, os, re

def extract_best_epoch_from_checkpoint(
    work_dir     : str, 
    model_name   : str,
    custom_model : bool
) -> Optional[int]:
    """
    This function extract the best number of epochs from saved checkpoint.

    Args:
        work_dir (str)      : Path to the root where the model checkpoint is stored.
        model_name (str)    : Subdirectory of model name that contains checkpoints.
        custom_model (bool) : To identify which model will be extracted.
    """

    # Merge path
    checkpoint = 'checkpoints'
    ckpt_dir   = os.path.join(work_dir, model_name, checkpoint)
    

    # Get all files in checkpoints folder
    ckpt_files = os.listdir(ckpt_dir)

    # Iterate through each files
    for f in ckpt_files:

        # Find file name
        if custom_model:
            match = re.search(r'MAPE-best-epoch=(\d+)', f)
        else:
            match = re.search(r'best-epoch=(\d+)', f)

        # Get the number of epochs
        if match:
            return int(match.group(1))
    return None
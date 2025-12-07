from src.forecasting.utils.libraries_others import os, shutil

def delete_model_folder(
    work_dir   : str,
    model_name : str
) -> None:
    """
    Function to delete model for simplify delete files rather than from terminal.
    Args:
        work_dir (str)   : Path to main folder that contain many folders of model_name.
        model_name (str) : Path to model_name folder that contain many folders and file inside.

    Returns:
        None: This function does not return anything.
    """
    for model in model_name:
        folder_path = os.path.join(work_dir, model)
        if os.path.exists(folder_path):
            print(f'❌ Deleting entire folder: {folder_path}')
            shutil.rmtree(folder_path)
        else:
            print(f'⚠️ Folder not found: {folder_path}')
    return None

if __name__ == "__main__":
    model_name = [
        'optuna_nbeats_ic48_oc12_bs96_st5_bl1_ly2_wd256_dp0.1_lr0.001_encodersFalse_stride12_vl0.2_cov5_onehotTrue_monitorMAPE'
    ]
    delete_model_folder(
        work_dir   = 'models/checkpoint_tuning_nbeats/',
        model_name = model_name
    )
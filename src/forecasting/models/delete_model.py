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
        'optuna_nbeats_ic216_oc24_bs32_st16_bl1_ly2_wd512_dp0.25_lr1e-05_encodersFalse_stride24_vl0.3_cov5_onehotTrue_monitorMAPE'
    ]
    delete_model_folder(
        work_dir   = 'models/checkpoint_tuning_nbeats/',
        model_name = model_name
    )
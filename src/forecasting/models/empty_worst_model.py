from src.forecasting.utils.libraries_data_handling import pd
from src.forecasting.utils.libraries_others import os, shutil

def empty_worst_model(
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
from src.forecasting.utils.libraries_data_handling import pd
from src.forecasting.utils.libraries_others import os

def recreate_folders_from_csv(
    csv_path   : str, 
    target_dir : str
) -> None:
    """
    Make folder from list in CSV based on 'model_name'.
    
    Args:
        csv_path (str)   : Path to CSV file that contain 'model_name'.
        target_dir (str) : Destination path to create folder 
    """

    df = pd.read_csv(csv_path)
    
    # Chek if model_name column is exist
    if 'model_name' not in df.columns:
        raise ValueError('CSV does not containt \'model_name\'')
    os.makedirs(target_dir, exist_ok=True)
    
    # Iterate through each folder name
    for name in df['model_name']:
        folder_path = os.path.join(target_dir, str(name))
        os.makedirs(folder_path, exist_ok=True)
        print(f'✅ Created: {folder_path}')

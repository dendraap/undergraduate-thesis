from src.forecasting.utils.recreate_model_folder import recreate_folders_from_csv

if __name__ == "__main__":
    
    recreate_folders_from_csv(
        csv_path   = 'reports/nbeats_folder_list.csv',
        target_dir = 'models/checkpoint_tuning_nbeats'
    )

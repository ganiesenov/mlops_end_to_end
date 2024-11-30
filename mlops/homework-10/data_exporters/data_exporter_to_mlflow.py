import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import pickle


MLFLOW_TRACKING_URI = 'http://mlflow:5000'

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("nyc-taxi-prediction")

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.
    """
    dv, lr, rmse = data
    
    # Create a models directory inside your project
    models_dir = Path('mage_data/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    vectorizer_path = models_dir / 'dict_vectorizer.bin'
    
    with mlflow.start_run():
        try:
            # Save vectorizer in the project directory
            with open(vectorizer_path, 'wb') as f_out:
                pickle.dump(dv, f_out)
            
            if vectorizer_path.exists():
                path = mlflow.log_artifact(str(vectorizer_path))
                print(f'dict_vectorizer saved at {path}')
                
                mlflow.log_metric('rmse', rmse)
                mlflow.sklearn.log_model(lr, 'model')
                print('successfully finished executing the block!')
            else:
                print('Error: vectorizer file was not created')
                
        except Exception as e:
            print(f'Error occurred: {str(e)}')
            raise
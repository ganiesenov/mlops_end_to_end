import pickle
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

MLFLOW_TRACKING_URI = 'http://mlflow:5000'

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
RUN_ID = 'eed269ffa30f431f992a69d0efb8a30c'

@data_loader
def load_data_from_api(*args, **kwargs):
    try:
        # Загружаем DictVectorizer
        dv_path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
        with open(dv_path, 'rb') as f_in:
            dv = pickle.load(f_in)
            
        # Загружаем модель
        logged_model = f'runs:/{RUN_ID}/model'
        model = mlflow.sklearn.load_model(logged_model)
        
        return {
            'vectorizer': dv,
            'model': model
        }
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert 'vectorizer' in output, 'Vectorizer not found in output'
    assert 'model' in output, 'Model not found in output'
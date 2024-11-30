import pickle 
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os

if 'data_loader' not in globals():
   from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
   from mage_ai.data_preparation.decorators import test

MLFLOW_TRACKING_URI = 'http://mlflow:5000'
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RUN_ID = 'ea492adbf1274bcc8136a92ecdb7d7e4'

@data_loader
def load_data_from_api(*args, **kwargs):
   try:
       try:
           client.get_run(RUN_ID)
           print("MLflow connection successful")
       except Exception as e:
           print(f"MLflow connection failed: {e}")
           raise
       try:
           artifacts = client.list_artifacts(RUN_ID)
           print(f"Available artifacts: {artifacts}")
       except Exception as e:
           print(f"Error listing artifacts: {e}")
           raise

       logged_model = f'runs:/{RUN_ID}/model'
       model = mlflow.pyfunc.load_model(logged_model)
       print("Model loaded successfully")

       print("Attempting to download vectorizer...")
       dv_path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
       print(f"Vectorizer download path: {dv_path}")
       
       with open(dv_path, 'rb') as f_in:
           dv = pickle.load(f_in)
       print("Vectorizer loaded successfully")
           
       current_dir = os.path.dirname(__file__)
       save_path = os.path.join(current_dir, 'dict_vectorizer.bin')
       
       with open(save_path, 'wb') as f_out:
           pickle.dump((dv, model), f_out)
       print(f"Models saved to: {save_path}")
           
       return {'status': 'Models saved successfully'}
       
   except Exception as e:
       print(f"Error in load_data_from_api: {str(e)}")
       raise

@test
def test_output(output, *args) -> None:
   assert output is not None, 'The output is undefined'
   current_dir = os.path.dirname(__file__)
   saved_file = os.path.join(current_dir, 'dict_vectorizer.bin')
   assert os.path.exists(saved_file), 'Model file was not saved'

import pandas as pd
import mlflow
@data_loader
def load_data(*args, **kwargs):

    logged_model = 'runs:/15bb1e1c099f4823806e07c3b962f2ab/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    loaded_model


        



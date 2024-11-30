import pickle 
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify
import pandas as pd

MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

client = MlflowClient()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RUN_ID = 'd246992101eb430ead1938a072b0ff9f'

# Загружаем сохраненные модели
with open('dict_vectorizer.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    
    features = prepare_features(ride)
    pred = predict(features)
    
    result = {
        'duration': pred,
        'model_version': RUN_ID
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
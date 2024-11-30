import pickle 
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify
import pandas as pd
import os
import sys  # добавляем импорт sys

# Получаем абсолютный путь к директории data_loaders
current_dir = os.path.dirname(os.path.abspath(__file__))
data_loaders_dir = os.path.join(os.path.dirname(current_dir), 'data_loaders')
vectorizer_path = os.path.join(data_loaders_dir, 'dict_vectorizer.bin')

print(f"Looking for vectorizer at: {vectorizer_path}")

# Загружаем модели
with open(vectorizer_path, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform([features])
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
        'model_version': 'd246992101eb430ead1938a072b0ff9f'
    }
    
    return jsonify(result)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9696
    app.run(debug=True, host='0.0.0.0', port=port)
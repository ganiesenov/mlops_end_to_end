import pickle
import mlflow 
from mlflow.tracking import MlflowClient 
from flask import Flask, request, jsonify, render_template
import pandas as pd 
import os 
import sys
from flask_cors import CORS
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, 'templates')
app = Flask('duration-prediction', template_folder=templates_dir)
CORS(app)

def load_model():
    vectorizer_path = os.path.join(current_dir, 'dict_vectorizer.bin')
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Model file not found at {vectorizer_path}")
        
    with open(vectorizer_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    # Log the features the vectorizer expects
    logger.info("Vectorizer feature names:")
    feature_names = dv.get_feature_names_out()
    logger.info(f"Total features: {len(feature_names)}")
    logger.info("Sample features:")
    for f in feature_names[:10]:  # Show first 10 features
        logger.info(f"- {f}")
        
    return dv, model

def prepare_features(ride):
    """Prepare features with numerical trip_distance"""
    logger.info(f"Preparing features for ride: {ride}")
    
    try:
        # Create feature dictionary with correct types
        features = {
            'PULocationID': str(ride['PULocationID']),
            'DOLocationID': str(ride['DOLocationID']),
            'trip_distance': float(ride['trip_distance'])
        }
        
        logger.info(f"Prepared features: {features}")
        
        # Verify trip_distance is being included
        logger.info(f"Trip distance value: {features['trip_distance']}")
        
        return features
        
    except KeyError as e:
        raise ValueError(f"Missing required field: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Invalid value in fields: {str(e)}")

def predict(features):
    """Make prediction with detailed feature logging"""
    try:
        logger.info(f"Making prediction for features: {features}")
        
        # Transform features
        X = dv.transform([features])
        
        # Log detailed feature information
        logger.info(f"Transformed feature matrix shape: {X.shape}")
        non_zero_indices = X.nonzero()[1]
        feature_names = dv.get_feature_names_out()
        
        logger.info("Non-zero features:")
        for idx, value in zip(non_zero_indices, X.data):
            feature_name = feature_names[idx]
            logger.info(f"- {feature_name}: {value}")
        
        # Make prediction
        pred = model.predict(X)
        logger.info(f"Prediction result: {pred[0]}")
        
        return float(pred[0])
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        ride = request.get_json()
        logger.info(f"Received prediction request: {ride}")
        
        features = prepare_features(ride)
        prediction = predict(features)
        
        result = {
            'duration': prediction,
            'model_version': getattr(model, '_model_id', 'unknown'),
            'status': 'success'
        }
        
        logger.info(f"Returning prediction: {result}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in prediction endpoint: {error_msg}")
        return jsonify({
            'status': 'error',
            'error': error_msg
        }), 500

# Load model
try:
    logger.info("Loading model...")
    dv, model = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    sys.exit(1)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9704
    logger.info(f"Starting server on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
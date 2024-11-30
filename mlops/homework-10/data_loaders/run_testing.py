if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import requests

@data_loader
def run_tests(*args, **kwargs):
    ride = {
        'PULocationID': 100,
        'DOLocationID': 200,
        'trip_distance': 10.0
    }

    url = 'http://localhost:9696/predict'
    
    try:
        response = requests.post(url, json=ride)
        if response.status_code == 200:
            result = response.json()
            print('Prediction success!')
            print(f'Input: {ride}')
            print(f'Predicted duration: {result["duration"]}')
            print(f'Model version: {result["model_version"]}')
            return result
        else:
            print(f'Error: {response.status_code}')
            print(f'Response: {response.text}')
    except requests.exceptions.ConnectionError:
        print(f'Error: Could not connect to prediction service at {url}')
    except Exception as e:
        print(f'Error occurred: {str(e)}')

@test
def test_output(output, *args) -> None:
    assert output is not None, 'No prediction result'
    assert 'duration' in output, 'No duration in prediction'
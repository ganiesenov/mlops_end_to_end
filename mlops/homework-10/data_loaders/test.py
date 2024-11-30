import requests

def test_prediction():
    # Тестовые данные
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
        else:
            print(f'Error: {response.status_code}')
            print(f'Response: {response.text}')
    except requests.exceptions.ConnectionError:
        print(f'Error: Could not connect to prediction service at {url}')
    except Exception as e:
        print(f'Error occurred: {str(e)}')

if __name__ == "__main__":
    test_prediction()
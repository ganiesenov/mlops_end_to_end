import requests

ride = {
    'PULocationID': 100,
    'DOLocationID': 200,
    'trip_distance': 10.0
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
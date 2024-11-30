if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform(data, *args, **kwargs):
    # Получаем модель и vectorizer из предыдущего блока
    model = data['model']
    dv = data['vectorizer']
    
    # Создаем тестовый пример для проверки
    test_ride = {
        'PULocationID': 100,
        'DOLocationID': 200,
        'trip_distance': 10.0
    }
    
    # Подготавливаем фичи
    features = {}
    features['PU_DO'] = f"{test_ride['PULocationID']}_{test_ride['DOLocationID']}"
    features['trip_distance'] = test_ride['trip_distance']
    
    # Делаем предсказание
    X = dv.transform([features])
    pred = model.predict(X)[0]
    
    return {
        'test_prediction': float(pred),
        'test_features': features,
        'status': 'model is ready for predictions'
    }

@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert 'test_prediction' in output, 'No test prediction in output'
    assert 'status' in output, 'No status in output'
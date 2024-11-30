if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer 
from sklearn.metrics import root_mean_squared_error


@transformer
def transform(df, *args, **kwargs):

    categorical = ['PULocationID', 'DOLocationID']
    dv = DictVectorizer()


    train_dict = df[categorical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    target = 'duration'
    y_train = df[target].values
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    rmse = root_mean_squared_error(y_pred, y_train)

    print(lr.intercept_)
    return dv, lr, rmse


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
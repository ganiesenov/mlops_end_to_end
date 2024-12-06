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
    # Include both categorical and numerical features
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    
    # Create dictionary with both categorical and numerical features
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    
    # Fit transform both categorical and numerical features
    dv = DictVectorizer()   
    X_train = dv.fit_transform(train_dicts)
    
    # Print feature names for verification
    feature_names = dv.get_feature_names_out()
    print("Features included in the model:")
    print(feature_names)
    
    target = 'duration'
    y_train = df[target].values      
    
    # Train model
    lr = LinearRegression()
    lr.fit(X_train, y_train)     
    
    # Calculate and print feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(lr.coef_)
    })
    print("\nFeature Importances:")
    print(feature_importance.sort_values('importance', ascending=False).head())
    
    # Calculate metrics
    y_pred = lr.predict(X_train)
    rmse = root_mean_squared_error(y_pred, y_train)     
    print(f"\nModel Intercept: {lr.intercept_}")
    print(f"RMSE: {rmse}")
    
    return dv, lr, rmse   

@test 
def test_output(output, *args) -> None:     
    """     
    Template code for testing the output of the block.     
    """     
    assert output is not None, 'The output is undefined'
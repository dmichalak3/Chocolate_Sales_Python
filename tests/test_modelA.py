# tests/test_modelA.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def test_sales_regression_logic():
    # Synthetic dataset mirroring our CSV structure
    data = {
        'Sales Person': [0, 1, 0],
        'Country': [0, 0, 1],
        'Product': [2, 1, 0],
        'Year': [2022, 2022, 2022],
        'Month': [1, 4, 7],
        'DayOfWeek': [3, 1, 5],
        'Boxes Shipped': [180, 94, 342],
        'Amount': [5320.00, 7896.00, 12726.00]
    }
    df_mock = pd.DataFrame(data)
    
    # Feature-target separation
    X = df_mock.drop(columns=['Amount'])
    y = df_mock['Amount']
    
    # Model initialization
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Validation of the fit-predict cycle
    model.fit(X, y)
    sample_input = X.iloc[[0]]
    prediction = model.predict(sample_input)
    
    # Functional Assertions
    assert len(prediction) == 1, "Model failed to return a single prediction"
    assert prediction[0] > 0, "Predicted amount should be positive"
    assert hasattr(model, "feature_importances_"), "Model failed to calculate feature significance"
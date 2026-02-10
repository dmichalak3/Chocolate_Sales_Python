# src/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

def train_model(df, target_col='Amount'):
    # Isolate features and the target variable
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Partition dataset into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and fit the RandomForestRegressor
    # Note: Hyperparameter tuning (e.g., GridSearch) can be implemented here if required
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Generate predictions for model evaluation
    rf_predictions = rf.predict(X_test)

    # DODANE: Linear Regression (baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_predictions = lr.predict(X_test)
    
    # Calculate key regression metrics
    mae = mean_absolute_error(y_test, rf_predictions)
    r2 = r2_score(y_test, rf_predictions)

    # DODANE: RMSE
    mse = mean_squared_error(y_test, rf_predictions)
    rmse = np.sqrt(mse)

    # DODANE: metryki Linear Regression
    lr_mae = mean_absolute_error(y_test, lr_predictions)
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_rmse = np.sqrt(lr_mse)
    lr_r2 = r2_score(y_test, lr_predictions)
    
    print(f"Model training complete. Performance metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    print(f"Linear Regression baseline -> MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}, R2: {lr_r2:.2f}")
    
    return rf

def save_model(model, filename='chocolate_sales_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model successfully serialized to {filename}")
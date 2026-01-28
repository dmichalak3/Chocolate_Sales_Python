# src/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

def train_model(df, target_col='Amount'):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tu możesz dodać logikę GridSearch jeśli chcesz, dla uproszczenia daję sam model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    predictions = rf.predict(X_test)
    
    # Ocena
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model wytrenowany. MAE: {mae:.2f}, R2: {r2:.2f}")
    
    return rf

def save_model(model, filename='chocolate_sales_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model zapisany jako {filename}")
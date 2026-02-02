# src/data_loader.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    # Load dataset from the specified CSV file path
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    # Sanitize 'Amount' column by removing currency symbols and casting to numeric
    if 'Amount' in df.columns and df['Amount'].dtype == 'object':
        df['Amount'] = df['Amount'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['Amount'] = pd.to_numeric(df['Amount'])

    # Perform feature engineering on 'Date' and drop the original timestamp
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df = df.drop(columns=['Date'])

    # Apply label encoding to categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Remove incomplete records to ensure data integrity
    df = df.dropna()
    return df
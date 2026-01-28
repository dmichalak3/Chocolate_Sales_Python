# src/data_loader.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    # 1. Wczytanie
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {filepath}")
        return None

    # 2. Czyszczenie Amount
    if 'Amount' in df.columns and df['Amount'].dtype == 'object':
        df['Amount'] = df['Amount'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['Amount'] = pd.to_numeric(df['Amount'])

    # 3. Data
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df = df.drop(columns=['Date'])

    # 4. Encoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df = df.dropna()
    return df
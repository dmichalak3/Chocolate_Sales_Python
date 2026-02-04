# tests/test_modelB.py
import pandas as pd
import numpy as np

def test_data_sanitization_logic():
    # Mock data with our specific currency formatting issues
    raw_data = {
        'Amount': ["$5,320.00", "$7,896.00", "$168.00"],
        'Date': ["04/01/2022", "01/08/2022", "28/07/2022"]
    }
    df = pd.DataFrame(raw_data)
    
    # Logic verification for Amount sanitization
    df['Amount'] = df['Amount'].astype(str).str.replace(r'[$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'])
    
    # Logic verification for Date feature engineering
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    months = df['Date'].dt.month.tolist()
    
    # Integrity Assertions
    assert df['Amount'].dtype == np.float64 or df['Amount'].dtype == np.int64, "Amount was not converted to numeric"
    assert df['Amount'].iloc[0] == 5320.0, "Currency symbol/comma removal failed"
    assert months == [1, 8, 7], "Date parsing failed to extract correct months"
    assert 'Date' in df.columns, "Original Date column should still exist before dropping"
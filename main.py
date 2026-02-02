# main.py
from src.data_loader import load_and_clean_data
from src.model_trainer import train_model, save_model

def main():
    # Define local file paths for the dataset
    data_path = 'data/Chocolate Sales (2).csv'
    
    # 1. Data Ingestion and Preprocessing
    print("--- Initializing data processing pipeline ---")
    df = load_and_clean_data(data_path)
    
    if df is not None:
        # 2. Exploratory Data Analysis (EDA) - Summary Statistics
        print(df.describe())
        
        # 3. Model Training and Optimization
        print("\n--- Starting model training phase ---")
        model = train_model(df)
        
        # 4. Model Persistence
        save_model(model)
        print("\n--- Pipeline execution completed successfully ---")

if __name__ == "__main__":
    main()
# main.py
from src.data_loader import load_and_clean_data
from src.model_trainer import train_model, save_model

def main():
    # Definiujemy ścieżki
    data_path = 'data/Chocolate Sales (2).csv'
    
    # 1. Wczytaj i wyczyść
    print("--- Rozpoczynam przetwarzanie danych ---")
    df = load_and_clean_data(data_path)
    
    if df is not None:
        # 2. Statystyki (opcjonalnie)
        print(df.describe())
        
        # 3. Trenuj
        print("\n--- Rozpoczynam trening modelu ---")
        model = train_model(df)
        
        # 4. Zapisz
        save_model(model)
        print("\n--- Gotowe! ---")

if __name__ == "__main__":
    main()
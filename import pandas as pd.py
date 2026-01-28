import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

# --- 1. IMPORT DANYCH ---
# Upewnij się, że plik CSV jest w tym samym folderze co skrypt
file_name = 'Chocolate Sales (2).csv' 

try:
    df = pd.read_csv(file_name)
    print("Dane wczytane pomyślnie!")
except FileNotFoundError:
    print(f"Błąd: Nie znaleziono pliku '{file_name}'. Sprawdź nazwę pliku.")
    # Tutaj przerywamy jeśli nie ma pliku, ale dla przykładu kod idzie dalej
    
# --- 2. STATYSTYKI ZBIORU DANYCH (EDA) ---
print("\n--- Podgląd Danych ---")
print(df.head())
print("\n--- Informacje o kolumnach ---")
print(df.info())
print("\n--- Statystyki opisowe ---")
print(df.describe())

# --- 3. PRZYGOTOWANIE DANYCH (DATA CLEANING) ---
# Problem w tym zbiorze: Kolumna 'Amount' (lub Revenue) często ma znaki '$' i ',' i jest tekstem
# Sprawdzamy nazwy kolumn, aby dopasować czyszczenie
if 'Amount' in df.columns and df['Amount'].dtype == 'object':
    print("\nCzyszczenie kolumny Amount...")
    df['Amount'] = df['Amount'].astype(str).str.replace(r'[$,]', '', regex=True)
    df['Amount'] = pd.to_numeric(df['Amount'])

# Konwersja daty na format datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)
    # Inżynieria cech: Wyciągamy rok, miesiąc, dzień tygodnia
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    # Usuwamy oryginalną datę, bo model jej nie zrozumie wprost
    df_model = df.drop(columns=['Date'])
else:
    df_model = df.copy()

# Kodowanie zmiennych tekstowych (Kraj, Sprzedawca, Produkt) na liczby
label_encoders = {}
categorical_cols = df_model.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le
    print(f"Zakodowano kolumnę: {col}")

# Usunięcie ewentualnych braków danych
df_model = df_model.dropna()

# --- 4. PODZIAŁ DANYCH ---
# Target (cel): Przewidujemy 'Amount' (Wartość sprzedaży)
target_col = 'Amount' 
if target_col not in df_model.columns:
    # Fallback jeśli kolumna nazywa się inaczej, np. 'Revenue'
    target_col = df_model.columns[-1] 

X = df_model.drop(columns=[target_col])
y = df_model[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nZbiór treningowy: {X_train.shape}, Zbiór testowy: {X_test.shape}")

# --- 5. TRENOWANIE MODELI ---
# Model 1: Regresja Liniowa (Prosty model bazowy)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Model 2: Random Forest (Bardziej zaawansowany)
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- 6. OCENA SKUTECZNOŚCI ---
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- Wyniki dla {name} ---")
    print(f"MAE (Średni błąd bezwzględny): {mae:.2f}")
    print(f"RMSE (Pierwiastek błędu kwadratowego): {rmse:.2f}")
    print(f"R2 Score (Dopasowanie): {r2:.2f}")

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# --- 7. TUNING HIPERPARAMETRÓW ---
print("\nRozpoczynam tuning modelu Random Forest (może to chwilę potrwać)...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")

# Sprawdzenie wyniku po tuningu
y_pred_best = best_rf.predict(X_test)
evaluate_model("Tuned Random Forest", y_test, y_pred_best)

# --- 8. EKSPORT MODELU I WIZUALIZACJA ---
# Zapis modelu
with open('chocolate_sales_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
print("\nModel został zapisany do pliku 'chocolate_sales_model.pkl'")

# Wykres Rzeczywiste vs Przewidywane
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Linia idealnego dopasowania
plt.xlabel("Rzeczywista Sprzedaż")
plt.ylabel("Przewidywana Sprzedaż")
plt.title("Wykres predykcji: Rzeczywiste vs Przewidywane")
plt.show()

# Wykres ważności cech (co wpływało na sprzedaż?)
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Ważność Cech (Co wpływa na sprzedaż?)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
import sys
import os
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# Import custom data loader
from src.data_loader import load_and_clean_data

# Define file paths
DATA_PATH = Path('data/Chocolate Sales (2).csv')
# Handle path logic to ensure it works from any execution directory
if not DATA_PATH.exists():
    # If running from src, adjust path to look in parent
    DATA_PATH = Path(parent_dir) / 'data' / 'Chocolate Sales (2).csv'

EDA_PATH = Path("data/eda")
FIG_PATH = EDA_PATH / "figures"

# Ensure output directories exist
if not EDA_PATH.is_absolute():
    EDA_PATH = Path(parent_dir) / EDA_PATH
if not FIG_PATH.is_absolute():
    FIG_PATH = Path(parent_dir) / FIG_PATH

EDA_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH.mkdir(parents=True, exist_ok=True)

def restore_country_labels(df, raw_data_path):

    try:
        # Check if 'Country' is numeric (encoded). If it's strings, we don't need to do anything.
        if pd.api.types.is_numeric_dtype(df['Country']):
            print("  -> Detected numeric Country IDs. Attempting to restore names from raw file...")
            
            # Load raw data just to get the unique country names
            raw_df = pd.read_csv(raw_data_path)
            
            # Get unique countries, sorted alphabetically (simulating LabelEncoder logic)
            # Note: We use dropna() to ignore empty rows that might have been cleaned out
            sorted_countries = sorted(raw_df['Country'].dropna().unique())
            
            # Create a dictionary mapping: {0: 'Australia', 1: 'Canada', ...}
            country_map = {i: name for i, name in enumerate(sorted_countries)}
            
            # Apply the mapping
            df['Country'] = df['Country'].map(country_map)
            print("  -> Country names restored successfully.")
            
    except Exception as e:
        print(f"  -> Warning: Could not restore country names. Error: {e}")
    
    return df

def run_eda():

    
    # 1. Load Data
    # We use the absolute path calculated earlier to avoid FileNotFoundError
    print(f"Loading data from: {DATA_PATH}")
    df = load_and_clean_data(DATA_PATH)
    
    if df is None:
        print("EDA Error: Could not load data.")
        return

    # 2. Fix Labels (Convert 0,1,2 back to 'India', 'USA', etc.)
    df = restore_country_labels(df, DATA_PATH)

    print(f"Data ready for EDA. Shape: {df.shape}")

    # 3. Descriptive Statistics
    stats = df.describe()
    
    # Group by Country for the plot
    country_revenue = df.groupby('Country')['Amount'].sum().reset_index().sort_values('Amount', ascending=False)

    # Save summary to text file
    summary_file = EDA_PATH / "eda_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=== STATISTICAL SUMMARY ===\n")
        f.write(stats.to_string())
        f.write("\n\n=== REVENUE PER COUNTRY ===\n")
        f.write(country_revenue.to_string())

    # 4. Visualization: Revenue by Country
    plt.figure(figsize=(12, 8)) # Slightly larger for better label fitting
    
    # Create Bar Plot
    # We use hue='Country' to avoid FutureWarning, but legend=False since x-axis labels are enough
    ax = sns.barplot(
        data=country_revenue, 
        x='Country', 
        y='Amount', 
        palette='viridis', 
        hue='Country', 
        legend=False
    )
    
    plt.title("Total Revenue by Country", fontsize=16, fontweight='bold')
    plt.xlabel("Country", fontsize=12)
    plt.ylabel("Total Revenue ($)", fontsize=12)
    
    # Formatting X-axis: Rotate labels 45 degrees to prevent overlapping
    plt.xticks(rotation=45, ha='right')
    
    # Formatting Y-axis: Use commas for thousands (e.g., 1,000,000 instead of 1e6)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(FIG_PATH / "revenue_by_country.png")
    plt.close()

    # 5. Visualization: Correlation Matrix
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation only on numeric columns
    # Note: 'Country' might be excluded if we successfully converted it back to String, which is correct for Pearson corr.
    corr_matrix = df.corr(numeric_only=True)
    
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIG_PATH / "correlation_matrix.png")
    plt.close()

    print(f"EDA completed. Artifacts saved in: {EDA_PATH} ✅")

if __name__ == "__main__":
    run_eda()
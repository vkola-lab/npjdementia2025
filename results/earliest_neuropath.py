import os
import sys
import pandas as pd

# Ensure repository root is on path to import utils
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def main():
    # Load the data
    data_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_data_filtered.csv')
    data = pd.read_csv(data_path)
    
    # Check if year_imag column exists
    if 'year_imag' in data.columns:
        # Find the earliest year
        earliest_year = data['year_imag'].min()
        print(f"Earliest year in year_imag column: {earliest_year}")
        
        # Also show some statistics about the year distribution
        print(f"Year range: {data['year_imag'].min()} - {data['year_imag'].max()}")
        print(f"Number of unique years: {data['year_imag'].nunique()}")
    else:
        print("year_imag column not found in the dataset")
    
    # Check if NACCADC column exists
    if 'NACCADC' in data.columns:
        # Count unique NACCADC values
        unique_naccadc_count = data['NACCADC'].nunique()
        print(f"Number of unique NACCADC values: {unique_naccadc_count}")
        
        # Also show the unique values themselves
        unique_naccadc_values = data['NACCADC'].unique()
        print(f"Unique NACCADC values: {sorted(unique_naccadc_values)}")
    else:
        print("NACCADC column not found in the dataset")
        print("Available columns:", data.columns.tolist())

if __name__ == '__main__':
    main()

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import csv

# Ensure repository root is on path to import utils
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from utils.load_data import load_feature_configs

def main():
    # Load feature configuration
    feature_config_path = os.path.join(repo_root, 'data', 'feature_config.json')
    feature_config = load_feature_configs(feature_config_path)
    all_features = feature_config['all_features']

    # Load and preprocess test data
    test_data_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_data_filtered_unique_naccid.csv')
    test_data = pd.read_csv(test_data_path)
    for col in ['his_SEX', 'his_RACE', 'his_HISPANIC', 'his_HISPOR']:
        if col in test_data.columns:
            test_data[col] = test_data[col].astype('category')

    # Convert NACCID to numeric form for merging with PET data
    if 'NACCID' in test_data.columns:
        test_data['NACCID'] = test_data['NACCID'].astype(str).str.replace('NACC', '').astype(int)
    else:
        print('Warning: NACCID column missing in test_data; merge with PET will fail.')

    # Features for AD model
    X_test = test_data[all_features]

    # Load AD model
    model_dir = os.path.join(repo_root, 'data', 'dev-model')
    ad_model = joblib.load(os.path.join(model_dir, 'xgb_model_AD.joblib'))
    X_test_ad = X_test[ad_model.get_booster().feature_names]
    prob_ad = ad_model.predict_proba(X_test_ad)[:, 1]
    test_data['prob_ad'] = prob_ad

    # Load PET data robustly
    pet = pd.read_csv(
        os.path.join(repo_root, 'data', 'ml_data', 'UCBERKELEY_AMYLOID_MRIFREE_GAAIN_15Jun2025.csv'),
    )
    print("PET data loaded, shape:", pet.shape)
    print(pet.head())
    # Clean up column names (remove spaces, quotes)
    pet.columns = [c.strip().replace('"', '').replace("'", '') for c in pet.columns]
    # Keep only rows with valid NACCID and CENTILOIDS
    pet = pet[pet['NACCID'].notnull() & pet['CENTILOIDS'].notnull()]
    # Ensure pet is a DataFrame
    if not isinstance(pet, pd.DataFrame):
        pet = pd.DataFrame(pet)


    # Convert NACCID to match test_data
    pet['NACCID'] = pet['NACCID'].astype(str).str.replace('NACC', '').astype(int)
    # Convert SCANDATE to datetime for latest scan selection
    pet['SCANDATE'] = pd.to_datetime(pet['SCANDATE'], errors='coerce')

    # Print the earliest scan date
    if 'SCANDATE' in pet.columns:
        earliest_date = pet['SCANDATE'].min()
        print(f"Earliest scan date: {earliest_date}")
    else:
        print("SCANDATE column not found in PET data")
    # Keep only the latest scan per NACCID
    if 'SCANDATE' in pet.columns:
        pet_latest = pet.sort_values('SCANDATE').groupby('NACCID').tail(1)
    else:
        print('SCANDATE column missing in PET data!')
        pet_latest = pet.groupby('NACCID').tail(1)

    # Merge test data with PET data on NACCID
    merged = test_data.merge(pet_latest[['NACCID', 'CENTILOIDS']], on='NACCID', how='inner')

    print(f"Merged dataframe shape: {merged.shape}")

    # Plot scatterplot
    plt.figure(figsize=(6, 5))
    plt.scatter(merged['CENTILOIDS'], merged['prob_ad'], alpha=0.6)
    plt.xlabel('Centiloids (PET)')
    plt.ylabel('Predicted AD Probability')
    plt.title('AD Model Probability vs PET Centiloids')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main() 
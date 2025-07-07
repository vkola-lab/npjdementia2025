# %%
import pandas as pd 
import numpy as np
import os
from utils.load_data import load_feature_configs

# Load feature configuration
feature_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'feature_config.json')
feature_config = load_feature_configs(feature_config_path)
demo_features = feature_config['demo_features']
vol_features = feature_config['vol_features']
wmh_features = feature_config['wmh_features']
imag_features = feature_config['imag_features']
all_features = feature_config['all_features']
labels = feature_config['labels'] if 'labels' in feature_config else ['AD', 'nAD']

# Data loading
train_data_path = '/projectnb/vkolagrp/spuduch/ml_data_updated.csv'
test_data_path = '/projectnb/vkolagrp/spuduch/ml_test_data_updated.csv'
data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Preprocessing (as before, but only keep relevant code)
data['ImageDate'] = pd.to_datetime(data['ImageDate'])
data['VisitDate'] = pd.to_datetime(data['VisitDate'])
data['diff'] = (data['VisitDate'] - data['ImageDate']).dt.days
data = data[~(data['month_imag'] == 'nodate')]
data = data[data['DE'] == 1]
test_data = test_data[test_data['DE'] == 1]

# Demographic preprocessing
for col in ['his_NACCAGE', 'his_SEX', 'his_RACE', 'his_HISPANIC', 'his_HISPOR', 'his_EDUC']:
    if col in data.columns:
        if col == 'his_NACCAGE':
            data[col] = data['NACCAGE'].replace({-4: np.nan, '-4':np.nan}).astype(float)
            test_data[col] = test_data['NACCAGE'].replace({-4: np.nan, '-4':np.nan}).astype(float)
        elif col == 'his_EDUC':
            data[col] = data['EDUC'].replace({99: np.nan, -4: np.nan, '-4':np.nan}).astype(float)
            test_data[col] = test_data['EDUC'].replace({99: np.nan, -4: np.nan, '-4':np.nan}).astype(float)
        else:
            data[col] = data[col].astype('category')
            test_data[col] = test_data[col].astype('category')

# nAD label
if 'VD' in data.columns and 'PRD' in data.columns and 'FTD' in data.columns:
    data['nAD'] = data['VD'] | data['PRD'] | data['FTD']
    test_data['nAD'] = test_data['VD'] | test_data['PRD'] | test_data['FTD']

# Save filtered data for use in other scripts
filtered_train_path = '/projectnb/vkolagrp/spuduch/ml_data_filtered.csv'
filtered_test_path = '/projectnb/vkolagrp/spuduch/ml_test_data_filtered.csv'
data.to_csv(filtered_train_path, index=False)
test_data.to_csv(filtered_test_path, index=False)

print('Data preprocessing complete. Use cv.py, final_train.py, or wandb_sweep.py for modeling.')

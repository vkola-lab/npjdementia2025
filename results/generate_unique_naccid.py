import os
import sys
import pandas as pd

# Determine repository root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.load_data import load_feature_configs

# Paths
data_dir = os.path.join(repo_root, 'data', 'ml_data')
input_path = os.path.join(data_dir, 'ml_data_filtered.csv')
output_path = os.path.join(data_dir, 'ml_data_filtered_unique_naccid.csv')
config_path = os.path.join(repo_root, 'data', 'feature_config.json')

# Load data and configuration
df = pd.read_csv(input_path)
feature_config = load_feature_configs(config_path)
all_features = feature_config['all_features']
vol_features = feature_config['vol_features']



# Compute number of missing values in feature columns
df['missing_count'] = df[all_features].isna().sum(axis=1)

# Ensure ImageDate is datetime for tie-breaking
if 'ImageDate' in df.columns:
    df['ImageDate'] = pd.to_datetime(df['ImageDate'])

# Sort: by NACCID, ascending missing_count, then latest ImageDate
sort_cols = ['NACCID', 'missing_count']
if 'ImageDate' in df.columns:
    sort_cols.append('ImageDate')

df_sorted = df.sort_values(sort_cols, ascending=[True, True, False])

# Drop duplicates, keep first in each group
df_unique = df_sorted.drop_duplicates(subset='NACCID', keep='first').copy()

# Remove helper column
df_unique.drop(columns=['missing_count'], inplace=True)

# print summary
print(f"Original dataset size: {len(df)}")
print(f"Filtered unique NACCIDs dataset size: {len(df_unique)}")

# t1 can be figured out via a column in vol_features - if its not na, then there was a t1 for that row
t1_col = vol_features[3]
num_t1 = df_unique[t1_col].notna().sum()
print(f"Number of unique NACCIDs with T1: {num_t1}")
# t2/flair can be figured out via the column 'total_wm_burden' - if its not na, then there was a t2/flair for that row
num_t2_flair = df_unique['total_wm_burden'].notna().sum()
print(f"Number of unique NACCIDs with T2/FLAIR: {num_t2_flair}")

# Save output
df_unique.to_csv(output_path, index=False)
print(f"Saved unique-NACCID data to {output_path}")

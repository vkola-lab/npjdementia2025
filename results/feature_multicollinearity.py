# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
import json
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # if notebook is in a subdir
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.load_data import load_feature_configs
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load feature configuration
feature_config_path = '/Users/spuduch/Research/K_lab/neuroradiology-radiomics/data/feature_config.json'
feature_config = load_feature_configs(feature_config_path)
all_features = feature_config['all_features']

# Load training data
train_data_path = '/Users/spuduch/Research/K_lab/neuroradiology-radiomics/data/ml_data/ml_data_filtered.csv'
data = pd.read_csv(train_data_path)

# Select only the features of interest
X = data[all_features]

categorical_features = ["his_NACCAGE", "his_SEX", "his_RACE", "his_HISPANIC", "his_HISPOR"]

# Load lobe mapping
lobe_mapping_path = '/Users/spuduch/Research/K_lab/neuroradiology-radiomics/data/lobe_mapping.json'
with open(lobe_mapping_path, 'r') as f:
    lobe_mapping = json.load(f)

# Load human-readable feature names map
feature_names_map_path = '/Users/spuduch/Research/K_lab/neuroradiology-radiomics/data/feature_names_map.json'
with open(feature_names_map_path, 'r') as f:
    feature_names_map = json.load(f)

# Helper to clean feature keys into display names
def clean_feature_name(feature: str) -> str:
    if feature.endswith('_vol'):
        base = feature[:-4]
        name = feature_names_map.get(base, base)
        return f"{name} Vol."
    elif feature.endswith('_wmh'):
        base = feature[:-4]
        name = feature_names_map.get(base, base)
        return f"{name} WMH"
    else:
        return feature_names_map.get(feature, feature)

# Prepare data: encode categorical features
X_enc = X.copy()
le_dict = {}
for col in X_enc.columns:
    if X_enc[col].dtype.name == 'category' or X_enc[col].dtype == object:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        le_dict[col] = le


# Get feature groups from config
vol_features = feature_config['vol_features']
wmh_features = feature_config['wmh_features']
demo_features = feature_config['demo_features']

# Define feature sets for the two plots
vol_demo_features = vol_features + demo_features
wmh_demo_features = wmh_features + demo_features

def group_features_by_lobe(features, lobe_mapping, demo_features):
    """
    Group features by brain lobe/region using the lobe mapping.
    Returns ordered feature list and group labels for heatmap annotation.
    """
    # Create reverse mapping: feature -> lobe
    feature_to_lobe = {}
    for lobe, regions in lobe_mapping.items():
        for region in regions:
            # Check both _vol and _wmh versions
            vol_feature = f"{region}_vol"
            wmh_feature = f"{region}_wmh"
            feature_to_lobe[vol_feature] = lobe
            feature_to_lobe[wmh_feature] = lobe
    
    # Group features by lobe
    grouped_features = {}
    uncategorized = []
    demographics = []
    
    for feature in features:
        if feature in demo_features:
            demographics.append(feature)
        elif feature in feature_to_lobe:
            lobe = feature_to_lobe[feature]
            if lobe not in grouped_features:
                grouped_features[lobe] = []
            grouped_features[lobe].append(feature)
        else:
            uncategorized.append(feature)
    
    # Create ordered feature list
    ordered_features = []
    group_boundaries = []
    current_pos = 0
    
    # Add demographics first
    if demographics:
        ordered_features.extend(demographics)
        group_boundaries.append(('Demographics', current_pos, current_pos + len(demographics)))
        current_pos += len(demographics)
    
    # Add each lobe group
    lobe_order = ['Frontal_Lobe', 'Parietal_Lobe', 'Occipital_Lobe','Temporal_Lobe',  'Subcortical',
                    'Brainstem', 'Cerebellum', 'Ventricular_CSF', 'Whole_Brain']
    
    for lobe in lobe_order:
        if lobe in grouped_features:
            lobe_features = sorted(grouped_features[lobe])
            ordered_features.extend(lobe_features)
            group_boundaries.append((lobe.replace('_', ' '), current_pos, current_pos + len(lobe_features)))
            current_pos += len(lobe_features)
    
    # Add uncategorized features
    if uncategorized:
        ordered_features.extend(sorted(uncategorized))
        group_boundaries.append(('Uncategorized', current_pos, current_pos + len(uncategorized)))
    if len(uncategorized) != 0:
        print(f"Warning: {len(uncategorized)} features not categorized into lobes: {uncategorized}")
    return ordered_features, group_boundaries

def reorder_matrix(matrix, feature_names, ordered_features, group_boundaries=None):
    """
    Reorder correlation matrix based on grouped features.
    If group_boundaries are passed, within each group, sort by mulicollinearity.
    """
    # Get indices for reordering
    feature_to_idx = {feature: i for i, feature in enumerate(feature_names)}
    new_indices = [feature_to_idx[feature] for feature in ordered_features if feature in feature_to_idx]
    
    if group_boundaries:
        # Sort within each group by multicollinearity
        for group_name, start, end in group_boundaries:
            if start < end:
                group_features = ordered_features[start:end]
                group_indices = [feature_to_idx[f] for f in group_features if f in feature_to_idx]
                # Use iloc for pandas DataFrame, np.ix_ for numpy array
                if hasattr(matrix, 'iloc'):
                    sub_matrix = matrix.iloc[group_indices, group_indices].values
                else:
                    sub_matrix = matrix[np.ix_(group_indices, group_indices)]
                sorted_indices = np.argsort(np.sum(-np.abs(sub_matrix), axis=0))
                new_indices[start:end] = [group_indices[i] for i in sorted_indices]
    # Reorder matrix - handle both pandas DataFrame and numpy array
    if hasattr(matrix, 'iloc'):  # pandas DataFrame
        reordered_matrix = matrix.iloc[new_indices, new_indices]
    else:  # numpy array
        reordered_matrix = matrix[np.ix_(new_indices, new_indices)]
    
    reordered_features = [feature_names[i] for i in new_indices]
    
    return reordered_matrix, reordered_features

def plot_grouped_heatmap(matrix, features, group_boundaries, title, figsize=None, region_fontsize=None, lobe_fontsize=None, save_to=None):
    """Plot heatmap with group boundaries marked and clean labels."""
    if figsize is None:
        figsize = (max(12, len(features)//3), max(10, len(features)//3))
    if region_fontsize is None:
        # scale it appropriately based on figure size
        region_fontsize = figsize[0] // 3
    if lobe_fontsize is None:
        lobe_fontsize = figsize[0] // 1

    # Prepare display labels
    display_names = [clean_feature_name(f) for f in features]
    plt.figure(figsize=figsize)
    
    # Create heatmap
    ax = sns.heatmap(matrix, cmap='viridis', square=True, linewidths=0.1,
                     xticklabels=display_names, yticklabels=display_names,
                     cbar_kws={'shrink': 0.8})

    # Add group boundaries
    count = 0
    for group_name, start, end in group_boundaries:

        # Add lines to separate groups
        ax.axhline(y=start, color='red', linewidth=4)
        ax.axvline(x=start, color='red', linewidth=4)
        ax.axhline(y=end, color='red', linewidth=4)
        ax.axvline(x=end, color='red', linewidth=4)

        # every other label should be offset
        if count % 2 == 0:
            stagger = 2
        else:
            stagger = -2
        
        # Add group labels on right side
        mid_point = (start + end) / 2
        x_offset = len(features) + len(features) * 0.02  # just outside the rightmost column
        ax.text(x_offset + stagger, mid_point, group_name, 
                rotation=-90, ha='left', va='center', 
                fontsize=lobe_fontsize, fontweight='bold', color='red')
        # top labels
        ax.text(mid_point, -len(features)*0.04 + stagger, group_name, 
                rotation=0, ha='center', va='center', 
                fontsize=lobe_fontsize, fontweight='bold', color='red')
        count += 1
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=region_fontsize)
    plt.yticks(rotation=0, fontsize=region_fontsize)

    # colorbar tick labels bigger
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=lobe_fontsize)
    cbar.set_label('Spearman Correlation Coefficient', fontsize=lobe_fontsize, fontweight='bold')
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, bbox_inches='tight')
    plt.show()

# Helper to compute MI matrix for a given feature set, parallelized

def compute_mi_row(i, features, X_enc, categorical_features):
    n_features = len(features)
    row = np.zeros(n_features)
    for j in range(n_features):
        if i == j:
            row[j] = np.nan
        else:
            x = X_enc[features[i]]
            y = X_enc[features[j]]
            indices_to_use = x.notna() & y.notna()
            x_ = x[indices_to_use]
            y_ = y[indices_to_use]
            if len(x_) == 0 or len(y_) == 0:
                row[j] = np.nan
                continue
            # Use classif if target is categorical, else regression
            if features[i] in categorical_features:
                mi_ij = mutual_info_classif(y_.values.reshape(-1, 1), x_, discrete_features=True)[0]
            else:
                mi_ij = mutual_info_regression(y_.values.reshape(-1, 1), x_, discrete_features=True)[0]
            if features[j] in categorical_features:
                mi_ji = mutual_info_classif(x_.values.reshape(-1, 1), y_, discrete_features=True)[0]
            else:
                mi_ji = mutual_info_regression(x_.values.reshape(-1, 1), y_, discrete_features=True)[0]
            row[j] = (mi_ij + mi_ji) / 2
    return row

def compute_mi_matrix_parallel(features, X_enc, categorical_features):
    n_features = len(features)
    results = Parallel(n_jobs=-1)(
        delayed(compute_mi_row)(i, features, X_enc, categorical_features) for i in tqdm(range(n_features), desc="Rows")
    )
    return np.vstack(results)

# %%
# Compute and plot for vol
features1 = [f for f in vol_features if f in X.columns]
# Option 1: Use correlation (faster)
correlation_matrix1 = X[features1].corr(method='spearman')
# Option 2: Use mutual information (slower but handles categorical features better)
# mi_matrix1 = compute_mi_matrix_parallel(features1, X_enc, categorical_features)

# Group features by lobe for vol+demo
ordered_features1, group_boundaries1 = group_features_by_lobe(features1, lobe_mapping, demo_features)
reordered_matrix1, reordered_features1 = reorder_matrix(correlation_matrix1, features1, ordered_features1, group_boundaries1)
# For MI version: reordered_matrix1, reordered_features1 = reorder_matrix(mi_matrix1, features1, ordered_features1)

# Plot grouped heatmap for vol+demo
plot_grouped_heatmap(reordered_matrix1, reordered_features1, group_boundaries1, 
                     title=None,
                     save_to='/Users/spuduch/Research/K_lab/neuroradiology-radiomics/figs/t1_vol_spearman_multicollinearity.png')

# %%
# Compute and plot for wmh+demo
features2 = [f for f in wmh_demo_features if f in X_enc.columns]
mi_matrix2 = compute_mi_matrix_parallel(features2, X_enc, categorical_features)

# Group features by lobe for wmh+demo
ordered_features2, group_boundaries2 = group_features_by_lobe(features2, lobe_mapping, demo_features)
reordered_matrix2, reordered_features2 = reorder_matrix(mi_matrix2, features2, ordered_features2)

# Plot grouped heatmap for wmh+demo
plot_grouped_heatmap(reordered_matrix2, reordered_features2, group_boundaries2, 
                     'Mutual Information: WMH + Demographics (Grouped by Brain Region)')
# %%

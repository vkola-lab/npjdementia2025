import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import joblib

# Ensure repo root is in path to import utils
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from utils.load_data import load_feature_configs

# Paths
feature_config_path = os.path.join(repo_root, 'data', 'feature_config.json')
model_dir = os.path.join(repo_root, 'data', 'dev-model')
test_data_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_test_data_filtered.csv')
figs_dir = os.path.join(repo_root, 'figs')
calib_error_path = os.path.join(repo_root, 'data', 'results_data', 'calibration_errors.json')
save_figs = False
save_ece = False

os.makedirs(figs_dir, exist_ok=True)

# Load config and data
feature_config = load_feature_configs(feature_config_path)
labels = feature_config.get('labels', ['AD', 'nAD'])
data = pd.read_csv(test_data_path)

categorical_features = ["his_NACCAGE", "his_SEX", "his_RACE", "his_HISPANIC", "his_HISPOR"]
for col in categorical_features:
    if col in data.columns:
        data[col] = data[col].astype('category')

# Load models
model_AD = joblib.load(os.path.join(model_dir, 'xgb_model_AD.joblib'))
model_nAD = joblib.load(os.path.join(model_dir, 'xgb_model_nAD.joblib'))
models = {'AD': model_AD, 'nAD': model_nAD}

# Prepare storage for calibration errors
errors = {}

# Plot settings
n_bins = 3  # Number of bins for calibration curve
plt.figure(figsize=(6, 6))
for label in labels:
    model = models[label]
    # Align features
    X_test = data[model.get_booster().feature_names]
    y_true = data[label].values
    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    # Define bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Assign each prediction to a bin
    bin_ids = np.digitize(y_prob, bin_edges) - 1

    # Compute expected calibration error (ECE)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.sum() > 0:
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            ece += (mask.sum() / len(y_prob)) * abs(avg_pred - avg_true)

    # Compute standard error for each prob_pred point
    std_err = []
    for p in prob_pred:
        # find bin index corresponding to this prob_pred
        idx = np.searchsorted(bin_edges, p, side='right') - 1
        idx = min(max(idx, 0), n_bins-1)
        mask = bin_ids == idx
        if mask.sum() > 0:
            avg_true = y_true[mask].mean()
            se = np.sqrt(avg_true * (1 - avg_true) / mask.sum())
        else:
            se = 0.0
        std_err.append(se)

    errors[label] = ece
    # Plot with error bars
    plt.errorbar(prob_pred, prob_true, yerr=std_err, fmt='o-', capsize=3,
                 label=f'{label} (ECE={ece:.3f})')

# Perfect calibration line
plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.xlabel('Mean Predicted Value')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves')
plt.legend(loc='best')
plt.tight_layout()
if save_figs:
    plt.savefig(os.path.join(figs_dir, 'calibration_curves.svg'))
    print(f'Calibration curves saved to {figs_dir}/calibration_curves.svg')
else:
    plt.show()
    print(f'Calibration curves displayed interactively.')
plt.close()

if save_ece:
    # Write calibration errors to JSON
    with open(calib_error_path, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f'Calibration curves saved to {figs_dir}/calibration_curves.svg')
    print(f'Calibration errors saved to {calib_error_path}')
else:
    print("Calibration errors not saved. Set save_ece=True to enable.")
    print("Calibration errors:", json.dumps(errors, indent=2))

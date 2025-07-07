import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
import joblib

# Ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Paths
thresholds_path = os.path.join(repo_root, 'data', 'thresholds_AD_nAD.json')
model_dir = os.path.join(repo_root, 'data' ,'model')
test_data_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_test_data_filtered.csv')
output_csv = os.path.join(repo_root, 'data', 'results_data', 'thresholded_performance.csv')

# Load thresholds
with open(thresholds_path, 'r') as f:
    thresholds = json.load(f)

# Load test data
data = pd.read_csv(test_data_path)
# True labels
y_true_AD = data['AD'].values
y_true_nAD = data['nAD'].values

# Load features from config
from utils.load_data import load_feature_configs
feature_config = load_feature_configs(os.path.join(repo_root, 'data', 'feature_config.json'))
all_features = feature_config['all_features']

def preprocess_test(df, model):
    # ensure same feature order
    feats = model.get_booster().feature_names
    return df[feats]

# Load models
model_AD = joblib.load(os.path.join(model_dir, 'xgb_model_AD.joblib'))
model_nAD = joblib.load(os.path.join(model_dir, 'xgb_model_nAD.joblib'))

# Predict probabilities
X_test = data[all_features]

categorical_features = ["his_NACCAGE", "his_SEX", "his_RACE", "his_HISPANIC", "his_HISPOR"]
for col in categorical_features:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('category')

prob_AD = model_AD.predict_proba(preprocess_test(X_test, model_AD))[:, 1]
prob_nAD = model_nAD.predict_proba(preprocess_test(X_test, model_nAD))[:, 1]

# Prepare results
records = []
for label, y_true, prob in [('AD', y_true_AD, prob_AD), ('nAD', y_true_nAD, prob_nAD)]:
    for method, thresh in thresholds.get(label, {}).items():
        thr = float(thresh)
        y_pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = recall_score(y_true, y_pred)
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        prec = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        name = 'AD' if label == 'AD' else 'OIED'
        records.append({
            'label': name,
            'method': method,
            'threshold': f'{thr:.3f}',
            'true_neg': int(tn),
            'false_pos': int(fp),
            'false_neg': int(fn),
            'true_pos': int(tp),
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec,
            'recall': sens,
            'f1_score': f1,
            # 'accuracy': acc,
            # 'balanced_accuracy': bal_acc
        })

# Save to CSV
df_perf = pd.DataFrame.from_records(records)
print(df_perf)
df_perf.to_csv(output_csv, index=False)
# also write to tsv so that it can be copy-pasted into a spreadsheet
df_perf.to_csv(output_csv.replace('.csv', '.tsv'), sep='\t', index=False)
print(f'Wrote thresholded performance to {output_csv}')

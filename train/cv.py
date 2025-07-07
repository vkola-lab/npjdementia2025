import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import json
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # if notebook is in a subdir
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from utils.load_data import load_feature_configs

feature_config_path = os.path.join(repo_root, 'data', 'feature_config.json')
hyperparams_AD_path = os.path.join(repo_root, 'data', 'hyperparams_AD.json')
hyperparams_nAD_path = os.path.join(repo_root, 'data', 'hyperparams_nAD.json')
# train_data_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_data_filtered.csv')
train_data_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_data_filtered_unique_naccid.csv')
figs_dir = os.path.join(repo_root, 'figs')
os.makedirs(figs_dir, exist_ok=True)

#options
write_thresholds = False  # write thresholds to file
plot_curves = False  # plot ROC and PR curves

# Load feature config
feature_config = load_feature_configs(feature_config_path)
demo_features = feature_config['demo_features']
imag_features = feature_config['imag_features']
all_features = feature_config['all_features']
labels = feature_config['labels'] if 'labels' in feature_config else ['AD', 'nAD']

# Load hyperparameters
with open(hyperparams_AD_path, 'r') as f:
    hyperparams_AD = json.load(f)
with open(hyperparams_nAD_path, 'r') as f:
    hyperparams_nAD = json.load(f)

# Load preprocessed data
data = pd.read_csv(train_data_path)
X_train = data[all_features + ['NACCID']]
y_train = data[labels]

# Ensure categorical features are treated correctly
categorical_features = ["his_NACCAGE", "his_SEX", "his_RACE", "his_HISPANIC", "his_HISPOR"]

for col in categorical_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')

# Cross-validation
X_train_, y_train_ = shuffle(X_train, y_train)
gkf = GroupKFold(n_splits=5)
tprs = {label: [] for label in labels}
aucs = {label: [] for label in labels}
mean_fpr = np.linspace(0, 1, 100)
prs = {label: [] for label in labels}
aps = {label: [] for label in labels}

def get_thresholds(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # Remove infinite threshold entries
    finite_mask = np.isfinite(thresholds)
    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    thresholds = thresholds[finite_mask]
    spec = 1 - fpr
    # Youden's J
    j_scores = tpr - fpr
    youden_idx = np.argmax(j_scores)
    youden_thresh = thresholds[youden_idx]
    
    # 95% sensitivity
    # make sure that tpr is sorted in ascending order
    assert np.all(np.diff(tpr) >= 0), "tpr should be sorted in ascending order"
    sens_idx_95 = np.where(tpr >= 0.95)[0]
    if sens_idx_95.size == 0: raise ValueError("No threshold found for 95% sensitivity")
    elif sens_idx_95.size >= 1: sens_thresh_95 = thresholds[sens_idx_95[0]] # take the first threshold that meets the condition
    # 80% sensitivity
    sens_idx_80 = np.where(tpr >= 0.80)[0]
    if sens_idx_80.size == 0: raise ValueError("No threshold found for 80% sensitivity")
    elif sens_idx_80.size >= 1: sens_thresh_80 = thresholds[sens_idx_80[0]] # take the first threshold that meets the condition


    # 95% specificity
    # make sure that spec is sorted in descending order
    assert np.all(np.diff(spec) <= 0), "spec should be sorted in descending order"
    spec_idx_95 = np.where(spec >= 0.95)[0]
    if spec_idx_95.size == 0: raise ValueError("No threshold found for 95% specificity")
    elif spec_idx_95.size >= 1: spec_thresh_95 = thresholds[spec_idx_95[-1]] # take the last threshold that meets the condition
    # 80% specificity
    spec_idx_80 = np.where(spec >= 0.80)[0]
    if spec_idx_80.size == 0: raise ValueError("No threshold found for 80% specificity")
    elif spec_idx_80.size >= 1: spec_thresh_80 = thresholds[spec_idx_80[-1]] # take the last threshold that meets the condition
    
    return {
        'youden': float(youden_thresh),
        'sens95': float(sens_thresh_95),
        'spec95': float(spec_thresh_95),
        'sens80': float(sens_thresh_80),
        'spec80': float(spec_thresh_80)
    }

# Store thresholds for each fold
thresholds_per_fold = {label: {'youden': [], 'sens95': [], 'spec95': [], 'sens80': [], 'spec80': []} for label in labels}

for fold, (train_index, test_index) in enumerate(gkf.split(X_train_, y_train_, groups=X_train_['NACCID'])):
    X_train_fold = X_train_.iloc[train_index][demo_features + imag_features]
    X_test_fold = X_train_.iloc[test_index][demo_features + imag_features]
    y_train_fold = y_train_.iloc[train_index]
    y_test_fold = y_train_.iloc[test_index]
    
    xgb_model_AD = xgb.XGBClassifier(enable_categorical=True, verbosity=2, **hyperparams_AD)
    xgb_model_nAD = xgb.XGBClassifier(enable_categorical=True, verbosity=2, **hyperparams_nAD)
    xgb_model_AD.fit(X_train_fold, y_train_fold['AD'])
    xgb_model_nAD.fit(X_train_fold, y_train_fold['nAD'])
    y_pred_proba_AD = xgb_model_AD.predict_proba(X_test_fold)[:, 1]
    y_pred_proba_nAD = xgb_model_nAD.predict_proba(X_test_fold)[:, 1]
    for y_test_fold_, y_pred_proba_, label in zip([y_test_fold['nAD'], y_test_fold['AD']], [y_pred_proba_nAD, y_pred_proba_AD], ['nAD', 'AD']):
        fpr, tpr, _ = roc_curve(y_test_fold_, y_pred_proba_)
        tprs[label].append(np.interp(mean_fpr, fpr, tpr))
        tprs[label][-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs[label].append(roc_auc)
        precision, recall, _ = precision_recall_curve(y_test_fold_, y_pred_proba_)
        pr_auc = average_precision_score(y_test_fold_, y_pred_proba_)
        prs[label].append((recall, precision))
        aps[label].append(pr_auc)
        # Compute thresholds for this fold
        ths = get_thresholds(y_test_fold_, y_pred_proba_)
        for k in ths:
            thresholds_per_fold[label][k].append(ths[k])

# Aggregate thresholds across folds (mean)
thresholds_agg = {}
for label in labels:
    thresholds_agg[label] = {k: float(np.mean(thresholds_per_fold[label][k])) for k in thresholds_per_fold[label]}

# Write thresholds to JSON
if write_thresholds:
    thresholds_path = os.path.join(repo_root, 'data', f'thresholds_AD_nAD.json')
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds_agg, f, indent=2)
    print(f"Thresholds written to {thresholds_path}")
else:
    print("Thresholds not written to file. Set write_thresholds=True to enable.")
    print("Thresholds aggregated across folds:")
    print(json.dumps(thresholds_agg, indent=2))

# Plot ROC curves
plt.figure(figsize=(5, 5))
label_name_dict = {'AD': 'AD', 'nAD': 'OIED'}
for label in labels:
    mean_tpr = np.mean(tprs[label], axis=0)
    std_tpr = np.std(tprs[label], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs[label])
    plt.plot(mean_fpr, mean_tpr, label=f'ROC curve {label_name_dict.get(label)} (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)
    color = plt.gca().lines[-1].get_color()
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2, label=f'± 1 SD')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
if plot_curves:
    plt.savefig(os.path.join(figs_dir, 'cv_roc_auc.svg'))
else:
    print("ROC curves not saved. Set plot_curves=True to enable.")
plt.show()

# Plot PR curves
plt.figure(figsize=(5, 5))
for label in labels:
    mean_ap = np.mean(aps[label])
    std_ap = np.std(aps[label])
    mean_pr_curve = np.mean([np.interp(np.linspace(0, 1, 100), pr[0][::-1], pr[1][::-1]) for pr in prs[label]], axis=0)
    plt.plot(np.linspace(0, 1, 100), mean_pr_curve, label=f'PR curve {label_name_dict.get(label)} (AP = {mean_ap:.2f} ± {std_ap:.2f})', lw=2)
    color = plt.gca().lines[-1].get_color()
    pr_upper = np.minimum(mean_pr_curve + std_ap, 1)
    pr_lower = np.maximum(mean_pr_curve - std_ap, 0)
    plt.fill_between(np.linspace(0, 1, 100), pr_lower, pr_upper, color=color, alpha=0.2, label=f'± 1 SD')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
if plot_curves:
    plt.savefig(os.path.join(figs_dir, 'cv_pr_auc.svg'))
else:
    print("PR curves not saved. Set plot_curves=True to enable.")
plt.show()
import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning

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
    test_data_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_test_data_filtered.csv')
    test_data = pd.read_csv(test_data_path)
    # Convert categorical columns
    for col in ['his_SEX', 'his_RACE', 'his_HISPANIC', 'his_HISPOR']:
        if col in test_data.columns:
            test_data[col] = test_data[col].astype('category')
    # Set CASEID index
    test_data['CASEID'] = test_data['CASEID'].str.replace('CASE_', '').astype(int)
    test_data.set_index('CASEID', inplace=True)

    # Features and labels
    X_test = test_data[all_features]
    y_test = test_data[['AD', 'nAD']]

    # Load trained models
    model_dir = os.path.join(repo_root, 'data', 'dev-model')
    ad_model = joblib.load(os.path.join(model_dir, 'xgb_model_AD.joblib'))
    nad_model = joblib.load(os.path.join(model_dir, 'xgb_model_nAD.joblib'))

    # Align test features for each model and predict probabilities
    X_test_ad = X_test[ad_model.get_booster().feature_names]
    X_test_nad = X_test[nad_model.get_booster().feature_names]
    prob_ad = ad_model.predict_proba(X_test_ad)[:, 1]
    prob_nad = nad_model.predict_proba(X_test_nad)[:, 1]

    # Combined mixed-dementia probability (joint probability)
    prob_mixed = prob_ad * prob_nad
    # True mixed-dementia label: both AD and nAD present
    y_mixed = (y_test['AD'] == 1) & (y_test['nAD'] == 1)

    # print the incidence of mixed dementia
    print(f"Mixed dementia incidence in test set: {y_mixed.mean():.2%}")

    # Evaluate
    roc_auc = roc_auc_score(y_mixed, prob_mixed)
    aupr = average_precision_score(y_mixed, prob_mixed)
    print(f"Mixed dementia ROC AUC: {roc_auc:.3f}")
    print(f"Mixed dementia AUPR: {aupr:.3f}")

    # print AD and nAD AUROC and AUPR for debugging
    ad_roc_auc = roc_auc_score(y_test['AD'], prob_ad)
    nad_roc_auc = roc_auc_score(y_test['nAD'], prob_nad)
    ad_aupr = average_precision_score(y_test['AD'], prob_ad)
    nad_aupr = average_precision_score(y_test['nAD'], prob_nad)
    print(f"AD ROC AUC: {ad_roc_auc:.3f}, AUPR: {ad_aupr:.3f}")
    print(f"nAD ROC AUC: {nad_roc_auc:.3f}, AUPR: {nad_aupr:.3f}")
    
    # Threshold-based mixed dementia performance
    thresholds_path = os.path.join(repo_root, 'data', 'thresholds_AD_nAD.json')
    with open(thresholds_path, 'r') as f:
        th = json.load(f)
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
    print("\nMixed Dementia Performance at various thresholds:")
    for method in th['AD'].keys():
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always", UndefinedMetricWarning)
            try:
                ad_t = th['AD'][method]
                nad_t = th['nAD'][method]
                pred_mixed = (prob_ad >= ad_t) & (prob_nad >= nad_t)
                y_true_mixed = y_mixed.astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true_mixed, pred_mixed).ravel()
                sens = recall_score(y_true_mixed, pred_mixed)
                spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
                prec = precision_score(y_true_mixed, pred_mixed)
                f1 = f1_score(y_true_mixed, pred_mixed)
                acc = accuracy_score(y_true_mixed, pred_mixed)
                bal_acc = balanced_accuracy_score(y_true_mixed, pred_mixed)
                print(f"{method}: thr_AD={ad_t:.3f}, thr_nAD={nad_t:.3f} | Sens={sens:.2f}, Spec={spec:.2f}, Prec={prec:.2f}, F1={f1:.2f}, Acc={acc:.2f}, BalAcc={bal_acc:.2f}")
                for warn in wlist:
                    if issubclass(warn.category, UndefinedMetricWarning):
                        print(f"{method}: Undefined metric due to no positive predictions")
            except ValueError:
                print(f"{method}: Undefined metric due to no positive predictions")
    
if __name__ == '__main__':
    main()
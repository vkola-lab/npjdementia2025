import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
import wandb
from utils.load_data import load_feature_configs

# Load feature config
feature_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'feature_config.json')
feature_config = load_feature_configs(feature_config_path)
demo_features = feature_config['demo_features']
imag_features = feature_config['imag_features']
all_features = feature_config['all_features']
labels = feature_config['labels'] if 'labels' in feature_config else ['AD', 'nAD']

# Load preprocessed data
train_data_path = '/projectnb/vkolagrp/spuduch/ml_data_filtered.csv'
data = pd.read_csv(train_data_path)
X_train = data[demo_features + imag_features + ['NACCID']]
y_train = data[labels]

def train_model():
    wandb.init()
    X_train_, y_train_ = shuffle(X_train, y_train)
    gkf = GroupKFold(n_splits=5)
    roc_auc_scores = []
    pr_auc_scores = []
    for train_index, test_index in gkf.split(X_train_, y_train_, groups=X_train_['NACCID']):
        X_train_fold = X_train_.iloc[train_index][demo_features + imag_features]
        X_test_fold = X_train_.iloc[test_index][demo_features + imag_features]
        y_train_fold = y_train_.iloc[train_index]['nAD']
        y_test_fold = y_train_.iloc[test_index]['nAD']
        xgb_model = xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            verbosity=2,
            n_jobs=4,
            max_depth=wandb.config.max_depth,
            learning_rate=wandb.config.learning_rate,
            n_estimators=wandb.config.n_estimators,
            gamma=wandb.config.gamma,
            subsample=wandb.config.subsample,
            colsample_bytree=wandb.config.colsample_bytree,
            min_child_weight=wandb.config.min_child_weight,
            reg_lambda=wandb.config.reg_lambda,
            reg_alpha=wandb.config.reg_alpha
        )
        xgb_model.fit(X_train_fold, y_train_fold)
        y_pred = xgb_model.predict_proba(X_test_fold)
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            roc_auc = roc_auc_score(np.array(y_test_fold), y_pred[:, 1])
            pr_auc = average_precision_score(np.array(y_test_fold), y_pred[:, 1])
        except ValueError as e:
            print(f"Error: {e}")
            roc_auc = np.nan
            pr_auc = np.nan
        wandb.log({'roc_auc': roc_auc, 'pr_auc': pr_auc})
        roc_auc_scores.append(roc_auc)
        pr_auc_scores.append(pr_auc)
    wandb.log({'mean_roc_auc': np.mean(roc_auc_scores), 'mean_pr_auc': np.mean(pr_auc_scores)})
    wandb.finish()

sweep_config = {
    "method": "bayes",
    "metric": {"name": "mean_roc_auc", "goal": "maximize"},
    "parameters": {
        "max_depth": {"values": [3, 5, 7, 10, 15]},
        "learning_rate": {"distribution": "uniform", "min": 0.01, "max": 0.2},
        "n_estimators": {"values": [200, 300, 400, 500, 600, 700, 800, 900, 1000]},
        "gamma": {"distribution": "uniform", "min": 0, "max": 0.5},
        "subsample": {"min": 0.5, "max": 1.0, "distribution": "uniform"},
        "colsample_bytree": {"min": 0.3, "max": 0.8, "distribution": "uniform"},
        "min_child_weight": {"min": 1, "max": 10, "distribution": "uniform"},
        "reg_lambda": {"min": 0, "max": 1, "distribution": "uniform"},
        "reg_alpha": {"min": 0, "max": 1, "distribution": "uniform"}
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="xgboost_nAD_5fold_cv")
    wandb.agent(sweep_id=sweep_id, function=train_model)

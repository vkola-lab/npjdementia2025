import os
import pandas as pd
import xgboost as xgb
import joblib
import json
import numpy as np
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from utils.load_data import load_feature_configs

# toggle saving models and training mixed
save_models = True
test_models = True
train_mixed = True  # set to True to train mixed-dementia model
model_output_dir = os.path.join(repo_root, 'data', 'dev-model')

# Load feature config
feature_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'feature_config.json')
feature_config = load_feature_configs(feature_config_path)
demo_features = feature_config['demo_features']
imag_features = feature_config['imag_features']
all_features = feature_config['all_features']
labels = feature_config['labels'] if 'labels' in feature_config else ['AD', 'nAD']

# Load preprocessed data
train_data_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_data_filtered_unique_naccid.csv')
# path for test set
test_data_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_test_data_filtered.csv')

data = pd.read_csv(train_data_path)
X_train = data[demo_features + imag_features]
y_train = data[labels]

# Hyperparameters load from config
hyperparams_AD_path = os.path.join(repo_root, 'data', 'hyperparams_AD.json')
hyperparams_nAD_path = os.path.join(repo_root, 'data', 'hyperparams_nAD.json')
with open(hyperparams_AD_path, 'r') as f:
    hyperparams_AD = json.load(f)
with open(hyperparams_nAD_path, 'r') as f:
    hyperparams_nAD = json.load(f)

# Ensure categorical features are treated correctly
categorical_features = ["his_NACCAGE", "his_SEX", "his_RACE", "his_HISPANIC", "his_HISPOR"]
for col in categorical_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')

# Final model training
xgb_model_AD = xgb.XGBClassifier(enable_categorical=True, verbosity=2, **hyperparams_AD)
xgb_model_nAD = xgb.XGBClassifier(enable_categorical=True, verbosity=2, **hyperparams_nAD)
models_to_save = {'AD': xgb_model_AD, 'nAD': xgb_model_nAD}

xgb_model_AD.fit(X_train, y_train['AD'])
xgb_model_nAD.fit(X_train, y_train['nAD'])

# Train mixed-dementia model if toggled
if train_mixed:
    y_train_mixed = (y_train['AD'] & y_train['nAD']).astype(int)
    xgb_model_mixed = xgb.XGBClassifier(enable_categorical=True, verbosity=2)
    xgb_model_mixed.fit(X_train, y_train_mixed)
    models_to_save['mixed'] = xgb_model_mixed

# Evaluate on test set
if test_models:
    from sklearn.metrics import roc_auc_score, average_precision_score
    import pandas as pd
    test_df = pd.read_csv(test_data_path) if os.path.exists(test_data_path) else None
    if test_df is not None:
        for col in ['his_NACCAGE', 'his_SEX', 'his_RACE', 'his_HISPANIC', 'his_HISPOR']:
            if col in test_df.columns:
                test_df[col] = test_df[col].astype('category')
        X_test = test_df[demo_features + imag_features]
        y_test = test_df[labels]
        # AD
        prob_ad_test = xgb_model_AD.predict_proba(X_test)[:,1]
        print(f"Test AD ROC AUC: {roc_auc_score(y_test['AD'], prob_ad_test):.3f}, AUPR: {average_precision_score(y_test['AD'], prob_ad_test):.3f}")
        # nAD
        prob_nad_test = xgb_model_nAD.predict_proba(X_test)[:,1]
        print(f"Test nAD ROC AUC: {roc_auc_score(y_test['nAD'], prob_nad_test):.3f}, AUPR: {average_precision_score(y_test['nAD'], prob_nad_test):.3f}")
        # mixed dementia
        if train_mixed:
            y_test_mixed = (y_test['AD'] & y_test['nAD']).astype(int)
            prob_mixed_test = xgb_model_mixed.predict_proba(X_test)[:,1]
            print(f"Test Mixed ROC AUC: {roc_auc_score(y_test_mixed, prob_mixed_test):.3f}, AUPR: {average_precision_score(y_test_mixed, prob_mixed_test):.3f}")

# Save models if requested
if save_models:
    os.makedirs(model_output_dir, exist_ok=True)
    for name, model in models_to_save.items():
        joblib.dump(model, os.path.join(model_output_dir, f'xgb_model_{name}.joblib'))
        model.save_model(os.path.join(model_output_dir, f'xgb_model_{name}.json'))
    print(f'Final models trained and saved to {model_output_dir}')
else:
    print('Final models trained; NOT saved.')

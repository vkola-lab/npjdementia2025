# %%
import pandas as pd 
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import os
print(os.environ)

print(np.__version__)
print(pd.__version__)
pd.set_option('display.max_columns', None)

# data = pd.read_csv('/projectnb/vkolagrp/spuduch/ml_data(w_nodate)(threshold_microbleed).csv')
data = pd.read_csv('/projectnb/vkolagrp/spuduch/ml_data_updated.csv')

#max diff between 'ImageDate' and 'VisitDate', which are both datetime
data['ImageDate'] = pd.to_datetime(data['ImageDate'])
data['VisitDate'] = pd.to_datetime(data['VisitDate'])
data['diff'] = (data['VisitDate'] - data['ImageDate']).dt.days
print(data['diff'].max())



# data['mri_name_mb'].notna().sum()
# %%
print(data.shape)
# drop rows if any of the following columns are equal to the string 'nodate': 'month_imag', 'day_imag','year_imag' AND 'mri_name_mb' is not null
data = data[~(data['month_imag'] == 'nodate')]
print(data.shape)
# %%
# data = pd.read_csv('/projectnb/vkolagrp/spuduch/ml_data(w_nodate).csv')
# test_data = pd.read_csv('/projectnb/vkolagrp/spuduch/ml_test_data(threshold_microbleed).csv')
test_data = pd.read_csv('/projectnb/vkolagrp/spuduch/ml_test_data_updated.csv')

data = data[data['DE'] == 1]
test_data = test_data[test_data['DE'] == 1]
# clinician_review_cases_test = pd.read_csv('/projectnb/vkolagrp/spuduch/clinician_review_cases_test.csv')

# clinician_review_cases_test = clinician_review_cases_test[clinician_review_cases_test['DE'] == 1]


data['his_NACCAGE'] = data['NACCAGE'].replace({-4: np.nan, '-4':np.nan}).astype(float)
data['his_SEX'] = data['SEX'].replace({1: 'male', 2: 'female', -4: np.nan, '-4':np.nan})
data['his_RACE'] = data['RACE'].replace({1:'whi', 2:'blk', 3:'ind', 4:'haw', 5:'asi', 50:'oth', 99:np.nan, -4: np.nan, '-4':np.nan})
data['his_HISPANIC'] = data['HISPANIC'].replace({0: 'no', 1: 'yes', 9:np.nan, -4: np.nan, '-4':np.nan, 'nan': np.nan})
data['his_HISPOR'] = data['HISPOR'].replace({50: 7, 88: 0, 99: np.nan, -4: np.nan, '-4':np.nan})
data['his_EDUC'] = data['EDUC'].replace({99: np.nan, -4: np.nan, '-4':np.nan}).astype(float)

test_data['his_NACCAGE'] = test_data['NACCAGE'].replace({-4: np.nan, '-4':np.nan}).astype(float)
test_data['his_SEX'] = test_data['SEX'].replace({1: 'male', 2: 'female', -4: np.nan, '-4':np.nan})
test_data['his_RACE'] = test_data['RACE'].replace({1:'whi', 2:'blk', 3:'ind', 4:'haw', 5:'asi', 50:'oth', 99:np.nan, -4: np.nan, '-4':np.nan})
test_data['his_HISPANIC'] = test_data['HISPANIC'].replace({0: 'no', 1: 'yes', 9:np.nan, -4: np.nan, '-4':np.nan, 'nan': np.nan})
test_data['his_HISPOR'] = test_data['HISPOR'].replace({50: 7, 88: 0, 99: np.nan, -4: np.nan, '-4':np.nan})
test_data['his_EDUC'] = test_data['EDUC'].replace({99: np.nan, -4: np.nan, '-4':np.nan}).astype(float)
# replae np.NaN with np.nan in test_data
test_data = test_data.replace({pd.NA: np.nan})

demo_features = [
    'his_NACCAGE', 'his_SEX', 'his_RACE', 'his_HISPANIC', 'his_HISPOR', 'his_EDUC',
]

for f in demo_features:
    print(f'{f}: {data[f].unique()}')
    print(f'{f}: {test_data[f].unique()}')
# feature columns
region_names = [ 
 '3rd-Ventricle',
 '4th-Ventricle',
 'Brain-Stem',
 'CSF',
 'Cbm_Left_CrusI',
 'Cbm_Left_CrusII',
 'Cbm_Left_IX',
 'Cbm_Left_I_IV',
 'Cbm_Left_V',
 'Cbm_Left_VI',
 'Cbm_Left_VIIIa',
 'Cbm_Left_VIIIb',
 'Cbm_Left_VIIb',
 'Cbm_Left_X',
 'Cbm_Right_CrusI',
 'Cbm_Right_CrusII',
 'Cbm_Right_IX',
 'Cbm_Right_I_IV',
 'Cbm_Right_V',
 'Cbm_Right_VI',
 'Cbm_Right_VIIIa',
 'Cbm_Right_VIIIb',
 'Cbm_Right_VIIb',
 'Cbm_Right_X',
 'Cbm_Vermis',
 'Cbm_Vermis_IX',
 'Cbm_Vermis_VI',
 'Cbm_Vermis_VII',
 'Cbm_Vermis_VIII',
 'Cbm_Vermis_X',
 'Left-Accumbens-area',
 'Left-Amygdala',
 'Left-Caudate',
 'Left-Cerebellum-Cortex',
 'Left-Cerebellum-White-Matter',
 'Left-Cerebral-White-Matter',
 'Left-Hippocampus',
 'Left-Inf-Lat-Vent',
 'Left-Lateral-Ventricle',
 'Left-Pallidum',
 'Left-Putamen',
 'Left-Thalamus',
 'Left-VentralDC',
 'Left-choroid-plexus',
 'Right-Accumbens-area',
 'Right-Amygdala',
 'Right-Caudate',
 'Right-Cerebellum-Cortex',
 'Right-Cerebellum-White-Matter',
 'Right-Cerebral-White-Matter',
 'Right-Hippocampus',
 'Right-Inf-Lat-Vent',
 'Right-Lateral-Ventricle',
 'Right-Pallidum',
 'Right-Putamen',
 'Right-Thalamus',
 'Right-VentralDC',
 'Right-choroid-plexus',
 'WM-hypointensities',
 'ctx-lh-caudalanteriorcingulate',
 'ctx-lh-caudalmiddlefrontal',
 'ctx-lh-cuneus',
 'ctx-lh-entorhinal',
 'ctx-lh-fusiform',
 'ctx-lh-inferiorparietal',
 'ctx-lh-inferiortemporal',
 'ctx-lh-insula',
 'ctx-lh-isthmuscingulate',
 'ctx-lh-lateraloccipital',
 'ctx-lh-lateralorbitofrontal',
 'ctx-lh-lingual',
 'ctx-lh-medialorbitofrontal',
 'ctx-lh-middletemporal',
 'ctx-lh-paracentral',
 'ctx-lh-parahippocampal',
 'ctx-lh-parsopercularis',
 'ctx-lh-parsorbitalis',
 'ctx-lh-parstriangularis',
 'ctx-lh-pericalcarine',
 'ctx-lh-postcentral',
 'ctx-lh-posteriorcingulate',
 'ctx-lh-precentral',
 'ctx-lh-precuneus',
 'ctx-lh-rostralanteriorcingulate',
 'ctx-lh-rostralmiddlefrontal',
 'ctx-lh-superiorfrontal',
 'ctx-lh-superiorparietal',
 'ctx-lh-superiortemporal',
 'ctx-lh-supramarginal',
 'ctx-lh-transversetemporal',
 'ctx-rh-caudalanteriorcingulate',
 'ctx-rh-caudalmiddlefrontal',
 'ctx-rh-cuneus',
 'ctx-rh-entorhinal',
 'ctx-rh-fusiform',
 'ctx-rh-inferiorparietal',
 'ctx-rh-inferiortemporal',
 'ctx-rh-insula',
 'ctx-rh-isthmuscingulate',
 'ctx-rh-lateraloccipital',
 'ctx-rh-lateralorbitofrontal',
 'ctx-rh-lingual',
 'ctx-rh-medialorbitofrontal',
 'ctx-rh-middletemporal',
 'ctx-rh-paracentral',
 'ctx-rh-parahippocampal',
 'ctx-rh-parsopercularis',
 'ctx-rh-parsorbitalis',
 'ctx-rh-parstriangularis',
 'ctx-rh-pericalcarine',
 'ctx-rh-postcentral',
 'ctx-rh-posteriorcingulate',
 'ctx-rh-precentral',
 'ctx-rh-precuneus',
 'ctx-rh-rostralanteriorcingulate',
 'ctx-rh-rostralmiddlefrontal',
 'ctx-rh-superiorfrontal',
 'ctx-rh-superiorparietal',
 'ctx-rh-superiortemporal',
 'ctx-rh-supramarginal',
 'ctx-rh-transversetemporal',
]
# imag_features actually needs _vol and _wmh appended to each
imag_features = []
vol_features = []
wmh_features = []
mb_features = []
for region in region_names:
    imag_features.append(region + '_vol')
    vol_features.append(region + '_vol')
    imag_features.append(region + '_wmh')
    wmh_features.append(region + '_wmh')
    # imag_features.append(region + '_mb')
    mb_features.append(region + '_mb')
imag_features.append('total_wm_burden')
wmh_features.append('total_wm_burden')
# imag_features.append('total_mb_burden')
mb_features.append('total_mb_burden')

imag_features = [f for f in imag_features if f in data.columns]
wmh_features = [f for f in wmh_features if f in data.columns]
vol_features = [f for f in vol_features if f in data.columns]
mb_features = [f for f in mb_features if f in data.columns]

print(imag_features)
# check if any of the features are missing
missing_features = set([col for col in data.columns if ('vol' in col or 'wmh' in col or 'mb' in col)]) - set(imag_features + ['mri_name_wmh', 'mri_name_vol', 'fname_wmh', 'NACCID_wmh', 'dy_wmh', 'mo_wmh', 'yr_wmh', 'NACCID_mb', 'dy_mb', 'yr_mb', 'mri_name_mb', 'mo_mb', 'fname_mb'])
# assert len(missing_features) == 0, f"Missing features: {missing_features}"
# # labels
labels = [
    'AD',
    'LBD',
    'VD',
    'PRD',
    'FTD',
    'NPH',
    'SEF',
    'PSY',
    'TBI',
    'ODE'
]
# check if any of the labels are missing
missing_labels = set(labels) - set(data.columns)
# assert len(missing_labels) == 0, f"Missing labels: {missing_labels}"
# define nAD as the presence of any of the following: LBD, VD, PRD, FTD, NPH, SEF, PSY, TBI, ODE
# data['nAD'] = data['LBD'] | data['VD'] | data['PRD'] | data['FTD'] | data['NPH'] | data['SEF'] | data['PSY'] | data['TBI'] | data['ODE']


# convert his_SEX, his_RACE, his_HISPANIC, his_HISPOR to category
data = data.astype({'his_SEX': 'category', 'his_RACE': 'category', 'his_HISPANIC': 'category', 'his_HISPOR': 'category'})
test_data = test_data.astype({'his_SEX': 'category', 'his_RACE': 'category', 'his_HISPANIC': 'category', 'his_HISPOR': 'category'})

print("data demo dtypes:")
print(data[demo_features].dtypes)
print("test data demo dtypes:")
print(test_data[demo_features].dtypes)

# assert that all imag features are float
for f in imag_features:
    assert data[f].dtype == 'float64', f"Expected {f} to be float64, got {data[f].dtype}"
    assert test_data[f].dtype == 'float64', f"Expected {f} to be float64, got {test_data[f].dtype}"
# %%
days = data['DateDifference']
days = days.apply(lambda x: x[:-5])
days = days.astype(int)

days.mean()

# %%
test_data.rename(columns = {'ID': 'NACCID'}, inplace = True)
# test_data.sort_values(by=['NACCID', 'mri_name_vol', 'mri_name_wmh', 'mri_name_mb'], ascending=[True, False, False, False], key=lambda x: x.notnull(), inplace=True)
test_data.sort_values(by=['NACCID', 'mri_name_vol', 'mri_name_wmh'], ascending=[True, False, False], key=lambda x: x.notnull(), inplace=True)

test_data = test_data.drop_duplicates(subset=['NACCID'])

# print the number of non-null values in Sequence_type_vol, Sequence_type_wmh, and Sequence_type_mb
# print(test_data[['mri_name_vol', 'mri_name_wmh', 'mri_name_mb']].notnull().sum())
print(test_data[['mri_name_vol', 'mri_name_wmh']].notnull().sum())
# %%

# test_data = test_data.drop_duplicates(subset=['NACCID'])
print(test_data[['mri_name_vol', 'mri_name_wmh']])
# print the breakdown of the labels
for label in labels:
    print(f'{label}: {data[label].sum()}')
print('\n')
for label in labels:
    print(f'{label}: {test_data[label].sum()}')

test_NACCIDs = set(test_data['NACCID'].unique())
print(f'Number of unique NACCIDs in clinician review: {len(test_NACCIDs)}')
print(test_NACCIDs)

# %%
# # pick the row from the test data where the mri_zip (stripped of the zip part) in clinician_review matches the fname in the test data
# clinician_review_cases_test['fname'] = clinician_review_cases_test['mri_zip'].str.replace('.zip', '')
# clinician_review_cases_test.rename(columns={'ID': 'NACCID'}, inplace=True)
# test_data.drop(columns=labels, inplace=True)
# print('AD' in clinician_review_cases_test.columns)
# print('AD' in test_data.columns)
# # test_data = test_data.merge(clinician_review_cases_test, on=['fname'], how='inner')

# # Perform right merge
# right_merged = test_data.merge(clinician_review_cases_test, on=['fname'], how='right')
# # print("\nRight Merge Result:")
# # print(right_merged)

# # Perform inner merge
# inner_merged = test_data.merge(clinician_review_cases_test, on=['fname'], how='inner')
# # print("Inner Merge Result:")
# # print(inner_merged)


# # Find rows only in the right merge but not in the inner merge
# # These rows will have nulls in the columns of the left DataFrame (test_data)
# rows_in_right_not_in_inner = right_merged[right_merged['imag_data_index'].isnull()]
# print("\nRows in Right Merge not in Inner Merge:")
# print(rows_in_right_not_in_inner)


# %%
print('AD' in test_data.columns)
print(f"size of test data: {len(test_data)}")
data = data[~data['NACCID'].isin(test_NACCIDs)]
print("After filtering:")
for label in labels:
    print(f'{label}: {data[label].sum()}')
print("\n")
print(f"test data:")
for label in labels:
    print(f'{label}: {test_data[label].sum()}')

# %%
# are there duplicate naccid in data?
print(data['NACCID'].value_counts().sort_values(ascending=False))
data = data.groupby('NACCID').head(100)
print(data['NACCID'].value_counts().sort_values(ascending=False))
# %%
# drop rows that have duplicate mri_name_vol and mri_name_wmh

# first identify duplicates
print(data[['NACCID', 'mri_name_vol', 'mri_name_wmh']].duplicated().sum())

# data = data.drop_duplicates(subset=['NACCID', 'mri_name_vol', 'mri_name_wmh'])

# data.columns
# %%
data['nAD'] = data['VD'] | data['PRD'] | data['FTD']  #| data['SEF'] | data['PSY'] | data['TBI'] | data['ODE']
# test_data = test_data[test_data['ODE'] == 0]
test_data['nAD'] = test_data['VD'] | test_data['PRD'] | test_data['FTD']  #| test_data['SEF'] | test_data['PSY'] | test_data['TBI'] | test_data['ODE']

# check that the nAD column is correct
for label in labels:
    print(f'{label}: {data[label].sum()}')
print(f'nAD: {data["nAD"].sum()}')

# check that the nAD column is correct
for label in labels:
    print(f'{label}: {test_data[label].sum()}')
print(f'nAD: {test_data["nAD"].sum()}')

labels = [
    'AD',
    # 'VD',
    # # 'LBD',
    # 'FTD',
    # 'PRD',
    'nAD' # nAD is VD or PRD or FTD
]

data.to_csv('/projectnb/vkolagrp/spuduch/ml_data_filtered.csv', index=False)
test_data.to_csv('/projectnb/vkolagrp/spuduch/ml_test_data_filtered.csv', index=False)

# print the breakdown of the labels for unique NACCIDs
print('train:')
# first drop duplicated NACCID
data_NACCID = data.drop_duplicates(subset=['NACCID'])
for label in labels: print(f'{label}: {data_NACCID[label].sum()}')
print('test:')
for label in labels: print(f'{label}: {test_data[label].sum()}')

# %%

# # do some feature masking: there are 3 modalities: vol, wmh, mb
# # for any row that has mb data, duplicate that entry and mask the vol and wmh data

# # Step 1: Identify the rows with `mb` modality present (non-null `mri_name_mb`)
# mb_rows = data[data['mri_name_mb'].notnull()]

# # Assert that we correctly identified the rows with the `mb` modality
# assert len(mb_rows) == len(data[data['mri_name_mb'].notnull()]), "Error: mb rows were not identified correctly"

# # Step 2: Duplicate the rows
# mb_rows_masked = mb_rows.copy()

# # Step 3: Mask the `vol` and `wmh` features by setting them to NaN
# mb_rows_masked[vol_features] = None
# mb_rows_masked[wmh_features] = None

# # Assert that the vol and wmh features in the duplicated rows are indeed masked
# for feature in vol_features + wmh_features:
#     assert mb_rows_masked[feature].isnull().all(), f"Error: {feature} was not masked correctly"

# # Step 4: Append the masked rows back to the original DataFrame
# data_with_masked = pd.concat([data, mb_rows_masked], ignore_index=True)

# # Test cases to ensure correctness

# # Test Case 1: Ensure the original number of rows + masked rows is correct
# expected_rows = len(data) + len(mb_rows)
# assert len(data_with_masked) == expected_rows, f"Error: Expected {expected_rows} rows, but got {len(data_with_masked)}"

# # Test Case 3: Ensure that the masked rows have the correct features masked
# for i in range(len(mb_rows)):
#     original_row_index = len(data) + i
#     original_row = data_with_masked.loc[original_row_index]
#     for feature in vol_features + wmh_features:
#         assert pd.isnull(original_row[feature]), f"Error: {feature} was not masked in duplicated row"
#     for feature in mb_features:
#         same = np.isclose(original_row[feature],mb_rows.iloc[i][feature])
#         if np.isnan(original_row[feature]) and np.isnan(mb_rows.iloc[i][feature]): same = True
#         assert same, f"Error: {feature} was incorrectly altered"

# # Print the final DataFrame for visual inspection (optional)
# print(data_with_masked)

# data = data_with_masked

# %%
# shuffle the data

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X_train = data[demo_features + imag_features + ['NACCID']]
y_train = data[labels]



X_test = test_data[demo_features + imag_features + ['NACCID']]
y_test = test_data[labels]


# %%
# print the NACCID with the most duplicates, sort by the NACCID wiht the most number of duplicates. print NACCID, and the number of duplicates, and the labels



# print(y_test)
# print the breakdown of the labels

# print('train:')
# # for label in labels: print(f'{label}: {y_train[label].sum()}')
# # print just AD, just nAD, and both
# print(f'AD: {y_train["AD"].sum()}')
# print(f'nAD: {y_train["nAD"].sum()}')
# print(f'Both: {(y_train["AD"] & y_train["nAD"]).sum()}')
# print('\n')
# print('test:')
# # for label in labels: print(f'{label}: {y_test[label].sum()}')
# # print just AD, just nAD, and both
# print(f'AD: {y_test["AD"].sum()}')
# print(f'nAD: {y_test["nAD"].sum()}')
# print(f'Both: {(y_test["AD"] & y_test["nAD"]).sum()}')

#%% hyperparams
hyperparams_nAD = {
  "max_depth": {"value": 10},
  "reg_alpha": {"value": 0.6133830639963205},
  "subsample": {"value": 0.7697909469413156},
  "reg_lambda": {"value": 0.01177052687225999},
  "n_estimators": {"value": 800},
  "learning_rate": {"value": 0.022605809313261414},
  "colsample_bytree": {"value": 0.35608984547073275},
  "min_child_weight": {"value": 4.331626469084744}
}
hyperparams_AD = {
  "max_depth": {"value": 7},
  "reg_alpha": {"value": 0.01279683607486526},
  "subsample": {"value": 0.9891093871482132},
  "reg_lambda": {"value": 0.02444612167772997},
  "n_estimators": {"value": 400},
  "learning_rate": {"value": 0.025486390351977094},
  "colsample_bytree": {"value": 0.3034053896570938},
  "min_child_weight": {"value": 1.1245785722271748}
}

hyperparams_nAD = {key: value["value"] for key, value in hyperparams_nAD.items()}
hyperparams_AD = {key: value["value"] for key, value in hyperparams_AD.items()}
# %%
xgb_model_AD = xgb.XGBClassifier(enable_categorical = True, verbosity = 2, tree_method="hist", **hyperparams_AD)
xgb_model_nAD = xgb.XGBClassifier(enable_categorical = True, verbosity = 2, **hyperparams_nAD)

# plot k fold cv ROC curves and PR curves for each label
# shuffle data
X_train_, y_train_ = shuffle(X_train, y_train)
gkf = GroupKFold(n_splits=5)

tprs = {label: [] for label in labels}
aucs = {label: [] for label in labels}
mean_fpr = np.linspace(0, 1, 100)

prs = {label: [] for label in labels}
aps = {label: [] for label in labels}

# Cross-validation
for fold, (train_index, test_index) in enumerate(gkf.split(X_train_, y_train_, groups=X_train_['NACCID'])):
    X_train_fold = X_train_.iloc[train_index][demo_features + imag_features]
    X_test_fold = X_train_.iloc[test_index][demo_features + imag_features]
    y_train_fold = y_train_.iloc[train_index]
    y_test_fold = y_train_.iloc[test_index]
    # print(y_train_fold)
    # print(y_test_fold)

    y_train_fold_nAD = y_train_fold['nAD']
    y_test_fold_nAD = y_test_fold['nAD']
    y_train_fold_AD = y_train_fold['AD']
    y_test_fold_AD = y_test_fold['AD']

    # print the breakdown of the labels
    print('train fold:')
    for label in labels: print(f'{label}: {y_train_fold[label].sum()}')
    print('test fold:')
    for label in labels: print(f'{label}: {y_test_fold[label].sum()}')

    # Fit the model
    xgb_model_AD.fit(X_train_fold, y_train_fold_AD)
    xgb_model_nAD.fit(X_train_fold, y_train_fold_nAD)

    # Predict probabilities
    y_pred_proba_AD = xgb_model_AD.predict_proba(X_test_fold)[:, 1]
    y_pred_proba_nAD = xgb_model_nAD.predict_proba(X_test_fold)[:, 1]
    
    # Compute ROC and PR curves for each label
    for y_test_fold_, y_pred_proba_, label in zip([y_test_fold_nAD, y_test_fold_AD], [y_pred_proba_nAD, y_pred_proba_AD], ['nAD', 'AD']):
        # Compute ROC curve and ROC area for each label
        fpr, tpr, _ = roc_curve(y_test_fold_, y_pred_proba_)
        tprs[label].append(np.interp(mean_fpr, fpr, tpr))
        tprs[label][-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        # calc roc auc with the roc auc score function
        roc_auc_s = roc_auc_score(y_test_fold_, y_pred_proba_)
        print(f'{label} | fold {fold} | ROC AUC: {roc_auc}')
        print(f'{label} | fold {fold} | ROC AUC score: {roc_auc_s}')
        aucs[label].append(roc_auc)
        
        # Compute PR curve and PR area for each label
        precision, recall, _ = precision_recall_curve(y_test_fold_, y_pred_proba_)
        pr_auc = average_precision_score(y_test_fold_, y_pred_proba_)
        prs[label].append((recall, precision))
        aps[label].append(pr_auc)
    
# print results
for label in labels:
    print(f'{label} | mean ROC AUC: {np.mean(aucs[label]):.2f} ± {np.std(aucs[label]):.2f}')
    print(f'{label} | mean PR AUC: {np.mean(aps[label]):.2f} ± {np.std(aps[label]):.2f}')

# %%
# @Krish to do: improve these ROC and AP plots for publication quality. show the standard deviation on the curve as a shaded region
import matplotlib.pyplot as plt
# Plot ROC curves
plt.figure(figsize=(5, 5))

label_name_dict = {'AD': 'AD', 'nAD': 'OIED'}
# Separate subplots for each model
for label in labels:
    # Calculate mean and standard deviation for TPRs
    mean_tpr = np.mean(tprs[label], axis=0)
    std_tpr = np.std(tprs[label], axis=0)
    mean_tpr[-1] = 1.0  # Ensure the last value is 1 for a proper ROC curve

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs[label])

    # Plot the mean ROC curve
    plt.plot(
        mean_fpr, mean_tpr,
        label=f'ROC curve {label_name_dict.get(label)} (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
        lw=2
    )

    # get the color of the current plot
    color = plt.gca().lines[-1].get_color()

    # Plotting the standard deviation as a shaded area
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2, label=f'± 1 Standard Deviation')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
# plt.title('Receiver Operating Characteristic', fontsize=10)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
# plt.show()
plt.savefig('/projectnb/vkolagrp/spuduch/plots/cv_roc_auc.svg')
# Plot Precision-Recall curves
plt.figure(figsize=(5, 5))
for label in labels:
    mean_ap = np.mean(aps[label])
    std_ap = np.std(aps[label])
    mean_pr_curve = np.mean([np.interp(np.linspace(0, 1, 100), pr[0][::-1], pr[1][::-1]) for pr in prs[label]], axis=0)
    plt.plot(np.linspace(0, 1, 100), mean_pr_curve,
             label=f'PR curve {label_name_dict.get(label)} (AP = {mean_ap:.2f} ± {std_ap:.2f})', lw=2)
    
    # get color
    color = plt.gca().lines[-1].get_color()
    # Plotting the standard deviation as a shaded area
    pr_upper = np.minimum(mean_pr_curve + std_ap, 1)
    pr_lower = np.maximum(mean_pr_curve - std_ap, 0)
    plt.fill_between(np.linspace(0, 1, 100), pr_lower, pr_upper, color=color, alpha=0.2, label=f'± 1 Standard Deviation')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
# plt.title('Precision-Recall Curve', fontsize=10)
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
# plt.show()
plt.savefig('/projectnb/vkolagrp/spuduch/plots/cv_pr_auc.svg')

# %%
xgb_model_AD = xgb.XGBClassifier(enable_categorical = True, verbosity = 2, **hyperparams_AD)
xgb_model_nAD = xgb.XGBClassifier(enable_categorical = True, verbosity = 2, **hyperparams_nAD)

xgb_model_AD.fit(X_train[demo_features + imag_features], y_train['AD'])
xgb_model_nAD.fit(X_train[demo_features + imag_features], y_train['nAD'])
# %%

# does the model use 'total_mb_burden'? print feature name and importance

# print the feature importance
feature_importance_AD = xgb_model_AD.feature_importances_
feature_importance_nAD = xgb_model_nAD.feature_importances_
feature_names = demo_features + imag_features

feature_importance_AD = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_AD})
feature_importance_nAD = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_nAD})

print(feature_importance_nAD.sort_values(by='importance', ascending=False))
# print the rows where feature includes 'mb' string
print(feature_importance_nAD[feature_importance_nAD['feature'].str.contains('mb')].sort_values(by='importance', ascending=False))


# %%
# save the model to disk at /projectnb/vkolagrp/spuduch/
import joblib
joblib.dump(xgb_model_AD, '/projectnb/vkolagrp/spuduch/xgb_model_AD.joblib')
xgb_model_AD.save_model('/projectnb/vkolagrp/spuduch/xgb_model_AD.json')

joblib.dump(xgb_model_nAD, '/projectnb/vkolagrp/spuduch/xgb_model_nAD.joblib')
xgb_model_nAD.save_model('/projectnb/vkolagrp/spuduch/xgb_model_nAD.json')
# %%

y_pred_AD = xgb_model_AD.predict_proba(X_test[demo_features + imag_features])[:,1]
y_pred_nAD = xgb_model_nAD.predict_proba(X_test[demo_features + imag_features])[:,1]
# combine them
y_pred = np.column_stack((y_pred_AD, y_pred_nAD))
y_pred = pd.DataFrame(y_pred, columns=['AD', 'nAD'])
print(y_pred)
# %%

# performance on the basis of roc auc and pr auc for each label

from sklearn.metrics import roc_auc_score, average_precision_score
roc_auc_scores = []
pr_auc_scores = []
for i, label in enumerate(labels):
    pred = y_pred[label]
    # for pred, label in zip([y_pred_AD, y_pred_nAD], ['AD', 'nAD']):
    roc_auc = roc_auc_score(y_test[label], pred)
    pr_auc = average_precision_score(y_test[label], pred)
    roc_auc_scores.append(roc_auc)
    pr_auc_scores.append(pr_auc)
    print(f'{label} | roc_auc: {roc_auc}, pr_auc: {pr_auc}')
# %%
#data leak hehe :)
# def train_model():
#     # Initialize a new wandb run
#     wandb.init()
#     # # multi output: ---------------
#     # Create the XGBClassifier model with wandb configuration
#     xgb_model = xgb.XGBClassifier(
#         tree_method="hist",
#         device='cuda',
#         enable_categorical=True,
#         verbosity=2,
#         n_jobs=4,
#         multi_strategy=wandb.config.multi_strategy,
#         max_depth=wandb.config.max_depth,
#         learning_rate=wandb.config.learning_rate,
#         n_estimators=wandb.config.n_estimators,
#         gamma=wandb.config.gamma,
#         subsample=wandb.config.subsample,
#         colsample_bytree=wandb.config.colsample_bytree,
#         min_child_weight=wandb.config.min_child_weight,
#         reg_lambda=wandb.config.reg_lambda,
#         reg_alpha=wandb.config.reg_alpha
#     )

#     # Fit the model
#     xgb_model.fit(X_train, y_train)

#     # Predict probabilities for ROC AUC and PR AUC calculations
#     y_pred = xgb_model.predict_proba(X_test)
#     assert y_pred.shape[1] == len(labels), f"Expected {len(labels)} columns in y_pred, got {y_pred.shape[1]}"
#     assert len(y_pred.shape) == 2, f"Expected 2 dimensions in y_pred, got {len(y_pred.shape)}"

#     # Calculate ROC AUC and PR AUC for each class
#     roc_auc_scores = []
#     pr_auc_scores = []

#     for i, label in enumerate(labels):
#         roc_auc = roc_auc_score(y_test[label], y_pred[:, i])
#         pr_auc = average_precision_score(y_test[label], y_pred[:, i])
#         roc_auc_scores.append(roc_auc)
#         pr_auc_scores.append(pr_auc)

#         wandb.log({f'{labels[i]}_roc_auc': roc_auc, f'{labels[i]}_pr_auc': pr_auc})

#     wandb.log({"mean_roc_auc": sum(roc_auc_scores) / len(roc_auc_scores),
#                "mean_pr_auc": sum(pr_auc_scores) / len(pr_auc_scores)})

#     # Finish the wandb run
#     wandb.finish()
#5fold cv
def train_model():
    # Initialize a new wandb run
    wandb.init()
    
    
    # Shuffle the train data but remember that X and y need to be shuffled together
    X_train_, y_train_ = shuffle(X_train, y_train)
    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    # Dicts to store list of scores
    # roc_auc_scores = {label: [] for label in labels}
    # pr_auc_scores = {label: [] for label in labels}
    roc_auc_scores = []
    pr_auc_scores = []
    # Perform k-fold cross-validation
    for train_index, test_index in gkf.split(X_train_, y_train_, groups=X_train_['NACCID']):
        X_train_fold, X_test_fold = X_train_.iloc[train_index][demo_features + imag_features], X_train_.iloc[test_index][demo_features + imag_features]
        y_train_fold, y_test_fold = y_train_.iloc[train_index]['nAD'], y_train_.iloc[test_index]['nAD']
        assert len(X_train_fold) == len(y_train_fold), f"Expected {len(X_train_fold)} rows in X_train_fold, got {len(y_train_fold)} rows in y_train_fold"
        assert len(X_test_fold) == len(y_test_fold), f"Expected {len(X_test_fold)} rows in X_test_fold, got {len(y_test_fold)} rows in y_test_fold"
        # print(y_train_fold)
        # print the sizes of the train and test sets
        print(f"train size: {len(X_train_fold)}, test size: {len(X_test_fold)}")
        
        # Create the XGBClassifier model with wandb configuration
        xgb_model = xgb.XGBClassifier(
            tree_method="hist",
            device='cuda',
            enable_categorical=True,
            verbosity=2,
            n_jobs=4,
            # multi_strategy=wandb.config.multi_strategy,
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
        # Fit the model on the fold
        xgb_model.fit(X_train_fold, y_train_fold)
        
        # Predict probabilities for ROC AUC and PR AUC calculations
        y_pred = xgb_model.predict_proba(X_test_fold)
        assert y_pred.shape[1] == 2, f"Expected 2 columns in y_pred, got {y_pred.shape[1]}"
        assert y_pred.shape[0] == len(y_test_fold), f"Expected {len(y_test_fold)} rows in y_pred, got {y_pred.shape[0]}"
        # assert np.array(y_test_fold).reshape(-1,1).shape == y_pred[:, 1].shape, f"Expected {np.array(y_test_fold).reshape(-1,1).shape} shape in y_test_fold, got {y_pred[:, 1].shape} in y_pred"
        # Calculate ROC AUC and PR AUC for each class
        # for i, label in enumerate(labels):
        #     try:
        #       roc_auc = roc_auc_score(y_test_fold[label], y_pred[:, i])
        #       pr_auc = average_precision_score(y_test_fold[label], y_pred[:, i])
        #     except ValueError:
        #       roc_auc = np.nan
        #       pr_auc = np.nan
        #     roc_auc_scores[label].append(roc_auc)
        #     pr_auc_scores[label].append(pr_auc)

        # Calculate ROC AUC and PR AUC for the single label
        # print(np.array(y_test_fold).reshape(-1,1))
        # print(y_pred[:, 1])
        try:
            roc_auc = roc_auc_score(np.array(y_test_fold), y_pred[:, 1])
            pr_auc = average_precision_score(np.array(y_test_fold), y_pred[:, 1])
        except ValueError as e:
            print(f"Error: {e}")
            roc_auc = np.nan
            pr_auc = np.nan
        print(f'roc_auc: {roc_auc}, pr_auc: {pr_auc}')
        wandb.log({f'roc_auc': roc_auc, f'pr_auc': pr_auc})
        roc_auc_scores.append(roc_auc)
        pr_auc_scores.append(pr_auc)
        
    wandb.log({f'mean_roc_auc': np.mean(roc_auc_scores), f'mean_pr_auc': np.mean(pr_auc_scores)})
    
    # for label, roc_auc_scores_list in roc_auc_scores.items():
    #     mean_roc_auc = np.nanmean(roc_auc_scores_list)
    #     wandb.log({f'{label}_roc_auc': mean_roc_auc}) 
    # for label, pr_auc_scores_list in pr_auc_scores.items():
    #     mean_pr_auc = np.nanmean(pr_auc_scores_list)
    #     wandb.log({f'{label}_pr_auc': mean_pr_auc})
    
    # mean_roc_auc = np.mean([np.nanmean(roc_auc_scores_list) for roc_auc_scores_list in roc_auc_scores.values()])
    # mean_pr_auc = np.mean([np.nanmean(pr_auc_scores_list) for pr_auc_scores_list in pr_auc_scores.values()])
    # # Log mean scores
    # wandb.log({"mean_roc_auc": mean_roc_auc, "mean_pr_auc": mean_pr_auc})
    
    # Finish the wandb run
    wandb.finish()


# Define the sweep configuration
sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "mean_roc_auc", "goal": "maximize"},  # Objective metric to maximize
    # "metric": {"name": "roc_auc", "goal": "maximize"},  # Objective metric to maximize
    # "parameters": {
    #     "max_depth": {"values": [3, 5, 7, 10]},
    #     "learning_rate": {"distribution": "uniform", "min": 0.01, "max": 0.1},
    #     "n_estimators": {"values": [100, 200, 300, 400]},
    #     "colsample_bytree": {"min": 0.3, "max": 0.8, "distribution": "uniform"}
    # },
    "parameters": {
    # "multi_strategy": {"values": ["multi_output_tree", "one_output_per_tree"]},
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

# Initialize the sweep
# sweep_id = wandb.sweep(sweep_config, project="xgboost_nAD 5-fold cv (updated)")
# print(sweep_id)
# wandb.agent(sweep_id=sweep_id, function=train_model)

# %%
# save model predicted probabilities
y_pred_df = y_pred.copy()
# remove index
y_pred_df.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)
y_pred_df['CASEID'] = test_data['CASEID']
y_pred_df['NACCID'] = test_data['NACCID']
assert len(y_pred_df) == 70, f"Expected 70 rows in y_pred_df, got {len(y_pred_df)}"
y_pred_df.to_csv('/projectnb/vkolagrp/spuduch/ml_test_data_AD_nAD(FTD,VD,PRD)_pred.csv', index=False)
# %%
# y_pred
# %%

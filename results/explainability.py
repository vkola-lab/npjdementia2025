# %% # load data, define features, load model, load feature_names_map.json
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import shap
import nibabel as nib
import numpy as np
import nilearn
from nilearn import plotting
import seaborn as sns
import scipy.stats as stats

# which xgboost version is installed?
print(xgb.__version__)
# Load the model
model_path_AD = '/projectnb/vkolagrp/spuduch/xgb_model_AD.joblib'
xgb_model_AD = joblib.load(model_path_AD)

model_path_nAD = '/projectnb/vkolagrp/spuduch/xgb_model_nAD.joblib'
xgb_model_nAD = joblib.load(model_path_nAD)

# json_model_path = '/projectnb/vkolagrp/spuduch/xgb_model.json'
# loaded_model = xgb.Booster({'nthread': 4})  # init model
# loaded_model.load_model(json_model_path)

demo_features = [
    'his_NACCAGE', 'his_SEX', 'his_RACE', 'his_HISPANIC', 'his_HISPOR', 'his_EDUC',
]

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



labels = ['AD', 'nAD']
# imag_features actually needs _vol and _wmh appended to each
imag_features = []
vol_features = []
wmh_features = []
for region in region_names:
    imag_features.append(region + '_vol')
    vol_features.append(region + '_vol')
    imag_features.append(region + '_wmh')
    wmh_features.append(region + '_wmh')
imag_features.append('total_wm_burden')
# imag_features.append('total_mb_burden')
imag_features = [f for f in imag_features]

test_data = pd.read_csv('/projectnb/vkolagrp/spuduch/ml_test_data_filtered.csv')
test_data = test_data.astype({'his_SEX': 'category', 'his_RACE': 'category', 'his_HISPANIC': 'category', 'his_HISPOR': 'category'})
# set index of test_data to CASEID str replace 'CASE_'
test_data['CASEID'] = test_data['CASEID'].str.replace('CASE_', '').astype(int)
test_data.set_index('CASEID', inplace=True)
# sort the index
test_data.sort_index(inplace=True)
print(test_data)
X_test = test_data[demo_features + imag_features]
y_test = test_data[labels]

data = pd.read_csv('/projectnb/vkolagrp/spuduch/ml_data_filtered.csv')
data = data.astype({'his_SEX': 'category', 'his_RACE': 'category', 'his_HISPANIC': 'category', 'his_HISPOR': 'category'})

train_VD_idx = data['VD'] 
train_PD_idx = data['PD']
train_FTD_idx = data['FTD']
X_train = data[demo_features + imag_features]
y_train = data[labels]

superstructures = [
 # superstructures
 'Right-Cerebellum-Cortex',
 'Right-Cerebellum-White-Matter',
 'Right-Cerebral-White-Matter',
 'Left-Cerebellum-Cortex',
 'Left-Cerebellum-White-Matter',
 'Left-Cerebral-White-Matter',

 #ventricle superstructures
 'CSF',
 'Left-Lateral-Ventricle',
 'Right-Lateral-Ventricle',
 'Left-Inf-Lat-Vent',
 'Right-Inf-Lat-Vent',
]

# move the superstructures to the start of the vol_featrues and wmh_features lists
for superstructure in superstructures:
    vol_features.remove(superstructure + '_vol')
    vol_features.insert(0, superstructure + '_vol')
    wmh_features.remove(superstructure + '_wmh')
    wmh_features.insert(0, superstructure + '_wmh')

# load the feature_names_map.json
json_dict = pd.read_json('/usr4/ugrad/spuduch/RadiologistRatings/dev/train/feature_names_map.json', typ='series').to_dict()

feature_names_map = {}
# in the mapping acccount for the suffixes
for key in json_dict.keys():
    if key == "WM-hypointensities":
        feature_names_map[key + "_vol"] = json_dict[key]
        continue
    if key in demo_features or key in imag_features:
        feature_names_map[key] = json_dict[key]
    if key + '_vol' in imag_features: feature_names_map[key + '_vol'] = json_dict[key] + ' Volume'
    if key + '_wmh' in imag_features: feature_names_map[key + '_wmh'] = 'WMH Volume in ' + json_dict[key]
# %% # predict on the test data

# Dmatrix_test = xgb.DMatrix(X_test, enable_categorical=True)
# Dmatrix_train = xgb.DMatrix(X_train, enable_categorical=True)

y_pred_AD = xgb_model_AD.predict_proba(X_test)[:, 1]
y_pred_nAD = xgb_model_nAD.predict_proba(X_test)[:, 1]

y_pred_df = pd.DataFrame({'AD': y_pred_AD, 'nAD': y_pred_nAD})
y_pred_df
# %% # Get feature importance for AD

feature_importance_AD = xgb_model_AD.get_booster().get_score(importance_type='weight')
feature_importance_AD_gain = xgb_model_AD.get_booster().get_score(importance_type='gain')
feature_importance_AD_cover = xgb_model_AD.get_booster().get_score(importance_type='cover')

# Sort features by importance
sorted_importance_AD = sorted(feature_importance_AD.items(), key=lambda x: x[1], reverse=True)

# Print feature importance
# for feature, importance in sorted_importance_AD:
#     print(f"Feature: {feature}, Importance: {importance}")

# save the top 20 toa csv
top_20_features = sorted_importance_AD[:20]
top_20_features_df = pd.DataFrame(top_20_features, columns=['Feature', 'Importance'])
# add the gain and cover columns
top_20_features_df['Gain'] = [feature_importance_AD_gain.get(f) for f, _ in top_20_features]
top_20_features_df['Cover'] = [feature_importance_AD_cover.get(f) for f, _ in top_20_features]

# # replace each feature name with the corresponding value in the feature_names_map
top_20_features_df['Feature'] = [feature_names_map.get(f) for f, _ in top_20_features]
print(top_20_features_df)
# con
top_20_features_df.to_csv('/projectnb/vkolagrp/spuduch/top_20_features_AD.csv', index=False)
# %% # Get feature importance for nAD

feature_importance_nAD = xgb_model_nAD.get_booster().get_score(importance_type='weight')
feature_importance_nAD_gain = xgb_model_nAD.get_booster().get_score(importance_type='gain')
feature_importance_nAD_cover = xgb_model_nAD.get_booster().get_score(importance_type='cover')

# Sort features by importance
sorted_importance_nAD = sorted(feature_importance_nAD.items(), key=lambda x: x[1], reverse=True)

# Print feature importance
# for feature, importance in sorted_importance_nAD:
#     print(f"Feature: {feature}, Importance: {importance}")

# save the top 20 to a csv
top_20_features = sorted_importance_nAD[:20]
top_20_features_df = pd.DataFrame(top_20_features, columns=['Feature', 'Importance'])
# add the gain and cover columns
top_20_features_df['Gain'] = [feature_importance_nAD_gain.get(f) for f, _ in top_20_features]
top_20_features_df['Cover'] = [feature_importance_nAD_cover.get(f) for f, _ in top_20_features]

# # replace each feature name with the corresponding value in the feature_names_map
top_20_features_df['Feature'] = [feature_names_map.get(f) for f, _ in top_20_features]
print(top_20_features_df)
top_20_features_df.to_csv('/projectnb/vkolagrp/spuduch/top_20_features_nAD.csv', index=False)
# %% # set up nilearn plotting info
# Path to the .stats file
stats_file_path = '/projectnb/vkolagrp/spuduch/mni/fastsurfer/stats/aseg+DKT.stats'

# Load the .stats file into a DataFrame
stats_df = pd.read_csv(stats_file_path, sep=r'\s+', comment='#', header=None)
stats_df.columns = ['Index', 'LabelID', 'Unknown1', 'Unknown2', 'RegionName', 'Unknown4', 'Unknown5', 'Unknown6', 'Unknown7', 'Unknown3']
stats_df = stats_df.drop(columns=['Index', 'Unknown1', 'Unknown2', 'Unknown3', 'Unknown4', 'Unknown5', 'Unknown6', 'Unknown7'])
# turn stats_df into a dictionary for RegionName: LabelID
stats_dict = dict(zip(stats_df['RegionName'], stats_df['LabelID']))

# cerebellum_stats_file_path
cerebellum_stats_file_path = '/projectnb/vkolagrp/spuduch/mni/fastsurfer/stats/cerebellum.CerebNet.stats'
cerebellum_stats_df = pd.read_csv(cerebellum_stats_file_path, sep=r'\s+', comment='#', header=None)
cerebellum_stats_df.columns = ['Index', 'LabelID', 'Unknown1', 'Unknown2', 'RegionName', 'Unknown4', 'Unknown5', 'Unknown6', 'Unknown7', 'Unknown3']
cerebellum_stats_df = cerebellum_stats_df.drop(columns=['Index', 'Unknown1', 'Unknown2', 'Unknown3', 'Unknown4', 'Unknown5', 'Unknown6', 'Unknown7'])

# add them to the stats_dict
for i, row in cerebellum_stats_df.iterrows():
    stats_dict[row['RegionName']] = row['LabelID']

print(stats_dict)
atlas_path = '/projectnb/vkolagrp/spuduch/mni/fastsurfer/mri/aparc.DKTatlas+aseg.deep.mgz'
atlas_img = nib.load(atlas_path)

atlas_data = atlas_img.get_fdata()

# %% # project _vol feature importance onto the atlas for AD

AD_atlas_heatmap_vol = np.zeros(atlas_data.shape)  # Initialize with zeros

for feature in vol_features:
    if feature in ['total_wm_burden', 'WM-hypointensities_vol']:
        continue
    # print(feature[:-4])
    atlas_index = stats_dict.get(feature[:-4])
    # print(atlas_index)
    AD_atlas_heatmap_vol[np.isin(atlas_data, atlas_index)] = feature_importance_AD.get(feature)

# Create a new Nifti image with the modified data
AD_atlas_heatmap_vol_img = nib.Nifti1Image(AD_atlas_heatmap_vol, atlas_img.affine, atlas_img.header)

# Save the modified atlas image
AD_atlas_heatmap_vol_img_path = '/projectnb/vkolagrp/spuduch/AD_atlas_heatmap_vol_img.mgz'
nib.save(AD_atlas_heatmap_vol_img, AD_atlas_heatmap_vol_img_path)

# %% # project _wmh feature importance onto the atlas for AD

AD_atlas_heatmap_wmh = np.zeros(atlas_data.shape)  # Initialize with zeros

for feature in wmh_features:
    # print(feature[:-4])
    atlas_index = stats_dict.get(feature[:-4])
    # print(atlas_index)
    AD_atlas_heatmap_wmh[np.isin(atlas_data, atlas_index)] = feature_importance_AD.get(feature)

# Create a new Nifti image with the modified data
AD_atlas_heatmap_wmh_img = nib.Nifti1Image(AD_atlas_heatmap_wmh, atlas_img.affine, atlas_img.header)

# Save the modified atlas image
AD_atlas_heatmap_wmh_img_path = '/projectnb/vkolagrp/spuduch/AD_atlas_heatmap_wmh_img.mgz'
nib.save(AD_atlas_heatmap_wmh_img, AD_atlas_heatmap_wmh_img_path)

# %% # plot
fig = plt.figure(figsize=(5, 5))
plotting.plot_stat_map(AD_atlas_heatmap_vol_img, bg_img= '/projectnb/vkolagrp/spuduch/mni/fastsurfer/mri/orig_nu.mgz', 
    threshold = 75, cmap='viridis',
    output_file= '/projectnb/vkolagrp/spuduch/plots/AD_vol_feature_weight.svg',
    radiological=True
    )
plt.close(fig)
fig = plt.figure(figsize=(5, 5))
plotting.plot_stat_map(AD_atlas_heatmap_wmh_img, bg_img= '/projectnb/vkolagrp/spuduch/mni/fastsurfer/mri/orig_nu.mgz', 
    threshold = 30, cmap='viridis', 
    output_file= '/projectnb/vkolagrp/spuduch/plots/AD_wmh_feature_weight.svg',
    radiological=True
    )
plt.close(fig)
# %% # nonimg features heatmap for AD
non_img_features = [
    'his_NACCAGE',
    'his_SEX',
    'his_RACE',
    'his_HISPANIC',
    'his_HISPOR',
    'his_EDUC',
    'total_wm_burden',
    'WM-hypointensities_vol',
]

non_img_feature_importance_AD = pd.DataFrame({k: feature_importance_AD[k] for k in non_img_features}, index=[0])

# rename the columsn with the feature_names_map dict
non_img_feature_importance_AD.columns = [feature_names_map.get(k) for k in non_img_features]
# make the seaborn heatmap vertically oriented and plt.cm.cold_hot
plt.figure(figsize=(3, 3))
sns.heatmap(non_img_feature_importance_AD.T, 
            annot=True, fmt='g', square=True, 
            xticklabels=False, yticklabels=True,
            cmap = 'viridis', #cbar_kws={},
            # linecolor='black', linewidth=0.5,
            annot_kws={'fontsize': 7},
)

plt.yticks(fontsize=7)
# cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='vertical')
# plt.xticks(fontsize=7)
# save the heatmap
# plt.savefig('/projectnb/vkolagrp/spuduch/plots/AD_nonimg_feature_weight.svg')


# %% # project _vol feature importance onto the atlas for nAD

nAD_atlas_heatmap_vol = np.zeros(atlas_data.shape)  # Initialize with zeros

for feature in vol_features:
    if feature in ['total_wm_burden', 'WM-hypointensities_vol']:
        continue
    # print(feature[:-4])
    atlas_index = stats_dict.get(feature[:-4])
    # print(atlas_index)
    nAD_atlas_heatmap_vol[np.isin(atlas_data, atlas_index)] = feature_importance_nAD.get(feature)

# Create a new Nifti image with the modified data
nAD_atlas_heatmap_vol_img = nib.Nifti1Image(nAD_atlas_heatmap_vol, atlas_img.affine, atlas_img.header)

# Save the modified atlas image
nAD_atlas_heatmap_vol_img_path = '/projectnb/vkolagrp/spuduch/nAD_atlas_heatmap_vol_img.mgz'
nib.save(nAD_atlas_heatmap_vol_img, nAD_atlas_heatmap_vol_img_path)

# %% # project _wmh feature importance onto the atlas for nAD

nAD_atlas_heatmap_wmh = np.zeros(atlas_data.shape)  # Initialize with zeros

for feature in wmh_features:
    # print(feature[:-4])
    atlas_index = stats_dict.get(feature[:-4])
    # print(atlas_index)
    nAD_atlas_heatmap_wmh[np.isin(atlas_data, atlas_index)] = feature_importance_nAD.get(feature)

# Create a new Nifti image with the modified data
nAD_atlas_heatmap_wmh_img = nib.Nifti1Image(nAD_atlas_heatmap_wmh, atlas_img.affine, atlas_img.header)

# Save the modified atlas image
nAD_atlas_heatmap_wmh_img_path = '/projectnb/vkolagrp/spuduch/nAD_atlas_heatmap_wmh_img.mgz'
nib.save(nAD_atlas_heatmap_wmh_img, nAD_atlas_heatmap_wmh_img_path)

# %% # plot
fig = plt.figure(figsize=(5, 5))
plotting.plot_stat_map(nAD_atlas_heatmap_vol_img, bg_img= '/projectnb/vkolagrp/spuduch/mni/fastsurfer/mri/orig_nu.mgz', 
    cmap='viridis', threshold = 60,
    output_file = '/projectnb/vkolagrp/spuduch/plots/nAD_vol_feature_weight.svg',
    radiological=True
    )
plt.close(fig)
fig = plt.figure(figsize=(5, 5))
plotting.plot_stat_map(nAD_atlas_heatmap_wmh_img, bg_img= '/projectnb/vkolagrp/spuduch/mni/fastsurfer/mri/orig_nu.mgz', 
    cmap='viridis', threshold = 25,
    output_file = '/projectnb/vkolagrp/spuduch/plots/nAD_wmh_feature_weight.svg',
    radiological=True
    )
plt.close(fig)
# %% # nonimg features heatmap for nAD

# features are the same
non_img_feature_importance_nAD = pd.DataFrame({k: feature_importance_nAD[k] for k in non_img_features}, index=[0])

# rename columns for better readability
non_img_feature_importance_nAD.columns = [feature_names_map.get(k) for k in non_img_features]

# figure size
plt.figure(figsize=(3, 3))
sns.heatmap(non_img_feature_importance_nAD.T, 
            annot=True, fmt='g', square=True, xticklabels=False, 
            cmap = 'viridis', 
            # cbar_kws={'label': 'Feature Importance'},           
            # linecolor='black', linewidth=0.5,
            annot_kws={'fontsize': 7}
)

plt.yticks(fontsize=7)

plt.savefig('/projectnb/vkolagrp/spuduch/plots/nAD_nonimg_feature_weight.svg')

# %% SHAP computation for train data
# SHAP explainer
AD_explainer = shap.TreeExplainer(xgb_model_AD)
shap_values_AD = AD_explainer(X_train)
# print(shap_values_AD.shape)

#absolute value of shap values
shap_values_AD_abs = np.abs(shap_values_AD.values)

# dataframe of means
shap_means_AD = pd.DataFrame(shap_values_AD_abs.mean(axis=0), index=X_train.columns, columns=['Mean SHAP Value'])
# sort
shap_means_AD = shap_means_AD.sort_values(by='Mean SHAP Value', ascending=False)
print(shap_means_AD)

# now nAD
nAD_explainer = shap.TreeExplainer(xgb_model_nAD)
shap_values_nAD = nAD_explainer(X_train)
# print(shap_values_nAD.shape)

#absolute value of shap values
shap_values_nAD_abs = np.abs(shap_values_nAD.values)

# dataframe of means
shap_means_nAD = pd.DataFrame(shap_values_nAD_abs.mean(axis=0), index=X_train.columns, columns=['Mean SHAP Value'])
# sort
shap_means_nAD = shap_means_nAD.sort_values(by='Mean SHAP Value', ascending=False)
print(shap_means_nAD)


# %% summary plots
# edit the feature names using the feature_names_map dict
shap_values_AD.feature_names = [feature_names_map.get(f) for f in X_train.columns]
shap_values_nAD.feature_names = [feature_names_map.get(f) for f in X_train.columns]

fig_AD = shap.summary_plot(shap_values_AD, X_train,
    max_display=15, sort=True, 
    show=False, plot_size=(7, 7), 
    )
# fig_AD = shap.plots.beeswarm(shap_values_AD, max_display=15, 
#     show=False, plot_size=(4, 5), order = shap.Explanation.abs.mean(0),
#     s=10, 
#     # do not group remaining features
#     )
plt.yticks(fontsize=10)
plt.savefig('/projectnb/vkolagrp/spuduch/plots/AD_shap_summary_plot.svg')
# plt.show()
plt.close()
fig_nAD = shap.summary_plot(shap_values_nAD, X_train, 
    max_display=15, sort=True, 
    show=False, plot_size=(7, 7),
)
plt.yticks(fontsize=10)
plt.savefig('/projectnb/vkolagrp/spuduch/plots/nAD_shap_summary_plot.svg')
plt.close()
# %%
# %% # SHAP for individual etiologies within OIED

top_k = 10

FTD_cases = X_train[train_FTD_idx == 1]
VD_cases = X_train[train_VD_idx == 1]
PD_cases = X_train[train_PD_idx == 1]

#print count 
print(f'FTD: {FTD_cases.shape[0]}, VD: {VD_cases.shape[0]}, PD: {PD_cases.shape[0]}')

FTD_explainer = shap.TreeExplainer(xgb_model_nAD)
shap_values_FTD = FTD_explainer(FTD_cases)
shap_values_FTD.feature_names = [feature_names_map.get(f) for f in FTD_cases.columns]
shap_values_FTD_abs = np.abs(shap_values_FTD.values)

VD_explainer = shap.TreeExplainer(xgb_model_nAD)
shap_values_VD = VD_explainer(VD_cases)
shap_values_VD.feature_names = [feature_names_map.get(f) for f in VD_cases.columns]
shap_values_VD_abs = np.abs(shap_values_VD.values)

PD_explainer = shap.TreeExplainer(xgb_model_nAD)
shap_values_PD = PD_explainer(PD_cases)
shap_values_PD.feature_names = [feature_names_map.get(f) for f in PD_cases.columns]
shap_values_PD_abs = np.abs(shap_values_PD.values)

shap_means_FTD = pd.DataFrame(shap_values_FTD_abs.mean(axis=0), index=[feature_names_map.get(f) for f in FTD_cases.columns], columns=['Mean |SHAP|'])
shap_means_FTD['Mean SHAP Value'] = shap_values_FTD.values.mean(axis=0)
shap_means_FTD = shap_means_FTD.sort_values(by='Mean |SHAP|', ascending=False)
print(shap_means_FTD)

shap_means_VD = pd.DataFrame(shap_values_VD_abs.mean(axis=0), index=[feature_names_map.get(f) for f in VD_cases.columns], columns=['Mean |SHAP|'])
shap_means_VD['Mean SHAP Value'] = shap_values_VD.values.mean(axis=0)
shap_means_VD = shap_means_VD.sort_values(by='Mean |SHAP|', ascending=False)
print(shap_means_VD)

shap_means_PD = pd.DataFrame(shap_values_PD_abs.mean(axis=0), index=[feature_names_map.get(f) for f in PD_cases.columns], columns=['Mean |SHAP|'])
shap_means_PD['Mean SHAP Value'] = shap_values_PD.values.mean(axis=0)
shap_means_PD = shap_means_PD.sort_values(by='Mean |SHAP|', ascending=False)
print(shap_means_PD)

# save the top k to csvs
shap_means_FTD[:top_k].to_csv('/projectnb/vkolagrp/spuduch/shap_means_FTD.csv')
shap_means_VD[:top_k].to_csv('/projectnb/vkolagrp/spuduch/shap_means_VD.csv')
shap_means_PD[:top_k].to_csv('/projectnb/vkolagrp/spuduch/shap_means_PD.csv')
# %% project onto atlas for AD vols

# project shap values onto the atlas for AD vols
shap_atlas_heatmap_AD_vol = np.zeros(atlas_data.shape)  # Initialize with zeros

for feature in vol_features:
    if feature in ['total_wm_burden', 'WM-hypointensities_vol']:
        continue
    # print(feature[:-4])
    atlas_index = stats_dict.get(feature[:-4])
    # print(atlas_index)
    shap_atlas_heatmap_AD_vol[np.isin(atlas_data, atlas_index)] = shap_means_AD.loc[feature, 'Mean SHAP Value']

# Create a new Nifti image with the modified data
shap_atlas_heatmap_AD_vol_img = nib.Nifti1Image(shap_atlas_heatmap_AD_vol, atlas_img.affine, atlas_img.header)

# Save the modified atlas image
shap_atlas_heatmap_AD_vol_img_path = '/projectnb/vkolagrp/spuduch/shap_atlas_heatmap_AD_vol_img.mgz'
nib.save(shap_atlas_heatmap_AD_vol_img, shap_atlas_heatmap_AD_vol_img_path)

# project shap values onto the atlas for AD wmhs
shap_atlas_heatmap_AD_wmh = np.zeros(atlas_data.shape)  # Initialize with zeros

for feature in wmh_features:
    # print(feature[:-4])
    atlas_index = stats_dict.get(feature[:-4])
    # print(atlas_index)
    shap_atlas_heatmap_AD_wmh[np.isin(atlas_data, atlas_index)] = shap_means_AD.loc[feature, 'Mean SHAP Value']

# Create a new Nifti image with the modified data
shap_atlas_heatmap_AD_wmh_img = nib.Nifti1Image(shap_atlas_heatmap_AD_wmh, atlas_img.affine, atlas_img.header)

# Save the modified atlas image
shap_atlas_heatmap_AD_wmh_img_path = '/projectnb/vkolagrp/spuduch/shap_atlas_heatmap_AD_wmh_img.mgz'
nib.save(shap_atlas_heatmap_AD_wmh_img, shap_atlas_heatmap_AD_wmh_img_path)

# %% # project shap values onto the atlas for nAD vols
shap_atlas_heatmap_nAD_vol = np.zeros(atlas_data.shape)  # Initialize with zeros

for feature in vol_features:
    if feature in ['total_wm_burden', 'WM-hypointensities_vol']:
        continue
    # print(feature[:-4])
    atlas_index = stats_dict.get(feature[:-4])
    # print(atlas_index)
    shap_atlas_heatmap_nAD_vol[np.isin(atlas_data, atlas_index)] = shap_means_nAD.loc[feature, 'Mean SHAP Value']

# Create a new Nifti image with the modified data
shap_atlas_heatmap_nAD_vol_img = nib.Nifti1Image(shap_atlas_heatmap_nAD_vol, atlas_img.affine, atlas_img.header)

# Save the modified atlas image
shap_atlas_heatmap_nAD_vol_img_path = '/projectnb/vkolagrp/spuduch/shap_atlas_heatmap_nAD_vol_img.mgz'
nib.save(shap_atlas_heatmap_nAD_vol_img, shap_atlas_heatmap_nAD_vol_img_path)

# project shap values onto the atlas for nAD wmhs
shap_atlas_heatmap_nAD_wmh = np.zeros(atlas_data.shape)  # Initialize with zeros

for feature in wmh_features:
    # print(feature[:-4])
    atlas_index = stats_dict.get(feature[:-4])
    # print(atlas_index)
    shap_atlas_heatmap_nAD_wmh[np.isin(atlas_data, atlas_index)] = shap_means_nAD.loc[feature, 'Mean SHAP Value']

# Create a new Nifti image with the modified data
shap_atlas_heatmap_nAD_wmh_img = nib.Nifti1Image(shap_atlas_heatmap_nAD_wmh, atlas_img.affine, atlas_img.header)

# Save the modified atlas image
shap_atlas_heatmap_nAD_wmh_img_path = '/projectnb/vkolagrp/spuduch/shap_atlas_heatmap_nAD_wmh_img.mgz'
nib.save(shap_atlas_heatmap_nAD_wmh_img, shap_atlas_heatmap_nAD_wmh_img_path)

# %% # plot
plotting.plot_stat_map(shap_atlas_heatmap_AD_vol_img, bg_img= '/projectnb/vkolagrp/spuduch/mni/fastsurfer/mri/orig_nu.mgz', 
    # cmap='RdBu', threshold = 0.1,
    )
plotting.plot_stat_map(shap_atlas_heatmap_AD_wmh_img, bg_img= '/projectnb/vkolagrp/spuduch/mni/fastsurfer/mri/orig_nu.mgz', 
    # cmap='RdBu', threshold = 0.075,
    )
# %%  # plot nonimage

# non_img_features are already defined





# %%
# # calculate spearman correlation between shap values and feature values and put it in a dataframe
# spearman_corr_AD = pd.DataFrame()
# for i, feature in enumerate(X_test.columns):
#     spearman_corr_AD.loc[i, 'Feature'] = feature
#     feature_data, shap_data = X_test[feature].values
#     spearman_corr_AD.loc[i, 'Spearman Correlation Coefficient'] = spearmanr

# spearman_corr_AD
# %% reading in radiologist data and defining the consensus

ratings_df = pd.read_csv('/usr4/ugrad/spuduch/RadiologistRatings/70_test_cases/cleaned_radiologist_data.csv')

rating_columns = list(ratings_df.iloc[:, 7:-34].columns)
ratings_df = ratings_df[['pt'] + rating_columns]

for column in ratings_df:
    print(column, ratings_df[column].unique())
# aggregate the ratings by column 'pt'
consensus_ratings_df = ratings_df.groupby('pt').median()
consensus_ratings_df
field_mapping = pd.read_json('/usr4/ugrad/spuduch/RadiologistRatings/dev/field_mapping.json', typ='series').to_dict()
field_mapping

feature_names_map = pd.read_json('/usr4/ugrad/spuduch/RadiologistRatings/dev/train/feature_names_map.json', typ='series').to_dict()

# which columns in consensus_ratings_df are not in field_mapping keys
[col for col in consensus_ratings_df.columns if col not in field_mapping.keys()]

# %% compute SHAP on test set
shap_values_test_AD = AD_explainer.shap_values(X_test)
shap_values_test_nAD = nAD_explainer.shap_values(X_test)

compiled_shap_AD = pd.DataFrame()
for Question, region_list in field_mapping.items():
    sum_SHAP = 0
    for region in region_list:
        sum_SHAP += shap_values_test_AD[:, X_test.columns.get_loc(region + '_vol')]
    compiled_shap_AD[Question] = sum_SHAP
compiled_shap_AD

compiled_shap_nAD = pd.DataFrame()
for Question, region_list in field_mapping.items():
    sum_SHAP = 0
    for region in region_list:
        sum_SHAP += shap_values_test_nAD[:, X_test.columns.get_loc(region + '_vol')]
    compiled_shap_nAD[Question] = sum_SHAP

# now compile 'Are there hyperintensities present in the T2 FLAIR sequence?' as the sum of total_wm_burden and all the wmh features
total_wm_burden_idx = X_test.columns.get_loc('total_wm_burden')
wmh_features_indices = [X_test.columns.get_loc(feature) for feature in wmh_features]

compiled_shap_AD['Are there hyperintensities present in the T2 FLAIR sequence?'] = shap_values_test_AD[:, total_wm_burden_idx] + np.sum(shap_values_test_AD[:, wmh_features_indices], axis=1)
compiled_shap_nAD['Are there hyperintensities present in the T2 FLAIR sequence?'] = shap_values_test_nAD[:, total_wm_burden_idx] + np.sum(shap_values_test_nAD[:, wmh_features_indices], axis=1)
# set the indices explicitly to index +1
compiled_shap_AD.index = compiled_shap_AD.index + 1
compiled_shap_nAD.index = compiled_shap_nAD.index + 1

compiled_shap_AD
# %% # calculate the correlation between the consensus ratings and the shap values
from sklearn.feature_selection import mutual_info_regression

p_val_to_star = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' if p < 0.1 else ''
# for AD
spearman_corr = pd.DataFrame()
region_df = pd.DataFrame()
for Question, region_list in field_mapping.items():
    # print data types
    # print(Question)
    # # count and n unique vals
    # print(consensus_ratings_df[Question].isna().sum(), consensus_ratings_df[Question].nunique())
    # print(compiled_shap_AD[Question].isna().sum())
    corr = stats.spearmanr(consensus_ratings_df[Question], compiled_shap_AD[Question])
    corr = str("{:.3f}".format(corr[0])) + p_val_to_star(corr[1])
    spearman_corr.loc[Question, 'ρ: SHAP_AD x Med(Atrophy Rating)'] = corr
    # mutual info
    # try:
    #     spearman_corr.loc[Question, 'Mutual Information'] = mutual_info_regression(compiled_shap_AD[Question].to_frame(), consensus_ratings_df[Question].values, discrete_features=[False]).astype(float)
    # except:
    #     spearman_corr.loc[Question, 'Mutual Information'] = np.nan
    
    # add the region_list to the spearman_corr dataframe
    # get the names from the feature_names_map dict

    region_df.loc[Question, 'DKT Atlas Regions'] = ', '.join([feature_names_map.get(region) for region in region_list])
# for nAD
# spearman_corr_nAD = pd.DataFrame()
for Question in field_mapping.keys():
    corr = stats.spearmanr(consensus_ratings_df[Question], compiled_shap_nAD[Question])
    corr = str("{:.3f}".format(corr[0])) + p_val_to_star(corr[1])
    spearman_corr.loc[Question, 'ρ: SHAP_OIED x Med(Atrophy Rating)'] = corr

# now to the hyperintensities
corr = stats.spearmanr(consensus_ratings_df['Are there hyperintensities present in the T2 FLAIR sequence?'], compiled_shap_AD['Are there hyperintensities present in the T2 FLAIR sequence?'])
# use 3 decimal places in the correlation coefficient
corr = str("{:.3f}".format(corr[0])) + p_val_to_star(corr[1])
spearman_corr.loc['Are there hyperintensities present in the T2 FLAIR sequence?', 'ρ: SHAP_AD x Med(Atrophy Rating)'] = corr

corr = stats.spearmanr(consensus_ratings_df['Are there hyperintensities present in the T2 FLAIR sequence?'], compiled_shap_nAD['Are there hyperintensities present in the T2 FLAIR sequence?'])
corr = str("{:.3f}".format(corr[0])) + p_val_to_star(corr[1])
spearman_corr.loc['Are there hyperintensities present in the T2 FLAIR sequence?', 'ρ: SHAP_OIED x Med(Atrophy Rating)'] = corr


# reorder the DKT atlas regions column to be first
# spearman_corr = spearman_corr[['DKT Atlas Regions', 'ρ: SHAP_AD x Med(Atrophy Rating)', 'ρ: SHAP_OIED x Med(Atrophy Rating)']]
spearman_corr.to_csv('/projectnb/vkolagrp/spuduch/spearman_corr.csv')
spearman_corr
# region_df.to_csv('/projectnb/vkolagrp/spuduch/supplemental_region_df.csv')
# %%

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# import ray
# ray.shutdown()
# ray.init(num_cpus=6)
# import modin.pandas as mpd

import warnings
warnings.filterwarnings("ignore")

def labels_conversion(row):
    flag = 0
    # NC
    if row['NACCUDSD'] == 1:
        row['NC'] = 1
    else:
        row['NC'] = 0

    # IMCI
    if row['NACCUDSD'] == 2:
        row['IMCI'] = 1
    else:
        row['IMCI'] = 0

    # MCI
    if row['NACCUDSD'] == 3:
        row['MCI'] = 1
    else:
        row['MCI'] = 0
    
    # DE
    if row['NACCUDSD'] == 4:
        row['DE'] = 1
    else:
        row['DE'] = 0 

    # AD
    if ((row['NACCALZP'] == 1) | (row['NACCALZP'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['AD'] = 1
    else:
        row['AD'] = 0

    # LBD
    if ((row['NACCLBDP'] == 1) | (row['NACCLBDP'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['LBD'] = 1
    else:
        row['LBD'] = 0
    
    # PDD
    if ((row['NACCLBDP'] == 1) | (row['NACCLBDP'] == 2)) & (row['PARK'] == 1) & (row['NACCUDSD'] == 4):
        flag = 1
        row['PDD'] = 1
    else:
        row['PDD'] = 0

    # DLB
    if ((row['NACCLBDP'] == 1) | (row['NACCLBDP'] == 2)) & (row['PARK'] == 0) & (row['NACCUDSD'] == 4):
        flag = 1
        row['DLB'] = 1
    else:
        row['DLB'] = 0

    # # PD
    # if (row['PARK'] == 1):
    #     flag = 1
    #     row['PD'] = 1
    # else:
    #     row['PD'] = 0

    # VD
    if (((row['CVDIF'] == 1) | (row['CVDIF'] == 2)) | ((row['VASCIF'] == 1) | (row['VASCIF'] == 2)) | ((row['VASCPSIF'] == 1) | (row['VASCPSIF'] == 2)) | ((row['STROKIF'] == 1) | (row['STROKIF'] == 2))) & (row['NACCUDSD'] == 4):
        flag = 1
        row['VD'] = 1
    else:
        row['VD'] = 0

    # Prion disease (CJD, other)
    if ((row['PRIONIF'] == 1) | (row['PRIONIF'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['PRD'] = 1
    else:
        row['PRD'] = 0

    # FTLD and its variants, including CBD and PSP, and with or without ALS
    if (((row['FTLDNOIF'] == 1) | (row['FTLDNOIF'] == 2)) | ((row['FTDIF'] == 1) | (row['FTDIF'] == 2)) | ((row['FTLDMOIF'] == 1) | (row['FTLDMOIF'] == 2)) | ((row['PPAPHIF'] == 1) | (row['PPAPHIF'] == 2)) | ((row['PSPIF'] == 1) | (row['PSPIF'] == 2)) | ((row['CORTIF'] == 1) | (row['CORTIF'] == 2))) & (row['NACCUDSD'] == 4):
        flag = 1
        row['FTD'] = 1
    else:
        row['FTD'] = 0

    # NPH
    if ((row['HYCEPHIF'] == 1) | (row['HYCEPHIF'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['NPH'] = 1
    else:
        row['NPH'] = 0

    # Infectious (HIV included), metabolic, substance abuse / alcohol, medications, systemic disease, delirium (Systemic and External Factors)
    if (((row['OTHCOGIF'] == 1) | (row['OTHCOGIF'] == 2)) | ((row['HIVIF'] == 1) | (row['HIVIF'] == 2)) | ((row['ALCDEMIF'] == 1) | (row['ALCDEMIF'] == 2)) | ((row['IMPSUBIF'] == 1) | (row['IMPSUBIF'] == 2)) | ((row['DYSILLIF'] == 1) | (row['DYSILLIF'] == 2)) | ((row['MEDSIF'] == 1) | (row['MEDSIF'] == 2)) | ((row['DELIRIF'] == 1) | (row['DELIRIF'] == 2))) & (row['NACCUDSD'] == 4):
        flag = 1
        row['SEF'] = 1
    else:
        row['SEF'] = 0

    
    # Psychiatric including schizophrenia, depression, bipolar, anxiety, PTSD
    if (((row['DEPIF'] == 1) | (row['DEPIF'] == 2)) | ((row['BIPOLDIF'] == 1) | (row['BIPOLDIF'] == 2)) | ((row['SCHIZOIF'] == 1) | (row['SCHIZOIF'] == 2)) | ((row['ANXIETIF'] == 1) | (row['ANXIETIF'] == 2)) | ((row['PTSDDXIF'] == 1) | (row['PTSDDXIF'] == 2)) | ((row['OTHPSYIF'] == 1) | (row['OTHPSYIF'] == 2))) & (row['NACCUDSD'] == 4):
        flag = 1
        row['PSY'] = 1
    else:
        row['PSY'] = 0

    # TBI
    if ((row['BRNINJIF'] == 1) | (row['BRNINJIF'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['TBI'] = 1
    else:
        row['TBI'] = 0

    # OTHER
    if  (row['NACCUDSD'] == 4) & ((flag == 0) | ((row['MSAIF'] == 1) | (row['MSAIF'] == 2)) | ((row['ESSTREIF'] == 1) | (row['ESSTREIF'] == 2)) | ((row['DOWNSIF'] == 1) | (row['DOWNSIF'] == 2)) | ((row['HUNTIF'] == 1) | (row['HUNTIF'] == 2)) | ((row['EPILEPIF'] == 1) | (row['EPILEPIF'] == 2)) | ((row['NEOPIF'] == 1) | (row['NEOPIF'] == 2)) | ((row['DEMUNIF'] == 1) | (row['DEMUNIF'] == 2)) | ((row['COGOTHIF'] == 1) | (row['COGOTHIF'] == 2)) | ((row['COGOTH2F'] == 1) | (row['COGOTH2F'] == 2)) | ((row['COGOTH3F'] == 1) | (row['COGOTH3F'] == 2))):
        row['ODE'] = 1
    else:
        row['ODE'] = 0
        
    return row


demo_features = [
    'NACCAGE', 'SEX', 'RACE', 'HISPANIC', 'HISPOR', 'EDUC',
]
# feature columns
imag_features = [ 
 '3rd-Ventricle',
 '4th-Ventricle',
 'Brain-Stem',
#  'CC_Anterior',
#  'CC_Central',
#  'CC_Mid_Anterior',
#  'CC_Mid_Posterior',
#  'CC_Posterior',
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
fs_data = pd.read_csv('/projectnb/vkolagrp/datasets/NACC/radiological_features/compiled_fs_volumes_updated.csv')
# wmh_data = pd.read_csv('/projectnb/vkolagrp/NACC_BIDS/radiological_features/compiled_wmh_burden.csv')
wmh_data = pd.read_csv('/projectnb/vkolagrp/datasets/NACC/radiological_features/compiled_wm_burden(p>0.5)_updated.csv')
# mb_data = pd.read_csv('/projectnb/vkolagrp/NACC_BIDS/radiological_features/compiled_mb_burden.csv')
# mb_data = pd.read_csv('/projectnb/vkolagrp/NACC_BIDS/radiological_features/compiled_mb_burden(p>0.8).csv')


# drop rows where mo or dy or yr are missing
fs_data.dropna(subset=['mo', 'dy', 'yr'], inplace=True)
wmh_data.dropna(subset=['mo', 'dy', 'yr'], inplace=True)
# mb_data.dropna(subset=['mo', 'dy', 'yr'], inplace=True)

fs_test_data = pd.read_csv('/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/compiled_fs_volumes.csv')
# wmh_test_data = pd.read_csv('/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/compiled_wm_burden.csv')
wmh_test_data = pd.read_csv('/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/compiled_wm_burden(p>0.5).csv')
# mb_test_data = pd.read_csv('/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/compiled_mb_burden.csv')
# mb_test_data = pd.read_csv('/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/compiled_mb_burden(p>0.8).csv')

demo_data = pd.read_csv('/projectnb/vkolagrp/datasets/NACC/csv/raw/investigator_ftldlbd_nacc65.csv')
# clinician_review_csv = pd.read_csv('/projectnb/vkolagrp/spuduch/clinician_review_cases_test.csv')
clinician_review_csv_radio = pd.read_csv("~/RadiologistRatings/70_test_cases/clinician_review_cases_converted_radio.csv")
# demo_data = demo_data.apply(labels_conversion, axis=1)
print('Loaded data ...')
# %%
# mb_data['mo'].value_counts()
# %%
# # merge the fs_data and wmh_data on fname, NACCID, mo, dy, yr. outer merge
# # in the case that there are duplicate column names, add suffixes to the columns

# common_cols = set(fs_data.columns) & set(wmh_data.columns) & set(mb_data.columns)
# for col in common_cols:
#     fs_data.rename(columns={col: f'{col}_vol'}, inplace=True)
#     wmh_data.rename(columns={col: f'{col}_wmh'}, inplace=True)
#     mb_data.rename(columns={col: f'{col}_mb'}, inplace=True)
# print(fs_data.columns, wmh_data.columns, mb_data.columns)
# print(fs_data.shape, wmh_data.shape, mb_data.shape)
# imag_data = pd.merge(fs_data, wmh_data, left_on=['fname_vol', 'NACCID_vol', 'mo_vol', 'dy_vol', 'yr_vol'], right_on= ['fname_wmh', 'NACCID_wmh', 'mo_wmh', 'dy_wmh', 'yr_wmh'],how='outer')
# imag_data = pd.merge(imag_data, mb_data, left_on=['fname_vol', 'NACCID_vol', 'mo_vol', 'dy_vol', 'yr_vol'], right_on= ['fname_mb', 'NACCID_mb', 'mo_mb', 'dy_mb', 'yr_mb'], how='outer')
# print(imag_data.shape, len(imag_data['fname_vol'].unique()))
# imag_data.rename(columns ={'fname_vol': 'fname', 'NACCID_vol': 'NACCID', 'mo_vol': 'mo', 'dy_vol': 'dy', 'yr_vol': 'yr'}, inplace=True)
# imag_data.drop(['fname_wmh', 'NACCID_wmh', 'mo_wmh', 'dy_wmh', 'yr_wmh', 'fname_mb', 'NACCID_mb', 'mo_mb', 'dy_mb', 'yr_mb'], axis=1, inplace=True)
# print(imag_data.shape)
# imag_data

merge_keys = ['fname', 'NACCID', 'mo', 'dy', 'yr']

# Check for duplicates in the keys
# for df, name in zip([fs_data, wmh_data, mb_data], ['fs_data', 'wmh_data', 'mb_data']):
#     duplicates = df.duplicated(subset=merge_keys, keep=False)
#     if duplicates.any():
#         print(f"Duplicate keys found in {name}:")
#         print(df[duplicates])

# Rename common columns to prevent issues during merging
# common_cols = set(fs_data.columns) & set(wmh_data.columns) & set(mb_data.columns) - set(merge_keys)
common_cols = set(fs_data.columns) & set(wmh_data.columns) - set(merge_keys)
for col in common_cols:
    fs_data.rename(columns={col: f'{col}_vol'}, inplace=True)
    wmh_data.rename(columns={col: f'{col}_wmh'}, inplace=True)
    # mb_data.rename(columns={col: f'{col}_mb'}, inplace=True)

# print("Columns after renaming:")
# print(fs_data.columns, wmh_data.columns, mb_data.columns)
print("Data shapes before merging:")
# print(fs_data.shape, wmh_data.shape, mb_data.shape)
print(fs_data.shape, wmh_data.shape)

# print(f"mb data points: {mb_data['total_mb_burden'].notnull().sum()}")

# First merge
imag_data = pd.merge(fs_data, wmh_data, left_on=['fname', 'NACCID', 'mo', 'dy', 'yr'],
                     right_on=['fname', 'NACCID', 'mo', 'dy', 'yr'], how='outer')
print("After first merge:", imag_data.shape)
print("Unique NACCIDs after first merge:", len(imag_data['NACCID'].unique()))

# Second merge
# imag_data = pd.merge(imag_data, mb_data, left_on=['fname', 'NACCID', 'mo', 'dy', 'yr'],
#                      right_on=['fname', 'NACCID', 'mo', 'dy', 'yr'], how='outer')
# print("After second merge:", imag_data.shape)
# print("Unique NACCIDs after second merge:", len(imag_data['NACCID'].unique()))


# Remove duplicates after merging (if necessary)
# imag_data = imag_data.drop_duplicates(subset=['fname', 'NACCID', 'mo', 'dy', 'yr'])
# print("After removing duplicates:", imag_data.shape)

# print(imag_data.head())
# print(f"mb data points: {imag_data['total_mb_burden'].notnull().sum()}")

# %%
# same for test


# common_cols = set(fs_test_data.columns) & set(wmh_test_data.columns) & set(mb_test_data.columns) - set(['CASEID'])
common_cols = set(fs_test_data.columns) & set(wmh_test_data.columns) - set(['CASEID'])
for col in common_cols:
    fs_test_data.rename(columns={col: f'{col}_vol'}, inplace=True)
    wmh_test_data.rename(columns={col: f'{col}_wmh'}, inplace=True)
    # mb_test_data.rename(columns={col: f'{col}_mb'}, inplace=True)

# print(fs_test_data.columns, wmh_test_data.columns, mb_test_data.columns)

# print([fs_test_data['CASEID'] == 38])

imag_test_data = pd.merge(fs_test_data, wmh_test_data, left_on=['CASEID'], right_on= ['CASEID'],how='outer')

# imag_test_data = pd.merge(imag_test_data, mb_test_data, left_on=['CASEID'], right_on= ['CASEID'], how='outer')
print(imag_test_data.shape, len(imag_test_data['CASEID'].unique()))
# imag_test_data.rename(columns ={'CASEID_vol': 'CASEID'}, inplace=True)
# imag_test_data.drop(['CASEID_wmh', 'CASEID_mb'], axis=1, inplace=True)

# print the row where CASEID == 38
# print(imag_test_data[imag_test_data['CASEID'] == 'CASE_38'])
# imag_test_data.head()
# %%
# how much data is missing volume, wmh, both, none
print(f"Length of imag_data: {len(imag_data)}")
print(f"Missing volume data: {imag_data['mri_name_vol'].isna().sum()}")
print(f"Missing WMH data: {imag_data['mri_name_wmh'].isna().sum()}")
# print(f"Missing MB data: {imag_data['mri_name_mb'].isna().sum()}")

print(f"Missing both volume and WMH data: {(imag_data['mri_name_vol'].isna() & imag_data['mri_name_wmh'].isna()).sum()}")
# print(f"Missing both volume and MB data: {(imag_data['mri_name_vol'].isna() & imag_data['mri_name_mb'].isna()).sum()}")
# print(f"Missing both WMH and MB data: {(imag_data['mri_name_wmh'].isna() & imag_data['mri_name_mb'].isna()).sum()}")

# print(f"Missing all three: {(imag_data['mri_name_vol'].isna() & imag_data['mri_name_wmh'].isna() & imag_data['mri_name_mb'].isna()).sum()}")

# print(f"Missing none (all three present): {(imag_data['mri_name_vol'].notna() & imag_data['mri_name_wmh'].notna() & imag_data['mri_name_mb'].notna()).sum()}")
# %%

# Convert date columns into single date columns
demo_data.rename(columns={'VISITYR': 
'year', 'VISITMO': 'month', 'VISITDAY': 'day'}, inplace=True)
demo_data['VisitDate'] = pd.to_datetime(demo_data[['year', 'month', 'day']], errors='coerce')

imag_data.rename(columns={'yr': 'year', 'mo': 'month', 'dy': 'day'}, inplace=True)
imag_data['ImageDate'] = pd.to_datetime(imag_data[['year', 'month', 'day']], errors='coerce')

# Filter out rows with 'nodate' in imag_data
imag_data = imag_data.dropna(subset=['ImageDate'])
demo_data = demo_data.dropna(subset=['VisitDate'])

imag_data.reset_index(inplace=True, drop=False)
imag_data.rename(columns={'index': 'imag_data_index'}, inplace=True)

print(len(imag_data), len(demo_data))
# print(imag_data['total_mb_burden'].notnull().sum())
# Print rows where date components might be missing
# print(imag_data[imag_data[['yr', 'mo', 'dy']].isnull().any(axis=1)])
# print(demo_data[demo_data[['VISITYR', 'VISITMO', 'VISITDAY']].isnull().any(axis=1)])

#%%
# Merge the dataframes on NACCID
merged_data = pd.merge(imag_data, demo_data, on='NACCID', suffixes=('_imag', '_demo'))
print(len(merged_data), len(merged_data['imag_data_index'].unique()))
merged_data
# Calculate the absolute difference in days between the dates
merged_data['DateDifference'] = (merged_data['ImageDate'] - merged_data['VisitDate']).abs()
merged_data['DateDifference'].isna().sum()
# For each 'NACCID' in imag_data, select the demo_data row with the closest VisitDate
result_df = pd.DataFrame()

# # Loop through each NACCID and process entries
# for naccid, group in merged_data.groupby('NACCID'):
#     # For each unique imag_data entry within this NACCID group, find the closest demo_data entry
#     for imag_index in group['imag_data_index'].unique():
#         sub_group = group[group['imag_data_index'] == imag_index]
#         # if DateDifference is nan, set the last visit date as the closest date
#         if sub_group['DateDifference'].isna().all():
#             closest_date = sub_group.loc[sub_group['VisitDate'].idxmax()]
#         else:
#             closest_date = sub_group.loc[sub_group['DateDifference'].idxmin()]
#         result_df = pd.concat([result_df, closest_date.to_frame().T], ignore_index=True)  # Transpose to keep DataFrame structure

# print(len(result_df), len(result_df['imag_data_index'].unique()))
# result_df
# Initialize counters
total_naccid = merged_data['NACCID'].nunique()
processed_naccid = 0
print(f"Total NACCIDs: {total_naccid}")
# Loop through each NACCID and process entries
for naccid, group in merged_data.groupby('NACCID'):
    processed_naccid += 1
    # Print the progress every 500 NACCIDs
    # if processed_naccid % 10 == 0 or processed_naccid == total_naccid:
    print(f"Processing NACCID {naccid}: {processed_naccid}/{total_naccid}; num rows: {len(group)}; num unique imag_data: {len(group['imag_data_index'].unique())}")
    # if processed_naccid == 345:
    #     # print(group['fname'].unique(), group['DateDifference'].unique(),group['mri_name_vol'].unique(), group['mri_name_mb'].unique(), group['mri_name_wmh'].unique(), group['VisitDate'].unique())
    #     print(len(group['imag_data_index'].unique()))
    #     raise ValueError
    # cache the last visit
    last_visit_date = group.loc[group['VisitDate'].idxmax()]
    # For each unique imag_data entry within this NACCID group, find the closest demo_data entry
    for imag_index in group['imag_data_index'].unique():
        sub_group = group[group['imag_data_index'] == imag_index]
        # if processed_naccid == 345:
        #     print(sub_group['fname'].unique)
        # if DateDifference is nan, set the last visit date as the closest date
        if sub_group['DateDifference'].isna().all():
            # print('All nan')
            closest_date = last_visit_date
        else:
            closest_date = sub_group.loc[sub_group['DateDifference'].idxmin()]
        result_df = pd.concat([result_df, closest_date.to_frame().T], ignore_index=True)  # Transpose to keep DataFrame structure

print(len(result_df), len(result_df['imag_data_index'].unique()))
result_df


result_df = result_df.apply(labels_conversion, axis=1)
result_df
# %%
result_df.to_csv('/projectnb/vkolagrp/spuduch/ml_data_updated.csv', index=False)


# %%
# merge the imag_test_data with the clinician_review_csv
# clinician_review_csv has column 'ID' while imag_test_data has column 'CASEID'

clinician_review_csv_radio.rename(columns={'case_number': 'CASEID'}, inplace=True)
print(len(imag_test_data['CASEID'].unique()))

merged_test_data = pd.merge(imag_test_data, clinician_review_csv_radio, on='CASEID', how='right')

# print the row where CASEID == 38
print(merged_test_data[merged_test_data['CASEID'] == 'CASE_38'])
# merged_test_data.head()
#%%
print(len(merged_test_data['ID'].unique()))

# remove lb form the column names for these: NC_lb	IMCI_lb	MCI_lb	DE_lb	AD_lb	LBD_lb	VD_lb	PRD_lb	FTD_lb	NPH_lb	SEF_lb	PSY_lb	TBI_lb	ODE_lb
merged_test_data.rename(columns={'NC_lb': 'NC', 'IMCI_lb': 'IMCI', 'MCI_lb': 'MCI', 'DE_lb': 'DE', 'AD_lb': 'AD', 'LBD_lb': 'LBD', 'VD_lb': 'VD', 'PRD_lb': 'PRD', 'FTD_lb': 'FTD', 'NPH_lb': 'NPH', 'SEF_lb': 'SEF', 'PSY_lb': 'PSY', 'TBI_lb': 'TBI', 'ODE_lb': 'ODE'}, inplace=True)
merged_test_data.rename(columns={'ID': 'NACCID'}, inplace=True)
merged_test_data.to_csv('/projectnb/vkolagrp/spuduch/ml_test_data_updated.csv', index=False)

# %%

#%%
import os
import pandas as pd
import re

def summarize_modalities(directory):
    modality_counts = {}

    # Walk through the directory structure
    for root, dirs, files in os.walk(directory):
        parts = root.split(os.sep)
        # Assuming the structure has variable depths, let's focus on capturing modalities correctly
        # We expect modality to be one of the last three segments, typically just before the file name level
        if len(parts) > 10:
            modality = parts[-2]  # Commonly the modality is second to last
            acquisition_type = parts[-1]      # Last part expected to be '3D', '2D', or similar acquisition type
                        
            # Initialize dictionary entries if they don't exist
            if modality not in modality_counts:
                modality_counts[modality] = {}
            if acquisition_type not in modality_counts[modality]:
                modality_counts[modality][acquisition_type] = 0
            
            # Count each .nii file in the current directory
            for file in files:
                if file.endswith('.nii'):
                    modality_counts[modality][acquisition_type] += 1
                # if modality == 'Diffusion':
                #     print(f'{root}/{file}')

    return modality_counts

def extract_metadata_from_path(path):
    parts = path.split('/')
    naccid = parts[-7]
    mri_name = parts[-3]
    Sequence_type = parts[-5]
    acq_type = parts[-4]
    fname_mo_dy_yr = parts[-6]
    fname_parts = fname_mo_dy_yr.split('_')
    date_parts = [part for part in fname_parts if part.startswith(('mo-', 'dy-', 'yr-'))]
    
    # Identify the fname
    fname_index = fname_mo_dy_yr.index('_'.join(date_parts[0:3])) if date_parts else len(fname_mo_dy_yr) - 6
    fname = fname_mo_dy_yr[6:fname_index-1].replace('-', '')
    
    # Extract date parts or handle nodate case
    if 'nodate' in fname_mo_dy_yr:
        mo, dy, yr = 'nodate', 'nodate', 'nodate'
    else:
        mo = [part.split('-')[1] for part in date_parts if part.startswith('mo-')][0]
        dy = [part.split('-')[1] for part in date_parts if part.startswith('dy-')][0]
        yr = [part.split('-')[1] for part in date_parts if part.startswith('yr-')][0]
    
    # # use regex to extrac tthe text from 'fname-' until either _nodate or _mo-xx_dy-xx_yr-xxxx
    # naccid = re.search(r'NACC\d+', path).group()
    # fname_mo_dy_yr = re.search(r'fname-.*?(_nodate|_mo-\d+_dy-\d+_yr-\d+)', path).group()
    # fname = re.search(r'fname-(.*?)_', fname_mo_dy_yr).group(1)
    # date_parts = re.findall(r'(mo-\d+|dy-\d+|yr-\d+)', fname_mo_dy_yr)
    # if 'nodate' in fname_mo_dy_yr:
    #     mo, dy, yr = 'nodate', 'nodate', 'nodate'
    # else:
    #     mo = [part.split('-')[1] for part in date_parts if part.startswith('mo-')][0]
    #     dy = [part.split('-')[1] for part in date_parts if part.startswith('dy-')][0]
    #     yr = [part.split('-')[1] for part in date_parts if part.startswith('yr-')][0]


    return naccid, fname, mo, dy, yr, mri_name, Sequence_type, acq_type

def extract_metadata_from_test_path(path):
    # basically the structure is /projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/CASE_x/Sequence_type/3D/mri_name/tool/file.csv
    parts = path.split('/')
    caseid = parts[-6]
    mri_name = parts[-3]
    Sequence_type = parts[-5]
    acq_type = parts[-4]

    return caseid, mri_name, Sequence_type, acq_type

def compile_fs_volumes(root_dir, filename_to_compile='fs_volumes.csv', test = False):
    compiled_data = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # print(f"Checking {filename}")
            if filename == filename_to_compile:
                print(f"Found {filename_to_compile} in {dirpath}")
                csv_path = os.path.join(dirpath, filename)
                if test:
                    caseid, mri_name, seq_type, acq_type = extract_metadata_from_test_path(csv_path)
                    df = pd.read_csv(csv_path)
                    df['CASEID'] = caseid
                    df['mri_name'] = mri_name
                    df['Sequence_type'] = seq_type
                    df['acq_type'] = acq_type
                    compiled_data.append(df)
                else:
                    naccid, fname, mo, dy, yr, mri_name, seq_type, acq_type = extract_metadata_from_path(csv_path)
                    df = pd.read_csv(csv_path)
                    df['NACCID'] = naccid
                    df['fname'] = fname
                    df['mo'] = mo
                    df['dy'] = dy
                    df['yr'] = yr
                    df['mri_name'] = mri_name
                    compiled_data.append(df)
    
    if compiled_data:
        compiled_df = pd.concat(compiled_data, ignore_index=True)
        return compiled_df
        # compiled_df.to_csv('/projectnb/vkolagrp/NACC_BIDS/radiological_features/compiled_fs_volumes.csv', index=False)
    else:
        print(f"No {filename_to_compile} files found.")


# %%

# Example usage
directory_path = '/projectnb/vkolagrp/datasets/NACC/MRI/raw'
modality_summary = summarize_modalities(directory_path)
print(modality_summary)

# %%
df = pd.read_csv('/projectnb/vkolagrp/datasets/NACC/radiological_features/fastsurfer_args.csv')
# df = pd.read_csv('/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/lpa_args.csv')
df
# %%
# max display width
pd.set_option('display.max_colwidth', None)
df.head()

# %%
## this is for compiling the individual dataframes into one
# root_dir = '/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features'
root_dir = '/projectnb/vkolagrp/datasets/NACC/radiological_features'
# compiled_df = compile_fs_volumes(root_dir, "wm_burden(p>0.5).csv", test=False)
compiled_df = compile_fs_volumes(root_dir, test=False)
compiled_df
# %%
# if value in column 'total_wm_burden' is not null, then set StructName to 'total_wm_burden' and MB Volume to the value in 'total_wm_burden'
# add some assertions to chekc if 'total_wm_burden' is not null, then 'StructName' should be 'total_wm_burden'
if 'total_wm_burden' in compiled_df.columns:
    compiled_df.loc[compiled_df['total_wm_burden'].notnull(), 'StructName'] = 'total_wm_burden'
    compiled_df.loc[compiled_df['total_wm_burden'].notnull(), 'WMH Volume'] = compiled_df['total_wm_burden']
    should_be_empty = compiled_df[(compiled_df['total_wm_burden'].notnull()) & (compiled_df['StructName'] != 'total_wm_burden')]
    assert should_be_empty.empty

    # the column 'total_wm_burden' is no longer needed
    compiled_df.drop('total_wm_burden', axis=1, inplace=True)
else:
    print("total_wm_burden column not found in the dataframe")
compiled_df
#%%
# compiled_df[compiled_df['total_wm_burden'].isnull() == False]
# compiled_df[compiled_df['StructName'] == 'total_wm_burden']
compiled_df.iloc[241:250]
# %%

## this is for converting the compiled dataframe into wide format 
wide_df = compiled_df.pivot_table(
                                    index=['fname', 'NACCID', 'mo', 'dy', 'yr', 'mri_name'], 
                                    # index = ['CASEID', 'mri_name', 'Sequence_type', 'acq_type'],
                                    columns='StructName', 
                                    values='Volume_mm3' # inthe case of fastsufer
                                    # values='WMH Volume'   # in the case of lpa MB burden                        
                                    )
# Reset index to flatten the DataFrame
wide_df.reset_index(inplace=True)
wide_df
#%%
wide_df
# %%
wide_df.to_csv('/projectnb/vkolagrp/datasets/NACC/radiological_features/compiled_fs_volumes_updated.csv', index=False)
# wide_df.to_csv('/projectnb/vkolagrp/datasets/NACC/radiological_features/compiled_wm_burden(p>0.5)_updated.csv', index=False)
# wide_df.to_csv('/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/compiled_wm_burden(p>0.5).csv', index=False)
# wide_df.to_csv('/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/compiled_fs_volumes.csv', index=False)
# %%
# checking the success of fastsurfer processing
found_count = 0
missing_count = 0
for index, row in df.iterrows():
    output_folder = row['full_output_path']
    fs_volumes = os.path.join(output_folder, 'fs_volumes.csv')
    if os.path.exists(fs_volumes):
        # print(f"Found {fs_volumes}")
        found_count += 1
    else:
        # print(f"Missing {fs_volumes}")
        missing_count += 1

print(f"Found: {found_count}, Missing: {missing_count}")

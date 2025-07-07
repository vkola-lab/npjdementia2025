# %%
import os
import subprocess
import pandas as pd

# pandas max col width
pd.set_option('display.max_colwidth', None)

def find_t1_3d_mris(directory, output_base):
    """
    Walks through the directory, explicitly parsing each hierarchical level to identify all MRI files 
    that are categorized as T1 and 3D, and then preps a csv for use in the array job.
    
    Args:
    directory (str): The root directory to search within.
    output_directory (str): The base directory where the output should be stored.
    """
    bash_script_path = "/usr4/ugrad/spuduch/RadiologistRatings/fs_docker_bash.sh"
    fastsurfer_args_df = pd.DataFrame()
    # Walk through the directory structure
    for root, dirs, files in os.walk(directory):
        parts = root.split(os.sep)
        
        # Check specific parts of the path for 'T1' and '3D'
        if len(parts) >= 8 and parts[-2] == 'T1' and parts[-1] == '3D':
            for file in files:
                if file.endswith('.nii'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, directory)
                    subject_id = os.path.join(relative_path, file[:-4], 'fastsurfer')
                    full_output_path = os.path.join(output_base, subject_id)


                    # Run the processing bash script
                    # subprocess.run([bash_script_path, full_path, subject_id, output_directory])

                    # # Filter the output using the custom script
                    # fs_scripts.filter_fs_output(full_output_path)

                    # im trying somehting else - maybe i can make sid the very last folder, and the outputdir the rest of the path
                    sid = full_output_path.split('/')[-1]
                    output_directory = '/'.join(full_output_path.split('/')[:-1])

                    fastsurfer_args_df = fastsurfer_args_df._append({
                        'bash_script_path': bash_script_path,
                        'full_path': full_path,
                        # 'subject_id': subject_id,
                        # 'output_directory': output_directory,
                        'sid': sid,
                        'output_directory': output_directory,
                        'full_output_path': full_output_path
                    }, ignore_index=True)
    return fastsurfer_args_df
def find_modality_and_t1_mris(directory, output_base, modalities='FLAIR', tool='lst'):
    """
    Walks through the directory, explicitly parsing each hierarchical level to identify all MRI files 
    that are categorized as FLAIR, and then preps a csv for use in the array job.
    It also checks the output_direcotry for the existence of fastsurfer/orig_nu.nii for the bias corrected t1.
    
    Args:
    directory (str): The root directory to search within.
    output_directory (str): The base directory where the output should be stored.
    """
    lpa_args_df = pd.DataFrame()
    # Walk through the directory structure
    for root, dirs, files in os.walk(directory):
        parts = root.split(os.sep)
        
        # Check specific parts of the path for 'FLAIR'
        if len(parts) >= 8 and parts[-2] in modalities:
            for file in files:
                if file.endswith('.nii'):
                    full_path = os.path.join(root, file)
                    if 'radiological_features' in full_path:
                        raise ValueError(full_path)
                    relative_path = os.path.relpath(root, directory)
                    subject_id = os.path.join(relative_path, file[:-4], tool)
                    full_output_path = os.path.join(output_base, subject_id)

                    # im trying somehting else - maybe i can make sid the very last folder, and the outputdir the rest of the path
                    sid = full_output_path.split('/')[-1]
                    output_directory = '/'.join(full_output_path.split('/')[:-1])

                    # find the bias corrected T1, if it exists. it will have filename orig_nu.nii
                    t1_bias_corrected = None
                    search_loc = '/'.join(full_output_path.split('/')[:-4])
                    print(search_loc)
                    for rootnew, dirsnew, filesnew in os.walk(search_loc):
                        if 'orig_nu.nii' in filesnew:
                            # make sure to have the full path
                            t1_bias_corrected = os.path.join(rootnew, 'orig_nu.nii')
                            assert os.path.exists(t1_bias_corrected)
                            assert ('fastsurfer' in rootnew)
                            break
                    lpa_args_df = lpa_args_df._append({
                        'full_path': full_path,
                        'sid': sid,
                        'output_directory': output_directory,
                        'full_output_path': full_output_path,
                        't1_bias_corrected': t1_bias_corrected
                    }, ignore_index=True)
    return lpa_args_df


directory_path = '/projectnb/vkolagrp/datasets/NACC/MRI/raw'
output_base_dir = '/projectnb/vkolagrp/datasets/NACC/radiological_features/'

# directory_path = '/projectnb/vkolagrp/spuduch/70_test_cases/raw'
# output_base_dir = '/projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/'

# make the directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)
fastsurfer_args_df = find_t1_3d_mris(directory_path, output_base_dir)
fastsurfer_args_df
# %%
fastsurfer_args_df.to_csv('/projectnb/vkolagrp/datasets/NACC/radiological_features/fastsurfer_args.csv', index=False)
# %%
microbleed_args_df = find_modality_and_t1_mris(directory_path, output_base_dir, modalities=['SWI', 'T2_STAR'], tool='microbleednet')
microbleed_args_df
# %%
naccids = microbleed_args_df['full_path'].str.split('/').str[-5]
naccids.value_counts()
# %%
microbleed_args_df.to_csv('/projectnb/vkolagrp/datasets/NACC/radiological_features/microbleednet_args.csv', index=False)
# %%
lpa_args_df = find_modality_and_t1_mris(directory_path, output_base_dir)
# %%
lpa_args_df
# %%
naccids = lpa_args_df['full_path'].str.split('/').str[-5]
naccids.value_counts()
# %%
lpa_args_df.to_csv('/projectnb/vkolagrp/datasets/NACC/radiological_features/lpa_args.csv', index=False)
# %%
microbleed_args_df.to_csv('/projectnb/vkolagrp/NACC_BIDS/radiological_features/microbleednet_args.csv', index=False)
pd.set_option('display.max_colwidth', None)
# lpa_args_df['full_path'].str.contains('radiological_features').sum()
microbleed_args_df

# %%
pd.read_csv('/projectnb/vkolagrp/NACC_BIDS/radiological_features/microbleednet_args.csv')
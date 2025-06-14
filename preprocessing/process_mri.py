# this is the master script, which should get called to process each set of MRIs for each patient
# this script will call the other scripts in the correct order

## T1
# there's a docker implimentation for FastSurfer
# FastSurfer segmentation to generate the volumetric data according to the DKT atlas
# should also generate the masks for each ROI

## FLAIR

# LST-AI should give us an estimate of the lesion load

## DTI

# Automated Fiber Quantification in Python (pyAFQ)

## SWI

# Extraction of cerebral microbleeds - relatively simple task

# simple unet or something to identify CMBs
#%%
# import torch
# print("PyTorch version:", torch.__version__)
# print("CUDA version supported by this PyTorch build:", torch.version.cuda)

# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
# print(os.getcwd())  # Check the current working directory


#%%
from re import S, sub
import subprocess
import gzip
import shutil
import os
import nibabel as nib
import numpy as np
import pandas as pd


# custom scripts
import lst_scripts
import fs_scripts
import swi_scripts

def compress_nifti(input_file_path, output_file_path=None):
    if output_file_path is None:
        output_file_path = input_file_path + '.gz'
    # check if the compressed file already exists
    if os.path.exists(output_file_path):
        print(f"Compressed file already exists: {output_file_path}")
        return

    with open(input_file_path, 'rb') as f_in:
        with gzip.open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Compressed file saved as: {output_file_path}")

subprocess.run('export MICROBLEEDNET_PRETRAINED_MODEL_PATH=\"tools/microbleed-detection/microbleednet_pretrained_models-20240628T030334Z-001/microbleednet_pretrained_models\"', shell=True)

# subprocess.run('python tools/microbleed-detection/setup.py install', shell=True)

#%%
# This stuff onwards is literally just for testing

# T1_filepath = 'sample_mris/CASE_2/CASE_2_AX_3D_MPR_3D_3.nii'
T1_filepath = 'sample_mris/CASE_5/CASE_5_MPR_CORONAL_3D_10.nii'
# FLAIR_filepath = 'sample_mris/CASE_2/CASE_2_AX_FLAIR_2D_2.nii'
FLAIR_filepath = 'sample_mris/CASE_5/CASE_5_3D_FLAIR_3D_7.nii'
DTI_filepath = ''
SWI_filepath = 'sample_mris/CASE_5/CASE_5_SWI_Images_3D_6.nii'

subject_id = "subject_5_test"
output_directory = "test_dir/"

#%%
# # T1 Fastsurfer segmentation
# # fastsurfer_docker.process_t1_mri(T1_filepath, "test_dir", 'case_2_test')
# if T1_filepath is not None:
#     bash_script_path = "fs_docker_bash.sh"
#     subprocess.run([bash_script_path, T1_filepath, subject_id, output_directory])
#     fs_scripts.filter_fs_output(output_directory + subject_id)
# %%
# # Flair LST-AI

# # first compress the T1 and flair files
# compress_nifti(T1_filepath)
# compress_nifti(FLAIR_filepath)

# command = [
#     'lst',  # Assuming 'lst' is the executable name
#     '--t1', T1_filepath + '.gz',
#     '--flair', FLAIR_filepath + '.gz',
#     '--output', output_directory + '/lst_output',
#     '--temp', output_directory + '/temp',
#     '--device', '0'
# ]

# # Run the command
# result = subprocess.run(command, capture_output=True, text=True)

# # Check if the command was successful
# if result.returncode == 0:
#     print("Command executed successfully!")
#     print("Output:", result.stdout)
# else:
#     print("Error:", result.stderr)


# %%

# mgz_image = nib.load('test_dir/subject_5/mri/orig_nu.mgz')

# nib.save(mgz_image, 'test_dir/subject_5/mri/orig_nu.nii')
# %%
t1_bias_corrected = f'{output_directory}/{subject_id}/orig_nu.nii'
# %%
# lst_scripts.run_matlab_command(t1_bias_corrected, FLAIR_filepath)
# #%%
# lst_scripts.clean_lst_output(FLAIR_filepath, f'{output_directory}/{subject_id}', os.path.exists(t1_bias_corrected))

# flair_filename = FLAIR_filepath.split('/')[-1]
# flair_filename = flair_filename.split('.')[0]
# if os.path.exists(t1_bias_corrected):   wm_prob_mask_path = f'{output_directory}/{subject_id}/ples_lpa_mr{flair_filename}.nii'
# else:    wm_prob_mask_path = f'{output_directory}/{subject_id}/ples_lpa_m{flair_filename}.nii'

# # print(wm_prob_mask_path)
# # %%

# region_wm_burden, total_wm_burden = lst_scripts.calc_wm_burden(t1_bias_corrected, wm_prob_mask_path, f'{output_directory}/{subject_id}/aparc.DKTatlas+aseg.deep.mgz', f'{output_directory}/{subject_id}/cerebellum.CerebNet.nii.gz')

# if os.path.exists(f'{output_directory}/{subject_id}/fs_volumes.csv') and region_wm_burden is not None:
#     region_names = pd.read_csv(f'{output_directory}/{subject_id}/fs_volumes.csv', usecols=['SegId', 'StructName'])
#     region_names_dict = region_names.set_index('SegId').to_dict()['StructName']
#     region_wm_burden = {region_names_dict[int(key)]: value for key, value in region_wm_burden.items()}

#     region_wm_burden['total_wm_burden'] = total_wm_burden

# if region_wm_burden is not None:
#     # save the wm burden to a csv file
#     wm_burden_df = pd.DataFrame(region_wm_burden.items(), columns=['StructName', 'WMH Volume'])
#     wm_burden_df.to_csv(f'{output_directory}/{subject_id}/wm_burden.csv', index=False)
# else:
#     #still save teh total wm burden as a csv file even if there are no regions
#     wm_burden_df = pd.DataFrame({'total_wm_burden': [total_wm_burden]})
#     wm_burden_df.to_csv(f'{output_directory}/{subject_id}/wm_burden.csv', index=False)

# %%
# SWI microbleed detection

if SWI_filepath is not None:
    # first register the SWI to the bias corrected T1 image with affine registration
    if T1_filepath is not None:
        print("Registering SWI to T1 image")
        if os.path.exists(t1_bias_corrected):
            swi_registered_path = f'{output_directory}/{subject_id}/swi_registered.nii.gz'
            subprocess.run(['flirt', '-in', SWI_filepath, '-ref', t1_bias_corrected, '-out', swi_registered_path, '-omat', f'{output_directory}/{subject_id}/swi_to_t1.mat'])
        else:
            print("Looks like there was an error with the T1 image processing. Skipping SWI registration")
            # this should prob get logged? it means that fastsurfer failed
            swi_registered_path = SWI_filepath

    # then prepare the microbleednet data
    print("Preparing SWI for microbleed detection")
    command = f"prepare_microbleednet_data {swi_registered_path} {output_directory}/{subject_id}/microbleednet_swi"
    subprocess.run(command, shell=True)

    # then evaluate the prepared image
    print("Evaluating microbleednet")

    os.environ['MICROBLEEDNET_PRETRAINED_MODEL_PATH'] = 'tools/microbleed-detection/microbleednet_pretrained_models-20240628T030334Z-001/microbleednet_pretrained_models'

    command = f""" \
        microbleednet evaluate \
        --inp_dir \"{output_directory}/{subject_id}\" \
        --model_name \"pre\" \
        --output_dir \"{output_directory}/{subject_id}\"
    """
    subprocess.run(command, shell=True)
else:
    print("No SWI file provided. Skipping microbleed detection")
# %%
# we can now compute the regional microbleed burden and total microbleed burden
# we can also save the microbleed burden to a csv file
region_mb_burden, total_mb_burden = swi_scripts.calc_microbleed_burden(t1_bias_corrected, f'{output_directory}/{subject_id}/Predicted_microbleednet_final_microbleednet_swi.nii.gz', f'{output_directory}/{subject_id}/aparc.DKTatlas+aseg.deep.mgz', f'{output_directory}/{subject_id}/cerebellum.CerebNet.nii.gz')

if os.path.exists(f'{output_directory}/{subject_id}/fs_volumes.csv') and region_mb_burden is not None:
    region_names = pd.read_csv(f'{output_directory}/{subject_id}/fs_volumes.csv', usecols=['SegId', 'StructName'])
    region_names_dict = region_names.set_index('SegId').to_dict()['StructName']
    region_mb_burden = {region_names_dict[int(key)]: value for key, value in region_mb_burden.items()}

    region_mb_burden['total_mb_burden'] = total_mb_burden

if region_mb_burden is not None:
    # save the mb burden to a csv file
    mb_burden_df = pd.DataFrame(region_mb_burden.items(), columns=['StructName', 'Microbleed Volume'])
    mb_burden_df.to_csv(f'{output_directory}/{subject_id}/mb_burden.csv', index=False)
else:
    #still save teh total mb burden as a csv file even if there are no regions
    mb_burden_df = pd.DataFrame({'total_mb_burden': [total_mb_burden]})
    mb_burden_df.to_csv(f'{output_directory}/{subject_id}/mb_burden.csv', index=False)

# %%

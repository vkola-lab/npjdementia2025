# %%

import subprocess
import nibabel as nib
import numpy as np
from nilearn import plotting
import os
import shutil
import pandas as pd

def run_matlab_command(t1_path, flair_path):
    # MATLAB command to run. Make sure to update the paths as needed.
    # matlab_command = f"ps_LST_lpa('{flair_path}', '{t1_path}', false);"

    # # Construct the full command to execute MATLAB from the command line
    # # -nodisplay -nojvm -nosplash are used to run MATLAB without the GUI and JVM
    # command = f"matlab -nodisplay -nosplash -r \"{matlab_command} exit;\""

    # # Run the command

    # subprocess.run(command, shell=True)
    '''
    If no reference image has been specified this algorithm saves a bias corrected version of the FLAIR image (m[FLAIR].nii). 
    If a reference image has been chosen the bias corrected coregistered FLAIR image (mr[FLAIR].nii) is saved. 
    Besides the lesion probability map (ples_lpa_m[r][FLAIR].nii) an additional .mat-file (LST_lpa_m[r][FLAIR].mat) 
    is produced which contains all necessary components that are needed for the longitudinal pipeline or for lesion filling. 
    If the user decided to do so a HTML report (report_LST_lpa_m[r][FLAIR].html) is saved as well as a folder 
    that contains images that are displayed in the report.
    '''
    spm_setup = """
    spm_dir = getenv('SCC_SPM_DIR'); 
    LST_dir = fullfile(spm_dir, 'toolbox', 'LST'); 
    addpath(LST_dir); 
    """
    lst_command = f"ps_LST_lpa('{flair_path}', '{t1_path}', false);"
    
    # Combine the setup and the command into a single MATLAB command
    matlab_command = f"{spm_setup} {lst_command} exit;"

    # Construct the full command to execute MATLAB from the command line
    command = f"matlab -nodisplay -nosplash -r \"{matlab_command}\""

    # Run the command using subprocess
    subprocess.run(command, shell=True)


def calc_wm_burden(t1_image_path, wmh_prob_mask_path, cortex_atlas_mask_path = None, cerebellum_atlas_mask_path = None):
    wmh_prob_mask = nib.load(wmh_prob_mask_path)
    if os.path.exists(t1_image_path) and wmh_prob_mask_path is not None and cortex_atlas_mask_path is not None:
        t1_image = nib.load(t1_image_path)
        atlas_mask = nib.load(cortex_atlas_mask_path)
        
        lobe_labels = np.unique(atlas_mask.get_fdata())[1:]  # Exclude 0 which usually stands for background

        # Dictionary to store WMH volumes by lobe
        wmh_volumes = {}

        for label in lobe_labels:
            lobe_mask = (atlas_mask.get_fdata() == label)
            lobe_wmh_prob = lobe_mask * wmh_prob_mask.get_fdata()  # Element-wise multiplication

            # Calculate the volume of WMH in this lobe
            voxel_volume = np.prod(t1_image.header.get_zooms())  # Calculate voxel volume from T1 metadata
            lobe_wmh_volume = np.nansum(lobe_wmh_prob) * voxel_volume  # Sum probabilities and multiply by voxel volume
            wmh_volumes[label] = lobe_wmh_volume
        if cerebellum_atlas_mask_path is not None:
            cerebellum_seg_mask = nib.load(cerebellum_atlas_mask_path)
            cerebellar_seg_labels = np.unique(cerebellum_seg_mask.get_fdata())[1:]
            for cerebellar_label in cerebellar_seg_labels:
                cerebellar_mask = (cerebellum_seg_mask.get_fdata() == cerebellar_label)
                cerebellar_wmh_prob = cerebellar_mask * wmh_prob_mask.get_fdata()
                voxel_volume = np.prod(t1_image.header.get_zooms())
                cerebellar_wmh_volume = np.nansum(cerebellar_wmh_prob) * voxel_volume
                wmh_volumes[cerebellar_label] = cerebellar_wmh_volume
    else:
        wmh_volumes = None

        voxel_volume = np.prod(wmh_prob_mask.header.get_zooms())
    
    # also calculate the total WMH volume
    total_wmh_volume = np.nansum(wmh_prob_mask.get_fdata()) * voxel_volume
    return wmh_volumes, total_wmh_volume

def clean_lst_output(flair_filepath, lst_output_dir, r=False):
    '''
    FLAIR is the filename of the flair_filepath
    LST_lpa_m[r][FLAIR].mat can be deleted
    mr[FLAIR].nii and ples_lpa_m[r][FLAIR].nii should be moved to lst_output_dir
    LST_tmp* folder should be deleted
    '''
    flair_input_dir = '/'.join(flair_filepath.split('/')[:-1])
    flair_filename = flair_filepath.split('/')[-1]
    flair_filename = flair_filename.strip('.nii')

    # Delete LST_lpa_m[r][FLAIR].mat
    if r: mat_file = f'{flair_input_dir}/LST_lpa_mr{flair_filename}.mat' 
    else: mat_file = f'{flair_input_dir}/LST_lpa_m{flair_filename}.mat'
    try:
        if os.path.exists(mat_file):
            os.remove(mat_file)
            print(f'Deleted: {mat_file}')
        else:
            print(f'File not found: {mat_file}')
    except Exception as e:
        print(f'Error deleting {mat_file}: {e}')

    
    # Move mr[FLAIR].nii and ples_lpa_m[r][FLAIR].nii to lst_output_dir
    if r: bias_corrected_flair_file = f'{flair_input_dir}/mr{flair_filename}.nii'
    else: bias_corrected_flair_file = f'{flair_input_dir}/m{flair_filename}.nii'
    
    if r: ples_file = f'{flair_input_dir}/ples_lpa_mr{flair_filename}.nii'
    else: ples_file = f'{flair_input_dir}/ples_lpa_m{flair_filename}.nii'
    # print(bias_corrected_flair_file)
    # create the lst_output_dir if it does not exist
    if not os.path.exists(lst_output_dir):
        os.makedirs(lst_output_dir)
        print(f'Created directory: {lst_output_dir}')
    try:
        if os.path.exists(bias_corrected_flair_file):
            # if the destination file already exists, delete it
            if os.path.exists(os.path.join(lst_output_dir, os.path.basename(bias_corrected_flair_file))):
                os.remove(os.path.join(lst_output_dir, os.path.basename(bias_corrected_flair_file)))
            shutil.move(bias_corrected_flair_file, lst_output_dir)
            print(f'Moved: {bias_corrected_flair_file}')
        else:
            print(f'File not found: {bias_corrected_flair_file}')  
    except Exception as e:
        print(f'Error moving {bias_corrected_flair_file}: {e}')

    try:
        if os.path.exists(ples_file):
            # if the destination file already exists, delete it
            if os.path.exists(os.path.join(lst_output_dir, os.path.basename(ples_file))):
                os.remove(os.path.join(lst_output_dir, os.path.basename(ples_file)))
            shutil.move(ples_file, lst_output_dir)
            print(f'Moved: {ples_file}')
        else:
            print(f'File not found: {ples_file}')
    except Exception as e:
        print(f'Error moving {ples_file}: {e}')

    # Delete LST_tmp* folder
    lst_tmp_folder = [f for f in os.listdir(flair_input_dir) if 'LST_tmp' in f]
    if len(lst_tmp_folder) > 0:
        lst_tmp_folder = os.path.join(flair_input_dir, lst_tmp_folder[0])
        try:
            shutil.rmtree(lst_tmp_folder)
            print(f'Deleted: {lst_tmp_folder}')
        except Exception as e:
            print(f'Error deleting {lst_tmp_folder}: {e}')

    # Check if the files were moved successfully
    if os.path.exists(mat_file):
        raise ValueError(f'WARNING: {mat_file} still exists')
    for file_name in [os.path.basename(f) for f in [bias_corrected_flair_file, ples_file]]:
        if not os.path.exists(os.path.join(lst_output_dir, file_name)):
            raise ValueError(f'WARNING: {file_name} does not exist in the destination')

    # make sure nothing else is left in the data dir other than the original flair file and its json
    for file in os.listdir(flair_input_dir):
        if file not in [f'{flair_filename}.nii', f'{flair_filename}.json']:
            raise ValueError(f'WARNING: {file} is still in the data directory')
    

    

def calc_and_save_wm_burden(t1_bias_corrected, wm_prob_mask_path, output_directory, cortex_atlas_mask_path = None, cerebellum_atlas_mask_path = None):
    if os.path.exists(t1_bias_corrected):
        t1_derivatives_dir = '/'.join(t1_bias_corrected.split('/')[:-1]) # get the fastsurfer derivatives directory by removing the filename from the t1 path
        cortex_atlas_mask_path = f'{t1_derivatives_dir}/aparc.DKTatlas+aseg.deep.mgz'
        cerebellum_atlas_mask_path = f'{t1_derivatives_dir}/cerebellum.CerebNet.nii.gz'
        assert os.path.exists(cortex_atlas_mask_path)
        assert os.path.exists(cerebellum_atlas_mask_path)
    else:
        t1_derivatives_dir = None

    region_wm_burden, total_wm_burden = calc_wm_burden(t1_bias_corrected, wm_prob_mask_path, cortex_atlas_mask_path, cerebellum_atlas_mask_path)

    if os.path.exists(f'{t1_derivatives_dir}/fs_volumes.csv') and region_wm_burden is not None:
        region_names = pd.read_csv(f'{t1_derivatives_dir}/fs_volumes.csv', usecols=['SegId', 'StructName'])
        region_names_dict = region_names.set_index('SegId').to_dict()['StructName']
        region_wm_burden = {region_names_dict[int(key)]: value for key, value in region_wm_burden.items()}

        region_wm_burden['total_wm_burden'] = total_wm_burden

    if region_wm_burden is not None:
        print('Calculated wm burden for regions. Saving to a csv file.')
        # save the wm burden to a csv file
        wm_burden_df = pd.DataFrame(region_wm_burden.items(), columns=['StructName', 'WMH Volume'])
        wm_burden_df.to_csv(f'{output_directory}/lst/wm_burden(p>0.5).csv', index=False)
    else:
        print('No regions found. Saving total wm burden to a csv file.')
        #still save teh total wm burden as a csv file even if there are no regions
        wm_burden_df = pd.DataFrame({'total_wm_burden': [total_wm_burden]})
        wm_burden_df.to_csv(f'{output_directory}/lst/wm_burden(p>0.5).csv', index=False)

# %%
# t1_image = nib.load('/home/spuduch/RadiologistRatings/test_dir/subject_5/mri/orig_nu.nii')
# wmh_prob_mask = nib.load('/home/spuduch/RadiologistRatings/sample_mris/CASE_5/ples_lpa_mrCASE_5_3D_FLAIR_3D_7.nii')
# atlas_mask = nib.load('/home/spuduch/RadiologistRatings/test_dir/subject_5/mri/aparc.DKTatlas+aseg.deep.mgz')


# # Assume atlas_mask has several labels, each corresponding to a different lobe
# lobe_labels = np.unique(atlas_mask.get_fdata())[1:]  # Exclude 0 which usually stands for background

# # Dictionary to store WMH volumes by lobe
# wmh_volumes = {}

# for label in lobe_labels:
#     lobe_mask = (atlas_mask.get_fdata() == label)
#     lobe_wmh_prob = lobe_mask * wmh_prob_mask.get_fdata()  # Element-wise multiplication
    
#     # Calculate the volume of WMH in this lobe
#     voxel_volume = np.prod(t1_image.header.get_zooms())  # Calculate voxel volume from T1 metadata
#     lobe_wmh_volume = np.nansum(lobe_wmh_prob) * voxel_volume  # Sum probabilities and multiply by voxel volume
#     wmh_volumes[label] = lobe_wmh_volume

# # %%
# wmh_volumes

# # %%
# print(np.where(np.isnan(wmh_prob_mask.get_fdata())))
# # print(np.isnan(atlas_mask.get_fdata()).any())
# data = wmh_prob_mask.get_fdata()

# # Determine where NaNs are located
# nan_indices = np.where(np.isnan(data))
# nan_mask = np.zeros_like(data)
# # nan_mask[nan_indices] = 1  # Set NaN positions to 1

# nan_mask_nii = nib.Nifti1Image(nan_mask, affine=wmh_prob_mask.affine)
# nib.save(nan_mask_nii, 'nan_mask.nii')

# display = plotting.plot_anat(wmh_prob_mask, title="NaN locations", display_mode='ortho', cut_coords=(0, 0, 0))
# display.add_overlay(nan_mask, cmap=plotting.cm.black_red)
# # display.savefig('NaN_visualization.png')  # Save the figure to a file
# display.show()
# display.close()

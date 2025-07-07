# %%
import os
import shutil
import pandas as pd
import nibabel as nib

def filter_fs_output(fs_output_dir):
    '''
    collects the fs_output files, filters them and deletes the irrelevant files
    input structure:
    [fs_output_dir]
      - mri
          - orig_nu.mgz
          - cerebellum.CerebNet.nii.gz
          - aparc.DKTatlas+aseg.deep.mgz
          - mask.mgz
      - scripts
      - stats
           - aseg+DKT.stats
           - cerebellum.CerebNet.stats

    output structure:
    [fs_output_dir]
      - orig_nu.nii
      - cerebellum.CerebNet.nii.gz
      - aparc.DKTatlas+aseg.deep.mgz
      - mask.mgz
      - csv of parsed stats files
      - txt file of the stats headers
    '''

    # Define paths for relevant directories and files
    mri_dir = os.path.join(fs_output_dir, 'mri')
    stats_dir = os.path.join(fs_output_dir, 'stats')
    output_csv_path = os.path.join(fs_output_dir, 'fs_volumes.csv')
    output_txt_path = os.path.join(fs_output_dir, 'fs_stats_headers.txt')

    # Move the orig_nu.nii file to the fs_output_dir
    orig_nu_path = os.path.join(mri_dir, 'orig_nu.mgz')
    mgz_image = nib.load(orig_nu_path)
    nib.save(mgz_image, f'{fs_output_dir}/orig_nu.nii')

    # shutil.move(orig_nu_path, fs_output_dir)

    # Move the cerebellum.CerebNet.nii.gz file to the fs_output_dir
    cerebellum_path = os.path.join(mri_dir, 'cerebellum.CerebNet.nii.gz')
    shutil.move(cerebellum_path, fs_output_dir)

    # Move the aparc.DKTatlas+aseg.deep.mgz file to the fs_output_dir
    aparc_path = os.path.join(mri_dir, 'aparc.DKTatlas+aseg.deep.mgz')
    shutil.move(aparc_path, fs_output_dir)

    # Move the mask.mgz file to the fs_output_dir
    mask_path = os.path.join(mri_dir, 'mask.mgz')
    shutil.move(mask_path, fs_output_dir)


    # Parse stats files and consolidate them into one DataFrame
    all_stats_df, all_headers_list = collect_statistics(stats_dir)

    # Save DataFrame to CSV
    all_stats_df.to_csv(output_csv_path, index=False)

    # Extract and save column headers to a text file
    with open(output_txt_path, 'w') as txt_file:
        txt_file.write('\n'.join(all_headers_list))

    # Clean up the directory by removing unnecessary subfolders
    for subdir in ['mri', 'scripts', 'stats']:
        shutil.rmtree(os.path.join(fs_output_dir, subdir))

def parse_stats_file(stats_file):
    """
    Parse the contents of a .stats file and extract relevant fields and values.
    """
    data = []
    headers = []
    with open(stats_file, 'r') as file:
        columns = []
        for line in file:
            if line.startswith('# ColHeaders'):
                columns = line.split()[2:]
                continue
            elif line.startswith('#'):
                headers.append(line.strip('#').strip())
            if not line.startswith('#') and line.strip():  # Ignore comments and empty lines
                values = line.split()
                if len(values) == len(columns):  # Ensure the row of data matches the columns
                    data.append(values)
    
    if not columns:
        return pd.DataFrame()  # Return an empty DataFrame if no data was parsed

    return pd.DataFrame(data, columns=columns).apply(pd.to_numeric, errors='ignore'), headers

def collect_statistics(root_dir):
    """
    Recursively collect statistics from all .stats files within the directory structure.
    """
    all_data = []
    all_headers = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.stats'):
                stats_path = os.path.join(dirpath, filename)
                df, list_of_headers = parse_stats_file(stats_path)
                if not df.empty:
                    df['File'] = os.path.basename(filename)  # Optional: keep track of source file
                    all_data.append(df)
                    all_headers.extend(list_of_headers)
    
    return pd.concat(all_data, ignore_index=True), all_headers

# %%
# Example usage:
# statistics_df, headers = collect_statistics('/home/spuduch/RadiologistRatings/test_dir/subject_5/stats')

# # %%
# print(statistics_df)

# # %%
# print(headers)
# # statistics_df.to_csv('/path/to/output/parsed_stats.csv', index=False)
# # %%
# filter_fs_output('/home/spuduch/RadiologistRatings/test_dir/subject_5')

    
        


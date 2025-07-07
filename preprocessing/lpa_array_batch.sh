#!/bin/bash -l
#$ -P vkolagrp
#$ -t 1-40
#$ -pe omp 16
#$ -l mem_per_core=16G
#$ -N lpa_ppmi_patched_second_pass
#$ -j y
#$ -m bes
#$ -l h_rt=12:00:00
module load matlab
module load spm
module load fsl

module load miniconda 
function conda_deactivate_all() {
    while [ -n "$CONDA_PREFIX" ]; do
        conda deactivate
    done
}
conda_deactivate_all
conda activate mri_radiology

export PYTHONPATH="/usr4/ugrad/spuduch/RadiologistRatings/:$PYTHONPATH"

# Calculate the starting and ending line indices for each task
# Each task processes Per_Task scans, except possibly the last one
PER_TASK=100 # Number of scans each task should handle
# nrows=$(($(wc -l < /projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/lpa_args.csv) - 1))
nrows=$(($(wc -l < /usr4/ugrad/spuduch/fmADRD/ppmi_lpa_args.csv) - 1))
start=$(( ($SGE_TASK_ID - 1) * PER_TASK + 2 ))  # +2 to account for header and 1-based indexing
end=$(( $SGE_TASK_ID * PER_TASK + 2 ))
if [ $end -gt $((nrows + 1)) ]; then
    end=$((nrows + 1))
fi

if [ $start -le $nrows ]; then
    echo "Task $SGE_TASK_ID processing rows $start to $((end - 1))"
    for (( i=$start; i<$end; i++ ))
    do
        # Read the fields from each line into variables
        # IFS=',' read -r full_path subject_id output_directory full_output_path t1_bias_corrected<<< $(awk -v line=$i 'NR == line' /projectnb/vkolagrp/spuduch/70_test_cases/radiological_features/lpa_args.csv)
        IFS=',' read -r full_path subject_id output_directory full_output_path t1_bias_corrected<<< $(awk -v line=$i 'NR == line' /usr4/ugrad/spuduch/fmADRD/ppmi_lpa_args.csv)
        echo "Processing row $i"
        echo "Full Path: $full_path"
        echo "Subject ID: $subject_id"
        echo "Output Directory: $output_directory"
        echo "Full Output Path: $full_output_path"
        echo "T1 Bias Corrected: $t1_bias_corrected"
        echo "-----------------------------------"
        # mkdir -p "$output_directory"

        # check if the wm_burden.csv file already exists
        if [ -e "${output_directory}/${subject_id}/wm_burden(p>0.5).csv" ]; then
            echo "WM burden already calculated for this subject"
            # set exists bool variable to true to skip the rest of the loop
            exists=true
        else
            exists=false
        fi
        SPM_DIR=$(echo $SCC_SPM_DIR)
        LST_DIR="$SPM_DIR/toolbox/LST"

        # Add LST directory to MATLAB path
        SPM_SETUP="addpath(genpath('${LST_DIR}'));"

        #verify that ps_LPA_lpa is in the path
        CHECK_FUNCTION="fprintf('ps_LST_lpa is located at: %s\\n', which('ps_LST_lpa'));"

        # Define the MATLAB command to execute
        if [ -z "$t1_bias_corrected" ]; then
            LST_COMMAND="ps_LST_lpa('${full_path}', '', false);"
            echo "T1 Bias Corrected not available"
        else
            LST_COMMAND="ps_LST_lpa('${full_path}', '${t1_bias_corrected}', false);"
            echo "T1 Bias Corrected available at ${t1_bias_corrected}"
        fi


        # Combine setup and command to form a full MATLAB command, then execute and exit MATLAB
        MATLAB_COMMAND="${SPM_SETUP} ${CHECK_FUNCTION} ${LST_COMMAND} exit;"

        flair_filename="${full_path##*/}"
        flair_filename="${flair_filename%.*}"
        input_dirname=$(dirname "$full_path")
        # Check if the T1 bias corrected file exists and define the wm_prob_mask_path accordingly
        if [ -e "$t1_bias_corrected" ]; then
            wm_prob_mask_path="${output_directory}/${subject_id}/ples_lpa_mr${flair_filename}.nii"
        else
            wm_prob_mask_path="${output_directory}/${subject_id}/ples_lpa_m${flair_filename}.nii"
        fi

        # Execute the MATLAB command without display or splash screen
        if $exists && [ -e "$wm_prob_mask_path" ]; then
            echo "WM burden already calculated for this subject, skipping matlab and calcing wm burden"
            export PYTHONPATH="/usr4/ugrad/spuduch/RadiologistRatings/:$PYTHONPATH"

            python -c "import lst_scripts; import os; lst_scripts.clean_lst_output('${full_path}', '${full_output_path}', os.path.exists('${t1_bias_corrected}'))"
            echo "Cleaned LST output"

        else
            matlab -nodisplay -nosplash -r "$MATLAB_COMMAND"
            conda activate mri_radiology

            export PYTHONPATH="/usr4/ugrad/spuduch/RadiologistRatings/:$PYTHONPATH"

            conda activate mri_radiology

            export PYTHONPATH="/usr4/ugrad/spuduch/RadiologistRatings/:$PYTHONPATH"

            python -c "import lst_scripts; import os; lst_scripts.clean_lst_output('${full_path}', '${full_output_path}', os.path.exists('${t1_bias_corrected}'))"
            echo "Cleaned LST output"


            echo "WM Prob Mask Path: $wm_prob_mask_path"
            export wm_prob_mask_path

            # Threshold the wm_prob_mask_path
            echo "Thresholding WM probability mask"
            export FSLOUTPUTTYPE=NIFTI
            THRESHOLD=0.5
            wmh_binary_mask_path="${output_directory}/${subject_id}/wmh_binary_mask_${THRESHOLD}.nii"
            fslmaths "${wm_prob_mask_path}" -thr $THRESHOLD -bin "${wmh_binary_mask_path}"

            export PYTHONPATH="/usr4/ugrad/spuduch/RadiologistRatings/:$PYTHONPATH"
            echo "Calculating WM burden"
            # Call the Python script with the appropriate arguments
            python -c "import lst_scripts; lst_scripts.calc_and_save_wm_burden('${t1_bias_corrected}', '${wmh_binary_mask_path}', '${output_directory}')"
            echo "Successfully calculated and saved WM burden"

            
        fi
        
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

    done
else
    echo "Task $SGE_TASK_ID has no rows to process because start index $start exceeds the number of rows $nrows."
fi

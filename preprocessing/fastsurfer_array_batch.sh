#!/bin/bash -l
#$ -P vkolagrp
#$ -t 1-4
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_c=6
#$ -l gpu_memory=32G
#$ -N ppmi_fastsurfer_patch
#$ -j y
#$ -m bes
#$ -l h_rt=12:00:00
module load miniconda 
conda activate mri_radiology
export PYTHONPATH="/usr4/ugrad/spuduch/RadiologistRatings/:$PYTHONPATH"

# Calculate the starting and ending line indices for each task
# Each task processes PER_TASK scans, except possibly the last one
PER_TASK=1000  # Number of scans each task should handle
nrows=$(($(wc -l < /usr4/ugrad/spuduch/fmADRD/ppmi_fastsurfer_args.csv) - 1))
start=$(( ($SGE_TASK_ID - 1) * PER_TASK + 2 ))  # +2 to account for header and 1-based indexing
end=$(( $SGE_TASK_ID * PER_TASK + 2 ))
if [ $end -gt $((nrows + 1)) ]; then
    end=$((nrows + 1))
fi

echo "Task $SGE_TASK_ID processing rows $start to $((end - 1))"

# Loop through the subset of lines for this task
for (( i=$start; i<$end; i++ ))
do
    IFS=',' read -r bash_script_path full_path subject_id output_directory full_output_path <<< $(awk -v line=$i 'NR == line' /usr4/ugrad/spuduch/fmADRD/ppmi_fastsurfer_args.csv)

    if [ ! -f "$full_output_path/fs_volumes.csv" ]; then
        echo "Output file missing, cleaning up and rerunning for row $i..."
        rm -rf "$full_output_path/*"  # Cleanup: remove all files in the directory
        mkdir -p "$full_output_path"  # Ensure the directory exists

        # escaped_full_path=$(echo "$full_path" | sed 's/,/\\,/g')

        echo "Running row number $i: bash $bash_script_path $full_path $subject_id $output_directory"
        bash "/usr4/ugrad/spuduch/RadiologistRatings/fs_docker_bash.sh" "$full_path" "$subject_id" "$output_directory"

        python -c "import fs_scripts; fs_scripts.filter_fs_output('$full_output_path')"
    else
        echo "Output for row $i already exists, skipping..."
    fi
done


#!/usr/bin/env bash

# Input parameters
t1_image=$1
subject_id=$2
output_dir=$3

# FastSurfer docker command
# docker run --gpus device=3 \
#     -v "$(dirname "${t1_image}")":/data \
#     -v "${output_dir}":/output \
#     -v /SeaExp_1-ayan/license.txt:/fs_license \
#     --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
#     --fs_license /fs_license \
#     --sd /output \
#     --sid "${subject_id}" \
#     --t1 /data/$(basename "${t1_image}") \
#     --seg_only

# singularity instead
singularity exec --nv \
                 --no-home \
                 -B "$(dirname "${t1_image}")":/data \
                 -B "${output_dir}":/output \
                 -B /usr4/ugrad/spuduch/license.txt:/fs_license \
                 /projectnb/vkolagrp/fastsurfer-gpu.sif \
                 ~/RadiologistRatings/tools/fastsurfer/run_fastsurfer.sh \
                 --sd /output \
                 --sid "${subject_id}" \
                 --seg_only \
                 --parallel \
                 --threads 4 \
                 --t1 /data/$(basename "${t1_image}") \


                 
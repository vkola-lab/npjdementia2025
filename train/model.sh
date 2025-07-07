#!/bin/bash -l

# Set SCC project
#$ -P vkolagrp
#$ -N wandb_nAD_sweep           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=6
#$ -m bes
#$ -l h_rt=2:00:00

module load miniconda
conda activate mri_radiology

python /usr4/ugrad/spuduch/RadiologistRatings/dev/train/model.py


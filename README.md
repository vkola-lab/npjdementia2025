# Neuroradiology Radiomics Pipeline

## Overview
This repository provides a comprehensive pipeline for preprocessing, feature extraction, modeling, and evaluation of neuroradiology imaging data. It integrates FreeSurfer-based volumetric analysis, white matter hyperintensity (WMH) burden estimation, and machine learning models (XGBoost) for Alzheimer’s disease (AD) and Other Imaging Evident Dementia (OIED) predictions.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Data Structure](#data-structure)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Results & Analysis](#results--analysis)
- [Utilities](#utilities)

## Environment Setup
1. Create a Conda environment:
   ```bash
   conda env create -f environment.yml
   # or for macOS specific
   conda env create -f environment-macos.yml
   ```
2. Activate the environment:
   ```bash
   conda activate neurorad-radiomics
   ```

## Data Structure
- `data/`
  - `ml_data/`: preprocessed tabular datasets (`ml_data_filtered.csv`, `ml_test_data_filtered.csv`, etc.)
  - `dev-model/`: saved XGBoost models and tuning parameters
  - `feature_config.json`: definitions of volumetric, WMH, imaging, and demographic features
  - `lobe_mapping.json`, `feature_names_map.json`, etc.

## Preprocessing
Scripts under `preprocessing/` handle MRI data conversion and feature extraction:
- `fs_scripts.py`: filter and convert FreeSurfer outputs, parse volume statistics
- `lst_scripts.py`: run LST (Lesion Segmentation Tool) for WMH estimation
- `NACC_mris.py` & `prep_array_job_csv_.py`: generate batch arguments for array jobs on NACC datasets

## Model Training
Training scripts are in `train/`:
- `model.py`: preprocessing and label generation
- `wandb_sweep.py`: hyperparameter sweep setup with Weights & Biases
- `cv.py`: cross-validation with GroupKFold
- `final_train.py`: train final AD, OIED models with hyperparmeters defined from sweeps

## Results & Analysis
Misc. analyses and plotting scripts are in `results/`

## Utilities
- `utils/load_data.py`: load and derive feature lists from JSON config
- `utils/dump_dkt_atlas.py`: export region names grouped by lobes

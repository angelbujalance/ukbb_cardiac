#!/bin/bash

#SBATCH --partition=fat_rome
#SBATCH --job-name=SEG_CMR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --output=output/segmentation_UKBB_CMR_full_data_all_ts_re_run.out

# Activate your environment
# source activate mae3
# pip install cudnn
# pip install cudatoolkit
# pip install tensorflow-cpu

DATA_DIR="$HOME/deep_risk/ukbb_cardiac/dataset_toy/1052287"
DATA_DIR='/scratch-shared/abujalancegome/CMR_data'
MOD_PATH="$HOME/deep_risk/ukbb_cardiac/trained_model/FCN_sa"

python deploy_network.py --data_dir ${DATA_DIR} --model_path ${MOD_PATH}
#!/bin/bash

#SBATCH --partition=fat_rome
#SBATCH --gpus=0
#SBATCH --job-name=CROP_CMR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=output/crop_images_cmr_time_all_2.out

DATA_DIR='/scratch-shared/abujalancegome/CMR_data'

python crop_images.py --data_path ${DATA_DIR}
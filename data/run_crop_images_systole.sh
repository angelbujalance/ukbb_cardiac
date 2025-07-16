#!/bin/bash

#SBATCH --partition=fat_rome
#SBATCH --gpus=0
#SBATCH --job-name=CROP_SYS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=output/crop_images_cmr_time_all_FULL_systole.out

DATA_DIR='/scratch-shared/abujalancegome/CL_data'

python crop_images_systole.py --data_path ${DATA_DIR} --max_height 85 --max_width 92
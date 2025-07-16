#!/bin/bash

#SBATCH --partition=fat_rome
#SBATCH --job-name=downl01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=100:00:00
#SBATCH --output=output/download_ukbb_data_long.out

# python download_data_ukbb_general.py

#  /scratch-shared/abujalancegome/CL_data

python download_ukbb_data.py --id_path /home/abujalancegome/patients_w_ecg_cmr.txt \
                             --out_dir /projects/prjs1252/UKBB_raw/CMR-long
                            

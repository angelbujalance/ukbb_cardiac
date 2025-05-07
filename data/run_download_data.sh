#!/bin/bash

#SBATCH --partition=fat_rome
#SBATCH --job-name=downl_CMR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=output/download_ukbb_data_CL.out

# python download_data_ukbb_general.py

python download_ukbb_data.py --id_path /home/abujalancegome/patients_w_ecg_cmr.txt \
                             --out_dir /scratch-shared/abujalancegome/CL_data

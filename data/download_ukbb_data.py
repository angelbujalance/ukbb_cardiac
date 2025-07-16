"""
    The script downloads the cardiac MR images for a UK Biobank Application and
    converts the DICOM into nifti images.
    Optimized for faster processing.
"""
import os
import glob
import pandas as pd
import shutil
from biobank_utils import *
import dateutil.parser
import concurrent.futures
import time
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Download UKBB Data', add_help=False)

    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--id_path', type=str)

    return parser


def process_patient(eid, data_root):
    """Process a single patient's data - separated as a function for parallelization"""
    start_time = time.time()
    eid = str(eid)
    print(f"\033[33m Processing patient with eid: {eid} \033[0m")

    # Patient directory
    data_dir = os.path.join(data_root, eid)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Patient DICOM directory
    dicom_dir = os.path.join(data_dir, 'dicom')
    if not os.path.exists(dicom_dir):
        os.makedirs(dicom_dir)

    # 209 -> short axis
    source_zip = f'/projects/0/aus20644/data/ukbiobank/imaging/cardiac_mri/20209_short_axis_heart/imaging_visit_array_0/{eid}_20209_2_0.zip'
                  # projects/0/aus20644/data/ukbiobank/imaging/cardiac_mri/20211_cine_tagging_images/imaging_visit_array_0/{eid}_20211_2_0
    # 211 -> tag images
    # source_zip = f'/projects/0/aus20644/data/ukbiobank/imaging/cardiac_mri/20211_cine_tagging_images/imaging_visit_array_0/{eid}_20211_2_0.zip'

    # 211 -> long axis
    source_zip = f'/projects/0/aus20644/data/ukbiobank/imaging/cardiac_mri/20208_long_axis_heart/imaging_visit_array_0/{eid}_20208_2_0.zip'
    shutil.copy(source_zip, dicom_dir)

    # Unpack the data (list the zip files for the different modalities)
    files = glob.glob(f'{dicom_dir}/{eid}_*.zip')  

    print("files: ", files)

    for f in files:
        print("Unzipping file: ", f)

        # Actually unzip the file ('>' part is to suppress the output)
        os.system(f'unzip -o {f} -d {dicom_dir} > /dev/null 2>&1')

        # Convert the cvs file to csv, since cvs doesn't make any sense    
        if os.path.exists(os.path.join(dicom_dir, 'manifest.cvs')):
            os.system(f'cp {os.path.join(dicom_dir, "manifest.cvs")} {os.path.join(dicom_dir, "manifest.csv")}')

        # process the manifest file and write it to manifest2.csv
        process_manifest(os.path.join(dicom_dir, 'manifest.csv'),
                         os.path.join(dicom_dir, 'manifest2.csv'))

        # read the manifest2.csv file into a dataframe
        df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'), on_bad_lines='skip')

        # Group the files into subdirectories for each imaging series
        for series_name, series_df in df2.groupby('series discription'):
            # Creates a directory for each series
            series_dir = os.path.join(dicom_dir, series_name)
            if not os.path.exists(series_dir):
                os.mkdir(series_dir)

            # Get the filenames for each series and move them to the series directories
            series_files = [os.path.join(dicom_dir, x) for x in series_df['filename']]
            # Use xargs instead of direct mv for better handling of large file lists
            with open(f"{dicom_dir}/filelist.txt", "w") as f:
                f.write("\n".join(series_files))
            os.system(f'cat {dicom_dir}/filelist.txt | xargs -I% mv % {series_dir}')

    # Convert dicom files to nifti images
    dset = Biobank_Dataset(dicom_dir)
    dset.read_dicom_images()
    dset.convert_dicom_to_nifti(data_dir)

    # Remove intermediate files
    os.system(f'rm -rf {dicom_dir}')
    os.system(f'rm -f {eid}_*.zip')

    end_time = time.time()
    print(f"\033[32m Completed patient {eid} in {end_time - start_time:.2f} seconds \033[0m")

    return eid


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()

    # Where the data will be downloaded
    data_root = args.out_dir

    # Read patient IDs
    ids_patients = args.id_path
    df = pd.read_csv(ids_patients, names=['IDs'])
    data_list = list(df['IDs'])
    data_list.sort()
    # data_list = data_list[len(data_list) // 2 :]

    # Performance improvement: Process patients in parallel
    max_workers = args.num_workers  # Adjust based on system resources

    total_start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for eid in data_list:
            futures.append(executor.submit(process_patient, eid, data_root))

        for future in concurrent.futures.as_completed(futures):
            try:
                eid = future.result()
                print(f"Completed processing for {eid}")
            except Exception as e:
                print(f"Error processing patient: {e}")

    total_end = time.time()
    print(f"\033[32m All patients processed in {total_end - total_start:.2f} seconds \033[0m")

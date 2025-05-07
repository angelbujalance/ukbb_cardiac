# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
    The script downloads the cardiac MR images for a UK Biobank Application and
    converts the DICOM into nifti images.
    """
import os
import glob
import pandas as pd
import shutil
from biobank_utils import *
import dateutil.parser


if __name__ == '__main__':
    # Where the data will be downloaded
    data_root = '/vol/vipdata/data/biobank/cardiac/Application_18545/data_path'

    # Path to the UK Biobank utilities directory
    # The utility programmes can be downloaded at http://biobank.ctsu.ox.ac.uk/crystal/download.cgi
    util_dir = '/vol/vipdata/data/biobank/cardiac/Application_18545/util_path'

    # The authentication file (application id + password) for downloading the data for a specific
    # UK Biobank application. You will get this file from the UK Biobank website after your
    # application has been approved.
    ukbkey = '/homes/wbai/ukbkey'

    # The spreadsheet which lists the anonymised IDs of the subjects.
    # You can download a very large spreadsheet from the UK Biobank website, which exceeds 10GB.
    # I normally first filter the spreadsheet, select only a subset of subjects with imaging data
    # and save them in a smaller spreadsheet.
    # csv_file = '/vol/vipdata/data/biobank/cardiac/Application_18545/downloaded/ukb9137_image_subset.csv'
    # df = pd.read_csv(os.path.join(csv_dir, csv_file), header=1)
    
    # Just takes a list of patient IDs
    # data_list = df['eid']
    
    
    # data_root = '/home/abujalancegome/deep_risk/ukbb_cardiac/CMR_data/'

    data_root = '/scratch-shared/abujalancegome/CMR_data'
    data_root = '/home/abujalancegome/deep_risk/ukbb_cardiac/data/dataset_tagged'


    ids_patients = '/home/abujalancegome/patients_w_ecg_cmr.txt'
    ids_patients = '/home/abujalancegome/deep_risk/cmr_pretrain/labels/ids.csv'

    df = pd.read_csv(ids_patients, names=['IDs'])

    data_list = list(df['IDs'])
    print(data_list[:10])

    # 100 = 2 GB
    data_list = [1000346]
    data_list = [1000346, 1041016, 1000605]
    data_list = [1993836]
    # 4865907, 1811541, 4609796, 2103667, 1685270, 3636526, 2640472, 3692444, 3267609, 1809025,
    # 1992086, 5786719, 1214870, 4178644, 3962267, 2545270, 5772667, 4845376, 5731954, 4164892,
    # 1955539, 3644551, 2703378, 2126811, 1960361, 2213341, 1052287, 1285515, 2996101, 3663187,
    # 5648476, 5468180, 5649382, 2017741, 3745052, 2104653, 5031602, 5826230, 2959892, 4691728,
    # ]

    # Download cardiac MR images for each subject
    start_idx = 0
    end_idx = len(data_list)

    # THIS HAPPENS FOR EACH SUBJECT
    for i in range(start_idx, end_idx):
        eid = str(data_list[i])

        print(f"\033[33m Processing patient with eid: {eid} \033[0m")

        # Patient directory
        data_dir = os.path.join(data_root, eid)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # # Patient DICOM directory
        dicom_dir = os.path.join(data_dir, 'dicom')
        if not os.path.exists(dicom_dir):
            os.makedirs(dicom_dir)

        # 209 -> short axis
        # shutil.copy(os.path.join(data_root, '/projects/0/aus20644/data/ukbiobank/imaging/cardiac_mri/20209_short_axis_heart/imaging_visit_array_0/{0}_20209_2_0.zip'.format(eid)), dicom_dir)
        shutil.copy(os.path.join(data_root, '/projects/0/aus20644/data/ukbiobank/imaging/cardiac_mri/20211_cine_tagging_images/imaging_visit_array_0/{0}_20211_2_0.zip'.format(eid)), dicom_dir)

        # Unpack the data (list the zip files for the different modalities)
        files = glob.glob('{1}/{0}_*.zip'.format(eid, dicom_dir))  
        
        print("files: ", files)
        
        for f in files:
            
            print("Unzipping file: ", f)
                        
            # Actually unzip the file ('>' part is to suppress the output)
            os.system('unzip -o {0} -d {1} > /dev/null 2>&1'.format(f, dicom_dir))
                        
            # Convert the cvs file to csv, since cvs doesn't make any sense    
            if os.path.exists(os.path.join(dicom_dir, 'manifest.cvs')):
                os.system('cp {0} {1}'.format(os.path.join(dicom_dir, 'manifest.cvs'),
                                              os.path.join(dicom_dir, 'manifest.csv')))

            # process the manifest file and write it to manifest2.csv
            process_manifest(os.path.join(dicom_dir, 'manifest.csv'),
                             os.path.join(dicom_dir, 'manifest2.csv'))

            # read the manifest2.csv file into a dataframe
            df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'), on_bad_lines='skip')

            # Patient ID and acquisition date
            pid = df2.at[0, 'patientid']
            date = dateutil.parser.parse(df2.at[0, 'date'][:11]).date().isoformat()
 
            # Group the files into subdirectories for each imaging series
            for series_name, series_df in df2.groupby('series discription'):
                # Creates a directory for each series, lovely
                series_dir = os.path.join(dicom_dir, series_name)
                if not os.path.exists(series_dir):
                    os.mkdir(series_dir)

                # Get the filenames for each series and move them to the series directories
                series_files = [os.path.join(dicom_dir, x) for x in series_df['filename']]
                os.system('mv {0} {1}'.format(' '.join(series_files), series_dir))

        # # Convert dicom files and annotations into nifti imagespip3 install tensorflow-gpu numpy scipy matplotlib seaborn pandas python-dateutil pydicom SimpleITK nibabel scikit-image opencv-python vtk

        # Create a Biobank_Dataset object based on all the series for a particular patient
        # It practically justs creates dictionaries for different series subclasses (SAX, LAX)
        dset = Biobank_Dataset(dicom_dir)
        dset.read_dicom_images()
        dset.convert_dicom_to_nifti(data_dir)

        # Remove intermediate files
        os.system('rm -rf {0}'.format(dicom_dir))
        os.system('rm -f {0}_*.zip'.format(eid))

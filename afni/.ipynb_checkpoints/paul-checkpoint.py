import os
import sys
import numpy as np
import boto3
import webdataset as wds
import nibabel as nib
import pickle as pkl
from subprocess import call

NUM_DATASETS = 1

# Connect to S3
s3 = boto3.client('s3')

# Set the bucket name and folder name
bucket_name = 'openneuro.org'

# List all folders in the parent directory
response = s3.list_objects_v2(Bucket=bucket_name, Prefix='', Delimiter='/')

# Extract the folder names from the response
folder_names = [x['Prefix'].split('/')[-2] for x in response.get('CommonPrefixes', [])]

sink = wds.ShardWriter("tars/%06d.tar")

for folder_name in folder_names[:NUM_DATASETS]:
    # List all objects in the folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    for obj in response.get('Contents', []):
        obj_key = obj['Key']

        if '_T1w.nii.gz' in obj_key or '_bold.nii.gz' in obj_key:
            print(folder_name, obj['Key'])
        if '_T1w.nii.gz' in obj_key: # Anatomical
            # Store subject number to verify anat/func match
            anat_subj = obj_key.split('/')[1]
            
            # Download the object to tmp location
            filename = os.path.join('openneuro', obj_key)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            s3.download_file(bucket_name, obj_key, filename)

            # store the head of anat_subj
            anat_header = nib.load(filename).header
            
        elif '_bold.nii.gz' in obj_key: # Functional bold
            # Verify func/anat subject number match
            func_subj = obj_key.split('/')[1]
            if anat_subj != func_subj:
                raise ValueError('Incompatible subject number found.')

            # Get the object key and download the object to tmp location
            filename = os.path.join('openneuro', obj_key)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            s3.download_file(bucket_name, obj_key, filename)
            
            # AFNI #
            afni_filename = filename[:-len('.nii.gz')] + '_aligned.nii.gz'
            
            command = f"align_epi_anat.py -anat tpl-MNI152NLin2009cAsym_res-03_T1w_brain.nii.gz -epi {filename} -epi_base 0 \
                        -epi2anat -rigid_body -ginormous_move -anat_has_skull no -epi_strip None -suffix _aligned -volreg on \
                        -tshift off -save_resample -master_epi 3.00"
            err
            call(command,shell=True)
            
            # transform AFNI outputs to nifti file
            command = f"3dAFNItoNIFTI -prefix {filename}_aligned.nii.gz {func}_aligned+tlrc"
            call(command,shell=True)
            
            # remove unnecessary AFNI outputs
            call("rm *+tlrc.*",shell=True)
            call("rm *vr_motion*",shell=True)
            call("rm *mat.aff*",shell=True)

            if os.path.exists(afni_filename):
                afni_data = nib.load(afni_filename).get_fdata()
                afni_data /= np.mean(np.abs(afni_data))
                afni_data = afni_data.astype(np.float16)

                sink.write({
                    "__key__": obj_key,
                    "func.npy": afni_data,
                    "header.pyd": anat_header,
                })
                
                # delete the nifti now that youve saved it to numpy
                call(f"rm {afni_filename}",shell=True)
            else:
                err

sink.close()
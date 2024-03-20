#!/usr/bin/env python
# coding: utf-8

# In[50]:


# Standard Library Imports
import os
import sys
from subprocess import call, check_output
import json
import time

# Third-Party Library Imports
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import webdataset as wds
import nibabel as nib
import pickle as pkl
import h5py
from PIL import Image
import matplotlib.pyplot as plt


# In[51]:


worker_id = int(sys.argv[1])
print(f"WORKER_ID={worker_id}")


# In[52]:


temp_dir = os.getcwd() + f"/temp{worker_id}" # the folder where the AFNI container will do its work
mni_dir = os.getcwd() + f"/MNIs{worker_id}" # the folder where MNI outputs will go

command = f"rm -r {temp_dir}"
call(command,shell=True)
command = f"rm -r {mni_dir}"
call(command,shell=True)

os.makedirs(temp_dir, exist_ok=True)
os.makedirs(mni_dir, exist_ok=True)
print(temp_dir)
print(mni_dir)


# In[53]:


s3 = boto3.client('s3')
bucket_name = 'proj-fmri'
prefix = 'fmri_foundation_datasets/parallel_openneuro/'

if os.path.exists(f"discarded_dataset_ids_{worker_id}.npy"):
    discarded_dataset_ids = np.load(f"discarded_dataset_ids_{worker_id}.npy").tolist()
else:
    discarded_dataset_ids = []
print("discarded_dataset_ids",discarded_dataset_ids)

paginator = s3.get_paginator('list_objects_v2')
file_name_list = []
for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in page.get('Contents', []):
        file_name = obj['Key']
        file_name_list.append(file_name)
print("len(file_name_list) =", len(file_name_list))

# subset to current worker
worker_id_idx = np.linspace(0,len(file_name_list),30)[worker_id:worker_id+2].astype(np.int32).tolist()
file_name_list = file_name_list[worker_id_idx[0]:worker_id_idx[1]]
print("len(file_name_list) =", len(file_name_list))


# In[54]:


max_wait = 500
overlap_cnt = 0
print("starting...")
for file_name in file_name_list:
    if file_name.endswith('_bold.nii.gz'):
        dataset_id = file_name.split('/')[2]
        
        if np.any(np.isin(dataset_id, discarded_dataset_ids)):
            continue
        
        func_path = file_name.split('/')[-1]
        temp_file_path = temp_dir + '/' + dataset_id + '/' + func_path
        mni_file_path = mni_dir + '/' + dataset_id + '/' + func_path
        
        os.makedirs(temp_dir + '/' + dataset_id, exist_ok=True)
        os.makedirs(mni_dir + '/' + dataset_id, exist_ok=True)
        
        afni_filename = mni_dir + '/' + dataset_id + '/' + func_path.split(".nii.gz")[0] + "_MNI.nii.gz"
        s3_afni_filename = f"s3://proj-fmri/fmri_foundation_datasets/openneuro_MNI/{dataset_id}/{func_path.split('.nii.gz')[0] + '_MNI.nii.gz'}"
        
        # check if MNI output already exists
        MNI_done = call(f"aws s3 ls {s3_afni_filename}",shell=True)
        if MNI_done==0:
            print(f"      done: {s3_afni_filename}")
            continue
            
        # download from s3
        print(f"downloading {temp_file_path}")
        try:
            #s3.download_file(bucket_name, file_name, temp_file_path)
            command = f"aws s3 cp s3://{bucket_name}/{file_name} {temp_file_path}"
            call(command,shell=True)
        except:
            print("failed to download? 1")

        while not os.path.exists(f"{temp_file_path}"):
            print(f"s3 download failed. trying again... {temp_file_path}")
            try:
                # s3.download_file(bucket_name, file_name, temp_file_path)
                command = f"aws s3 cp s3://{bucket_name}/{file_name} {temp_file_path}"
                call(command,shell=True)
            except:
                print("failed to download? 2")
            time.sleep(5)

        # Wait for AFNI to be complete
        print(f'waiting for {afni_filename}')
        waiting_time = 0
        while not os.path.exists(afni_filename):
            time.sleep(5)     
            waiting_time += 5
            if waiting_time > max_wait:
                break

        if waiting_time <= max_wait:
            time.sleep(5) # wait to ensure file was fully created
            with open(mni_file_path.split(".nii.gz")[0] + "_overlap.txt", 'r') as file:
                try:
                    overlap = file.readlines()
                    overlap = np.array(overlap).astype(np.float32)[0]
                except:
                    print("overlap error!")
                    overlap = 0 # in case some weird error occurs where overlap txt is empty, assume its ok
            
            # if overlap >20%, discard outputs and skip this dataset
            if overlap>20:
                overlap_cnt += 1
                if overlap_cnt>5:
                    discarded_dataset_ids.append(dataset_id)
                    print("discarded_dataset_ids")
                    print(discarded_dataset_ids)
                    np.save(f"discarded_dataset_ids_{worker_id}.npy",discarded_dataset_ids)
                    overlap_cnt = 0
            else:   
                overlap_cnt = 0
                
                command = f"aws s3 cp {afni_filename} {s3_afni_filename}"
                call(command,shell=True)
        else:
            print("waiting time exceeded...")
            
        # remove files
        command = f"rm {temp_file_path}"
        call(command,shell=True)
        
        command = f"rm {afni_filename}"
        call(command,shell=True)


# In[45]:


end_dir = os.getcwd() + f"/END_{worker_id}"
os.makedirs(end_dir, exist_ok=True)


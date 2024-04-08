#!/bin/bash

worker_id=$1
TEMP_PATH="temp${worker_id}/"
OUT_PATH="MNIs${worker_id}/"
END_PATH="END_${worker_id}/"
TEMPLATE_PATH="tpl-MNI152NLin2009cAsym_res-02_T1w_brain.nii.gz"
suffix="_MNI"

echo "Worker ID is $worker_id"
echo "OUT_PATH is $OUT_PATH"
echo "TEMP_PATH is $TEMP_PATH"
echo "END_PATH is $END_PATH"

while true; do
    # Check if the "END" folder exists
    if [ -d $END_PATH]; then
        echo "END folder found. Stopping the script."
        break
    fi

    # Find all .nii.gz files in the folder (including subdirectories)
    readarray -t files < <(find $TEMP_PATH -type f -name "*_bold.nii.gz")
    for func in "${files[@]}"; do
        # Remove the file extension for use in file naming
        func_base=$(basename "$func" ".nii.gz")
        echo "func_base" $func_base

        # Extract the dataset ID (the name of the parent directory)
        dataset_id=$(basename $(dirname "$func"))
    
        # Align the functional image to the anatomical template
        align_epi_anat.py -anat ${TEMPLATE_PATH} -epi $func -epi_base 0 -epi_strip 3dAutomask -epi2anat -ginormous_move -anat_has_skull no -suffix $suffix -volreg on -tshift off -save_resample -master_epi 2.00 -output_dir ${TEMP_PATH}/${dataset_id}

        # Convert AFNI format to NIFTI format
        3dAFNItoNIFTI -prefix ${TEMP_PATH}/${dataset_id}/${func_base}${suffix}.nii.gz ${TEMP_PATH}/${dataset_id}/${func_base}${suffix}'+tlrc'

        # if file exists, then continue...
        if [ -f ${TEMP_PATH}/${dataset_id}/${func_base}${suffix}.nii.gz ]; then
            # Move and remove files
            mv ${TEMP_PATH}/${dataset_id}/${func_base}${suffix}.nii.gz ${OUT_PATH}/${dataset_id}/${func_base}${suffix}.nii.gz
            rm ${TEMP_PATH}/${dataset_id}/${func_base}*
        
            # Calculate and log the overlap
            3dABoverlap -no_automask ${TEMPLATE_PATH} ${OUT_PATH}/${dataset_id}/${func_base}${suffix}.nii.gz | awk 'NR==3 {print $7}' >> ${OUT_PATH}/${dataset_id}/${func_base}_overlap.txt
        else
            echo "no output. retry."
        fi
    done
    echo "waiting 10 seconds..."
    sleep 10 # wait 10 seconds before iterating again
done
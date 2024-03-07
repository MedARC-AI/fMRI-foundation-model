#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=a40x
#SBATCH --job-name=afni
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=350:00:00       # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err       # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --container-image=afni/afni_make_build
#SBATCH --container-mounts=/weka/proj-fmri/paulscotti,/admin/home-paulscotti,/opt/slurm/etc

cd /weka/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation/afni_conversion 

. afni_watch.sh NSD
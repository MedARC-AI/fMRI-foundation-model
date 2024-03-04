# fMRI Foundation Model

In-progress -- this repo is under active development in the MedARC discord server. https://medarc.ai/fmri

1. Download contents of https://huggingface.co/datasets/pscotti/fmrifoundation and place them in a folder. You will need to specify the paths to this folder in the code.

```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="pscotti/fmrifoundation", repo_type = "dataset", revision="main", cache_dir = "./cache" ,
    local_dir= "your_local_dir", local_dir_use_symlinks = False, resume_download = True)
```

2. Run setup.sh to create a new "fmri" virtual environment
3. Activate the virtual environment with "source fmri/bin/activate"

# Running Jepa

To run Jepa, make sure to edit the yamls files in `jepa/configs/pretrain`. Specifically, logging -> folder to your logging folder path and data -> datasets to dataset path.

After that, run the following command:

```
PYTHONPATH=jepa python -m jepa.app.main --fname $config_path --devices cuda:0
```

where $config_path is the path to your config in `jepa/configs/pretrain`.

For a distruibted run, perform the following:

```
python main_distributed.py \
  --fname $config_path \
  --folder $path_to_save_submitit_logs \
  --partition $slurm_partition \
  --nodes 2 --tasks-per-node 8 \
  --time 1000
```

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

4. Install causal_conv1d and mamba from VideoMamba 
```
git clone https://github.com/OpenGVLab/VideoMamba.git
cd VideoMamba
```
Install causal_conv1d and mamba 
```
pip install -r requirements.txt
pip install -e causal-conv1d
pip install -e mamba
```
We don't actually need VideoMamba so you can delete that repo once you've installed causal_conv1d and mamba 

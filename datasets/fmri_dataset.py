import os
import torch
import tarfile
import webdataset as wds
from skimage import io
from concurrent.futures import ThreadPoolExecutor, as_completed

class FMRIDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, are_tars=True):
        self.root = root
        self.transform = transform
        self.folders = os.listdir(root)
        self.are_tars = are_tars

        self.files = []
        if self.are_tars:
            self.folders = [folder for folder in self.folders if folder[-4:] == ".tar"]
        
        self._load_input_files()

    def _load_from_tar(self, folder):
            with tarfile.open(os.path.join(self.root, folder), 'r') as tar:
                files = tar.getmembers()
                return [(os.path.join(self.root, folder), file.name) for file in files if "func" in file.name]

    def _load_from_directory(self, folder):
        folder_path = os.path.join(self.root, folder)
        return [(folder_path, f) for f in os.listdir(folder_path) if "func" in f and os.path.isfile(os.path.join(folder_path, f))]

    def _load_input_files(self):
        with ThreadPoolExecutor() as executor:
            futures = []
            for folder in self.folders:
                if self.are_tars:
                    futures.append(executor.submit(self._load_from_tar, folder))
                else:
                    futures.append(executor.submit(self._load_from_directory, folder))

            for future in as_completed(futures):
                self.files.extend(future.result())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path_tuple = self.files[idx]
        fmri = self.load_file(file_path_tuple)        
        if self.transform:
            fmri = self.transform(fmri)
        
        return fmri

    def load_file(self, file_path_tuple):
        if self.are_tars:
            tar_path, member_name = file_path_tuple
            with tarfile.open(tar_path, 'r') as tar:
                member = tar.getmember(member_name)
                f = tar.extractfile(member)
                fmri = io.imread(f)
        else:
            file_path = os.path.join(*file_path_tuple)
            fmri = io.imread(file_path)
        
        return fmri

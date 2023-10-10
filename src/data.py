import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, input_file, target_file):
        self.inputs = np.loadtxt(input_file)
        self.targets = np.loadtxt(target_file)

    #     Normalize input data
    #     self.inputs = self.normalize(self.inputs)

    def normalize(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        target_sample = self.targets[idx]
        input_tensor = torch.tensor(input_sample, dtype=torch.float32)
        target_tensor = torch.tensor(target_sample, dtype=torch.float32)
        return input_tensor, target_tensor

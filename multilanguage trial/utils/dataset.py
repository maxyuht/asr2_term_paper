# utils/dataset.py

import os
import torch
from torch.utils.data import Dataset

class FeatureEmotionDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        data = torch.load(self.feature_files[idx])
        feature = data["feature"]        # shape: [T, 768]
        label = data["label"]            # int: 情绪类别
        return feature, label

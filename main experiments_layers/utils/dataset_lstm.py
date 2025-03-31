import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class FeatureEmotionDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        data = torch.load(self.feature_files[idx])
        feature = data["feature"]  # shape: [T, 768]
        label = data["label"]
        return feature, label

# ✅ 用于 LSTM：padding + 返回原始长度
def collate_fn(batch):
    features, labels = zip(*batch)
    lengths = [f.shape[0] for f in features]  # 原始 T 长度
    features_padded = pad_sequence(features, batch_first=True)  # shape: [B, T_max, D]
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return features_padded, lengths, labels

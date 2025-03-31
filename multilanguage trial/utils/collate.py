from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn_lstm(batch):
    features = [item[0] for item in batch]  # [T_i, 768]
    labels = [item[1] for item in batch]
    lengths = [f.shape[0] for f in features]

    padded = pad_sequence(features, batch_first=True)  # → [B, max_T, 768]
    labels = torch.tensor(labels)
    return padded, labels, lengths  # 送给 LSTM

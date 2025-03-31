# scripts/split_dataset_v1_3th_layer_1.py

import os
import random
import shutil
import torch
from collections import defaultdict
import pandas as pd

# ✅ 设置路径
features_dir = "./features_v1_3th_layer_1"
splits_dir = "./splits_v1_3th_layer_1"
split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

# ✅ 创建 split 文件夹
for split in split_ratio:
    os.makedirs(os.path.join(splits_dir, split), exist_ok=True)

# ✅ 读取 .pt 文件，并按 label 分类
label_to_files = defaultdict(list)

print("Reading 3th-layer features...")
for file in os.listdir(features_dir):
    if file.endswith(".pt"):
        path = os.path.join(features_dir, file)
        try:
            data = torch.load(path)
            label = int(data["label"])
            label_to_files[label].append(file)
        except Exception as e:
            print(f"❌ Failed: {file}, error: {e}")

# ✅ 划分
split_stats = defaultdict(lambda: defaultdict(int))

print("\nSplitting the data...")
for label, files in label_to_files.items():
    random.shuffle(files)
    n_total = len(files)
    n_train = int(split_ratio["train"] * n_total)
    n_val = int(split_ratio["val"] * n_total)
    n_test = n_total - n_train - n_val

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    for split_name, split_files in splits.items():
        for f in split_files:
            src = os.path.join(features_dir, f)
            dst = os.path.join(splits_dir, split_name, f)
            shutil.copy(src, dst)
            split_stats[split_name][label] += 1

# ✅ 输出统计
df = pd.DataFrame(split_stats).fillna(0).astype(int)
df.index.name = "Emotion Label"
df.columns.name = "Subset"
print("\n✅ Split result for 3th-layer features:")
print(df)

# ✅ 保存统计
df.to_csv("split_stats_v1_3th_layer_1.csv")

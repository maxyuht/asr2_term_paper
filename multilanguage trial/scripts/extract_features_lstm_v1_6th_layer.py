import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import random

# ✅ 加载 HuBERT 预训练模型
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained(
    "facebook/hubert-base-ls960",
    output_hidden_states=True  # 🟢 启用中间层输出
).eval().to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 标签映射
label_map = {"happy": 0, "sad": 1, "surprised": 2, "calm": 3}
label_names = {v: k for k, v in label_map.items()}

# ✅ 路径设置
data_dir = "./data_v1"
save_dir = "./features_lstm_v1"
os.makedirs(save_dir, exist_ok=True)

# ✅ 用于可视化的采样帧和标签（每类不超过 50 条帧）
tsne_features = []
tsne_labels = {0: [], 1: [], 2: [], 3: []}  # 情绪 → 对应帧

print("🚀 HuBERT is extracting frame-level features...\n")
for emotion in os.listdir(data_dir):
    emo_path = os.path.join(data_dir, emotion)
    if not os.path.isdir(emo_path): continue

    if emotion not in label_map:
        print(f"⚠️ Skipping unknown emotion: {emotion}")
        continue

    label = label_map[emotion]
    print(f"🎧 Processing emotion: {emotion}")
    file_list = [f for f in os.listdir(emo_path) if f.endswith(".wav")]

    for wav in tqdm(file_list, desc=f"{emotion}", ncols=80):
        wav_path = os.path.join(emo_path, wav)
        waveform, _ = librosa.load(wav_path, sr=16000)

        # ✅ 提取帧级特征 [1, T, 768] → [T, 768]
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            hidden_states = hubert(inputs.input_values.to(hubert.device), output_hidden_states=True).hidden_states[6]
            feature = hidden_states.squeeze(0).cpu()  # [T, 768]

        # ✅ 保存 .pt 特征
        save_path = os.path.join(save_dir, wav.replace(".wav", ".pt"))
        torch.save({"feature": feature, "label": label}, save_path)

        # ✅ 可视化采样帧（每类最多保留 50 个帧）
        if len(tsne_labels[label]) < 50:
            frame_indices = random.sample(range(feature.size(0)), min(5, feature.size(0)))
            for idx in frame_indices:
                tsne_features.append(feature[idx].numpy())
                tsne_labels[label].append(1)

print(f"\n✅ All features saved to: {save_dir}")
print("🧠 Collected {} total frame-level features for t-SNE.".format(len(tsne_features)))

# ✅ 特征可视化（t-SNE）
print("\n🎨 Feature visualization (t-SNE)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced = tsne.fit_transform(np.array(tsne_features))

# ✅ 绘图
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange']
for label_id, color in zip(range(4), colors):
    indices = [i for i, feat_label in enumerate(tsne_labels[label_id]) if feat_label == 1]
    if indices:
        reduced_subset = reduced[sum(len(tsne_labels[i]) for i in range(label_id)):sum(len(tsne_labels[i]) for i in range(label_id + 1))]
        plt.scatter(reduced_subset[:, 0], reduced_subset[:, 1], label=label_names[label_id], color=color, alpha=0.7, s=30)

plt.title("t-SNE Visualization of HuBERT Frame-level Features")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_visualization_lstm_v1.png")
plt.show()

print("📍 t-SNE plot saved as tsne_visualization_lstm_v1.png")

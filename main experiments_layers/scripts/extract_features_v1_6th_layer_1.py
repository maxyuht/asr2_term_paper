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

# ✅ 加载预训练模型（启用 hidden_states）
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained(
    "facebook/hubert-base-ls960",
    output_hidden_states=True
).eval().to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 标签映射
label_map = {"happy": 0, "sad": 1, "surprised": 2, "calm": 3}
label_names = {v: k for k, v in label_map.items()}

# ✅ 路径设置
data_dir = "./data_v1"
save_dir = "./features_v1_6th_layer_1"
os.makedirs(save_dir, exist_ok=True)

# ✅ 用于可视化（仍用 mean）
all_features = []
all_labels = []

print("HuBERT is extracting 6th-layer features for attention pooling...\n")
for emotion in os.listdir(data_dir):
    emo_path = os.path.join(data_dir, emotion)
    if not os.path.isdir(emo_path): continue

    print(f"Working on: {emotion}")
    file_list = [f for f in os.listdir(emo_path) if f.endswith(".wav")]

    for wav in tqdm(file_list, desc=f"{emotion}", ncols=80):
        wav_path = os.path.join(emo_path, wav)
        waveform, _ = librosa.load(wav_path, sr=16000)

        # ✅ 提取第6层特征
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = hubert(inputs.input_values.to(hubert.device))
            hidden_states = outputs.hidden_states[6]  # 第6层

        seq_feature = hidden_states.squeeze(0).cpu()  # shape: [T, 768]

        # ✅ 保存
        save_path = os.path.join(save_dir, wav.replace(".wav", ".pt"))
        torch.save({"feature": seq_feature, "label": label_map[emotion]}, save_path)

        # ✅ 可视化仍用 mean
        mean_feature = seq_feature.mean(dim=0).numpy()
        all_features.append(mean_feature)
        all_labels.append(label_map[emotion])

    print(f"Finished {emotion}, processed {len(file_list)} audios.\n")

print("✅ All 6th-layer features saved to ./features_v1_6th_layer_1/\n")

# ✅ 可视化
print("Feature visualization (t-SNE)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced = tsne.fit_transform(np.array(all_features))

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange']
for i, label_name in label_names.items():
    idx = [j for j, l in enumerate(all_labels) if l == i]
    plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label_name, alpha=0.7, s=30, color=colors[i])

plt.title("t-SNE of HuBERT 6th-Layer Features")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_visualization_v1_6th_layer_1.png")
plt.show()

print("📊 t-SNE saved as tsne_visualization_v1_6th_layer_1.png")

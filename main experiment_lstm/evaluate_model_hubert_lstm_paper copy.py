import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from utils.dataset_lstm import FeatureEmotionDataset, collate_fn
from train_classifier_hubert_lstm_paper import EmotionClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence

# ✅ 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载测试集
test_set = FeatureEmotionDataset("./splits_v1_6th_layer_1/test")
test_loader = DataLoader(test_set, batch_size=16, collate_fn=collate_fn)

# ✅ 初始化并加载 HuBERT-LSTM 模型
model = EmotionClassifier().to(device)
model.load_state_dict(torch.load("./models_hubert_lstm_paper/best_model.pt", map_location=device))
model.eval()

# ✅ 评估过程
all_preds, all_labels = [], []

with torch.no_grad():
    for x, lengths, y in test_loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        outputs = model(x, lengths)
        preds = outputs.argmax(1)
        all_preds += preds.cpu().tolist()
        all_labels += y.cpu().tolist()

# ✅ 分类报告
target_names = ["happy", "sad", "surprised", "calm"]
report = classification_report(all_labels, all_preds, target_names=target_names, labels=[0, 1, 2, 3])
print("\n✅ Classification Report (Test Set - HuBERT-LSTM Paper):")
print(report)

# ✅ 保存报告
with open("test_classification_report_hubert_lstm_paper.txt", "w") as f:
    f.write("Test Set - Classification Report (HuBERT-LSTM Paper)\n")
    f.write(report)

# ✅ 混淆矩阵
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Oranges")
plt.title("Test Set - Confusion Matrix (HuBERT-LSTM Paper)")
plt.grid(False)
plt.savefig("test_confusion_matrix_hubert_lstm_paper.png")
plt.show()

# ✅ 修复后的 evaluate_model_v2.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from utils.dataset_non_lstm import FeatureEmotionDataset
from train_classifier_v2_6th_layer import EmotionClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ✅ 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载测试集数据
test_set = FeatureEmotionDataset("./splits_v2_6th_layer/test")
test_loader = DataLoader(test_set, batch_size=16)

# ✅ 初始化并加载训练好的模型（best）
model = EmotionClassifier()
model.load_state_dict(torch.load("./models_v2_6th_layer/best_model_v2_6th_layer.pt", map_location=device))
model.to(device)
model.eval()

# ✅ 评估过程
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        all_preds += preds.cpu().tolist()
        all_labels += y.cpu().tolist()

# ✅ 生成报告和混淆矩阵
target_names = ["happy", "sad", "angry", "neutral"]
report = classification_report(all_labels, all_preds, target_names=target_names, labels=[0, 1, 2, 3])
print("\n✅ Classification Report (Test Set):")
print(report)

# ✅ 保存报告
with open("test_classification_report_v2_6th_layer.txt", "w") as f:
    f.write("Test Set - Classification Report\n")
    f.write(report)

# ✅ 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Oranges")
plt.title("Test Set - Confusion Matrix")
plt.grid(False)
plt.savefig("test_confusion_matrix_v2_6th_layer.png")
plt.show()

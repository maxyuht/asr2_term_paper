# scripts/evaluate_model_lstm.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from utils.dataset import FeatureEmotionDataset
from utils.collate import collate_fn_lstm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ✅ 定义与训练中一致的模型结构
class EmotionLSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=1, num_classes=4):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                  batch_first=True, bidirectional=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: [B, T, 768]
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[0], hn[1]), dim=1)  # [B, H*2]
        return self.classifier(hn)

# ✅ 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载测试数据
test_set = FeatureEmotionDataset("./splits_lstm_v1/test")
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, collate_fn=collate_fn_lstm)

# ✅ 加载模型
model = EmotionLSTMClassifier()
model.load_state_dict(torch.load("./models_lstm_v1/best_model_lstm_v1.pt", map_location=device))
model.to(device)
model.eval()

# ✅ 模型推理
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y, lengths in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = outputs.argmax(1)
        all_preds += preds.cpu().tolist()
        all_labels += y.cpu().tolist()

# ✅ 输出分类报告
target_names = ["happy", "sad", "surprised", "calm"]
report = classification_report(all_labels, all_preds, target_names=target_names, labels=[0, 1, 2, 3])
print("\n✅ Classification Report (Test Set - LSTM):")
print(report)

# ✅ 保存报告
with open("test_classification_report_lstm_v1.txt", "w") as f:
    f.write("Test Set - Classification Report (LSTM)\n")
    f.write(report)

# ✅ 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Oranges")
plt.title("Test Set - Confusion Matrix (LSTM)")
plt.grid(False)
plt.savefig("test_confusion_matrix_lstm_v1.png")
plt.show()

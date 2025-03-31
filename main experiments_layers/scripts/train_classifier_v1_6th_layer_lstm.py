import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_lstm import FeatureEmotionDataset, collate_fn
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ✅ 强化版 LSTM 分类器
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2, num_classes=4, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=0.3)

        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor + input_dim, 128),  # 拼接 lstm + mean pooled
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lengths):  # x: [B, T, D], lengths: [B]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hn, _) = self.lstm(packed)
        if self.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)  # [B, 2*hidden]
        else:
            hn = hn[-1]  # [B, hidden]

        # 加入 mean pooling 作为补充
        mean_pooled = torch.sum(x, dim=1) / lengths.unsqueeze(1)  # [B, D]
        out = torch.cat([hn, mean_pooled], dim=1)
        return self.fc(out)

if __name__ == "__main__":
    # ✅ 加载数据
    train_set = FeatureEmotionDataset("./splits_v1_6th_layer_1/train")
    val_set = FeatureEmotionDataset("./splits_v1_6th_layer_1/val")

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=16, collate_fn=collate_fn)

    print(f"Loaded {len(train_set)} training samples, {len(val_set)} validation samples.")

    # ✅ 类别权重
    label_counter = Counter()
    for _, label in train_set:
        label_counter[int(label)] += 1
    total = sum(label_counter.values())
    weights = [total / label_counter[i] for i in range(4)]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # ✅ 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # ✅ 更稳健的学习率

    train_accs, train_losses, val_accs = [], [], []
    best_val_acc = 0.0
    patience = 7
    no_improve_epochs = 0

    os.makedirs("./models_v1_6th_lstm", exist_ok=True)
    log_file = open("train_log_v1_6th_lstm.txt", "w")

    # ✅ 主训练循环
    for epoch in range(30):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x, lengths)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # ✅ Clip grad
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

        acc = correct / total
        train_losses.append(total_loss)
        train_accs.append(acc)

        # ✅ 验证
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                outputs = model(x, lengths)
                preds = outputs.argmax(1)
                all_preds += preds.cpu().tolist()
                all_labels += y.cpu().tolist()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={acc:.4f}, Val Acc={val_acc:.4f}")
        log_file.write(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={acc:.4f}, Val Acc={val_acc:.4f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./models_v1_6th_lstm/best_model.pt")
            print("✅ New best model saved!")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("⛔ Early stopping triggered. Training stopped.")
            break

    log_file.close()
    torch.save(model.state_dict(), "./models_v1_6th_lstm/final_model.pt")

    # ✅ 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy", color='blue')
    plt.plot(val_accs, label="Val Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curve_v1_6th_lstm.png")
    plt.show()

    # ✅ 分类报告 & 混淆矩阵
    target_names = ["happy", "sad", "surprised", "calm"]
    report = classification_report(all_labels, all_preds, target_names=target_names, labels=[0, 1, 2, 3])
    print("\n✅ Classification Report (Validation Set):")
    print(report)

    with open("val_classification_report_v1_6th_lstm.txt", "w") as f:
        f.write("Validation Set - Classification Report\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title("Validation Set - Confusion Matrix (LSTM Strong)")
    plt.grid(False)
    plt.savefig("val_confusion_matrix_v1_6th_lstm.png")
    plt.show()

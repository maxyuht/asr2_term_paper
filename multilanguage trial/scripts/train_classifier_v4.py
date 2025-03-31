import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_non_lstm import FeatureEmotionDataset
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ✅ 模型定义
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    # ✅ 数据加载
    train_set = FeatureEmotionDataset("./splits_v4/train")
    val_set = FeatureEmotionDataset("./splits_v4/val")

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    print(f"Loaded {len(train_set)} training samples, {len(val_set)} validation samples.")

    # ✅ 类别权重
    label_counter = Counter()
    for _, label in train_set:
        label_counter[int(label)] += 1
    total = sum(label_counter.values())
    weights = [total / label_counter[i] for i in range(4)]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # ✅ 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_accs, train_losses, val_accs = [], [], []
    best_val_acc = 0.0
    patience = 7
    no_improve_epochs = 0

    os.makedirs("./models_v4", exist_ok=True)
    log_file = open("train_log_v4.txt", "w")

    # ✅ 训练主循环
    for epoch in range(30):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
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
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
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
            torch.save(model.state_dict(), "./models_v4/best_model_v4.pt")
            print("✅ New best model saved!")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("⛔ Early stopping triggered. Training stopped.")
            break

    log_file.close()
    torch.save(model.state_dict(), "./models_v4/classifier_v4.pt")

    # ✅ 可视化训练曲线
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
    plt.savefig("training_curve_v4.png")
    plt.show()

    # ✅ 验证集报告和混淆矩阵
    target_names = ["happy", "sad", "angry", "neutral"]
    report = classification_report(all_labels, all_preds, target_names=target_names, labels=[0, 1, 2, 3])
    print("\n✅ Classification Report (Validation Set):")
    print(report)

    with open("val_classification_report_v4.txt", "w") as f:
        f.write("Validation Set - Classification Report\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title("Validation Set - Confusion Matrix")
    plt.grid(False)
    plt.savefig("val_confusion_matrix_v4.png")
    plt.show()

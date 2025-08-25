import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from BirdsongDataset_mel import dataset, train_loader, val_loader
from model import EfficientNetV2
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------
# 配置 GPU
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EfficientNetV2(num_classes=len(dataset.classes)).to(device)
# ---------------------------
# 训练设置
# ---------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-6)

num_epochs = 150
best_score = -float("inf")
alpha = 5.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)  # 乘以batch size以保持一致性
        predicted = torch.argmax(outputs, dim=1)
        running_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    scheduler.step()
    avg_train_loss = running_loss / total_samples
    train_acc = running_correct / total_samples * 100
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_train_loss:.4f} - Training Accuracy: {train_acc:.2f}%")

    # ---------------------------
    # 每epoch都进行验证
    # ---------------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / total
    val_accuracy = 100 * correct / total

    # 计算评估指标
    val_precision = precision_score(all_labels, all_preds, average='weighted')
    val_recall = recall_score(all_labels, all_preds, average='weighted')
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    # ---------------------------
    # 保存最佳模型
    # ---------------------------
    composite_score = val_accuracy - alpha * avg_val_loss
    if composite_score > best_score:
        best_score = composite_score
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'accuracy': val_accuracy
        }, 'birdsong_model_V2(mel+stft+mfcc）基线-余弦退火-Urbansound8k.pth')
        print(f"New best model saved (Score: {composite_score:.2f})")
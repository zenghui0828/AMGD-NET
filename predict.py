import torch
import torch.nn as nn
from tqdm import tqdm
from BirdsongDataset_mel import dataset, val_loader
from model import EfficientNetV2
from sklearn.metrics import confusion_matrix, f1_score

import seaborn as sns
from matplotlib import rcParams
# ---------------------------
# 配置 GPU
# ---------------------------
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 使用文泉驿正黑
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# 加载模型
# ---------------------------
num_classes = len(dataset.classes)
model = EfficientNetV2(num_classes=num_classes).to(device)
checkpoint = torch.load('birdsong_model_V2(mel+stft+mfcc）基线-余弦退火-Urbansound8k.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # 只加载模型权重部分
model.eval()


# ---------------------------
# 评估函数（计算准确率、Loss、F1、混淆矩阵）
# ---------------------------
def evaluate_model(model, val_loader, criterion, class_names):
    predictions = []
    true_labels = [] 
    total = 0
    correct = 0
    val_loss = 0.0
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 计算预测结果
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # 计算准确率和Loss
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            # 统计每个类别的正确预测数
            for true, pred in zip(labels, predicted):
                class_total[true.item()] += 1
                if true == pred:
                    class_correct[true.item()] += 1

    # 计算指标
    accuracy = correct / total
    avg_loss = val_loss / total
    f1 = f1_score(true_labels, predictions, average='macro')
    cm = confusion_matrix(true_labels, predictions)

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'f1': f1,
        'confusion_matrix': cm,
        'class_correct': class_correct,
        'class_total': class_total
    }

# ---------------------------
# 假设你有一个从常见名到学名的映射
# 这里只是一个示例，你需要替换成实际的学名数据
# ---------------------------

# 示例：创建一个从常见名到学名的映射字典
# 你需要根据你的实际数据替换这些值
#scientific_names = [
#    "EO",    # 白腹蓝鹟
#    "Ec",      # 北红尾鸲
#    "AT",         # 红喉歌鸲
#    "AC",         # 红胁蓝尾鸲
#    "DH",    # 黄眉柳莺
#    "CB",   # 黄腰柳莺
#    "UE",           # 栗耳鹀
#    "GC",        # 乌鹟
#    "CE",          # 小鹀
#    "GG",       # 银喉长尾山雀
#    "HC",               # 远东山雀
#    "Aa",
#    "FP",
#    "DF",
#    "FS",
#    "FM",
#    "MG",
#    "DL",
#    "HD",
#    "AF",
#    "Ue",
#    "HE",
#    "CC",
#    "DC",
#    "OS",
#    "FT",
#    "LC",
#    "SC",
#    "SL",
#    "CS",
#    "PS",
#    "AA",
#    "OL",
#    "TB",
#    "PC",
#    "Ps",
#    "PF",
#    "EC"
#]
scientific_names = [
    "AC",
    "CH",
    "CP",
    "DB",
    "D",
    "EI",
    "GS",
    "J",
    "S",
    "SM"
]
# ---------------------------
# 修改可视化混淆矩阵函数
# ---------------------------
def plot_confusion_matrix(cm, class_names):
    # 设置更大的画布和字体
    plt.figure(figsize=(16, 14))  # 增大画布尺寸（宽，高）
    plt.rcParams['font.size'] = 12  # 全局字体大小



    # 绘制热力图（颜色基于归一化值，但显示原始数值）
    ax = sns.heatmap(
        cm,  # 显示原始数值
        annot=True,
        fmt='d',  # 显示整数
        cmap='Blues',
        xticklabels=scientific_names,  # 使用学名作为x轴标签
        yticklabels=scientific_names,  # 使用学名作为y轴标签
        vmin=0,
        vmax=200,  # 颜色范围 0-100%
      # 隐藏 0 值
        cbar_kws={ 'shrink': 0.8},  # 调整颜色条大小
        annot_kws={'fontsize': 10},  # 调整注释字体大小
        square=True,  # 使单元格为正方形
        linewidths=0.5,  # 单元格边框线宽
        linecolor='lightgray'  # 边框颜色
    )

    # 调整标签和标题
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16, pad=20)

    # 调整刻度标签（旋转45度，避免重叠）
    plt.xticks(rotation=0, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # 自动调整布局，防止标签被截断
    plt.tight_layout()

    plt.show()

# ---------------------------
# 运行评估
# ---------------------------
criterion = nn.CrossEntropyLoss()
class_names = dataset.classes

results = evaluate_model(model, val_loader, criterion, class_names)

# 打印整体指标
print("验证集评估结果:")
print(f"Loss: {results['loss']:.4f}")
print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
print(f"F1 Score (宏平均): {results['f1']:.4f}")

# 打印每个类别的准确率
print("各类别准确率:")
for i in range(num_classes):
    total = results['class_total'][i]
    correct = results['class_correct'][i]
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"{class_names[i]:<15}: 样本数={total:3d}, 准确率={acc:.2f}%")

# 绘制混淆矩阵
plot_confusion_matrix(results['confusion_matrix'], class_names)
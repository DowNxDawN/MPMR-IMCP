import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import time
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from timm.layers import SwiGLUPacked
from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.metrics import print_metrics
from sklearn.manifold import TSNE
from torch.amp import autocast, GradScaler  # 不再使用torch.cuda.amp
from scipy.interpolate import interp1d


# 清理内存函数
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


# 替换复杂的字体处理为简化版本


# 添加这个简化的字体设置函数
def set_plotting_style():
    # 使用简单的配置，避免字体问题
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "FreeSans", "Arial", "Helvetica", "sans-serif"]

    # 确保PDF和PS输出使用嵌入式字体
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    # 使用清爽的seaborn风格
    sns.set_style("whitegrid")

    print("Plot style set to use system sans-serif fonts")


# 替换原来的字体设置代码段
set_plotting_style()

# 检查可用GPU数量
print(f"可用GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 设置参数
num_classes = 2  # 类别数量
class_names = ["CD10", "N-CD10"]  
base_save_path = "/path/to/your/result"  # 结果保存路径
local_dir = "/path/to/your/assets/ckpts/uni2/"  # 修改为UNI2的目录
model_path = os.path.join(local_dir, "UNI2-h.pth")  # 修改为UNI2的权重文件名
dataroot = "/path/to/your/data"  # 数据集路径
patch_size = 224  # 补丁大小
stride = 224  # 步长
batch_size = 32  # 增大批次大小
num_epochs = 100  # 训练轮数
early_stopping_patience = 5  # 早停机制的耐心
min_delta = 0.01  # 早停机制的最小变化值
num_workers = 8  # 增加工作线程数量

# 允许处理大图像
Image.MAX_IMAGE_PIXELS = None

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存路径
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(base_save_path, current_time)
os.makedirs(save_path, exist_ok=True)

# 下载 UNI 权重并创建模型
import timm
from huggingface_hub import login, hf_hub_download
from peft import LoraConfig, get_peft_model

os.makedirs(local_dir, exist_ok=True)  # 如果目录不存在，则创建


# 修改initialize_model函数
def initialize_model():
    print("加载UNI2模型.../Loading UNI2 model...")
    # 使用UNI2的模型配置
    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
        "in_chans": 3,
    }

    base_model = timm.create_model(pretrained=False, **timm_kwargs)

    # 加载预训练权重
    base_model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)

    # 更保守的LoRA微调设置，减少数值不稳定性
    lora_config = LoraConfig(
        r=16,  # 从32降低到16
        lora_alpha=16,  # 从32降低到16
        target_modules=["qkv", "fc1", "fc2"],
        lora_dropout=0.05,  # 降低dropout率
        bias="none",
    )
    model = nn.Sequential(
        get_peft_model(base_model, lora_config),
        nn.Dropout(p=0.2),  # 降低dropout以增加稳定性
        nn.Linear(base_model.num_features, num_classes),
    )

    # 在模型初始化后添加：
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个 GPU 进行训练")
        model = nn.DataParallel(model)

    # 使用更高效的内存格式
    model = model.to(device, memory_format=torch.channels_last)

    return model


# 添加NaN检测和处理的函数
def forward_with_nan_check(model, inputs):
    """在前向传播过程中检测和处理NaN值"""
    outputs = model(inputs)
    if torch.isnan(outputs).any():
        print("检测到NaN输出，尝试恢复...")
        # 保存问题检查点以便后期分析
        torch.save(
            {
                "inputs": inputs,
            },
            os.path.join(save_path, "nan_debug.pt"),
        )
        # 替换NaN值
        outputs = torch.nan_to_num(outputs, nan=0.0)
    return outputs


# 设置图像变换
print("设置图像变换.../Setting up image transformations...")
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# 创建数据集
print("创建数据集.../Creating dataset...")
train_dir = os.path.join(dataroot, "train")
test_dir = os.path.join(dataroot, "test")
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)

# 优化DataLoader配置
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=3,  # 预加载更多批次
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size * 2,  # 测试时使用更大的batch size
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=3,
)

# 创建数据集后
print("训练集类别映射: ", train_dataset.class_to_idx)
print("测试集类别映射: ", test_dataset.class_to_idx)

# 打印类别标签和样本数量
print(f"训练集类别: {train_dataset.classes}")
print(f"训练集每个类别的样本数量: {np.bincount(train_dataset.targets)}")
print(f"测试集类别: {test_dataset.classes}")
print(f"测试集每个类别的样本数量: {np.bincount(test_dataset.targets)}")

# 定义损失函数和优化器
# 计算类别权重
class_weights = torch.tensor(
    [
        len(train_dataset) / np.bincount(train_dataset.targets)[0],
        len(train_dataset) / np.bincount(train_dataset.targets)[1],
    ],
    device=device,
    dtype=torch.float32,
)
criterion = nn.CrossEntropyLoss(weight=class_weights)
model = initialize_model()

# 修改优化器，降低学习率
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # 从0.001降低到0.0001

# 修改GradScaler配置
scaler = GradScaler(init_scale=2**10, growth_factor=1.5, backoff_factor=0.5, growth_interval=100)

# 初始化模型
model = initialize_model()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 添加学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

best_auc = 0
epochs_no_improve = 0

# 记录训练过程
train_losses = []
test_aucs = []
test_accuracies = []
test_sensitivities = []
test_specificities = []

# 在训练循环中添加梯度累积
accumulation_steps = 1  # 减少累积步数，适配更大的batch_size

for epoch in range(num_epochs):
    # 每个epoch开始时清理CUDA缓存
    torch.cuda.empty_cache()

    start_time = time.time()
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(
        tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", ncols=100)
    ):
        # 使用channels_last内存格式
        inputs = inputs.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)

        # 使用autocast，显式指定float16
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = forward_with_nan_check(model, inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        # 使用scaler处理反向传播
        scaler.scale(loss).backward()

        # 累积梯度指定步数后才更新参数
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            # 添加梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 从1.0改为0.5

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

        running_loss += loss.item() * accumulation_steps  # 还原原始损失值

        # 定期清理不需要的中间变量
        if i % 10 == 0:
            torch.cuda.empty_cache()

    train_losses.append(running_loss / len(train_dataloader))

    model.eval()
    test_running_loss = 0.0
    all_labels = []
    all_preds = []
    y_probs = []

    with torch.no_grad():
        for test_inputs, test_labels in tqdm(test_dataloader, desc=f"Testing Epoch {epoch+1}/{num_epochs}", ncols=100):
            # 使用channels_last内存格式
            test_inputs = test_inputs.to(device, memory_format=torch.channels_last)
            test_labels = test_labels.to(device)

            # 使用autocast进行评估，显式指定float16
            with autocast(device_type="cuda", dtype=torch.float16):
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)

            test_running_loss += test_loss.item()

            y_prob = torch.softmax(test_outputs, dim=1).cpu().numpy()
            y_pred = np.argmax(y_prob, axis=1)
            all_labels.extend(test_labels.cpu().numpy())
            all_preds.extend(y_pred)
            y_probs.extend(y_prob)

    test_loss = test_running_loss / len(test_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    test_accuracies.append(accuracy)

    # 在计算ROC曲线之前添加检查
    y_probs = np.array(y_probs)
    if np.any(np.isnan(y_probs)):
        print("警告: 模型输出包含NaN值")
        y_probs = np.nan_to_num(y_probs, nan=0.0)  # 将NaN替换为0.0

    # 继续计算ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, y_probs[:, 1])
    auc_value = auc(fpr, tpr)
    test_aucs.append(auc_value)

    # 更新学习率
    scheduler.step(auc_value)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # 召回率
    specificity = tn / (tn + fp)  # 特异度
    test_sensitivities.append(sensitivity)
    test_specificities.append(specificity)

    end_time = time.time()
    epoch_time = end_time - start_time

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Test Loss: {test_loss:.4f}, AUC: {test_aucs[-1]:.4f}, Accuracy: {test_accuracies[-1]:.4f}, Sensitivity: {test_sensitivities[-1]:.4f}, Specificity: {test_specificities[-1]:.4f}, Time: {epoch_time:.2f}s"
    )

    if auc_value > best_auc + min_delta:
        best_auc = auc_value
        epochs_no_improve = 0
        # 保存当前最佳模型状态
        best_model_state_dict = model.state_dict()
        # 保存最佳模型
        torch.save(model.state_dict(), os.path.join(save_path, "best_vit_classifier_model.pth"))
        print(f"Saved current best model, AUC: {best_auc:.4f}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(f"Early stopping triggered, training stopped at epoch {epoch+1}")
        break

# 保存最后一个epoch的评估结果，用于绘图和报告
final_labels = all_labels
final_preds = all_preds
final_probs = y_probs

# 保存最佳AUC值对应的模型状态
torch.save(best_model_state_dict, os.path.join(save_path, "best_vit_classifier_model.pth"))

# 在绘图之前添加seaborn样式设置
sns.set(style="whitegrid")

# 将五个指标合并到一张图中，而不是分散在子图中

# 绘制损失值、AUC值、准确率、灵敏度和特异性的趋势曲线图 - 合并成单一图表
plt.figure(figsize=(14, 10))

# 获取训练的轮数
epochs = np.arange(len(train_losses))
max_epochs = len(train_losses) + 5  # 加5为了图表右侧留白

# 定义颜色方案和线型
colors = ["#FF5252", "#4CAF50", "#2196F3", "#9C27B0", "#FF9800"]
line_styles = ["-", "-", "-", "-", "-"]
line_widths = [2.5, 2, 2, 2, 2]

# 绘制训练损失 - 使用辅助Y轴
ax1 = plt.gca()
ax2 = ax1.twinx()

# 在主Y轴上绘制AUC, 准确率, 灵敏度和特异度
metrics = [
    (test_aucs, "Test AUC", colors[1], line_styles[1], line_widths[1]),
    (test_accuracies, "Test Accuracy", colors[2], line_styles[2], line_widths[2]),
    (test_sensitivities, "Test Sensitivity", colors[3], line_styles[3], line_widths[3]),
    (test_specificities, "Test Specificity", colors[4], line_styles[4], line_widths[4]),
]

for i, (metric_data, label, color, ls, lw) in enumerate(metrics):
    if len(epochs) > 3:  # 使用样条插值平滑曲线
        smooth_func = interp1d(epochs, metric_data, kind="cubic")
        smooth_epochs = np.linspace(0, len(metric_data) - 1, 300)
        ax1.plot(smooth_epochs, smooth_func(smooth_epochs), label=label, color=color, linestyle=ls, linewidth=lw)
    else:
        ax1.plot(epochs, metric_data, label=label, color=color, linestyle=ls, linewidth=lw)

# 在辅助Y轴上绘制损失
if len(epochs) > 3:
    losses_smooth = interp1d(epochs, train_losses, kind="cubic")
    smooth_epochs = np.linspace(0, len(train_losses) - 1, 300)
    loss_line = ax2.plot(
        smooth_epochs,
        losses_smooth(smooth_epochs),
        label="Training Loss",
        color=colors[0],
        linestyle=line_styles[0],
        linewidth=line_widths[0],
    )
else:
    loss_line = ax2.plot(
        epochs, train_losses, label="Training Loss", color=colors[0], linestyle=line_styles[0], linewidth=line_widths[0]
    )

# 设置坐标轴范围
ax1.set_xlim(0, max_epochs)
ax1.set_ylim(0.5, 1.05)
ax2.set_ylim(0, max(train_losses) * 1.2)

# 添加标签和标题
ax1.set_xlabel("Epochs", fontsize=14)
ax1.set_ylabel("Metrics (AUC, Accuracy, Sensitivity, Specificity)", fontsize=14)
ax2.set_ylabel("Training Loss", fontsize=14, color=colors[0])
ax2.tick_params(axis="y", labelcolor=colors[0])
plt.title("Training and Evaluation Metrics", fontsize=16)

# 突出显示每个指标的最大值
for i, (metric_data, label, color, _, _) in enumerate(metrics):
    max_val = max(metric_data)
    max_idx = metric_data.index(max_val)
    ax1.scatter(max_idx, max_val, color=color, s=100, zorder=5, edgecolor="white")
    ax1.annotate(
        f"{max_val:.3f}",
        (max_idx, max_val),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color=color,
    )

# 处理图例
# 获取ax1的图例句柄和标签
handles1, labels1 = ax1.get_legend_handles_labels()
# 获取ax2的图例句柄和标签
handles2, labels2 = ax2.get_legend_handles_labels()
# 合并图例
all_handles = handles1 + handles2
all_labels = labels1 + labels2

# 配置单一图例
plt.legend(all_handles, all_labels, loc="upper right", fontsize=12, frameon=True, fancybox=True, framealpha=0.7)

# 添加网格线
ax1.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "metrics_trends.png"), dpi=300, bbox_inches="tight")
plt.close()

# 绘制单次训练的ROC曲线
plt.figure(figsize=(12, 10))
# 计算ROC曲线
fpr, tpr, _ = roc_curve(final_labels, final_probs[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2.5, alpha=0.8, label=f"ROC (AUC = {roc_auc:.3f})")

# 使用bootstrap计算AUC的置信区间
bootstrapped_aucs = []
n_bootstraps = 1000
for _ in range(n_bootstraps):
    indices = resample(np.arange(len(final_labels)), replace=True)
    if len(np.unique(np.array(final_labels)[indices])) < 2:
        continue
    score = auc(*roc_curve(np.array(final_labels)[indices], np.array(final_probs)[indices, 1])[:2])
    bootstrapped_aucs.append(score)

# 计算置信区间
sorted_aucs = np.sort(bootstrapped_aucs)
ci_lower = sorted_aucs[int(0.025 * len(sorted_aucs))]
ci_upper = sorted_aucs[int(0.975 * len(sorted_aucs))]
print(f"95% AUC CI: [{ci_lower:.3f} - {ci_upper:.3f}]")

plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curve", fontsize=16)
plt.legend(
    loc="lower right",
    fontsize=12,
    frameon=True,
    fancybox=True,
    framealpha=0.7,
    title=f"95% AUC CI: [{ci_lower:.3f} - {ci_upper:.3f}]",
)
plt.grid(True)
plt.savefig(os.path.join(save_path, "roc_curve.png"), dpi=300)
plt.close()


# 绘制混淆矩阵
def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300)
    plt.close()


plot_confusion_matrix(final_labels, final_preds, class_names, save_path)


# 可视化特征并保存
def visualize_features(model, dataloader, device, save_path):
    model.eval()
    features = []
    labels = []
    nan_count = 0

    # 检查是否是DataParallel模型
    if isinstance(model, nn.DataParallel):
        # 如果是DataParallel，访问基础模型
        base_model = model.module
    else:
        base_model = model

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features for t-SNE visualization", ncols=100):
            inputs = inputs.to(device, memory_format=torch.channels_last)
            # 使用autocast
            with autocast(device_type="cuda", dtype=torch.float16):
                # 使用序列的第一部分 (基础模型提取特征)
                if isinstance(base_model, nn.Sequential):
                    outputs = base_model[0](inputs)
                else:
                    # 如果模型结构不是Sequential，尝试直接访问特征提取器
                    outputs = (
                        base_model.forward_features(inputs)
                        if hasattr(base_model, "forward_features")
                        else base_model(inputs)
                    )

            # 检查并处理NaN值
            if torch.isnan(outputs).any():
                nan_count += torch.isnan(outputs).sum().item()
                outputs = torch.nan_to_num(outputs, nan=0.0)

            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    if nan_count > 0:
        print(f"警告: 在特征提取过程中发现 {nan_count} 个NaN值并已替换")

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 检查并处理特征中的NaN值
    if np.isnan(features).any():
        print(f"警告: 连接后的特征包含 {np.isnan(features).sum()} 个NaN值，替换为0")
        features = np.nan_to_num(features, nan=0.0)

    # 使用PCA降维前检查是否有无穷大值
    if np.isinf(features).any():
        print(f"警告: 特征包含 {np.isinf(features).sum()} 个无穷大值，替换为0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # 使用PCA降维后再应用t-SNE以加快速度
    from sklearn.decomposition import PCA

    if features.shape[1] > 50:
        print(f"应用PCA降维从 {features.shape[1]} 到 50 维...")
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)

    print("执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    colors = ["#FF9999", "#66B2FF"]  # 使用更美观的配色
    markers = ["o", "s"]

    for i, class_name in enumerate(class_names):
        plt.scatter(
            features_2d[labels == i, 0],
            features_2d[labels == i, 1],
            label=class_name,
            alpha=0.7,
            color=colors[i],
            marker=markers[i],
            edgecolors="w",
            s=70,
        )

    plt.legend(prop={"size": 12}, frameon=True, fancybox=True, framealpha=0.7)
    plt.title("t-SNE Visualization of Features", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "tsne_features.png"), dpi=300, bbox_inches="tight")
    plt.close()


# 重新加载最佳模型用于特征可视化
print("加载最佳模型进行特征可视化...")
model = initialize_model()
model.load_state_dict(best_model_state_dict)
model.eval()

# 限制特征可视化的样本数以加快处理速度
test_subset_size = min(500, len(test_dataset))  # 最多使用500个样本
test_subset_indices = np.random.choice(len(test_dataset), test_subset_size, replace=False)
test_subset = torch.utils.data.Subset(test_dataset, test_subset_indices)
test_subset_loader = torch.utils.data.DataLoader(
    test_subset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True
)

# 生成特征可视化
visualize_features(model, test_subset_loader, device, save_path)
print("特征可视化完成！")

print("\n=================== Balanced Classification Evaluation ===================")

# 1. 为每个类别单独计算性能指标
all_labels_np = np.array(final_labels)
all_preds_np = np.array(final_preds)
all_probs_np = np.array(final_probs)

print("\nPerformance for each class:")

for i, class_name in enumerate(class_names):
    # 将当前类别视为正类(one-vs-rest)
    binary_labels = (all_labels_np == i).astype(int)
    binary_preds = (all_preds_np == i).astype(int)
    class_probs = all_probs_np[:, i]

    # 计算各项指标
    precision = precision_score(binary_labels, binary_preds)
    recall = recall_score(binary_labels, binary_preds)
    f1 = f1_score(binary_labels, binary_preds)
    class_auc = roc_auc_score(binary_labels, class_probs)

    # 打印类别样本数
    class_count = np.sum(binary_labels)
    total = len(binary_labels)

    print(f"  Class '{class_name}' (samples: {class_count}/{total}, {class_count/total*100:.1f}%):")
    print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}, AUC: {class_auc:.3f}")

# 2. 计算平衡精度和MCC
balanced_acc = balanced_accuracy_score(final_labels, final_preds)
mcc = matthews_corrcoef(final_labels, final_preds)

print(f"\nBalanced Accuracy (considering class imbalance): {balanced_acc:.3f}")
print(f"Matthews Correlation Coefficient (MCC, suitable for imbalanced data): {mcc:.3f}")

# 3. 绘制精确率-召回率曲线
plt.figure(figsize=(10, 8))

for i, class_name in enumerate(class_names):
    # 计算二元标签
    binary_labels = (all_labels_np == i).astype(int)
    class_probs = all_probs_np[:, i]

    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(binary_labels, class_probs)

    # 计算平均精确率
    avg_precision = np.sum(np.diff(recall) * np.array(precision)[:-1])

    # 绘制每个类别的PR曲线
    plt.plot(recall, precision, label=f"{class_name} (AP = {avg_precision:.3f})", lw=2, alpha=0.8)

plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision-Recall Curve", fontsize=16)
plt.legend(loc="best", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig(os.path.join(save_path, "precision_recall_curve.png"), dpi=300)
plt.close()

# 将评估结果保存到文件
with open(os.path.join(save_path, "evaluation_results.txt"), "w") as f:
    f.write(f"平衡精度: {balanced_acc:.3f}\n")
    f.write(f"马修斯相关系数(MCC): {mcc:.3f}\n")
    f.write(f"AUC 95%置信区间: [{ci_lower:.3f} - {ci_upper:.3f}]\n\n")

    for i, class_name in enumerate(class_names):
        class_count = np.sum(all_labels_np == i)
        total = len(all_labels_np)
        f.write(f"类别 '{class_name}' (样本数: {class_count}/{total}, {class_count/total*100:.1f}%):\n")

        binary_labels = (all_labels_np == i).astype(int)
        binary_preds = (all_preds_np == i).astype(int)
        class_probs = all_probs_np[:, i]

        precision = precision_score(binary_labels, binary_preds)
        recall = recall_score(binary_labels, binary_preds)
        f1 = f1_score(binary_labels, binary_preds)
        class_auc = roc_auc_score(binary_labels, class_probs)

        f.write(f"  精确率: {precision:.3f}\n")
        f.write(f"  召回率: {recall:.3f}\n")
        f.write(f"  F1分数: {f1:.3f}\n")
        f.write(f"  AUC: {class_auc:.3f}\n\n")

# 添加在评估部分之后，创建一个额外的平衡测试集评估

print("\n=================== Balanced Test Set Evaluation ===================")


# 创建一个平衡的测试子集
def create_balanced_test_subset(dataset, class_indices):
    labels = np.array(dataset.targets)

    # 找出每个类别的样本索引
    class_indices = {}
    for i in range(len(class_names)):
        class_indices[i] = np.where(labels == i)[0]

    # 确定每个类别要选择的样本数量（使用少数类的数量）
    min_class_size = min([len(indices) for indices in class_indices.values()])

    # 为每个类别随机选择相同数量的样本
    balanced_indices = []
    for i in range(len(class_names)):
        selected_indices = np.random.choice(class_indices[i], min_class_size, replace=False)
        balanced_indices.extend(selected_indices)

    # 创建一个平衡的子集
    balanced_subset = torch.utils.data.Subset(dataset, balanced_indices)

    return balanced_subset, min_class_size


# 创建平衡测试集
balanced_test_subset, samples_per_class = create_balanced_test_subset(test_dataset, {})
print(f"Created balanced test subset with {samples_per_class} samples per class")

# 为平衡测试集创建数据加载器
balanced_test_loader = torch.utils.data.DataLoader(
    balanced_test_subset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

# 在平衡测试集上评估模型
model.eval()
balanced_labels = []
balanced_preds = []
balanced_probs = []

with torch.no_grad():
    for test_inputs, test_labels in tqdm(balanced_test_loader, desc="Evaluating on balanced test set", ncols=100):
        test_inputs = test_inputs.to(device, memory_format=torch.channels_last)
        test_labels = test_labels.to(device)

        with autocast(device_type="cuda", dtype=torch.float16):
            test_outputs = model(test_inputs)

        y_prob = torch.softmax(test_outputs, dim=1).cpu().numpy()
        y_pred = np.argmax(y_prob, axis=1)
        balanced_labels.extend(test_labels.cpu().numpy())
        balanced_preds.extend(y_pred)
        balanced_probs.extend(y_prob)

# 转换为numpy数组
balanced_labels = np.array(balanced_labels)
balanced_preds = np.array(balanced_preds)
balanced_probs = np.array(balanced_probs)

# 计算平衡测试集上的指标
balanced_accuracy = accuracy_score(balanced_labels, balanced_preds)
balanced_fpr, balanced_tpr, _ = roc_curve(balanced_labels, balanced_probs[:, 1])
balanced_auc = auc(balanced_fpr, balanced_tpr)
balanced_cm = confusion_matrix(balanced_labels, balanced_preds)
tn, fp, fn, tp = balanced_cm.ravel()
balanced_sensitivity = tp / (tp + fn)
balanced_specificity = tn / (tn + fp)

print(f"\nBalanced test set results:")
print(f"  Accuracy: {balanced_accuracy:.3f}")
print(f"  AUC: {balanced_auc:.3f}")
print(f"  Sensitivity: {balanced_sensitivity:.3f}")
print(f"  Specificity: {balanced_specificity:.3f}")

# 绘制平衡测试集的ROC曲线
plt.figure(figsize=(12, 10))
# 绘制原始测试集ROC
plt.plot(fpr, tpr, lw=2.5, alpha=0.8, label=f"Original test set (AUC = {roc_auc:.3f})")
# 绘制平衡测试集ROC
plt.plot(
    balanced_fpr, balanced_tpr, lw=2.5, alpha=0.8, linestyle="--", label=f"Balanced test set (AUC = {balanced_auc:.3f})"
)

plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curves - Original vs. Balanced Test Set", fontsize=16)
plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, framealpha=0.7)
plt.grid(True)
plt.savefig(os.path.join(save_path, "roc_curve_comparison.png"), dpi=300)
plt.close()

# 将平衡测试集结果添加到评估结果文件
with open(os.path.join(save_path, "evaluation_results.txt"), "a") as f:
    f.write("\n平衡测试集结果:\n")
    f.write(f"  准确率: {balanced_accuracy:.3f}\n")
    f.write(f"  AUC: {balanced_auc:.3f}\n")
    f.write(f"  敏感度: {balanced_sensitivity:.3f}\n")
    f.write(f"  特异度: {balanced_specificity:.3f}\n")

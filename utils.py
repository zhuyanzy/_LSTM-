import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
from typing import List, Tuple


def set_seed(seed: int = 42) -> None:
    """设置所有相关库的随机种子，保证实验可复现。

    涵盖 Python 内置随机数、NumPy、PyTorch（CPU + GPU）以及 Python 哈希种子。

    Args:
        seed: 全局随机种子数值，默认为 42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_results(
    train_losses: List[float],
    val_losses: List[float],
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: str = 'lstm_prediction_results.png'
) -> None:
    """绘制训练结果可视化图。

    包含两个子图：
    1. 训练/验证损失曲线对比
    2. 测试集预测值与真实值对比，并显示 95% 置信区间

    Args:
        train_losses: 每轮训练损失列表，长度为 epochs。
        val_losses: 每轮验证损失列表，长度为 epochs。
        predictions: 测试集预测值数组。
        actuals: 测试集真实值数组。
        save_path: 图片保存路径，默认为 'lstm_prediction_results.png'。
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    axes[0].plot(train_losses, 'b-', label='训练损失', linewidth=2)
    axes[0].plot(val_losses, 'r-', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('训练/验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(np.arange(0, len(train_losses), 5))

    x_axis = np.arange(len(predictions))
    axes[1].plot(x_axis, actuals, 'b-', label='真实值', linewidth=1.5, alpha=0.7)
    axes[1].plot(x_axis, predictions, 'r-', label='预测值', linewidth=1.5, alpha=0.8)

    residuals = actuals.flatten() - predictions.flatten()
    std_error = np.std(residuals)
    upper_bound = predictions.flatten() + 1.96 * std_error
    lower_bound = predictions.flatten() - 1.96 * std_error

    axes[1].fill_between(
        x_axis, lower_bound, upper_bound,
        color='gray', alpha=0.2, label='95%置信区间'
    )

    axes[1].set_xlabel('时间步', fontsize=12)
    axes[1].set_ylabel('数值', fontsize=12)
    axes[1].set_title('测试集预测值 vs 真实值对比', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n结果图已保存为: {save_path}')
    plt.close()


def cleanup_files(file_paths: List[str]) -> None:
    """清理临时文件。

    用于删除训练过程中生成的临时文件，如归一化器和模型权重文件。

    Args:
        file_paths: 待删除的文件路径列表。
    """
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)

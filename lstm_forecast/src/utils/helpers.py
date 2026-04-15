"""工具函数模块。

包含绘图、模型保存加载、随机种子设置和评估指标计算等辅助功能。
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import Config


@dataclass
class Metrics:
    """评估指标数据类。
    
    Attributes:
        mae: 平均绝对误差 (Mean Absolute Error)。
        rmse: 均方根误差 (Root Mean Squared Error)。
        mape: 平均绝对百分比误差 (Mean Absolute Percentage Error)。
    """
    mae: float
    rmse: float
    mape: float
    
    def __str__(self) -> str:
        """格式化输出指标。"""
        return (
            f"MAE  (平均绝对误差):      {self.mae:.6f}\n"
            f"RMSE (均方根误差):        {self.rmse:.6f}\n"
            f"MAPE (平均绝对百分比误差): {self.mape:.2f}%"
        )


def set_seed(seed: Optional[int] = 42) -> None:
    """设置随机种子以确保实验可复现。
    
    设置 Python、NumPy 和 PyTorch 的随机种子。
    
    Args:
        seed: 随机种子值，默认为 42。
    
    Example:
        >>> set_seed(42)
        >>> # 后续代码将产生可复现的结果
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(
    model: nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None
) -> None:
    """保存模型状态。
    
    保存模型权重和可选的训练状态信息。
    
    Args:
        model: 要保存的 PyTorch 模型。
        path: 保存路径。
        optimizer: 优化器状态（可选）。
        epoch: 当前轮数（可选）。
        loss: 当前损失值（可选）。
    
    Example:
        >>> model = LSTMModel(input_size=1, hidden_size=128, num_layers=2)
        >>> save_model(model, Path("models/model.pth"), epoch=10, loss=0.01)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model)
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    
    # 确保目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model(
    model: nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Optional[int], Optional[float]]:
    """加载模型状态。
    
    从文件加载模型权重和可选的训练状态信息。
    
    Args:
        model: 要加载权重的 PyTorch 模型。
        path: 模型文件路径。
        optimizer: 要加载状态的优化器（可选）。
        device: 加载模型的设备（可选）。
    
    Returns:
        包含以下内容的元组：
        - model: 加载权重后的模型。
        - epoch: 保存时的轮数（如果存在）。
        - loss: 保存时的损失值（如果存在）。
    
    Example:
        >>> model = LSTMModel(input_size=1, hidden_size=128, num_layers=2)
        >>> model, epoch, loss = load_model(model, Path("models/model.pth"))
    """
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch')
    loss = checkpoint.get('loss')
    
    return model, epoch, loss


def calculate_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Metrics:
    """计算评估指标。
    
    计算 MAE、RMSE 和 MAPE 指标。
    
    Args:
        predictions: 预测值数组，形状为 (n_samples, 1)。
        actuals: 真实值数组，形状为 (n_samples, 1)。
    
    Returns:
        包含评估指标的 Metrics 对象。
    
    Example:
        >>> predictions = np.array([[1.0], [2.0], [3.0]])
        >>> actuals = np.array([[1.1], [2.1], [2.9]])
        >>> metrics = calculate_metrics(predictions, actuals)
        >>> print(metrics.mae)
    """
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # 避免除以零
    mask = actuals != 0
    mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    
    return Metrics(mae=mae, rmse=rmse, mape=mape)


def plot_results(
    train_losses: List[float],
    val_losses: List[float],
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: Path,
    show_plot: bool = False
) -> None:
    """绘制训练结果。
    
    绘制训练/验证损失曲线和预测值与真实值对比图。
    
    Args:
        train_losses: 每轮的训练损失列表。
        val_losses: 每轮的验证损失列表。
        predictions: 测试集预测值，形状为 (n_samples, 1)。
        actuals: 测试集真实值，形状为 (n_samples, 1)。
        save_path: 图片保存路径。
        show_plot: 是否显示图形，默认为 False。
    
    Example:
        >>> train_losses = [0.1, 0.05, 0.03]
        >>> val_losses = [0.12, 0.06, 0.04]
        >>> predictions = np.random.randn(100, 1)
        >>> actuals = np.random.randn(100, 1)
        >>> plot_results(train_losses, val_losses, predictions, actuals, Path("output.png"))
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 图1: 训练/验证损失曲线
    axes[0].plot(train_losses, 'b-', label='训练损失', linewidth=2)
    axes[0].plot(val_losses, 'r-', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('训练/验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(np.arange(0, len(train_losses), max(1, len(train_losses) // 10)))
    
    # 图2: 测试集预测对比
    x_axis = np.arange(len(predictions))
    axes[1].plot(x_axis, actuals, 'b-', label='真实值', linewidth=1.5, alpha=0.7)
    axes[1].plot(x_axis, predictions, 'r-', label='预测值', linewidth=1.5, alpha=0.8)
    
    # 计算置信区间 (95%置信区间: ±1.96*std)
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
    
    plt.tight_layout()
    
    # 保存图片
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n结果图已保存为: {save_path}')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def cleanup_temp_files(config: Config) -> None:
    """清理临时文件。
    
    删除模型和归一化器临时文件。
    
    Args:
        config: 项目配置对象。
    """
    paths_to_remove = [
        config.paths.scaler_path,
        config.paths.model_path
    ]
    
    for path in paths_to_remove:
        if path.exists():
            os.remove(path)
            print(f"已删除临时文件: {path}")

"""
LSTM 时间序列预测项目 - 辅助函数模块

该模块提供项目通用的辅助函数，包括：
- 随机种子设置
- 模型保存与加载
- 结果可视化
- 评估指标计算
"""

import os
import pickle
from typing import Tuple, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from lstm_project.config import Config


def set_seed(seed: int) -> None:
    """
    设置全局随机种子以确保实验可复现。

    Args:
        seed: 随机种子值。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[随机种子] 已设置随机种子: {seed}")


def save_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None
) -> None:
    """
    保存模型检查点。

    Args:
        model: 要保存的模型。
        path: 保存路径。
        optimizer: 优化器（可选）。
        epoch: 当前轮次（可选）。
        loss: 当前损失（可选）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if loss is not None:
        checkpoint["loss"] = loss
    
    torch.save(checkpoint, path)
    print(f"[模型保存] 模型已保存至: {path}")


def load_model(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Optional[int], Optional[float]]:
    """
    加载模型检查点。

    Args:
        model: 模型实例。
        path: 检查点路径。
        device: 计算设备。
        optimizer: 优化器实例（可选）。

    Returns:
        Tuple: 包含模型、优化器、轮次、损失的元组。
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", None)
    loss = checkpoint.get("loss", None)
    
    print(f"[模型加载] 模型已从 {path} 加载")
    return model, optimizer, epoch, loss


def save_scaler(scaler: object, path: str) -> None:
    """
    保存归一化器。

    Args:
        scaler: 归一化器实例。
        path: 保存路径。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[归一化器保存] 已保存至: {path}")


def load_scaler(path: str) -> object:
    """
    加载归一化器。

    Args:
        path: 归一化器路径。

    Returns:
        object: 归一化器实例。
    """
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[归一化器加载] 已从 {path} 加载")
    return scaler


def calculate_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray
) -> Tuple[float, float, float]:
    """
    计算回归评估指标。

    Args:
        actuals: 真实值数组。
        predictions: 预测值数组。

    Returns:
        Tuple[float, float, float]: MAE, RMSE, MAPE 指标。
    """
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    actuals_flat = actuals.flatten()
    predictions_flat = predictions.flatten()
    mask = actuals_flat != 0
    mape = np.mean(np.abs((actuals_flat[mask] - predictions_flat[mask]) / actuals_flat[mask])) * 100
    
    return mae, rmse, mape


def print_metrics(metrics: Tuple[float, float, float]) -> None:
    """
    打印评估指标。

    Args:
        metrics: 包含 MAE, RMSE, MAPE 的元组。
    """
    mae, rmse, mape = metrics
    print("\n" + "=" * 60)
    print("测试集评估指标:")
    print(f"MAE  (平均绝对误差):       {mae:.6f}")
    print(f"RMSE (均方根误差):         {rmse:.6f}")
    print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
    print("=" * 60)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    绘制训练和验证损失曲线。

    Args:
        train_losses: 训练损失列表。
        val_losses: 验证损失列表。
        save_path: 图片保存路径（可选）。
        show: 是否显示图片。
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="训练损失", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="验证损失", linewidth=2)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("训练/验证损失曲线", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    tick_interval = max(1, len(train_losses) // 10)
    ax.set_xticks(range(0, len(train_losses) + 1, tick_interval))
    
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[图片保存] 训练曲线已保存至: {save_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_predictions(
    actuals: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
    confidence_interval: bool = True
) -> None:
    """
    绘制预测结果对比图。

    Args:
        actuals: 真实值数组。
        predictions: 预测值数组。
        save_path: 图片保存路径（可选）。
        show: 是否显示图片。
        confidence_interval: 是否绘制置信区间。
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_axis = np.arange(len(predictions))
    ax.plot(x_axis, actuals.flatten(), "b-", label="真实值", linewidth=1.5, alpha=0.7)
    ax.plot(x_axis, predictions.flatten(), "r-", label="预测值", linewidth=1.5, alpha=0.8)
    
    if confidence_interval:
        residuals = actuals.flatten() - predictions.flatten()
        std_error = np.std(residuals)
        upper_bound = predictions.flatten() + 1.96 * std_error
        lower_bound = predictions.flatten() - 1.96 * std_error
        
        ax.fill_between(
            x_axis, lower_bound, upper_bound,
            color="gray", alpha=0.2, label="95%置信区间"
        )
    
    ax.set_xlabel("时间步", fontsize=12)
    ax.set_ylabel("数值", fontsize=12)
    ax.set_title("测试集预测值 vs 真实值对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[图片保存] 预测结果已保存至: {save_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_combined_results(
    train_losses: List[float],
    val_losses: List[float],
    actuals: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    绘制组合结果图（训练曲线 + 预测对比）。

    Args:
        train_losses: 训练损失列表。
        val_losses: 验证损失列表。
        actuals: 真实值数组。
        predictions: 预测值数组。
        save_path: 图片保存路径（可选）。
        show: 是否显示图片。
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-", label="训练损失", linewidth=2)
    axes[0].plot(epochs, val_losses, "r-", label="验证损失", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("MSE Loss", fontsize=12)
    axes[0].set_title("训练/验证损失曲线", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    tick_interval = max(1, len(train_losses) // 10)
    axes[0].set_xticks(range(0, len(train_losses) + 1, tick_interval))
    
    x_axis = np.arange(len(predictions))
    axes[1].plot(x_axis, actuals.flatten(), "b-", label="真实值", linewidth=1.5, alpha=0.7)
    axes[1].plot(x_axis, predictions.flatten(), "r-", label="预测值", linewidth=1.5, alpha=0.8)
    
    residuals = actuals.flatten() - predictions.flatten()
    std_error = np.std(residuals)
    upper_bound = predictions.flatten() + 1.96 * std_error
    lower_bound = predictions.flatten() - 1.96 * std_error
    
    axes[1].fill_between(
        x_axis, lower_bound, upper_bound,
        color="gray", alpha=0.2, label="95%置信区间"
    )
    
    axes[1].set_xlabel("时间步", fontsize=12)
    axes[1].set_ylabel("数值", fontsize=12)
    axes[1].set_title("测试集预测值 vs 真实值对比", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[图片保存] 结果图已保存至: {save_path}")
    
    if show:
        plt.show()
    plt.close()


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    计算模型参数数量。

    Args:
        model: PyTorch 模型。
        trainable_only: 是否只计算可训练参数。

    Returns:
        int: 参数数量。
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def cleanup_temp_files(config: Config) -> None:
    """
    清理临时文件。

    Args:
        config: 配置实例。
    """
    files_to_remove = [config.path.model_path, config.path.scaler_path]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[清理] 已删除临时文件: {file_path}")

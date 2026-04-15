"""数据加载与预处理模块。

该模块包含时间序列数据生成、数据集类定义和数据预处理功能。
"""

import pickle
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.config import Config


class TimeSeriesDataset(Dataset):
    """时间序列数据集类，支持滑窗采样。
    
    继承自 torch.utils.data.Dataset，用于将时间序列数据转换为
    适合 LSTM 模型输入的格式。
    
    Attributes:
        data: 归一化后的时间序列数据，形状为 (n_samples, n_features)。
        seq_len: 输入序列长度。
        pred_len: 预测步长。
    
    Example:
        >>> data = np.random.randn(1000, 1)
        >>> dataset = TimeSeriesDataset(data, seq_len=24, pred_len=1)
        >>> x, y = dataset[0]
        >>> print(x.shape)  # torch.Size([24, 1])
        >>> print(y.shape)  # torch.Size([1, 1])
    """
    
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int) -> None:
        """初始化时间序列数据集。
        
        Args:
            data: 时间序列数据，形状为 (n_samples, n_features)。
            seq_len: 输入序列长度（时间步长）。
            pred_len: 预测步长。
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
    
    def __len__(self) -> int:
        """返回数据集长度。
        
        Returns:
            可用于采样的序列数量。
        """
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本。
        
        Args:
            idx: 样本索引。
        
        Returns:
            包含输入序列和目标值的元组 (x, y)。
            - x: 输入序列，形状为 (seq_len, n_features)。
            - y: 目标值，形状为 (pred_len, n_features)。
        """
        # 输入序列：[idx, idx+seq_len)
        x = self.data[idx:idx + self.seq_len]
        # 目标值：[idx+seq_len, idx+seq_len+pred_len)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


def generate_time_series(
    n_samples: int = 3500,
    noise_level: float = 0.15,
    seed: Optional[int] = None
) -> np.ndarray:
    """生成模拟时间序列数据。
    
    生成包含指数趋势、周期性波动和随机噪声的合成时间序列数据。
    
    Args:
        n_samples: 样本数量。
        noise_level: 噪声水平。
        seed: 随机种子，用于可复现性。
    
    Returns:
        生成的合成时间序列数据，形状为 (n_samples, 1)。
    
    Example:
        >>> data = generate_time_series(n_samples=1000, noise_level=0.1, seed=42)
        >>> print(data.shape)  # (1000, 1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n_samples)
    
    # 1. 指数趋势项
    trend = 0.0005 * np.exp(0.002 * t)
    
    # 2. 周期性波动项（多个频率叠加）
    seasonality_1 = 0.3 * np.sin(2 * np.pi * t / 100)
    seasonality_2 = 0.15 * np.sin(2 * np.pi * t / 30)
    seasonality_3 = 0.1 * np.sin(2 * np.pi * t / 7)
    seasonality = seasonality_1 + seasonality_2 + seasonality_3
    
    # 3. 随机噪声
    noise = noise_level * np.random.randn(n_samples)
    
    # 4. 组合所有成分
    data = trend + seasonality + noise
    
    return data.reshape(-1, 1)


def preprocess_data(
    data: np.ndarray,
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """预处理时间序列数据。
    
    执行数据划分、归一化和 DataLoader 创建。注意时序数据
    不能随机划分，必须按时间顺序划分。
    
    Args:
        data: 原始时间序列数据，形状为 (n_samples, n_features)。
        config: 项目配置对象。
    
    Returns:
        包含以下内容的元组：
        - train_loader: 训练集 DataLoader。
        - val_loader: 验证集 DataLoader。
        - test_loader: 测试集 DataLoader。
        - scaler: 拟合在训练集上的 StandardScaler 对象。
    
    Example:
        >>> data = generate_time_series(n_samples=3500)
        >>> train_loader, val_loader, test_loader, scaler = preprocess_data(data, config)
        >>> print(len(train_loader))  # 训练集批次数
    """
    data_cfg = config.data
    train_cfg = config.training
    paths_cfg = config.paths
    
    # 划分数据集（时序数据不能随机划分）
    n_samples = len(data)
    train_end = int(n_samples * data_cfg.train_ratio)
    val_end = train_end + int(n_samples * data_cfg.val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Z-Score 归一化（仅使用训练集统计量）
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    # 保存归一化器参数用于后续逆变换
    with open(paths_cfg.scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 创建数据集
    train_dataset = TimeSeriesDataset(
        train_scaled, data_cfg.seq_len, data_cfg.pred_len
    )
    val_dataset = TimeSeriesDataset(
        val_scaled, data_cfg.seq_len, data_cfg.pred_len
    )
    test_dataset = TimeSeriesDataset(
        test_scaled, data_cfg.seq_len, data_cfg.pred_len
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory
    )
    
    return train_loader, val_loader, test_loader, scaler

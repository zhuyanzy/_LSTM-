"""
LSTM 时间序列预测项目 - 数据加载与预处理模块

该模块负责数据的生成、预处理和加载，包括：
- 时间序列数据生成
- 自定义 Dataset 类
- DataLoader 创建
- 数据归一化处理
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from lstm_project.config import Config


def generate_time_series(
    n_samples: int = 3500,
    noise_level: float = 0.15,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成模拟时间序列数据。

    数据由以下成分组合而成：
    1. 指数趋势项
    2. 多频率周期性波动
    3. 随机噪声

    Args:
        n_samples: 样本数量。
        noise_level: 噪声水平（标准差倍数）。
        seed: 随机种子（可选）。

    Returns:
        np.ndarray: 形状为 (n_samples, 1) 的时间序列数据。
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n_samples)
    
    trend = 0.0005 * np.exp(0.002 * t)
    
    seasonality_1 = 0.3 * np.sin(2 * np.pi * t / 100)
    seasonality_2 = 0.15 * np.sin(2 * np.pi * t / 30)
    seasonality_3 = 0.1 * np.sin(2 * np.pi * t / 7)
    seasonality = seasonality_1 + seasonality_2 + seasonality_3
    
    noise = noise_level * np.random.randn(n_samples)
    
    data = trend + seasonality + noise
    
    return data.reshape(-1, 1)


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集类。

    支持滑窗采样方式构建监督学习样本，适用于时间序列预测任务。

    Attributes:
        data: 归一化后的时间序列数据。
        seq_len: 输入序列长度。
        pred_len: 预测序列长度。

    Example:
        >>> data = np.random.randn(1000, 1)
        >>> dataset = TimeSeriesDataset(data, seq_len=24, pred_len=1)
        >>> x, y = dataset[0]
        >>> x.shape  # torch.Size([24, 1])
        >>> y.shape  # torch.Size([1, 1])
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int
    ) -> None:
        """
        初始化时间序列数据集。

        Args:
            data: 时间序列数据，形状为 (n_samples, n_features)。
            seq_len: 输入序列长度（历史窗口大小）。
            pred_len: 预测序列长度（预测步数）。

        Raises:
            ValueError: 如果数据长度不足以构建样本。
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        min_length = seq_len + pred_len
        if len(data) < min_length:
            raise ValueError(
                f"数据长度 ({len(data)}) 必须大于等于 seq_len + pred_len ({min_length})"
            )

    def __len__(self) -> int:
        """
        返回数据集样本数量。

        Returns:
            int: 可构建的样本数量。
        """
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的样本。

        Args:
            idx: 样本索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - x: 输入序列，形状为 (seq_len, n_features)
                - y: 目标序列，形状为 (pred_len, n_features)
        """
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y)
        )


class DataProcessor:
    """
    数据处理器类。

    负责数据的划分、归一化和 DataLoader 创建。

    Attributes:
        config: 配置实例。
        scaler: 归一化器实例。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。
        test_loader: 测试数据加载器。
    """

    def __init__(self, config: Config) -> None:
        """
        初始化数据处理器。

        Args:
            config: 项目配置实例。
        """
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def process(
        self,
        data: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
        """
        执行完整的数据处理流程。

        包括数据划分、归一化和 DataLoader 创建。

        Args:
            data: 原始时间序列数据。

        Returns:
            Tuple: (train_loader, val_loader, test_loader, scaler)
        """
        train_data, val_data, test_data = self._split_data(data)
        
        train_scaled, val_scaled, test_scaled, self.scaler = self._normalize_data(
            train_data, val_data, test_data
        )
        
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders(
            train_scaled, val_scaled, test_scaled
        )
        
        return self.train_loader, self.val_loader, self.test_loader, self.scaler

    def _split_data(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        按时间顺序划分数据集。

        Args:
            data: 完整时间序列数据。

        Returns:
            Tuple: (train_data, val_data, test_data)
        """
        n_samples = len(data)
        train_end = int(n_samples * self.config.data.train_ratio)
        val_end = train_end + int(n_samples * self.config.data.val_ratio)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        print(f"[数据划分] 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
        
        return train_data, val_data, test_data

    def _normalize_data(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """
        对数据进行 Z-Score 归一化。

        使用训练集的统计量对验证集和测试集进行归一化，
        避免数据泄露。

        Args:
            train_data: 训练数据。
            val_data: 验证数据。
            test_data: 测试数据。

        Returns:
            Tuple: 归一化后的数据和归一化器。
        """
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        val_scaled = scaler.transform(val_data)
        test_scaled = scaler.transform(test_data)
        
        print(f"[归一化] 均值: {scaler.mean_[0]:.6f}, 标准差: {scaler.scale_[0]:.6f}")
        
        return train_scaled, val_scaled, test_scaled, scaler

    def _create_dataloaders(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器。

        Args:
            train_data: 训练数据。
            val_data: 验证数据。
            test_data: 测试数据。

        Returns:
            Tuple: (train_loader, val_loader, test_loader)
        """
        train_dataset = TimeSeriesDataset(
            train_data,
            self.config.data.seq_len,
            self.config.data.pred_len
        )
        val_dataset = TimeSeriesDataset(
            val_data,
            self.config.data.seq_len,
            self.config.data.pred_len
        )
        test_dataset = TimeSeriesDataset(
            test_data,
            self.config.data.seq_len,
            self.config.data.pred_len
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )
        
        print(f"[DataLoader] 训练批次数: {len(train_loader)}, "
              f"验证批次数: {len(val_loader)}, 测试批次数: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader


def create_dataloaders(
    config: Config,
    data: Optional[np.ndarray] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, np.ndarray]:
    """
    便捷函数：创建数据加载器的完整流程。

    Args:
        config: 项目配置实例。
        data: 可选的预加载数据。如果为 None，则自动生成。

    Returns:
        Tuple: (train_loader, val_loader, test_loader, scaler, raw_data)
    """
    if data is None:
        data = generate_time_series(
            n_samples=config.data.n_samples,
            noise_level=config.data.noise_level,
            seed=config.seed
        )
    
    processor = DataProcessor(config)
    train_loader, val_loader, test_loader, scaler = processor.process(data)
    
    return train_loader, val_loader, test_loader, scaler, data

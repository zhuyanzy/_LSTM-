import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple, Optional
from config import Config


def generate_time_series(n_samples: int = 3500, noise_level: float = 0.15) -> np.ndarray:
    """生成模拟时间序列数据。

    组合三种成分：指数趋势项 + 多频率周期性波动 + 高斯噪声。

    Args:
        n_samples: 时间序列的总样本点数。
        noise_level: 高斯噪声的幅度系数。

    Returns:
        np.ndarray: 形状为 (n_samples, 1) 的时间序列数组。
    """
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
    """时间序列数据集类，支持滑窗采样。

    按照指定的序列长度和预测长度，将连续的时间序列划分为
    输入-输出对，用于 LSTM 模型的训练和预测。

    Attributes:
        data: 归一化后的时间序列数据。
        seq_len: 输入序列长度（历史窗口大小）。
        pred_len: 预测序列长度（未来步数）。
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int) -> None:
        """初始化数据集。

        Args:
            data: 归一化后的时间序列数组。
            seq_len: 输入序列长度。
            pred_len: 预测序列长度。
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        """返回数据集的总样本数。"""
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本对。

        Args:
            idx: 样本起始索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (输入序列, 目标序列)。
                输入形状: (seq_len, 1)
                目标形状: (pred_len, 1)
        """
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y)
        )


def preprocess_data(
    data: np.ndarray,
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """数据预处理完整流程。

    1. 按时序划分训练/验证/测试集（不随机打乱）
    2. 使用 StandardScaler 进行 Z-Score 归一化（仅使用训练集统计量）
    3. 保存归一化器用于后续逆变换
    4. 创建数据集和 DataLoader

    Args:
        data: 原始时间序列数据，形状为 (n_samples, 1)。
        config: 配置对象。

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
            - 训练集 DataLoader
            - 验证集 DataLoader
            - 测试集 DataLoader
            - 拟合完成的归一化器
    """
    n_samples = len(data)
    train_end = int(n_samples * config.train_ratio)
    val_end = train_end + int(n_samples * config.val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    with open(config.scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    train_dataset = TimeSeriesDataset(train_scaled, config.seq_len, config.pred_len)
    val_dataset = TimeSeriesDataset(val_scaled, config.seq_len, config.pred_len)
    test_dataset = TimeSeriesDataset(test_scaled, config.seq_len, config.pred_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, scaler

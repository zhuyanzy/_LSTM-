"""
LSTM 时间序列预测项目 - 配置模块

该模块定义了项目的所有超参数和配置项，包括：
- 数据参数
- 模型参数
- 训练参数
- 路径配置
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import os


@dataclass
class DataConfig:
    """
    数据相关配置参数。

    Attributes:
        n_samples: 样本数量。
        seq_len: 输入序列长度（时间步长）。
        pred_len: 预测步长。
        train_ratio: 训练集比例。
        val_ratio: 验证集比例。
        test_ratio: 测试集比例。
        noise_level: 噪声水平。
    """
    n_samples: int = 3500
    seq_len: int = 24
    pred_len: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    noise_level: float = 0.15

    def __post_init__(self) -> None:
        """初始化后验证比例之和是否为1。"""
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(ratio_sum - 1.0) < 1e-6:
            raise ValueError(f"数据集比例之和必须为1，当前为 {ratio_sum}")


@dataclass
class ModelConfig:
    """
    模型相关配置参数。

    Attributes:
        input_size: 输入特征维度。
        hidden_size: LSTM 隐藏层维度。
        num_layers: LSTM 层数。
        output_size: 输出维度。
        dropout: Dropout 比率。
        bidirectional: 是否使用双向 LSTM。
    """
    input_size: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 1
    dropout: float = 0.1
    bidirectional: bool = False

    def __post_init__(self) -> None:
        """初始化后验证参数有效性。"""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size 必须大于0，当前为 {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers 必须大于0，当前为 {self.num_layers}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout 必须在 [0, 1) 范围内，当前为 {self.dropout}")


@dataclass
class TrainingConfig:
    """
    训练相关配置参数。

    Attributes:
        batch_size: 批次大小。
        epochs: 训练轮数。
        learning_rate: 学习率。
        weight_decay: 权重衰减（L2正则化）。
        patience: 早停耐心值。
        scheduler_factor: 学习率衰减因子。
        scheduler_patience: 学习率调度耐心值。
        num_workers: DataLoader 工作进程数。
        pin_memory: 是否将数据固定在内存中。
        device: 计算设备。
    """
    batch_size: int = 64
    epochs: int = 35
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    num_workers: int = 0
    pin_memory: bool = False
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))

    def __post_init__(self) -> None:
        """初始化后自动检测并设置最优设备。"""
        self.device = get_device()


@dataclass
class PathConfig:
    """
    路径相关配置参数。

    Attributes:
        output_dir: 输出目录。
        model_path: 模型保存路径。
        scaler_path: 归一化器保存路径。
        result_image_path: 结果图片保存路径。
    """
    output_dir: str = "outputs"
    model_path: str = "outputs/lstm_model.pth"
    scaler_path: str = "outputs/scaler.pkl"
    result_image_path: str = "outputs/lstm_prediction_results.png"

    def __post_init__(self) -> None:
        """初始化后确保输出目录存在。"""
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class Config:
    """
    项目总配置类。

    聚合所有子配置模块，提供统一的配置访问接口。

    Attributes:
        data: 数据配置。
        model: 模型配置。
        training: 训练配置。
        path: 路径配置。
        seed: 随机种子。
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    path: PathConfig = field(default_factory=PathConfig)
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """
        从字典创建配置实例。

        Args:
            config_dict: 配置字典。

        Returns:
            Config: 配置实例。
        """
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        path_config = PathConfig(**config_dict.get("path", {}))
        seed = config_dict.get("seed", 42)

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            path=path_config,
            seed=seed
        )


def get_device() -> torch.device:
    """
    自动检测并返回最优计算设备。

    优先级：CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: 可用的计算设备。
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[设备] 检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[设备] 检测到 Apple MPS 设备")
    else:
        device = torch.device("cpu")
        print("[设备] 使用 CPU 设备")
    return device

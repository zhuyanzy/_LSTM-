"""配置模块，存放所有超参数和路径配置。

该模块定义了 LSTM 时间序列预测项目的所有配置参数，
包括数据参数、模型参数、训练参数和路径配置。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class DataConfig:
    """数据相关配置参数。
    
    Attributes:
        n_samples: 样本数量，用于生成模拟数据。
        seq_len: 输入序列长度（时间步长）。
        pred_len: 预测步长。
        train_ratio: 训练集比例。
        val_ratio: 验证集比例。
        test_ratio: 测试集比例。
        noise_level: 模拟数据的噪声水平。
    """
    n_samples: int = 3500
    seq_len: int = 24
    pred_len: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    noise_level: float = 0.15


@dataclass
class ModelConfig:
    """模型相关配置参数。
    
    Attributes:
        input_size: 输入特征维度。
        hidden_size: LSTM 隐藏层维度。
        num_layers: LSTM 层数。
        dropout: Dropout 比率。
        output_size: 输出维度。
        bidirectional: 是否使用双向 LSTM。
    """
    input_size: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    output_size: int = 1
    bidirectional: bool = False


@dataclass
class TrainingConfig:
    """训练相关配置参数。
    
    Attributes:
        batch_size: 批次大小。
        epochs: 训练轮数。
        learning_rate: 学习率。
        weight_decay: 权重衰减（L2 正则化）。
        patience: 学习率调度器的 patience。
        factor: 学习率衰减因子。
        num_workers: 数据加载器的工作进程数。
        pin_memory: 是否使用 pin_memory 加速数据传输。
        device: 训练设备（cuda 或 cpu）。
    """
    batch_size: int = 64
    epochs: int = 35
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    patience: int = 5
    factor: float = 0.5
    num_workers: int = 0
    pin_memory: bool = False
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))


@dataclass
class PathConfig:
    """路径相关配置参数。
    
    Attributes:
        project_root: 项目根目录。
        model_dir: 模型保存目录。
        output_dir: 输出文件目录。
        log_dir: 日志文件目录。
        scaler_path: 归一化器保存路径。
        model_path: 模型权重保存路径。
        plot_path: 结果可视化保存路径。
    """
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    def __post_init__(self) -> None:
        """初始化后创建必要的目录。"""
        self.model_dir: Path = self.project_root / "models"
        self.output_dir: Path = self.project_root / "outputs"
        self.log_dir: Path = self.project_root / "logs"
        
        # 创建目录
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.scaler_path: Path = self.model_dir / "scaler.pkl"
        self.model_path: Path = self.model_dir / "lstm_model.pth"
        self.plot_path: Path = self.output_dir / "lstm_prediction_results.png"


@dataclass
class Config:
    """项目主配置类，聚合所有子配置。
    
    Attributes:
        data: 数据配置。
        model: 模型配置。
        training: 训练配置。
        paths: 路径配置。
        seed: 随机种子。
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    seed: Optional[int] = 42


# 全局配置实例
cfg = Config()

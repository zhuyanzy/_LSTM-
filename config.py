from dataclasses import dataclass
import torch


@dataclass
class Config:
    """LSTM 时间序列预测项目配置类。

    包含数据参数、模型结构参数、训练超参数以及路径配置。
    设备会自动检测 CUDA 是否可用，优先使用 GPU。

    Attributes:
        n_samples: 生成时间序列的样本数量。
        seq_len: 输入序列的时间步长（历史窗口大小）。
        pred_len: 预测步长（未来预测的时间步数）。
        train_ratio: 训练集占总数据的比例。
        val_ratio: 验证集占总数据的比例。
        test_ratio: 测试集占总数据的比例。
        input_size: LSTM 输入特征维度。
        hidden_size: LSTM 隐藏层神经元数量。
        num_layers: LSTM 层数。
        dropout: Dropout 正则化比率。
        batch_size: 训练批次大小。
        epochs: 最大训练轮数。
        learning_rate: 优化器初始学习率。
        device: 训练设备（CUDA 如果可用，否则 CPU）。
        scaler_path: 归一化器保存路径。
        model_path: 最佳模型权重保存路径。
        random_seed: 随机种子，保证实验可复现。
    """
    n_samples: int = 3500
    seq_len: int = 24
    pred_len: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    input_size: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    batch_size: int = 64
    epochs: int = 35
    learning_rate: float = 0.001
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler_path: str = 'scaler.pkl'
    model_path: str = 'lstm_model.pth'
    random_seed: int = 42

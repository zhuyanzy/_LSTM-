"""
LSTM 时间序列预测项目 - 模型定义模块

该模块定义了 LSTM 神经网络模型，包括：
- LSTM 模型类
- 模型工厂函数
"""

from typing import Optional
import torch
import torch.nn as nn

from lstm_project.config import ModelConfig


class LSTMModel(nn.Module):
    """
    多层 LSTM 时间序列预测模型。

    该模型由多层 LSTM 网络和全连接输出层组成，
    支持单向和双向 LSTM，适用于单变量和多变量时间序列预测。

    Attributes:
        hidden_size: LSTM 隐藏层维度。
        num_layers: LSTM 层数。
        bidirectional: 是否使用双向 LSTM。
        lstm: LSTM 层。
        dropout: Dropout 层。
        fc: 全连接输出层。

    Example:
        >>> model = LSTMModel(input_size=1, hidden_size=128, num_layers=2)
        >>> x = torch.randn(32, 24, 1)  # (batch, seq_len, features)
        >>> output = model(x)
        >>> output.shape  # torch.Size([32, 1, 1])
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ) -> None:
        """
        初始化 LSTM 模型。

        Args:
            input_size: 输入特征维度。
            hidden_size: LSTM 隐藏层维度。
            num_layers: LSTM 层数。
            output_size: 输出维度（预测步长）。
            dropout: Dropout 比率（仅在 num_layers > 1 时应用于 LSTM 层间）。
            bidirectional: 是否使用双向 LSTM。
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)。
            hidden: 可选的初始隐藏状态 (h_0, c_0)。

        Returns:
            torch.Tensor: 预测输出，形状为 (batch_size, pred_len, output_size)。
        """
        lstm_out, _ = self.lstm(x, hidden)
        
        last_out = lstm_out[:, -1, :]
        
        last_out = self.dropout(last_out)
        
        out = self.fc(last_out)
        
        return out.unsqueeze(-1)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化隐藏状态。

        Args:
            batch_size: 批次大小。
            device: 计算设备。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (h_0, c_0) 初始隐藏状态。
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        return (h_0, c_0)


class StackedLSTMModel(nn.Module):
    """
    堆叠式 LSTM 模型（带残差连接）。

    该模型在多层 LSTM 之间添加残差连接，
    有助于缓解梯度消失问题，适合更深层的网络。

    Attributes:
        lstm_layers: LSTM 层列表。
        layer_norms: 层归一化列表。
        dropout: Dropout 层。
        fc: 全连接输出层。
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1
    ) -> None:
        """
        初始化堆叠式 LSTM 模型。

        Args:
            input_size: 输入特征维度。
            hidden_size: LSTM 隐藏层维度。
            num_layers: LSTM 层数。
            output_size: 输出维度。
            dropout: Dropout 比率。
        """
        super(StackedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size))
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（带残差连接）。

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)。

        Returns:
            torch.Tensor: 预测输出，形状为 (batch_size, pred_len, output_size)。
        """
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            lstm_out, _ = lstm(x)
            lstm_out = norm(lstm_out)
            
            if i > 0:
                lstm_out = lstm_out + x
            
            x = self.dropout(lstm_out)
        
        last_out = x[:, -1, :]
        out = self.fc(last_out)
        
        return out.unsqueeze(-1)


def create_model(
    config: ModelConfig,
    model_type: str = "lstm"
) -> nn.Module:
    """
    模型工厂函数。

    根据配置创建对应的模型实例。

    Args:
        config: 模型配置实例。
        model_type: 模型类型，可选 "lstm" 或 "stacked_lstm"。

    Returns:
        nn.Module: 模型实例。

    Raises:
        ValueError: 如果 model_type 不支持。
    """
    if model_type == "lstm":
        model = LSTMModel(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        )
    elif model_type == "stacked_lstm":
        model = StackedLSTMModel(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            dropout=config.dropout
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model

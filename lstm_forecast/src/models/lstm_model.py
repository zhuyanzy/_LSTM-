"""LSTM 模型定义模块。

该模块定义了用于时间序列预测的多层 LSTM 模型。
"""

from typing import Tuple

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """多层 LSTM 时间序列预测模型。
    
    该模型包含多层 LSTM、Dropout 正则化和全连接输出层，
    用于单变量或多变量时间序列预测任务。
    
    Attributes:
        hidden_size: LSTM 隐藏层维度。
        num_layers: LSTM 层数。
        lstm: LSTM 层。
        dropout: Dropout 层。
        fc: 全连接输出层。
    
    Example:
        >>> model = LSTMModel(
        ...     input_size=1,
        ...     hidden_size=128,
        ...     num_layers=2,
        ...     output_size=1,
        ...     dropout=0.1
        ... )
        >>> x = torch.randn(32, 24, 1)  # (batch_size, seq_len, input_size)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 1, 1])
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ) -> None:
        """初始化 LSTM 模型。
        
        Args:
            input_size: 输入特征维度。
            hidden_size: LSTM 隐藏层维度。
            num_layers: LSTM 层数。
            output_size: 输出维度，默认为 1。
            dropout: Dropout 比率，默认为 0.1。
            bidirectional: 是否使用双向 LSTM，默认为 False。
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 多层 LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        # 如果是双向 LSTM，输入维度需要翻倍
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)。
        
        Returns:
            输出张量，形状为 (batch_size, output_size, 1)。
        """
        # LSTM 前向传播
        # lstm_out: (batch_size, seq_len, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_size)
        # cell: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出用于预测
        # last_out: (batch_size, hidden_size * num_directions)
        last_out = lstm_out[:, -1, :]
        
        # Dropout 正则化
        last_out = self.dropout(last_out)
        
        # 全连接层输出预测值
        # out: (batch_size, output_size)
        out = self.fc(last_out)
        
        # 增加维度以匹配目标形状 (batch_size, output_size, 1)
        return out.unsqueeze(-1)
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量。
        
        Returns:
            模型总参数数量。
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量。
        
        Returns:
            可训练参数数量。
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

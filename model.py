import torch
import torch.nn as nn
from typing import Tuple


class LSTMModel(nn.Module):
    """多层 LSTM 时间序列预测模型。

    网络结构：多层单向 LSTM -> Dropout -> 全连接输出层。
    使用最后一个时间步的隐藏状态进行预测。

    Attributes:
        hidden_size: LSTM 隐藏层维度。
        num_layers: LSTM 堆叠层数。
        lstm: PyTorch LSTM 层。
        dropout: Dropout 正则化层。
        fc: 最终输出的全连接层。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
        dropout: float = 0.1
    ) -> None:
        """初始化 LSTM 模型。

        Args:
            input_size: 输入特征维度，单变量预测为 1。
            hidden_size: LSTM 隐藏层神经元数量。
            num_layers: LSTM 堆叠层数。
            output_size: 输出维度（预测步长），单步预测为 1。
            dropout: Dropout 比率，仅层数 > 1 时生效。
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """模型前向传播。

        Args:
            x: 输入批次张量，形状为 (batch_size, seq_len, input_size)。

        Returns:
            torch.Tensor: 预测结果，形状为 (batch_size, pred_len, 1)。
        """
        lstm_out, _ = self.lstm(x)

        last_out = lstm_out[:, -1, :]

        last_out = self.dropout(last_out)

        out = self.fc(last_out)

        return out.unsqueeze(-1)

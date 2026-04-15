"""
LSTM 时间序列预测项目 - 运行脚本

该脚本提供简单的项目入口，可直接运行完整训练流程。

运行方式:
    python run.py
    
或使用命令行参数:
    python run.py --epochs 50 --hidden_size 256 --batch_size 32
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lstm_project.main import main

if __name__ == "__main__":
    main()

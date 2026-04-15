"""
LSTM 时间序列预测项目

基于 PyTorch 实现的多层 LSTM 模型用于单变量时间序列预测。

模块结构:
    - config: 配置管理
    - data_loader: 数据加载与预处理
    - model: LSTM 模型定义
    - train: 训练与验证
    - utils: 辅助函数
    - main: 主入口

使用方法:
    from lstm_project import Config, run_pipeline
    
    config = Config()
    run_pipeline(config)
    
或者使用命令行:
    python -m lstm_project.main --epochs 50 --hidden_size 256
"""

from lstm_project.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    PathConfig,
    get_device
)
from lstm_project.data_loader import (
    generate_time_series,
    TimeSeriesDataset,
    DataProcessor,
    create_dataloaders
)
from lstm_project.model import (
    LSTMModel,
    StackedLSTMModel,
    create_model
)
from lstm_project.train import (
    Trainer,
    EarlyStopping,
    train_model,
    predict
)
from lstm_project.utils import (
    set_seed,
    save_model,
    load_model,
    save_scaler,
    load_scaler,
    calculate_metrics,
    print_metrics,
    plot_training_curves,
    plot_predictions,
    plot_combined_results,
    count_parameters,
    cleanup_temp_files
)
from lstm_project.main import run_pipeline, parse_args, create_config_from_args

__version__ = "1.0.0"
__author__ = "LSTM Project Team"

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "PathConfig",
    "get_device",
    "generate_time_series",
    "TimeSeriesDataset",
    "DataProcessor",
    "create_dataloaders",
    "LSTMModel",
    "StackedLSTMModel",
    "create_model",
    "Trainer",
    "EarlyStopping",
    "train_model",
    "predict",
    "set_seed",
    "save_model",
    "load_model",
    "save_scaler",
    "load_scaler",
    "calculate_metrics",
    "print_metrics",
    "plot_training_curves",
    "plot_predictions",
    "plot_combined_results",
    "count_parameters",
    "cleanup_temp_files",
    "run_pipeline",
    "parse_args",
    "create_config_from_args",
]

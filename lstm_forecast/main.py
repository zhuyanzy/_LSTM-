"""LSTM 时间序列预测项目主入口。

基于 PyTorch 实现的多层 LSTM 模型用于单变量时间序列预测。

运行步骤:
    1. 安装依赖: pip install -r requirements.txt
    2. 运行项目: python main.py

项目结构:
    lstm_forecast/
    ├── src/
    │   ├── config.py          # 配置参数
    │   ├── data/              # 数据处理模块
    │   │   └── data_loader.py # 数据加载与预处理
    │   ├── models/            # 模型定义模块
    │   │   └── lstm_model.py  # LSTM 模型
    │   ├── training/          # 训练模块
    │   │   └── trainer.py     # 训练器
    │   └── utils/             # 工具函数模块
    │       └── helpers.py     # 辅助函数
    ├── models/                # 模型保存目录
    ├── outputs/               # 输出文件目录
    ├── logs/                  # 日志文件目录
    └── main.py                # 项目入口
"""

import sys
from pathlib import Path

# 添加 src 到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import cfg
from src.data.data_loader import generate_time_series, preprocess_data
from src.models.lstm_model import LSTMModel
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, plot_results, cleanup_temp_files


def main() -> None:
    """主函数，执行完整的训练和评估流程。"""
    print('=' * 60)
    print('LSTM 时间序列预测项目')
    print('=' * 60)
    
    # 设置随机种子
    set_seed(cfg.seed)
    
    # 1. 生成数据
    print('\n[1/5] 生成模拟时间序列数据...')
    data = generate_time_series(
        n_samples=cfg.data.n_samples,
        noise_level=cfg.data.noise_level,
        seed=cfg.seed
    )
    print(f'生成数据形状: {data.shape}')
    print(f'数据范围: [{data.min():.4f}, {data.max():.4f}]')
    
    # 2. 数据预处理
    print('\n[2/5] 数据预处理与加载...')
    train_loader, val_loader, test_loader, scaler = preprocess_data(data, cfg)
    print(f'训练集批次数: {len(train_loader)}')
    print(f'验证集批次数: {len(val_loader)}')
    print(f'测试集批次数: {len(test_loader)}')
    
    # 3. 创建模型
    print('\n[3/5] 创建 LSTM 模型...')
    model = LSTMModel(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        output_size=cfg.model.output_size,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional
    )
    
    total_params = model.get_num_parameters()
    trainable_params = model.get_trainable_parameters()
    print(f'模型总参数数量: {total_params:,}')
    print(f'可训练参数数量: {trainable_params:,}')
    print(f'模型结构:\n{model}')
    
    # 4. 训练模型
    print('\n[4/5] 模型训练...')
    trainer = Trainer(model, cfg)
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    # 5. 预测与评估
    print('\n[5/5] 模型预测与评估...')
    predictions, actuals, metrics = trainer.evaluate(test_loader, scaler)
    
    # 6. 绘制结果
    plot_results(
        train_losses,
        val_losses,
        predictions,
        actuals,
        cfg.paths.plot_path
    )
    
    # 清理临时文件
    cleanup_temp_files(cfg)
    
    print('\n' + '=' * 60)
    print('项目执行完成!')
    print('=' * 60)


if __name__ == '__main__':
    main()

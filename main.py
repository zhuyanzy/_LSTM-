import warnings
warnings.filterwarnings('ignore')

import torch
from config import Config
from data_loader import generate_time_series, preprocess_data
from model import LSTMModel
from train import train_model, predict_and_evaluate
from utils import set_seed, plot_results, cleanup_files


def main() -> None:
    print('='*60)
    print('LSTM 时间序列预测项目')
    print('='*60)

    config = Config()

    set_seed(config.random_seed)

    print(f'\n使用设备: {config.device}')
    if torch.cuda.is_available():
        print(f'GPU型号: {torch.cuda.get_device_name(0)}')

    print('\n[1/5] 生成模拟时间序列数据...')
    data = generate_time_series(n_samples=config.n_samples)
    print(f'生成数据形状: {data.shape}')
    print(f'数据范围: [{data.min():.4f}, {data.max():.4f}]')

    print('\n[2/5] 数据预处理与加载...')
    train_loader, val_loader, test_loader, scaler = preprocess_data(data, config)
    print(f'训练集批次数: {len(train_loader)}')
    print(f'验证集批次数: {len(val_loader)}')
    print(f'测试集批次数: {len(test_loader)}')

    print('\n[3/5] 创建LSTM模型...')
    model = LSTMModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.pred_len,
        dropout=config.dropout
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数数量: {total_params:,}')
    print(f'模型结构:\n{model}')

    print('\n[4/5] 模型训练...')
    train_losses, val_losses = train_model(model, train_loader, val_loader, config)

    print('\n[5/5] 模型预测与评估...')
    predictions, actuals, metrics = predict_and_evaluate(model, test_loader, scaler, config)

    plot_results(train_losses, val_losses, predictions, actuals)

    cleanup_files([config.scaler_path, config.model_path])

    print('\n' + '='*60)
    print('项目执行完成!')
    print('='*60)


if __name__ == '__main__':
    main()

"""
LSTM 时间序列预测项目 - 主入口模块

该模块是项目的入口点，负责：
- 解析命令行参数
- 协调各模块完成完整训练流程
- 输出结果和可视化
"""

import argparse
from typing import Optional
import torch

from lstm_project.config import Config, get_device
from lstm_project.data_loader import generate_time_series, create_dataloaders
from lstm_project.model import create_model
from lstm_project.train import Trainer
from lstm_project.utils import (
    set_seed,
    count_parameters,
    calculate_metrics,
    print_metrics,
    plot_combined_results,
    save_scaler,
    cleanup_temp_files
)


def run_pipeline(config: Config, verbose: bool = True) -> None:
    """
    执行完整的训练预测流程。

    Args:
        config: 项目配置实例。
        verbose: 是否打印详细信息。
    """
    if verbose:
        print("=" * 60)
        print("LSTM 时间序列预测项目")
        print("=" * 60)
    
    set_seed(config.seed)
    
    if verbose:
        print("\n[1/5] 生成模拟时间序列数据...")
    data = generate_time_series(
        n_samples=config.data.n_samples,
        noise_level=config.data.noise_level,
        seed=config.seed
    )
    if verbose:
        print(f"生成数据形状: {data.shape}")
        print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")
    
    if verbose:
        print("\n[2/5] 数据预处理与加载...")
    train_loader, val_loader, test_loader, scaler, _ = create_dataloaders(config, data)
    
    save_scaler(scaler, config.path.scaler_path)
    
    if verbose:
        print("\n[3/5] 创建LSTM模型...")
    model = create_model(config.model, model_type="lstm")
    model = model.to(config.training.device)
    
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    if verbose:
        print(f"模型参数数量: {total_params:,} (可训练: {trainable_params:,})")
        print(f"模型结构:\n{model}")
    
    if verbose:
        print("\n[4/5] 模型训练...")
    trainer = Trainer(model, config)
    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        verbose=verbose,
        print_interval=5
    )
    
    if verbose:
        print("\n[5/5] 模型预测与评估...")
    predictions, actuals = trainer.predict(test_loader, scaler)
    
    metrics = calculate_metrics(actuals, predictions)
    print_metrics(metrics)
    
    plot_combined_results(
        train_losses,
        val_losses,
        actuals,
        predictions,
        save_path=config.path.result_image_path,
        show=False
    )
    
    cleanup_temp_files(config)
    
    if verbose:
        print("\n" + "=" * 60)
        print("项目执行完成!")
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数。
    """
    parser = argparse.ArgumentParser(
        description="LSTM 时间序列预测项目",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--n_samples", type=int, default=3500,
        help="样本数量"
    )
    parser.add_argument(
        "--seq_len", type=int, default=24,
        help="输入序列长度"
    )
    parser.add_argument(
        "--pred_len", type=int, default=1,
        help="预测步长"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128,
        help="LSTM 隐藏层维度"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2,
        help="LSTM 层数"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="批次大小"
    )
    parser.add_argument(
        "--epochs", type=int, default=35,
        help="训练轮数"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001,
        help="学习率"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout 比率"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式"
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> Config:
    """
    从命令行参数创建配置实例。

    Args:
        args: 命令行参数。

    Returns:
        Config: 配置实例。
    """
    from lstm_project.config import DataConfig, ModelConfig, TrainingConfig, PathConfig
    
    data_config = DataConfig(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        pred_len=args.pred_len
    )
    
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    path_config = PathConfig(
        output_dir=args.output_dir,
        model_path=f"{args.output_dir}/lstm_model.pth",
        scaler_path=f"{args.output_dir}/scaler.pkl",
        result_image_path=f"{args.output_dir}/lstm_prediction_results.png"
    )
    
    return Config(
        data=data_config,
        model=model_config,
        training=training_config,
        path=path_config,
        seed=args.seed
    )


def main() -> None:
    """主函数入口。"""
    args = parse_args()
    config = create_config_from_args(args)
    
    run_pipeline(config, verbose=not args.quiet)


if __name__ == "__main__":
    main()

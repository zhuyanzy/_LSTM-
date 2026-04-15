import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from config import Config
from model import LSTMModel


def train_model(
    model: LSTMModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config
) -> Tuple[List[float], List[float]]:
    """模型训练与验证完整流程。

    包含：MSE 损失函数、Adam 优化器、ReduceLROnPlateau 学习率调度器。
    每轮训练后进行验证，保存验证损失最优的模型权重。

    Args:
        model: 待训练的 LSTM 模型实例。
        train_loader: 训练集 DataLoader。
        val_loader: 验证集 DataLoader。
        config: 配置对象。

    Returns:
        Tuple[List[float], List[float]]: (每轮训练损失列表, 每轮验证损失列表)。
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print('='*60)
    print('开始训练...')
    print('='*60)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.device)
                batch_y = batch_y.to(config.device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{config.epochs}] '
                  f'Train Loss: {train_loss:.6f} '
                  f'Val Loss: {val_loss:.6f} '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print('='*60)
    print(f'训练完成! 最佳验证损失: {best_val_loss:.6f}')
    print('='*60)

    return train_losses, val_losses


def predict_and_evaluate(
    model: LSTMModel,
    test_loader: DataLoader,
    scaler: StandardScaler,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """在测试集上进行预测并计算评估指标。

    1. 加载训练过程中保存的最佳模型权重
    2. 在测试集上进行批量预测
    3. 逆归一化恢复原始尺度
    4. 计算 MAE、RMSE、MAPE 三种评估指标

    Args:
        model: LSTM 模型实例。
        test_loader: 测试集 DataLoader。
        scaler: 训练时拟合的归一化器，用于逆变换。
        config: 配置对象。

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
            - predictions_original: 原始尺度的预测值，形状 (n_samples, 1)
            - actuals_original: 原始尺度的真实值，形状 (n_samples, 1)
            - metrics: (MAE, RMSE, MAPE) 评估指标元组
    """
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.device)
            outputs = model(batch_x)

            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(batch_y.squeeze().cpu().numpy())

    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    predictions_original = scaler.inverse_transform(predictions)
    actuals_original = scaler.inverse_transform(actuals)

    mae = mean_absolute_error(actuals_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(actuals_original, predictions_original))
    mape = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100

    print('\n' + '='*60)
    print('测试集评估指标:')
    print(f'MAE  (平均绝对误差):  {mae:.6f}')
    print(f'RMSE (均方根误差):   {rmse:.6f}')
    print(f'MAPE (平均绝对百分比误差): {mape:.2f}%')
    print('='*60)

    return predictions_original, actuals_original, (mae, rmse, mape)

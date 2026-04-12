"""
LSTM 时间序列预测项目
基于 PyTorch 实现的多层 LSTM 模型用于单变量时间序列预测

运行环境: Conda 环境 ts_env
执行步骤:
    1. conda create -n ts_env python=3.10
    2. conda activate ts_env
    3. pip install -r requirements.txt
    4. python lstm_time_series_prediction.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 配置参数
# =============================================================================
class Config:
    # 数据参数
    n_samples = 3500          # 样本数量
    seq_len = 24              # 时间步长（输入序列长度）
    pred_len = 1              # 预测步长
    train_ratio = 0.7         # 训练集比例
    val_ratio = 0.1           # 验证集比例
    test_ratio = 0.2          # 测试集比例
    
    # 模型参数
    input_size = 1            # 输入特征维度
    hidden_size = 128         # 隐藏层维度
    num_layers = 2            # LSTM层数
    dropout = 0.1             # Dropout率
    
    # 训练参数
    batch_size = 64           # 批次大小
    epochs = 35               # 训练轮数
    learning_rate = 0.001     # 学习率
    device = torch.device('cpu')  # 使用CPU
    
    # 保存路径
    scaler_path = 'scaler.pkl'
    model_path = 'lstm_model.pth'

config = Config()

# =============================================================================
# 2. 数据生成模块
# =============================================================================
def generate_time_series(n_samples=3500, noise_level=0.15):
    """
    生成模拟时间序列数据
    组合：指数趋势 + 周期性波动 + 噪声
    """
    t = np.arange(n_samples)
    
    # 1. 指数趋势项
    trend = 0.0005 * np.exp(0.002 * t)
    
    # 2. 周期性波动项（多个频率叠加）
    seasonality_1 = 0.3 * np.sin(2 * np.pi * t / 100)
    seasonality_2 = 0.15 * np.sin(2 * np.pi * t / 30)
    seasonality_3 = 0.1 * np.sin(2 * np.pi * t / 7)
    seasonality = seasonality_1 + seasonality_2 + seasonality_3
    
    # 3. 随机噪声
    noise = noise_level * np.random.randn(n_samples)
    
    # 4. 组合所有成分
    data = trend + seasonality + noise
    
    return data.reshape(-1, 1)

# =============================================================================
# 3. 数据集定义
# =============================================================================
class TimeSeriesDataset(Dataset):
    """
    时间序列数据集，支持滑窗采样
    """
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # 输入序列：[idx, idx+seq_len)
        x = self.data[idx:idx+self.seq_len]
        # 目标值：[idx+seq_len, idx+seq_len+pred_len)
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y)
        )

# =============================================================================
# 4. 数据预处理模块
# =============================================================================
def preprocess_data(data, config):
    """
    数据预处理：归一化 + 数据集划分 + DataLoader创建
    """
    # 划分数据集（时序数据不能随机划分）
    n_samples = len(data)
    train_end = int(n_samples * config.train_ratio)
    val_end = train_end + int(n_samples * config.val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Z-Score归一化（仅使用训练集统计量）
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    # 保存归一化器参数用于后续逆变换
    with open(config.scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 创建数据集
    train_dataset = TimeSeriesDataset(train_scaled, config.seq_len, config.pred_len)
    val_dataset = TimeSeriesDataset(val_scaled, config.seq_len, config.pred_len)
    test_dataset = TimeSeriesDataset(test_scaled, config.seq_len, config.pred_len)
    
    # 创建DataLoader（CPU优化：使用多线程加载）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows下设为0避免多进程问题
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader, scaler

# =============================================================================
# 5. 模型定义
# =============================================================================
class LSTMModel(nn.Module):
    """
    多层LSTM时间序列预测模型
    包含: 多层LSTM + Dropout + 全连接输出层
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多层LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        前向传播
        x: [batch_size, seq_len, input_size]
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出用于预测
        last_out = lstm_out[:, -1, :]
        
        # Dropout正则化
        last_out = self.dropout(last_out)
        
        # 全连接层输出预测值
        out = self.fc(last_out)
        
        return out.unsqueeze(-1)

# =============================================================================
# 6. 训练模块
# =============================================================================
def train_model(model, train_loader, val_loader, config):
    """
    模型训练函数
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
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            
            optimizer.zero_grad(set_to_none=True)  # CPU优化
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
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
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_path)
        
        # 打印训练信息
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{config.epochs}] '
                  f'Train Loss: {train_loss:.6f} '
                  f'Val Loss: {val_loss:.6f} '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    print('='*60)
    print(f'训练完成! 最佳验证损失: {best_val_loss:.6f}')
    print('='*60)
    
    return train_losses, val_losses

# =============================================================================
# 7. 预测与评估模块
# =============================================================================
def predict_and_evaluate(model, test_loader, scaler, config):
    """
    预测与评估
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
    
    # 逆归一化
    predictions_original = scaler.inverse_transform(predictions)
    actuals_original = scaler.inverse_transform(actuals)
    
    # 计算评估指标
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

# =============================================================================
# 8. 可视化模块
# =============================================================================
def plot_results(train_losses, val_losses, predictions, actuals):
    """
    绘制训练损失曲线和预测对比图
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 图1: 训练/验证损失曲线
    axes[0].plot(train_losses, 'b-', label='训练损失', linewidth=2)
    axes[0].plot(val_losses, 'r-', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('训练/验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(np.arange(0, len(train_losses), 5))
    
    # 图2: 测试集预测对比
    x_axis = np.arange(len(predictions))
    axes[1].plot(x_axis, actuals, 'b-', label='真实值', linewidth=1.5, alpha=0.7)
    axes[1].plot(x_axis, predictions, 'r-', label='预测值', linewidth=1.5, alpha=0.8)
    
    # 计算置信区间 (95%置信区间: ±1.96*std)
    residuals = actuals.flatten() - predictions.flatten()
    std_error = np.std(residuals)
    upper_bound = predictions.flatten() + 1.96 * std_error
    lower_bound = predictions.flatten() - 1.96 * std_error
    
    axes[1].fill_between(
        x_axis, lower_bound, upper_bound,
        color='gray', alpha=0.2, label='95%置信区间'
    )
    
    axes[1].set_xlabel('时间步', fontsize=12)
    axes[1].set_ylabel('数值', fontsize=12)
    axes[1].set_title('测试集预测值 vs 真实值对比', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.savefig('lstm_prediction_results.png', dpi=300, bbox_inches='tight')
    print('\n结果图已保存为: lstm_prediction_results.png')
    plt.close()

# =============================================================================
# 9. 主函数
# =============================================================================
def main():
    print('='*60)
    print('LSTM 时间序列预测项目')
    print('='*60)
    
    # 1. 生成数据
    print('\n[1/5] 生成模拟时间序列数据...')
    data = generate_time_series(n_samples=config.n_samples)
    print(f'生成数据形状: {data.shape}')
    print(f'数据范围: [{data.min():.4f}, {data.max():.4f}]')
    
    # 2. 数据预处理
    print('\n[2/5] 数据预处理与加载...')
    train_loader, val_loader, test_loader, scaler = preprocess_data(data, config)
    print(f'训练集批次数: {len(train_loader)}')
    print(f'验证集批次数: {len(val_loader)}')
    print(f'测试集批次数: {len(test_loader)}')
    
    # 3. 创建模型
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
    
    # 4. 训练模型
    print('\n[4/5] 模型训练...')
    train_losses, val_losses = train_model(model, train_loader, val_loader, config)
    
    # 5. 预测与评估
    print('\n[5/5] 模型预测与评估...')
    predictions, actuals, metrics = predict_and_evaluate(model, test_loader, scaler, config)
    
    # 6. 绘制结果
    plot_results(train_losses, val_losses, predictions, actuals)
    
    # 清理临时文件
    if os.path.exists(config.scaler_path):
        os.remove(config.scaler_path)
    if os.path.exists(config.model_path):
        os.remove(config.model_path)
    
    print('\n' + '='*60)
    print('项目执行完成!')
    print('='*60)

if __name__ == '__main__':
    main()

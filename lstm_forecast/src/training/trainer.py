"""训练器模块。

包含模型训练循环、验证逻辑和 Trainer 类。
"""

from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.models.lstm_model import LSTMModel
from src.utils.helpers import save_model, calculate_metrics, Metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """执行一个训练轮次。
    
    Args:
        model: 要训练的模型。
        train_loader: 训练数据加载器。
        criterion: 损失函数。
        optimizer: 优化器。
        device: 训练设备。
    
    Returns:
        平均训练损失。
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)
    
    return total_loss / total_samples


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """执行一个验证轮次。
    
    Args:
        model: 要验证的模型。
        val_loader: 验证数据加载器。
        criterion: 损失函数。
        device: 验证设备。
    
    Returns:
        平均验证损失。
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
    
    return total_loss / total_samples


class Trainer:
    """模型训练器类。
    
    封装完整的训练流程，包括训练循环、验证、学习率调度和模型保存。
    
    Attributes:
        model: 要训练的模型。
        config: 项目配置。
        criterion: 损失函数。
        optimizer: 优化器。
        scheduler: 学习率调度器。
        device: 训练设备。
        train_losses: 训练损失历史。
        val_losses: 验证损失历史。
        best_val_loss: 最佳验证损失。
    
    Example:
        >>> model = LSTMModel(input_size=1, hidden_size=128, num_layers=2)
        >>> trainer = Trainer(model, config)
        >>> train_losses, val_losses = trainer.train(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> None:
        """初始化训练器。
        
        Args:
            model: 要训练的模型。
            config: 项目配置。
            criterion: 损失函数，默认为 MSELoss。
            optimizer: 优化器，默认为 Adam。
            scheduler: 学习率调度器，默认为 ReduceLROnPlateau。
        """
        self.model = model.to(config.training.device)
        self.config = config
        self.device = config.training.device
        
        # 损失函数
        self.criterion = criterion or nn.MSELoss()
        
        # 优化器
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.training.factor,
            patience=config.training.patience
        )
        
        # 训练历史
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss: float = float('inf')
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[List[float], List[float]]:
        """执行完整训练流程。
        
        Args:
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器。
        
        Returns:
            包含训练损失列表和验证损失列表的元组。
        """
        print('=' * 60)
        print('开始训练...')
        print(f'设备: {self.device}')
        print(f'模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}')
        print('=' * 60)
        
        for epoch in range(self.config.training.epochs):
            # 训练阶段
            train_loss = train_epoch(
                self.model, train_loader, self.criterion,
                self.optimizer, self.device
            )
            self.train_losses.append(train_loss)
            
            # 验证阶段
            val_loss = validate_epoch(
                self.model, val_loader, self.criterion, self.device
            )
            self.val_losses.append(val_loss)
            
            # 学习率调度
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_model(
                    self.model,
                    self.config.paths.model_path,
                    self.optimizer,
                    epoch + 1,
                    val_loss
                )
            
            # 打印训练信息
            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f'Epoch [{epoch+1}/{self.config.training.epochs}] '
                    f'Train Loss: {train_loss:.6f} '
                    f'Val Loss: {val_loss:.6f} '
                    f'LR: {current_lr:.6f}'
                )
        
        print('=' * 60)
        print(f'训练完成! 最佳验证损失: {self.best_val_loss:.6f}')
        print('=' * 60)
        
        return self.train_losses, self.val_losses
    
    def evaluate(
        self,
        test_loader: DataLoader,
        scaler: StandardScaler
    ) -> Tuple[np.ndarray, np.ndarray, Metrics]:
        """评估模型。
        
        在测试集上进行预测并计算评估指标。
        
        Args:
            test_loader: 测试数据加载器。
            scaler: 用于逆归一化的 StandardScaler。
        
        Returns:
            包含预测值、真实值和评估指标的元组。
        """
        # 加载最佳模型
        checkpoint = torch.load(self.config.paths.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        predictions: List[np.ndarray] = []
        actuals: List[np.ndarray] = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(batch_y.squeeze().cpu().numpy())
        
        predictions_arr = np.array(predictions).reshape(-1, 1)
        actuals_arr = np.array(actuals).reshape(-1, 1)
        
        # 逆归一化
        predictions_original = scaler.inverse_transform(predictions_arr)
        actuals_original = scaler.inverse_transform(actuals_arr)
        
        # 计算评估指标
        metrics = calculate_metrics(predictions_original, actuals_original)
        
        print('\n' + '=' * 60)
        print('测试集评估指标:')
        print(metrics)
        print('=' * 60)
        
        return predictions_original, actuals_original, metrics

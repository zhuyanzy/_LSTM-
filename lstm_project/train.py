"""
LSTM 时间序列预测项目 - 训练模块

该模块包含模型训练、验证和预测的核心逻辑，包括：
- 训练器类
- 训练循环
- 验证逻辑
- 预测函数
"""

from typing import Tuple, List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from lstm_project.config import Config
from lstm_project.utils import save_model, load_model


class EarlyStopping:
    """
    早停机制类。

    监控验证损失，在连续多轮没有改善时停止训练，
    防止过拟合并节省训练时间。

    Attributes:
        patience: 耐心值（允许无改善的轮数）。
        min_delta: 最小改善阈值。
        counter: 当前计数器。
        best_loss: 最佳验证损失。
        early_stop: 是否触发早停。
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ) -> None:
        """
        初始化早停机制。

        Args:
            patience: 耐心值，允许无改善的轮数。
            min_delta: 最小改善阈值。
            mode: 监控模式，"min" 或 "max"。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False

    def __call__(self, loss: float) -> bool:
        """
        检查是否应该早停。

        Args:
            loss: 当前验证损失。

        Returns:
            bool: 是否触发早停。
        """
        if self.best_loss is None:
            self.best_loss = loss
            return False
        
        if self.mode == "min":
            improved = loss < self.best_loss - self.min_delta
        else:
            improved = loss > self.best_loss + self.min_delta
        
        if improved:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    模型训练器类。

    封装了完整的训练流程，包括训练循环、验证、
    学习率调度、早停和模型保存。

    Attributes:
        model: 待训练的模型。
        config: 项目配置。
        criterion: 损失函数。
        optimizer: 优化器。
        scheduler: 学习率调度器。
        early_stopping: 早停机制。
        device: 计算设备。
        train_losses: 训练损失历史。
        val_losses: 验证损失历史。
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """
        初始化训练器。

        Args:
            model: 待训练的模型。
            config: 项目配置实例。
            criterion: 损失函数（默认为 MSELoss）。
            optimizer: 优化器（默认为 Adam）。
        """
        self.model = model.to(config.training.device)
        self.config = config
        self.device = config.training.device
        
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience
        )
        
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=1e-6,
            mode="min"
        )
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss: float = float("inf")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个 epoch。

        Args:
            train_loader: 训练数据加载器。

        Returns:
            float: 平均训练损失。
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        return total_loss / total_samples

    def validate(self, val_loader: DataLoader) -> float:
        """
        在验证集上评估模型。

        Args:
            val_loader: 验证数据加载器。

        Returns:
            float: 平均验证损失。
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                batch_size = batch_x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        return total_loss / total_samples

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
        print_interval: int = 5
    ) -> Tuple[List[float], List[float]]:
        """
        执行完整训练流程。

        Args:
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器。
            verbose: 是否打印训练信息。
            print_interval: 打印间隔（每隔多少个 epoch）。

        Returns:
            Tuple[List[float], List[float]]: 训练损失和验证损失历史。
        """
        if verbose:
            print("=" * 60)
            print("开始训练...")
            print("=" * 60)
        
        for epoch in range(self.config.training.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_model(
                    self.model,
                    self.config.path.model_path,
                    self.optimizer,
                    epoch,
                    val_loss
                )
            
            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch + 1}/{self.config.training.epochs}] "
                    f"Train Loss: {train_loss:.6f} "
                    f"Val Loss: {val_loss:.6f} "
                    f"LR: {current_lr:.6f}"
                )
            
            if self.early_stopping(val_loss):
                if verbose:
                    print(f"\n[早停] 在第 {epoch + 1} 轮触发早停机制")
                break
        
        if verbose:
            print("=" * 60)
            print(f"训练完成! 最佳验证损失: {self.best_val_loss:.6f}")
            print("=" * 60)
        
        return self.train_losses, self.val_losses

    def predict(
        self,
        test_loader: DataLoader,
        scaler: StandardScaler
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在测试集上进行预测。

        Args:
            test_loader: 测试数据加载器。
            scaler: 归一化器（用于逆变换）。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 预测值和真实值（原始尺度）。
        """
        self.model, _, _, _ = load_model(
            self.model,
            self.config.path.model_path,
            self.device
        )
        self.model.eval()
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(batch_y.squeeze().cpu().numpy())
        
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        
        predictions_original = scaler.inverse_transform(predictions)
        actuals_original = scaler.inverse_transform(actuals)
        
        return predictions_original, actuals_original


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    便捷函数：训练模型的简化接口。

    Args:
        model: 待训练的模型。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。
        config: 项目配置。

    Returns:
        Tuple: (训练后的模型, 训练损失历史, 验证损失历史)
    """
    trainer = Trainer(model, config)
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    return trainer.model, train_losses, val_losses


def predict(
    model: nn.Module,
    test_loader: DataLoader,
    scaler: StandardScaler,
    config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：模型预测的简化接口。

    Args:
        model: 训练好的模型。
        test_loader: 测试数据加载器。
        scaler: 归一化器。
        config: 项目配置。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 预测值和真实值。
    """
    trainer = Trainer(model, config)
    return trainer.predict(test_loader, scaler)

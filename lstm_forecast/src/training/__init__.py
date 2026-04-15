"""训练模块。

包含模型训练和评估功能。
"""

from src.training.trainer import Trainer, train_epoch, validate_epoch

__all__ = ["Trainer", "train_epoch", "validate_epoch"]

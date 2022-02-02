from turtle import forward
import torch
import torch.nn as nn


class MAEScaledLoss(nn.Module):
    """
    MAE scaled loss
    """
    def __init__(self, factor: float = 100) -> None:
        """
        init function.

        Args:
            factor (float): factor to multiply error. Defaults to 1000.
        """
        super().__init__()
        self.loss = nn.L1Loss()
        self.factor = factor
    
    def forward(self, y_pred: torch.nn, y_true: torch.nn) -> torch.nn:
        """
        forward step for loss

        Args:
            y_pred (torch.nn): predictions from model.
            y_true (torch.nn): ground truth.

        Returns:
            torch.nn: error
        """
        error = self.loss(y_pred, y_true)
        return self.factor * error

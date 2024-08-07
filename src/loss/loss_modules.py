import torch
import torch.nn as nn


class WingLoss3D(nn.Module):
    def __init__(self, omega=5, epsilon=2):
        super(WingLoss3D, self).__init__()
        self.omega = torch.tensor(
            omega, dtype=torch.float32
        )  # Convert omega to a tensor
        self.epsilon = torch.tensor(
            epsilon, dtype=torch.float32
        )  # Convert epsilon to a tensor

    def forward(self, pred, target, mask=None):
        # Compute the L1 distance between predicted and target coordinates
        delta_y = torch.abs(pred - target)

        # Compute the loss for small errors
        small_mask = delta_y < self.omega
        loss_small = self.omega * torch.log(1 + delta_y / self.epsilon) * small_mask

        # Compute the loss for large errors
        large_mask = delta_y >= self.omega
        C = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon)
        loss_large = (delta_y - C) * large_mask

        # Compute the total loss for all dimensions and mean over batch and points
        loss = loss_small + loss_large

        if mask is not None:
            loss = loss[mask]

        return loss.mean()


def get_loss_fn(loss_fn: str):
    if loss_fn == "L1":
        return nn.L1Loss()
    elif loss_fn == "WingLoss":
        return WingLoss3D()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

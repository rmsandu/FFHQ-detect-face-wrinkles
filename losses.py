import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_focal_loss(pred, target, alpha=0.9, gamma=1.0):
    """
    Binary focal loss
    For y=1: -α(1-p)ᵧ log(p)
    For y=0: -(1-α)pᵧ log(1-p)
    """

    # Ensure inputs are in the right shape
    pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
    target = target.unsqueeze(1) if target.dim() == 3 else target

    assert (
        pred.shape == target.shape
    ), f"Shape mismatch: pred {pred.shape}, target {target.shape}"

    # Get probabilities
    pred_probs = torch.sigmoid(pred)

    # Clip probabilities to prevent log(0)
    eps = 1e-6
    pred_probs = torch.clamp(pred_probs, eps, 1.0 - eps)

    # Calculate focal weights
    focal_weights_pos = (1 - pred_probs) ** gamma  # For positive class
    focal_weights_neg = pred_probs**gamma  # For negative class

    # Calculate log probabilities
    log_p = torch.log(pred_probs)
    log_1_p = torch.log(1 - pred_probs)

    # Calculate loss for positive and negative classes
    loss_pos = -alpha * focal_weights_pos * log_p
    loss_neg = -(1 - alpha) * focal_weights_neg * log_1_p

    # Combine losses based on target
    focal_loss = torch.where(target == 1, loss_pos, loss_neg)

    return focal_loss.mean()


def dice_loss(pred, target, epsilon=1e-6):
    """
    Compute Dice loss between predicted probabilities and target
    pred: Already sigmoided predictions
    target: Ground truth binary masks
    """
    # Ensure inputs are in the right shape
    pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
    target = target.unsqueeze(1) if target.dim() == 3 else target

    # Flatten predictions and targets
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    # Calculate intersection and union
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    # Calculate Dice coefficient
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return 1.0 - dice.mean()

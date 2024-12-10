import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, focal_alpha=0.25):
        """
        Initialize the Combined Loss function.

        Args:
            alpha: Weighting factor for the Dice loss in the combination.
            gamma: Focusing parameter for the Focal loss.
            focal_alpha: Weighting factor for the positive class in Focal loss.
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_alpha = focal_alpha

    def dice_loss(self, pred, target, epsilon=1e-6):
        """
        Compute the Dice loss.

        Args:
            pred: Predicted logits or probabilities. Shape: (batch_size, height, width).
            target: Ground truth binary mask. Shape: (batch_size, height, width).
            epsilon: Small constant to prevent division by zero.

        Returns:
            Dice loss value.
        """
        # Ensure pred and target are of shape (batch_size, 1, height, width)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # Add channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add channel dimension

        # Convert to probabilities if necessary
        pred = torch.sigmoid(pred)

        # Compute intersection and union
        intersection = (pred * target).sum(dim=(2, 3))  # Element-wise multiplication
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2.0 * intersection + epsilon) / (union + epsilon)
        return 1.0 - dice.mean()

    def focal_loss(self, pred, target, epsilon=1e-6):
        """
        Compute the Focal loss.

        Args:
            pred: Predicted logits (not probabilities). Shape: (batch_size, 1, height, width).
            target: Ground truth binary mask. Shape: (batch_size, 1, height, width).
            epsilon: Small constant to prevent division by zero.

        Returns:
            Focal loss value.
        """
        # Ensure pred and target are of shape (batch_size, 1, height, width)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # Add channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add channel dimension

        # Use binary_cross_entropy_with_logits to combine sigmoid and BCE
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Compute probabilities from logits for focal loss calculation
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)  # Probabilities for the true class

        # Compute focal loss
        focal_loss = (self.focal_alpha * (1 - p_t) ** self.gamma * bce_loss).mean()

        return focal_loss

    def forward(self, pred, target):
        """
        Compute the combined loss.

        Args:
            pred: Predicted logits. Shape: (batch_size, height, width).
            target: Ground truth binary mask. Shape: (batch_size, height, width).

        Returns:
            Combined loss value, Dice loss value, and Focal loss value.
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        combined = self.alpha * dice + (1 - self.alpha) * focal
        return combined, dice, focal

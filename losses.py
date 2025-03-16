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
        """Compute Dice loss with proper shape handling"""
        # Ensure inputs are in the right shape (B, C, H, W)
        pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
        target = target.unsqueeze(1) if target.dim() == 3 else target

        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)

        # Flatten predictions and targets for easier computation
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + epsilon) / (union + epsilon)
        return 1.0 - dice.mean()

    def focal_loss(self, pred, target, epsilon=1e-6):
        """Compute Focal loss with proper shape handling"""
        # Ensure inputs are in the right shape (B, C, H, W)
        pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
        target = target.unsqueeze(1) if target.dim() == 3 else target

        # Compute BCE with logits
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Get probabilities for focal term
        pred_prob = torch.sigmoid(pred)
        # Calculate p_t: probability of the true class (both for positive and negative cases)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        # Apply focusing parameter gamma
        focal_term = (1 - p_t) ** self.gamma

        # Apply class weights: focal_alpha for positive class, (1-focal_alpha) for negative class
        alpha_t = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)

        # Combine all terms
        focal = alpha_t * focal_term * bce
        return focal.mean()

    def forward(self, pred, target):
        """Combined Dice and Focal loss"""
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.alpha * dice + (1 - self.alpha) * focal, dice, focal

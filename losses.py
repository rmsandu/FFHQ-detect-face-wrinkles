import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, focal_alpha=0.75, pos_weight=None):
        """
        Args:
            alpha: Weight between Dice (alpha) and Focal loss (1-alpha)
            gamma: Focal loss focusing parameter
            focal_alpha: Weight for positive class (wrinkles)
            pos_weight: Optional tensor of positive class weights for BCE
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_alpha = focal_alpha
        # Initialize pos_weight but don't move to device yet
        self.pos_weight = pos_weight if pos_weight is not None else torch.tensor([5.0])

    def dice_loss(self, pred, target, epsilon=1e-6):
        """Compute Dice loss with proper shape handling"""
        # Move pos_weight to the same device as the input
        pos_weight = self.pos_weight.to(pred.device)

        # Ensure inputs are in the right shape (B, C, H, W)
        pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
        target = target.unsqueeze(1) if target.dim() == 3 else target

        # Use pos_weight that's now on the correct device
        weight_map = target * pos_weight
        weight_map = weight_map + 1.0  # Background weight is 1

        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)

        # Flatten predictions and targets for easier computation
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        weight_map = weight_map.view(weight_map.size(0), -1)

        intersection = (pred * target * weight_map).sum(dim=1)
        union = (pred * weight_map).sum(dim=1) + (target * weight_map).sum(dim=1)

        dice = (2.0 * intersection + epsilon) / (union + epsilon)
        return 1.0 - dice.mean()

    def focal_loss(self, pred, target, epsilon=1e-6):
        """Compute Focal loss with proper shape handling"""
        # Move pos_weight to the same device as the input
        pos_weight = self.pos_weight.to(pred.device)

        # Ensure inputs are in the right shape (B, C, H, W)
        pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
        target = target.unsqueeze(1) if target.dim() == 3 else target

        # Use pos_weight that's now on the correct device
        bce = F.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=pos_weight,
            reduction="none",
        )

        # Get probabilities for focal term
        pred_prob = torch.sigmoid(pred)
        # Calculate p_t: probability of the true class (both for positive and negative cases)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        # Apply focusing parameter gamma
        focal_term = (1 - p_t) ** self.gamma

        # Increased weight for positive class
        alpha_t = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)

        # Combine all terms
        focal = alpha_t * focal_term * bce
        return focal.mean()

    def forward(self, pred, target):
        """Combined Dice and Focal loss"""
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.alpha * dice + (1 - self.alpha) * focal, dice, focal

import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    """
    Compute the Dice coefficient for binary segmentation.

    Args:
        input: Predicted tensor of shape (N, H, W) or (N, C, H, W).
        target: Ground truth tensor of the same shape as `input`.
        reduce_batch_first: Whether to average the Dice score across the batch.
        epsilon: Small value to prevent division by zero.

    Returns:
        Dice coefficient as a Tensor.
    """
    assert input.size() == target.size(), "Input and target must have the same shape"
    sum_dim = (-1, -2) if input.dim() == 3 or not reduce_batch_first else (-1, -2, -3)

    # Compute intersection and union
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)  # Handle edge cases where sets_sum is zero

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    """
    Compute the Dice coefficient for multiclass segmentation.

    Args:
        input: Predicted tensor of shape (N, C, H, W).
        target: Ground truth tensor of the same shape as `input`.
        reduce_batch_first: Whether to average the Dice score across the batch.
        epsilon: Small value to prevent division by zero.

    Returns:
        Dice coefficient as a Tensor.
    """
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor:
    """
    Compute Dice loss, which is (1 - Dice coefficient).

    Args:
        input: Predicted tensor of shape (N, C, H, W) for multiclass, or (N, H, W) for binary.
        target: Ground truth tensor of the same shape as `input`.
        multiclass: Whether to compute Dice loss for multiclass segmentation.

    Returns:
        Dice loss as a Tensor.
    """
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

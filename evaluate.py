import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)
from tqdm import tqdm
import wandb

# Local imports
from losses import CombinedLoss


@torch.inference_mode()
def evaluate(
    net,
    dataloader,
    device,
    amp=True,
    log_images=False,
    epoch=0,
    save_images=False,
    run_name=None,
    mode="val",
):
    """
    Evaluates the performance of the model on the validation dataset.

    Args:
        net: PyTorch model.
        dataloader: DataLoader for the validation set.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        amp: Use Automatic Mixed Precision (default: True).
        log_images: Whether to log validation images to W&B (default: False).
        epoch: Current training epoch (used for logging).
        save_images: Whether to save validation images locally (default: False).
        run_name: Name of the current run (used for saving images).
        mode: Mode of evaluation ('val' or 'test').

    Returns:
        A dictionary of metrics (Dice, Precision, Recall, F1, AUC).

    """
    net.eval()
    num_val_batches = len(dataloader)

    # Initialize metrics with proper device
    metrics = {
        "precision": BinaryPrecision().to(device),
        "recall": BinaryRecall().to(device),
        "f1": BinaryF1Score().to(device),
        "auc": BinaryAUROC().to(device),
        "iou": BinaryJaccardIndex().to(device),
    }

    # Initialize loss tracking
    losses = {"dice_loss": 0.0, "focal_loss": 0.0, "combined_loss": 0.0}

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{mode} round")):
            images = batch["image"].to(device=device, dtype=torch.float32)
            true_masks = batch["mask"].to(device=device, dtype=torch.float32)

            # Forward pass
            pred_masks = net(images)  # Raw logits

            # Ensure proper shapes
            pred_masks = (
                pred_masks.squeeze(1)
                if pred_masks.dim() == 4 and pred_masks.size(1) == 1
                else pred_masks
            )
            true_masks = (
                true_masks.squeeze(1)
                if true_masks.dim() == 4 and true_masks.size(1) == 1
                else true_masks
            )

            # Calculate losses separately
            focal_loss_val = binary_focal_loss(pred_masks, true_masks)
            dice_loss_val = dice_loss(torch.sigmoid(pred_masks), true_masks)
            loss = focal_loss_val + dice_loss_val

            # Update loss tracking
            losses["dice_loss"] += dice_loss_val.item()
            losses["focal_loss"] += focal_loss_val.item()

            # Get predictions for metrics
            pred_probs = torch.sigmoid(pred_masks)
            pred_binary = (pred_probs > 0.5).float()

            # Update metrics
            for name, metric in metrics.items():
                if name == "auc":
                    metric.update(pred_probs, true_masks.long())
                else:
                    metric.update(pred_binary, true_masks)

            # Handle image logging
            if (log_images or save_images) and batch_idx == 0:
                _log_images(
                    images,
                    true_masks,
                    pred_binary,
                    epoch,
                    mode,
                    save_dir,
                    log_images,
                    save_images,
                )

    # Compute final metrics
    results = {name: metric.compute().item() for name, metric in metrics.items()}

    # Add averaged losses
    for name, value in losses.items():
        results[name] = value / num_val_batches

    # Log to wandb if enabled
    if log_images:
        wandb.log({f"{mode}/{k}": v for k, v in results.items()})

    net.train()
    return results


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
) -> Tensor:
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
    sets_sum = torch.where(
        sets_sum == 0, inter, sets_sum
    )  # Handle edge cases where sets_sum is zero

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def _binary_dice_score(pred_masks: Tensor, true_masks: Tensor) -> Tensor:
    """
    Compute Dice score for binary segmentation.

    Args:
        pred_masks: Logits or probabilities (before or after sigmoid).
        true_masks: Binary ground truth masks.

    Returns:
        Dice score as a Tensor.

    """
    pred_probs = torch.sigmoid(pred_masks)  # Convert logits to probabilities
    pred_bin = (pred_probs > 0.5).float()  # Threshold probabilities to get binary mask

    # Ensure masks have the same shape and remove unnecessary dimensions
    if true_masks.dim() == 4 and true_masks.size(1) == 1:
        true_masks = true_masks.squeeze(1)  # (B, H, W)

    if pred_bin.dim() == 4 and pred_bin.size(1) == 1:
        pred_bin = pred_bin.squeeze(1)

    assert (
        pred_bin.shape == true_masks.shape
    ), f"Shape mismatch: {pred_bin.shape} vs {true_masks.shape}"

    return dice_coeff(pred_bin, true_masks, reduce_batch_first=True)


def create_overlay(image, true_mask, pred_mask):
    """
    Create an overlay of the input image with true and predicted masks.

    Args:
        image: NumPy array of the input image (HWC, normalized to [0, 1]).
        true_mask: NumPy array of the true mask (HW, binary).
        pred_mask: NumPy array of the predicted mask (HW, binary).

    Returns:
        A NumPy array of the overlayed image.
    """
    true_mask_color = np.zeros_like(image)
    true_mask_color[:, :, 1] = true_mask  # Green channel for true mask

    pred_mask_color = np.zeros_like(image)
    pred_mask_color[:, :, 0] = pred_mask  # Red channel for predicted mask

    overlay = 0.6 * image + 0.2 * true_mask_color + 0.2 * pred_mask_color
    return np.clip(overlay, 0, 1)  # Ensure values are within [0, 1]


def save_image(image, path):
    """
    Save an image or mask as a PNG file.

    Args:
        image: NumPy array representing the image/mask.
        path: Path to save the image.
    """

    plt.imsave(path, image)


def _log_images(
    images, true_masks, pred_masks, epoch, mode, save_dir, log_wandb, save_local
):
    """Helper function to log/save validation images"""
    for i in range(min(len(images), 5)):  # Limit to 5 images
        img = images[i].cpu().permute(1, 2, 0).numpy()  # Convert to HWC
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        true_mask = true_masks[i].cpu().numpy()
        pred_mask = pred_masks[i].cpu().numpy()

        # Create overlay
        overlay_img = create_overlay(img, true_mask, pred_mask)

        if log_wandb:
            wandb.log(
                {
                    f"{mode}/Overlay_Image_{i}_Epoch_{epoch}": wandb.Image(
                        overlay_img,
                        caption=f"Overlay of Input, True Mask, and Prediction ({mode})",
                    )
                }
            )

        if save_local:
            plt.imsave(save_dir / f"overlay_{i}.png", overlay_img)
            np.save(save_dir / f"pred_mask_{i}.npy", pred_mask)

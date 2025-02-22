import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import wandb
import numpy as np
from utils.dice_score import dice_coeff, multiclass_dice_coeff
from losses import CombinedLoss
import matplotlib.pyplot as plt
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
)
from torch.utils.data import DataLoader


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
    total_dice_score = 0.0
    total_dice_loss = 0.0
    total_focal_loss = 0.0

    # Initialize metrics
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)
    loss_fn = CombinedLoss(alpha=0.7, gamma=2.0, focal_alpha=0.25)
    # Set up local directory for saving images
    if save_images:
        run_name = run_name or wandb.run.name if wandb.run else "default_run"
        save_dir = Path(f"{mode}/{run_name}/epoch_{epoch}")
        save_dir.mkdir(parents=True, exist_ok=True)

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                total=num_val_batches,
                desc="Validation round",
                unit="batch",
                leave=False,
            )
        ):
            # Move inputs and labels to the correct device
            images = batch["image"].to(device=device, dtype=torch.float32)
            true_masks = batch["mask"].to(device=device, dtype=torch.float32)

            # Predict masks
            pred_masks = net(images)

            # Reshape masks for consistency
            if (
                pred_masks.dim() == 4 and pred_masks.size(1) == 1
            ):  # Binary segmentation output
                pred_masks = pred_masks.squeeze(1)  # Shape: (B, H, W)
            if (
                true_masks.dim() == 4 and true_masks.size(1) == 1
            ):  # Handle mask with extra channel
                true_masks = true_masks.squeeze(1)

            # Compute Loss Components
            _, dice_loss_value, focal_loss_value = loss_fn(pred_masks, true_masks)
            total_dice_loss += dice_loss_value.item()
            total_focal_loss += focal_loss_value.item()

            # Compute Dice score
            if net.n_classes == 1:
                total_dice_score += _binary_dice_score(pred_masks, true_masks)
            else:
                total_dice_score += _multiclass_dice_score(
                    pred_masks, true_masks, net.n_classes
                )

            pred_probs = torch.sigmoid(pred_masks)  # Convert logits to probabilities
            pred_labels = (pred_probs > 0.5).float()
            precision.update(pred_labels, true_masks)
            recall.update(pred_labels, true_masks)
            f1.update(pred_labels, true_masks)
            auc.update(
                pred_probs, true_masks.long()
            )  # AUC expects probabilities and integer labels

            # Update metrics
            precision.update(pred_labels, true_masks)
            recall.update(pred_labels, true_masks)
            f1.update(pred_labels, true_masks)
            auc.update(
                pred_probs, true_masks.long()
            )  # AUC expects probabilities and integer labels

            # Log and save images for the first batch
            if (log_images or save_images) and batch_idx == 0:
                for i in range(min(len(images), 5)):  # Limit to 5 images per batch
                    img = images[i].cpu().permute(1, 2, 0).numpy()  # Convert to HWC
                    img = (img - img.min()) / (
                        img.max() - img.min()
                    )  # Normalize to [0, 1]
                    true_mask = true_masks[i].cpu().numpy()
                    pred_mask = (torch.sigmoid(pred_masks[i]) > 0.5).cpu().numpy()

                    # Create overlay
                    overlay_img = create_overlay(img, true_mask, pred_mask)

                    if log_images:
                        wandb.log(
                            {
                                f"{mode.capitalize()}/Overlay_Image_{i}_Epoch_{epoch}": wandb.Image(
                                    overlay_img,
                                    caption="Overlay of Input, True Mask, and Prediction {mode} )",
                                )
                            }
                        )

                    if save_images:
                        # Save overlay locally
                        overlay_path = save_dir / f"overlay_{i}.png"
                        save_image(overlay_img, overlay_path)

    # Compute final metrics
    dice_score = total_dice_score / max(num_val_batches, 1)
    precision_score = precision.compute().item()
    recall_score = recall.compute().item()
    f1_score = f1.compute().item()
    auc_score = auc.compute().item()
    avg_dice_loss = total_dice_loss / max(num_val_batches, 1)
    avg_focal_loss = total_focal_loss / max(num_val_batches, 1)

    # Log metrics
    if log_images:
        wandb.log(
            {
                f"{mode}/dice": dice_score,
                f"{mode}/dice": precision_score,
                f"{mode}/recall": recall_score,
                f"{mode}/f1": f1_score,
                f"{mode}/auc": auc_score,
                f"{mode}/dice_loss": avg_dice_loss,
                f"{mode}/focal_loss": avg_focal_loss,
            }
        )

    net.train()

    return {
        "dice": dice_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "auc": auc_score,
        "dice_loss": avg_dice_loss,
        "focal_loss": avg_focal_loss,
        "combined_loss": avg_dice_loss + avg_focal_loss,
    }


def _binary_dice_score(pred_masks, true_masks):
    """
    Compute Dice score for binary segmentation.
    """
    pred_probs = torch.sigmoid(pred_masks)  # Convert logits to probabilities
    pred_bin = (pred_probs > 0.5).float()  # Threshold probabilities
    if true_masks.dim() == 4:  # If true_masks has a channel dimension
        true_masks = true_masks.squeeze(1)
    assert (
        pred_bin.shape == true_masks.shape
    ), f"Shape mismatch: {pred_bin.shape} vs {true_masks.shape}"
    return dice_coeff(pred_bin, true_masks, reduce_batch_first=False)


def _multiclass_dice_score(pred_masks, true_masks, n_classes):
    """
    Compute Dice score for multiclass segmentation.
    Ignores the background class (index 0).
    """
    assert (
        true_masks.min() >= 0 and true_masks.max() < n_classes
    ), f"True mask indices should be in range [0, {n_classes - 1}]"

    true_masks = F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float()
    pred_masks = (
        F.one_hot(pred_masks.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
    )

    # Exclude the background class (index 0) from the Dice score calculation
    return multiclass_dice_coeff(
        pred_masks[:, 1:], true_masks[:, 1:], reduce_batch_first=False
    )


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

import os
import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryJaccardIndex,
)
from tqdm import tqdm
from pathlib import Path
import wandb
import numpy as np
import matplotlib.pyplot as plt
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
    total_dice_score = 0.0
    total_dice_loss = 0.0
    total_focal_loss = 0.0

    # Initialize metrics
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)
    iou = BinaryJaccardIndex().to(device)
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
            total_dice_loss += dice_loss_value.detach()
            total_focal_loss += focal_loss_value.detach()

            # Compute Dice score
            if net.n_classes == 1:
                total_dice_score += _binary_dice_score(pred_masks, true_masks)
            else:
                # TODO: Implement multiclass dice score
                print("Multiclass dice score not implemented yet")
                pass

            pred_probs = torch.sigmoid(pred_masks)  # Convert logits to probabilities
            pred_labels = (pred_probs > 0.5).float()
            precision.update(pred_labels, true_masks)
            recall.update(pred_labels, true_masks)
            f1.update(pred_labels, true_masks)
            auc.update(pred_probs, true_masks.long())
            iou.update(pred_labels, true_masks)

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
                        np.save(save_dir / f"pred_mask_{i}.npy", pred_mask)

    # Compute final metrics
    dice_score = total_dice_score / max(num_val_batches, 1)
    precision_score = precision.compute().item()
    recall_score = recall.compute().item()
    f1_score = f1.compute().item()
    auc_score = auc.compute().item()
    iou_score = iou.compute().item()

    avg_dice_loss = total_dice_loss / max(num_val_batches, 1)
    avg_focal_loss = total_focal_loss / max(num_val_batches, 1)

    # Log metrics
    if log_images:
        wandb.log(
            {
                f"{mode}/dice_score": dice_score,
                f"{mode}/precision_score": precision_score,
                f"{mode}/recall_score": recall_score,
                f"{mode}/f1": f1_score,
                f"{mode}/auc": auc_score,
                f"{mode}/dice_loss": avg_dice_loss,
                f"{mode}/focal_loss": avg_focal_loss,
                f"{mode}/iou": iou_score,
                f"{mode}/total_combined_loss": avg_dice_loss + avg_focal_loss,
            }
        )

    net.train()

    return {
        "dice": dice_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "auc": auc_score,
        "iou": iou_score,
        "dice_loss": avg_dice_loss,
        "focal_loss": avg_focal_loss,
        "total_combined_val_loss": avg_dice_loss + avg_focal_loss,
    }


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

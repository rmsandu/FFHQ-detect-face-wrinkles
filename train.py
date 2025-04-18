import os
import logging
import yaml
import torch
import wandb
from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from utils.dataset_loading import (
    WrinkleDataset,
    get_debug_transforms,
    get_augmentation_transforms,
)

from evaluate import evaluate
from losses import binary_focal_loss, dice_loss


# Load YAML configuration
def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def train_model(model, device, config):
    """Trains the U-Net model on the wrinkle segmentation task."""
    # Extract configuration
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    val_percent = config["val_percent"]
    weight_decay = config["weight_decay"]
    gradient_clipping = config["gradient_clipping"]
    amp = config["amp"]
    dilate_mask = config.get("dilate_masks", False)
    checkpoint_dir = Path(config["checkpoint_dir"])
    image_dir = Path(config["image_dir"])
    mask_dir = Path(config["mask_dir"])

    # Initialize wandb first
    wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        config=config,
    )

    # Validate data directories
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Image or mask directory does not exist: {image_dir}, {mask_dir}"
        )

    image_files = (
        list(image_dir.glob("*.[pP][nN][gG]"))
        + list(image_dir.glob("*.[jJ][pP][gG]"))
        + list(image_dir.glob("*.[jJ][pP][eE][gG]"))
    )

    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in: {image_dir}")

    # Create dataset and splits
    transform = (
        get_augmentation_transforms()
        if config["augmentation"]
        else get_debug_transforms()
    )
    dataset = WrinkleDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
        calculate_weights=False,
        dilate_mask=False,
    )

    # Calculate splits
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    # Create splits with fixed seed
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(dataset)), [n_train, n_val], generator=generator
    )

    train_dataset = WrinkleDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
        calculate_weights=False,
        dilate_mask=False,  # <-- Only train gets dilation
    )

    val_dataset = WrinkleDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
        calculate_weights=False,
        dilate_mask=False,
    )
    # Create subsets
    train_set = torch.utils.data.Subset(train_dataset, train_indices)
    val_set = torch.utils.data.Subset(val_dataset, val_indices)
    # Create dataloaders
    loader_args = {
        "batch_size": batch_size,
        "num_workers": min(os.cpu_count(), 8),  # Limit max workers
        "pin_memory": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.75,
        patience=10,
        verbose=True,
        min_lr=1e-6,
    )

    # Mixed Precision
    grad_scaler = torch.amp.GradScaler(enabled=amp)

    # Training state
    best_val_iou = 0.0  # For tracking best IoU
    best_model_info = None  # Store best model's info
    patience_counter = 0
    global_step = 0
    epoch = 0
    val_loss = 0
    val_iou = 0
    val_score = {}

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            with tqdm(
                total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
            ) as pbar:
                for batch in train_loader:
                    images = batch["image"].to(device=device, dtype=torch.float32)
                    true_masks = batch["mask"].to(device=device, dtype=torch.float32)

                    try:
                        # Forward pass with AMP
                        with torch.amp.autocast(
                            device.type if device.type != "mps" else "cpu", enabled=amp
                        ):
                            masks_pred = model(images)  # Raw logits
                            # Calculate losses separately
                            focal_loss_val = binary_focal_loss(masks_pred, true_masks)
                            dice_loss_val = dice_loss(
                                torch.sigmoid(masks_pred), true_masks
                            )
                            loss = focal_loss_val + dice_loss_val

                        # Backward pass
                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()

                        if amp:
                            grad_scaler.unscale_(optimizer)

                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clipping
                        )
                        if grad_norm > gradient_clipping:
                            logging.warning(f"Large gradient norm: {grad_norm:.3f}")

                        grad_scaler.step(optimizer)
                        grad_scaler.update()

                        # Logging
                        epoch_loss += loss.item()
                        global_step += 1

                        # Log metrics
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/dice_loss": dice_loss_val.item(),
                                "train/focal_loss": focal_loss_val.item(),
                                "train/pos_ratio": true_masks.mean().item(),
                                "train/step": global_step,
                            }
                        )

                        pbar.update(images.shape[0])
                        pbar.set_postfix(loss=f"{loss.item():.4f}")

                        # Prediction monitoring (separate from gradient monitoring)
                        with torch.no_grad():
                            pred_probs = torch.sigmoid(masks_pred)
                            pred_binary = (pred_probs > 0.5).float()
                            pred_pos_ratio = pred_binary.mean().item()
                            if pred_pos_ratio > 0.5:
                                logging.warning(
                                    f"High positive prediction ratio: {pred_pos_ratio:.3f}"
                                )

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logging.warning("GPU OOM, skipping batch")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise e

            # Validation phase
            val_score = evaluate(
                model,
                val_loader,
                device,
                amp=amp,
                log_images=(epoch % 5 == 0),
                save_images=(epoch % 5 == 0),
                epoch=epoch,
                run_name=wandb.run.name,
                mode="val",
            )

            # Get validation metrics
            val_loss = (
                val_score["focal_loss"] + val_score["dice_loss"]
            )  # Total loss for scheduler
            val_iou = val_score["iou"]

            # Step scheduler with total validation loss
            scheduler.step(val_loss)

            # Log all validation metrics to W&B
            wandb.log(
                {
                    **{f"val/{k}": v for k, v in val_score.items()},
                    "val/total_loss": val_loss,  # Add total loss to logging
                    "val/epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # Save best model based on IoU
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                patience_counter = 0

                # Store best model's info
                best_model_info = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_iou": best_val_iou,
                    "all_metrics": val_score,
                }

                # Save best model
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_info, checkpoint_dir / "best_checkpoint.pth")

                # Log best metrics
                wandb.log(
                    {
                        "best_val_iou": best_val_iou,
                        "best_val_epoch": epoch,
                    }
                )

                logging.info(f"New best model saved! (IoU: {best_val_iou:.4f})")
            else:
                patience_counter += 1

            # Save regular checkpoint (optional, for recovery)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_iou": val_iou,
                    "best_val_iou": best_val_iou,
                    "all_metrics": val_score,
                },
                checkpoint_dir / "last_checkpoint.pth",
            )

            if patience_counter >= config["patience"]:
                logging.info(f"Early stopping after {epoch + 1} epochs")
                break

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    finally:
        # If training was interrupted, ensure we still have the best model saved
        if best_model_info is not None:
            torch.save(best_model_info, checkpoint_dir / "best_checkpoint.pth")

        # Save final state in a separate file
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "final_val_iou": val_iou,
                "best_val_iou": best_val_iou,
                "all_metrics": val_score,
            },
            checkpoint_dir / "final_checkpoint.pth",
        )

        wandb.finish()

    return best_val_iou


if __name__ == "__main__":
    # Load configuration from YAML file

    config = load_config("config.yaml")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize model
    model = UNet(
        n_channels=3,
        n_classes=1,
        bilinear=False,
        pretrained=config.get("pretrained", True),
        freeze_encoder=config.get("freeze_encoder", True),
    )
    model.to(device)

    train_model(model, device, config)

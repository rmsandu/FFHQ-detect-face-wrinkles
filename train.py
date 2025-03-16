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
    calculate_class_weights,
)

from evaluate import evaluate
from losses import CombinedLoss


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
    checkpoint_dir = Path(config["checkpoint_dir"])
    image_dir = Path(config["image_dir"])
    mask_dir = Path(config["mask_dir"])

    # Initialize wandb first
    wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        config=config,
        name=f"unet_wrinkle_{wandb.util.generate_id()}",
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
        image_dir=image_dir, mask_dir=mask_dir, transform=transform
    )

    # Calculate splits
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    # Create splits with fixed seed
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    # Create dataloaders
    loader_args = {
        "batch_size": batch_size,
        "num_workers": min(os.cpu_count(), 8),  # Limit max workers
        "pin_memory": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # Calculate class weights from training set
    if config.get("use_class_weights", False):
        pos_weight = calculate_class_weights(train_set)
    else:
        pos_weight = None  # No class weights

    # Initialize loss function
    loss_fn = CombinedLoss(
        alpha=config["loss_function"]["alpha"],
        gamma=config["loss_function"]["gamma"],
        focal_alpha=config["loss_function"]["focal_alpha"],
        pos_weight=pos_weight,
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.75,  # More gradual reduction
        patience=10,
        verbose=True,
        min_lr=1e-6,
    )

    # Mixed Precision
    grad_scaler = torch.amp.GradScaler(enabled=amp)

    # Training state
    best_val_score = 0.0
    patience_counter = 0
    global_step = 0

    logging.info(f"Starting training: {epochs} epochs, {n_train} training samples")

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
                        # Validation checks
                        if not torch.all((true_masks >= 0) & (true_masks <= 1)):
                            raise ValueError("Mask values must be binary (0 or 1)")

                        # Forward pass with AMP
                        with torch.amp.autocast(
                            device.type if device.type != "mps" else "cpu", enabled=amp
                        ):
                            masks_pred = model(images)
                            loss, dice_loss, focal_loss = loss_fn(
                                masks_pred.squeeze(1), true_masks
                            )

                        # Backward pass
                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()

                        if amp:
                            grad_scaler.unscale_(optimizer)

                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clipping
                        )
                        grad_scaler.step(optimizer)
                        grad_scaler.update()

                        # Logging
                        epoch_loss += loss.item()
                        global_step += 1

                        # Log metrics
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/dice_loss": dice_loss.item(),
                                "train/focal_loss": focal_loss.item(),
                                "train/pos_ratio": true_masks.mean().item(),
                                "train/step": global_step,
                            }
                        )

                        pbar.update(images.shape[0])
                        pbar.set_postfix(loss=f"{loss.item():.4f}")

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

            scheduler.step(val_score["dice"])

            # Log validation metrics
            wandb.log(
                {
                    "val/dice": val_score["dice"],
                    "val/epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # Save best model
            if val_score["dice"] > best_val_score:
                best_val_score = val_score["dice"]
                patience_counter = 0
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_score": best_val_score,
                    },
                    checkpoint_dir / "best_model.pth",
                )

                wandb.log({"best_val_dice": best_val_score})
            else:
                patience_counter += 1

            if patience_counter >= config["patience"]:
                logging.info(f"Early stopping after {epoch + 1} epochs")
                break

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    finally:
        # Save final model state
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_score": best_val_score,
            },
            checkpoint_dir / "final_model.pth",
        )

        wandb.finish()

    return best_val_score


if __name__ == "__main__":
    # Load configuration from YAML file

    config = load_config("config.yaml")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize model
    model = UNet(n_channels=3, n_classes=1)
    model.to(device)

    train_model(model, device, config)

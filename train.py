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
from utils.sanity_check_dimensions import check_dimensions_and_sanity
from utils.dataset_loading import (
    WrinkleDataset,
    get_debug_transforms,
    get_augmentation_transforms,
)
from utils.dataset_metrics import analyze_class_imbalance
from evaluate import evaluate
from losses import CombinedLoss


# Load YAML configuration
def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def train_model(model, device, config):
    """
    Trains the U-Net model on the wrinkle segmentation task.

    Args:
        model: PyTorch model to train.
        device: Device to use for training ('cuda' or 'cpu').
        config: Configuration dictionary loaded from the YAML file.
    """
    # Extract configuration
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    val_percent = config["val_percent"]
    val_test = config["test_percent"]
    weight_decay = config["weight_decay"]
    gradient_clipping = config["gradient_clipping"]
    amp = config["amp"]
    checkpoint_dir = Path(config["checkpoint_dir"])
    image_dir = Path(config["image_dir"])
    mask_dir = Path(config["mask_dir"])

    if config["loss_function"]["name"] == "CombinedLoss":
        loss_fn = CombinedLoss(
            alpha=config["loss_function"]["alpha"],
            gamma=config["loss_function"]["gamma"],
            focal_alpha=config["loss_function"]["focal_alpha"],
        )
    elif config["loss_function"]["name"] == "BCEWithLogitsLoss":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(
            f"Unsupported loss function: {config['loss_function']['name']}"
        )

    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"])

    # Validate data directories
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Image or mask directory does not exist: {image_dir}, {mask_dir}"
        )
    if (
        len(list(image_dir.glob("*.PNG"))) == 0
        or len(list(mask_dir.glob("*.png"))) == 0
    ):
        raise RuntimeError(
            f"No Image (png, jpeg, jpg) files found in the specified directories: {image_dir}, {mask_dir}"
        )

    # Create dataset
    transform = (
        get_augmentation_transforms()
        if config["augmentation"]
        else get_debug_transforms()
    )
    dataset = WrinkleDataset(
        image_dir=image_dir, mask_dir=mask_dir, transform=transform
    )

    # Analyze class imbalance
    analyze_class_imbalance(DataLoader(dataset, batch_size=batch_size, shuffle=False))
    # Split dataset
    n_val = int(len(dataset) * val_percent)
    n_test = int(len(dataset) * val_test)  # 10% for testing
    n_train = len(dataset) - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    # DataLoaders
    loader_args = {
        "batch_size": batch_size,
        "num_workers": os.cpu_count(),
        "pin_memory": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    check_dimensions_and_sanity(train_loader, device=device)
    # Optimizer, Scheduler, and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)

    # Mixed Precision
    grad_scaler = torch.amp.GradScaler(enabled=amp)

    # Start WandB Experiment
    wandb.config.update(config)

    logging.info("Starting training: %d epochs, batch size %d", epochs, batch_size)

    best_val_score = 0.0  # Track the best validation Dice score
    counter = 0  # Number of epochs without improvement
    # Initialize the combined loss
    loss_fn = CombinedLoss(alpha=0.7, gamma=2.0, focal_alpha=0.25)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
                images = batch["image"].to(device, dtype=torch.float32)
                true_masks = batch["mask"].to(device, dtype=torch.float32)

                # verify the batch size for images and masks
                assert (
                    images.shape[0] == true_masks.shape[0]
                ), "Batch size mismatch between images and masks"

                # Forward Pass with AMP
                with torch.amp.autocast(device_type="cuda", enabled=amp):
                    masks_pred = model(images)
                    if config["loss_function"]["name"] == "CombinedLoss":
                        loss, dice_loss_value, focal_loss_value = loss_fn(
                            masks_pred.squeeze(1), true_masks
                        )
                        wandb.log(
                            {
                                "train_dice_loss": dice_loss_value.item(),
                                "train_focal_loss": focal_loss_value.item(),
                            }
                        )
                    else:
                        loss = loss_fn(masks_pred.squeeze(1), true_masks)

                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()

                # Gradient Clipping
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Update Progress Bar
                epoch_loss += loss.item()
                pbar.update(images.shape[0])
                pbar.set_postfix({"loss": loss.item()})
                wandb.log({"train_loss": loss.item()})

        logging.info(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_loader):.4f}")

        # Validation
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
        val_dice_score = val_score["dice"]
        scheduler.step(val_dice_score)
        wandb.log({"val_dice": val_dice_score})

        # Save the best model weights
        if val_dice_score > best_val_score:
            best_val_score = val_dice_score
            counter = 0  # Reset counter on improvement
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = checkpoint_dir / f"best_checkpoint.pth"
            torch.save(model.state_dict(), best_checkpoint_path)
            logging.info(f"New best model saved with Dice Score: {best_val_score:.4f}")
        else:
            counter += 1
            logging.info(f"No improvement for {counter} epochs")

        # Early stopping
        if counter >= config["patience"]:
            logging.info(
                f"Stopping training after {epoch + 1} epochs without improvement."
            )
            break

    logging.info(
        "Training completed! Best Validation Dice Score: {:.4f}".format(best_val_score)
    )

    ## Test set evaluation
    # Load best model weights
    checkpoint_path = Path(config["checkpoint_dir"]) / "best_checkpoint.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f"Loaded best model from {checkpoint_path}")
    else:
        logging.warning(
            f"No checkpoint found at {checkpoint_path}. Using current model."
        )

    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    test_metrics = evaluate(
        net=model,
        dataloader=test_loader,
        device=device,
        amp=config["amp"],
        log_images=True,
        save_images=True,
        run_name=wandb.run.name,
        mode="test",
    )
    logging.info(f"Test Set Metrics: {test_metrics}")


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

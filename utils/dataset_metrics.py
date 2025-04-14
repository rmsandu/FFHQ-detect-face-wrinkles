import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb


# Convert tensor to NumPy for visualization
def tensor_to_numpy(image_tensor):
    """Convert a PyTorch tensor image to a NumPy array (HWC format)."""
    image = image_tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0,1]
    return image


def analyze_class_imbalance(dataloader):
    total_pixels = 0
    wrinkle_pixels = 0
    background_pixels = 0

    for batch in dataloader:
        masks = batch["mask"].numpy()  # Convert to NumPy for easier pixel counting
        total_pixels += masks.size
        wrinkle_pixels += (masks == 1).sum()
        background_pixels += (masks == 0).sum()

    print("Dataset Size", len(dataloader.dataset))
    print(f"Total Pixels: {total_pixels}")
    print(
        f"Wrinkle Pixels: {wrinkle_pixels} ({(wrinkle_pixels / total_pixels) * 100:.2f}%)"
    )
    print(
        f"Background Pixels: {background_pixels} ({(background_pixels / total_pixels) * 100:.2f}%)"
    )

    # Save class distribution to a text file
    class_distribution_path = "class_distribution.txt"
    with open(class_distribution_path, "w", encoding="utf-8") as f:
        f.write(f"Total Pixels: {total_pixels}\n")
        f.write(
            f"Wrinkle Pixels: {wrinkle_pixels} ({(wrinkle_pixels / total_pixels) * 100:.2f}%)\n"
        )
        f.write(
            f"Background Pixels: {background_pixels} ({(background_pixels / total_pixels) * 100:.2f}%)\n"
        )

    # Log the text file to wandb
    # wandb.log({"class_distribution.txt": wandb.Artifact(class_distribution_path, type="dataset")})

    # Plot Class Distribution
    labels = ["Background", "Wrinkles"]
    counts = [background_pixels, wrinkle_pixels]

    plt.bar(labels, counts, color=["blue", "orange"])
    plt.title("Class Distribution")
    plt.ylabel("Pixel Count")

    # Save the plot
    plot_path = "class_distribution.png"
    plt.savefig(plot_path)
    plt.close()

    # Log the plot to wandb
    wandb.log({"Class Distribution": wandb.Image(plot_path)})

    # Visuaize one example from the dataset and transformations work as expected before training:

    image, mask = dataloader.dataset[0]["image"], dataloader.dataset[0]["mask"]

    image_np = tensor_to_numpy(image)
    mask_np = mask.squeeze().cpu().numpy()

    # Define individual augmentations (One by One)
    augmentations = {
        "Original": A.NoOp(),  # No augmentation
        "Resize (512x512)": A.Resize(height=512, width=512),
        "Horizontal Flip": A.HorizontalFlip(p=1.0),
        "Affine Transform": A.Affine(
            scale=(0.9, 1.1), translate_percent=(0.05, 0.1), rotate=(-15, 15), p=1.0
        ),
        "Elastic Transform": A.ElasticTransform(alpha=30, sigma=120 * 0.05, p=1.0),
        "CLAHE (Contrast Enhancement)": A.CLAHE(
            clip_limit=2.0, tile_grid_size=(8, 8), p=1.0
        ),
        "Gaussian Blur": A.GaussianBlur(blur_limit=(5, 5), p=1.0),
        "Gaussian Noise": A.GaussNoise(var_limit=(50.0, 100.0), p=0.5),  # Add noise
    }

    # Plot all augmentations in a grid
    num_augmentations = len(augmentations)
    fig, axes = plt.subplots(2, num_augmentations // 2, figsize=(15, 6))

    for i, (name, transform) in enumerate(augmentations.items()):
        # Apply transformation
        augmented = transform(image=image_np, mask=mask_np)
        aug_image, aug_mask = augmented["image"], augmented["mask"]

        aug_image = (aug_image * 255).astype(np.uint8)
        # Convert to NumPy for visualization
        if isinstance(
            aug_image, np.ndarray
        ):  # Some augmentations return NumPy directly
            aug_image = (aug_image - aug_image.min()) / (
                aug_image.max() - aug_image.min()
            )  # Normalize
        else:
            aug_image = aug_image.permute(1, 2, 0).cpu().numpy()

        # Plot augmentation
        ax = axes[i // (num_augmentations // 2), i % (num_augmentations // 2)]
        ax.imshow(aug_image)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

        # Save each augmentation separately
        cv2.imwrite(
            f"augmentations/{name.replace(' ', '_')}.png",
            (aug_image * 255).astype(np.uint8),
        )

    plt.tight_layout()
    plt.savefig("augmentations_preview.png")  # Save full visualization
    plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="your_project_name", entity="your_username")

    # Create a DataLoader (replace with your actual DataLoader)
    dataloader = DataLoader(...)

    # Analyze class imbalance
    analyze_class_imbalance(dataloader)

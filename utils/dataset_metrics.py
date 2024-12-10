from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb


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
    print(f"Final Tensor Image shape: {image.shape}, Final Mask shape: {mask.shape}")
    plt.subplot(1, 2, 1)
    image = (image - image.min()) / (image.max() - image.min())  # normalize image
    plt.imshow(image.permute(1, 2, 0))  # Convert from CHW to HWC
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.title("Mask")
    plt.savefig("dataset_image_mask_preview_example0.png")  # Save the preview
    # Visualize how each augmentation looks like


# Example usage
if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="your_project_name", entity="your_username")

    # Create a DataLoader (replace with your actual DataLoader)
    dataloader = DataLoader(...)

    # Analyze class imbalance
    analyze_class_imbalance(dataloader)

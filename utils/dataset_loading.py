import os
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


class WrinkleDataset(Dataset):
    """Custom dataset for loading wrinkle images and masks."""

    def __init__(self, image_dir, mask_dir, transform=None, calculate_weights=False):
        """
        Args:
            image_dir (str): Path to the directory containing RGB images.
            mask_dir (str): Path to the directory containing binary masks.
            transform (callable, optional): Optional transform to be applied on an image/mask pair.
            calculate_weights (bool, optional): Flag to calculate class weights.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.calculate_weights = calculate_weights

        # Load image and mask filenames
        self.images = sorted(
            [
                file
                for file in os.listdir(image_dir)
                if file.endswith(".png")
                or file.endswith(".jpg")
                or file.endswith(".jpeg")
            ]
        )
        self.masks = sorted(
            [
                file
                for file in os.listdir(mask_dir)
                if file.endswith(".png")
                or file.endswith(".jpg")
                or file.endswith(".jpeg")
            ]
        )

        if len(self.images) == 0 or len(self.masks) == 0:
            raise RuntimeError(
                f"No images or masks found in {image_dir} or {mask_dir}."
            )
        if len(self.images) != len(self.masks):
            raise RuntimeError("The number of images and masks must be equal.")

        logging.info("Loaded %d image-mask pairs.", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Check if image and mask names match
        assert (
            self.images[idx].split(".")[0] == self.masks[idx].split(".")[0]
        ), f"Image and mask names do not match: {self.images[idx]} vs {self.masks[idx]}"

        # Load image and mask using PIL first
        pil_image = Image.open(img_path).convert("RGB")
        pil_mask = Image.open(mask_path).convert("L")

        # Resize masks to match image size if they don't match
        if pil_image.size != pil_mask.size:
            logging.warning(f"Resizing mask to match image size for {self.images[idx]}")
            pil_image = pil_image.resize(
                (512, 512), Image.Resampling.LANCZOS
            )  # Better quality for images
            pil_mask = pil_mask.resize(
                (512, 512), Image.Resampling.NEAREST
            )  # Preserve binary values for masks

        # Convert to numpy arrays after ensuring sizes match
        image = np.array(pil_image)
        mask = np.array(pil_mask)

        # Now apply any additional resizing if needed (though we've already resized)
        basic_transform = A.Compose(
            [
                A.Resize(
                    height=512,
                    width=512,
                    interpolation=cv2.INTER_LANCZOS4,  # For image
                    mask_interpolation=cv2.INTER_NEAREST,
                )  # For mask
            ]
        )
        transformed = basic_transform(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]

        # Verify image dimensions
        assert (
            len(image.shape) == 3
        ), f"Image should have 3 dimensions, got {len(image.shape)}"
        assert (
            image.shape[-1] == 3
        ), f"Image should have 3 channels, got {image.shape[-1]}"

        # Check and normalize mask values to [0, 1]
        mask = (mask > 127).astype(np.float32)
        assert np.all((mask >= 0) & (mask <= 1)), "Mask values must be between 0 and 1"
        unique_values = np.unique(mask)
        assert len(unique_values) <= 2 and np.all(
            np.isin(unique_values, [0, 1])
        ), f"Mask should only contain 0s and 1s, got {unique_values}"

        # Ensure mask and image have same spatial dimensions
        assert (
            image.shape[:2] == mask.shape[:2]
        ), f"Image shape {image.shape[:2]} does not match mask shape {mask.shape[:2]}"

        if self.calculate_weights:
            # For weight calculation, we only need the basic mask without transforms
            mask = torch.from_numpy(mask).float()
            mask = mask.unsqueeze(0)  # Add channel dimension (1, H, W)
            return {"mask": mask}

        # Regular processing for training/validation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

            # Ensure mask has channel dimension after transform
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        else:
            # If no transforms, manually convert to tensor and normalize
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
            image = image / 255.0
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(image)

            mask = torch.from_numpy(mask).float()
            mask = mask.unsqueeze(0)  # Add channel dimension (1, H, W)

        # Final assertions after transformations
        assert (
            len(image.shape) == 3
        ), f"Image should have 3 dimensions (C,H,W), got {image.shape}"
        assert (
            image.shape[0] == 3
        ), f"Image should have 3 channels as first dim, got {image.shape}"
        assert (
            len(mask.shape) == 3
        ), f"Mask should have 3 dimensions (1,H,W), got {mask.shape}"
        assert (
            mask.shape[0] == 1
        ), f"Mask should have 1 channel as first dim, got {mask.shape}"
        assert (
            image.shape[1:] == mask.shape[1:]
        ), f"Image and mask spatial dimensions mismatch: {image.shape[1:]} vs {mask.shape[1:]}"

        return {"image": image, "mask": mask}


# Transforms for debugging (minimal preprocessing, no augmentation)
def get_debug_transforms():
    """Returns a set of minimal transforms for debugging."""
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(transpose_mask=True),  # This will ensure proper mask dimensions
        ]
    )


# Transforms with basic augmentations for training
def get_augmentation_transforms():
    """Returns a set of augmentation transforms for training."""
    return A.Compose(
        [
            A.Resize(height=512, width=512),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1), translate_percent=(0.05, 0.1), rotate=(-15, 15), p=0.5
            ),
            A.ElasticTransform(alpha=30, sigma=120 * 0.05, p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(transpose_mask=True),  # This will ensure proper mask dimensions
        ]
    )


def calculate_class_weights(dataset):
    """
    Calculate class weights based on dataset statistics.

    Args:
        dataset: WrinkleDataset instance or Subset of WrinkleDataset
    Returns:
        torch.Tensor: Weight for positive class based on class distribution
    """
    total_pixels = 0
    wrinkle_pixels = 0
    logging.info("Calculating class weights...")

    # Handle both Dataset and Subset cases
    if hasattr(dataset, "dataset"):  # If it's a Subset
        original_dataset = dataset.dataset
    else:  # If it's the original Dataset
        original_dataset = dataset

    # Create a dataloader without transforms to get raw masks
    basic_dataset = WrinkleDataset(
        image_dir=original_dataset.image_dir,
        mask_dir=original_dataset.mask_dir,
        transform=None,
        calculate_weights=True,
    )

    dataloader = DataLoader(basic_dataset, batch_size=1, num_workers=4)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing class distribution"):
            mask = batch["mask"]
            # Ensure mask is 2D for counting (remove batch and channel dimensions if present)
            if mask.dim() == 4:  # (B, C, H, W)
                mask = mask.squeeze(0).squeeze(0)
            elif mask.dim() == 3:  # (B, H, W)
                mask = mask.squeeze(0)

            total_pixels += mask.numel()
            wrinkle_pixels += mask.sum().item()

    background_pixels = total_pixels - wrinkle_pixels
    ratio = background_pixels / wrinkle_pixels
    pos_weight = torch.tensor([ratio])

    logging.info(f"Class distribution - Background:Wrinkle = {ratio:.2f}:1")
    return pos_weight

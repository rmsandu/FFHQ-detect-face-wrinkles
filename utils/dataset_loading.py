import os
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class WrinkleDataset(Dataset):
    """Custom dataset for loading wrinkle images and masks."""

    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing RGB images.
            mask_dir (str): Path to the directory containing binary masks.
            transform (callable, optional): Optional transform to be applied on an image/mask pair.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

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
        # check if the image and mask have the same name
        assert (
            self.images[idx].split(".")[0] == self.masks[idx].split(".")[0]
        ), f"Image and mask names do not match: {self.images[idx]} vs {self.masks[idx]}"

        image = np.array(
            Image.open(img_path).convert("RGB")
        )  # Shape: (height, width, channels)
        mask = np.array(Image.open(mask_path).convert("L"))  # Shape: (height, width)
        mask = (mask > 127).astype(np.float32)  # Normalize mask to [0, 1]
        # print("mask shape before loading", mask.shape)
        # print("image shape before loading", image.shape)

        height, width = image.shape[:2]
        if mask.shape[:2] != (height, width):
            mask = np.resize(mask, (height, width))

        # Apply transforms if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # print(
        #    f"After transform -> Image shape: {image.shape}, Mask shape: {mask.shape}"
        # )

        # Ensure tensors are returned
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float32)

        if len(mask.shape) == 2:  # Add channel dimension to mask if missing
            mask = mask.unsqueeze(0)
        assert (
            image.shape[1:] == mask.shape[1:]
        ), f"Image and mask dimensions mismatch: {image.shape[1:]} vs {mask.shape[1:]}"

        return {"image": image, "mask": mask}


# Transforms for debugging (minimal preprocessing, no augmentation)
def get_debug_transforms():
    """Returns a set of minimal transforms for debugging."""
    return A.Compose(
        [
            A.Resize(512, 512),  # Resize images and masks to a fixed size
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # Normalize RGB images
            ToTensorV2(),  # Convert to PyTorch tensors
        ],
        additional_targets={"mask": "mask"},
    )


# Transforms with basic augmentations for training
def get_augmentation_transforms():
    """Returns a set of augmentation transforms for training."""
    return A.Compose(
        [
            A.Resize(height=512, width=512),  # Resize to model input size
            A.HorizontalFlip(p=0.5),  # Flip horizontally
            A.Affine(
                scale=(0.9, 1.1), translate_percent=(0.05, 0.1), rotate=(-15, 15), p=0.5
            ),  # Geometric changes
            A.ElasticTransform(alpha=30, sigma=120 * 0.05, p=0.4),  # Distortion
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),  # Enhance contrast
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),  # Brightness/contrast
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),  # Blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Noise
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # Normalize images
            ToTensorV2(),  # Convert to PyTorch tensors
        ],
        additional_targets={"mask": "mask"},
    )

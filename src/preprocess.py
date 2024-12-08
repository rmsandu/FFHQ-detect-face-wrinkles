import os
import cv2
import torch
from torchvision import transforms
from albumentations import Compose, HorizontalFlip, Normalize
from albumentations.pytorch import ToTensorV2

# Directories
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

# Albumentations transforms
transform = Compose([
    HorizontalFlip(p=0.5),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def load_data(image_dir, mask_dir):
    images, masks = [], []
    for file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file)
        if os.path.exists(mask_path):
            # Load and preprocess image and mask
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)  # Load mask as grayscale
            augmented = transform(image=image, mask=mask)
            images.append(augmented['image'])
            masks.append(augmented['mask'])
    return torch.stack(images), torch.stack(masks)

# Example usage
images, masks = load_data(IMAGE_DIR, MASK_DIR)
print(f"Loaded {len(images)} images and masks.")

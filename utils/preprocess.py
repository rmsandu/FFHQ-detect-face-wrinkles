import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms

class WrinkleDataset(Dataset):
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
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # Open images
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        return image, mask

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images and masks to a fixed size
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
        transforms.ToTensor(),  # Convert to PyTorch tensors
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize RGB images
          # Convert to PyTorch tensors
    ], additional_targets={'mask': 'mask'})  # Specify that masks are not regular images
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class UnderwaterDataset(Dataset):
    """
    PyTorch Dataset for Underwater Image Enhancement
    Loads paired input (underwater) and target (enhanced) images.
    """

    def __init__(self, input_dir, target_dir, augment=False):
        """
        Args:
            input_dir (str): Folder containing input images (.npy)
            target_dir (str): Folder containing target images (.npy)
            augment (bool): Apply data augmentation (only for training)
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.augment = augment

        self.image_names = sorted(os.listdir(input_dir))
        if len(self.image_names) == 0:
            raise ValueError(f"No images found in {input_dir}")

    def __len__(self):
        return len(self.image_names)

    def augment_pair(self, input_img, target_img):
        """Apply random flips and rotations to input and target"""
        # Horizontal flip
        if random.random() > 0.5:
            input_img = np.fliplr(input_img)
            target_img = np.fliplr(target_img)
        # Vertical flip
        if random.random() > 0.5:
            input_img = np.flipud(input_img)
            target_img = np.flipud(target_img)
        # Rotation 90, 180, 270 degrees
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            input_img = np.rot90(input_img, k)
            target_img = np.rot90(target_img, k)
        return input_img, target_img

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # Load processed .npy files
        input_path = os.path.join(self.input_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)

        input_img = np.load(input_path)   # Already normalized [0,1]
        target_img = np.load(target_path)

        # Apply augmentation if training
        if self.augment:
            input_img, target_img = self.augment_pair(input_img, target_img)

        # Convert to tensors and change shape to C,H,W
        input_tensor = torch.from_numpy(input_img).permute(2,0,1).float()
        target_tensor = torch.from_numpy(target_img).permute(2,0,1).float()

        return input_tensor, target_tensor

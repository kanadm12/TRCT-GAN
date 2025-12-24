"""
Dataset class for TRCT-GAN
Loads biplane X-ray images and corresponding CT volumes
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class XRayCTDataset(Dataset):
    """
    Dataset for X-ray to CT reconstruction
    
    Expected directory structure:
    data_path/
        xray_frontal/
            sample_001.png
            sample_002.png
            ...
        xray_lateral/
            sample_001.png
            sample_002.png
            ...
        ct_volumes/
            sample_001.nii.gz
            sample_002.nii.gz
            ...
    """
    
    def __init__(self, data_path, augmentation=None, normalize=None):
        self.data_path = data_path
        self.augmentation = augmentation
        self.normalize = normalize
        
        # Get list of samples
        self.frontal_dir = os.path.join(data_path, 'xray_frontal')
        self.lateral_dir = os.path.join(data_path, 'xray_lateral')
        self.ct_dir = os.path.join(data_path, 'ct_volumes')
        
        # List all samples
        self.samples = []
        if os.path.exists(self.frontal_dir):
            for filename in sorted(os.listdir(self.frontal_dir)):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    sample_id = os.path.splitext(filename)[0]
                    self.samples.append(sample_id)
        
        print(f"Found {len(self.samples)} samples in {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def load_xray(self, path):
        """Load X-ray image"""
        if not os.path.exists(path):
            # Return dummy data if file doesn't exist (for testing)
            return torch.randn(1, 128, 128)
        
        img = Image.open(path).convert('L')  # Grayscale
        img = img.resize((128, 128), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        # Apply normalization
        if self.normalize:
            xray_min = self.normalize.get('xray_min', -1.0)
            xray_max = self.normalize.get('xray_max', 1.0)
            img = img * (xray_max - xray_min) + xray_min
        
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        return img
    
    def load_ct(self, path):
        """Load CT volume"""
        if not os.path.exists(path):
            # Return dummy data if file doesn't exist (for testing)
            return torch.randn(1, 128, 128, 128)
        
        # Load NIfTI file
        nii = nib.load(path)
        volume = nii.get_fdata()
        
        # Resize to target size (128, 128, 128)
        # Note: This is a simple approach; you might want to use proper 3D interpolation
        from scipy.ndimage import zoom
        target_shape = (128, 128, 128)
        zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
        volume = zoom(volume, zoom_factors, order=1)
        
        # Normalize
        volume = volume.astype(np.float32)
        
        # Clip HU values (typical CT range)
        volume = np.clip(volume, -1000, 3000)
        
        # Normalize to [0, 1]
        volume = (volume + 1000) / 4000
        
        # Apply normalization
        if self.normalize:
            ct_min = self.normalize.get('ct_min', -1.0)
            ct_max = self.normalize.get('ct_max', 1.0)
            volume = volume * (ct_max - ct_min) + ct_min
        
        volume = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)
        return volume
    
    def augment(self, xray_frontal, xray_lateral, ct_volume):
        """Apply data augmentation"""
        if not self.augmentation or not self.augmentation.get('enabled', False):
            return xray_frontal, xray_lateral, ct_volume
        
        # Random horizontal flip
        if self.augmentation.get('random_flip', False) and random.random() > 0.5:
            xray_frontal = TF.hflip(xray_frontal)
            xray_lateral = TF.hflip(xray_lateral)
            ct_volume = torch.flip(ct_volume, [3])  # Flip width dimension
        
        # Random rotation (small angles)
        if self.augmentation.get('random_rotation', 0) > 0:
            angle = random.uniform(
                -self.augmentation['random_rotation'],
                self.augmentation['random_rotation']
            )
            xray_frontal = TF.rotate(xray_frontal, angle)
            xray_lateral = TF.rotate(xray_lateral, angle)
            # Note: Rotating 3D volume is more complex, skipping for simplicity
        
        # Random brightness
        if self.augmentation.get('random_brightness', 0) > 0:
            brightness_factor = 1.0 + random.uniform(
                -self.augmentation['random_brightness'],
                self.augmentation['random_brightness']
            )
            xray_frontal = TF.adjust_brightness(xray_frontal, brightness_factor)
            xray_lateral = TF.adjust_brightness(xray_lateral, brightness_factor)
        
        # Random contrast
        if self.augmentation.get('random_contrast', 0) > 0:
            contrast_factor = 1.0 + random.uniform(
                -self.augmentation['random_contrast'],
                self.augmentation['random_contrast']
            )
            xray_frontal = TF.adjust_contrast(xray_frontal, contrast_factor)
            xray_lateral = TF.adjust_contrast(xray_lateral, contrast_factor)
        
        return xray_frontal, xray_lateral, ct_volume
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # Load X-rays
        frontal_path = os.path.join(self.frontal_dir, f"{sample_id}.png")
        lateral_path = os.path.join(self.lateral_dir, f"{sample_id}.png")
        
        xray_frontal = self.load_xray(frontal_path)
        xray_lateral = self.load_xray(lateral_path)
        
        # Load CT
        ct_path = os.path.join(self.ct_dir, f"{sample_id}.nii.gz")
        if not os.path.exists(ct_path):
            ct_path = os.path.join(self.ct_dir, f"{sample_id}.nii")
        
        ct_volume = self.load_ct(ct_path)
        
        # Apply augmentation
        xray_frontal, xray_lateral, ct_volume = self.augment(
            xray_frontal, xray_lateral, ct_volume
        )
        
        return {
            'xray_frontal': xray_frontal,
            'xray_lateral': xray_lateral,
            'ct_volume': ct_volume,
            'sample_id': sample_id
        }


if __name__ == "__main__":
    # Test dataset
    print("Testing XRayCTDataset...")
    
    # Create dummy dataset
    dataset = XRayCTDataset(
        data_path='data/train',
        augmentation={'enabled': True, 'random_flip': True},
        normalize={'xray_min': -1.0, 'xray_max': 1.0, 'ct_min': -1.0, 'ct_max': 1.0}
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Frontal X-ray shape: {sample['xray_frontal'].shape}")
        print(f"Lateral X-ray shape: {sample['xray_lateral'].shape}")
        print(f"CT volume shape: {sample['ct_volume'].shape}")
        print(f"Sample ID: {sample['sample_id']}")
    else:
        print("No samples found. Please check your data path.")
    
    print("✓ Dataset test completed!")

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
    
    Expected directory structure (patient-based):
    data_path/
        patient_001/
            patient_001.nii.gz
            patient_001_pa_drr.png
            patient_001_lat_drr.png
        patient_002/
            patient_002.nii.gz
            patient_002_pa_drr.png
            patient_002_lat_drr.png
        ...
    """
    
    def __init__(self, data_path, augmentation=None, normalize=None):
        self.data_path = data_path
        self.augmentation = augmentation
        self.normalize = normalize
        
        # List all samples (each patient directory)
        self.samples = []
        if os.path.exists(data_path):
            for patient_dir in sorted(os.listdir(data_path)):
                patient_path = os.path.join(data_path, patient_dir)
                if os.path.isdir(patient_path):
                    # Check if all required files exist
                    nii_file = os.path.join(patient_path, f"{patient_dir}.nii.gz")
                    pa_file = os.path.join(patient_path, f"{patient_dir}_pa_drr.png")
                    lat_file = os.path.join(patient_path, f"{patient_dir}_lat_drr.png")
                    
                    if os.path.exists(nii_file) and os.path.exists(pa_file) and os.path.exists(lat_file):
                        self.samples.append(patient_dir)
                    else:
                        print(f"Warning: Skipping {patient_dir} - missing files")
        
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
        
        # Flip vertically (DRRs are upside down)
        img = np.flipud(img)
        
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
        patient_path = os.path.join(self.data_path, sample_id)
        
        # Load X-rays
        frontal_path = os.path.join(patient_path, f"{sample_id}_pa_drr.png")
        lateral_path = os.path.join(patient_path, f"{sample_id}_lat_drr.png")
        
        xray_frontal = self.load_xray(frontal_path)
        xray_lateral = self.load_xray(lateral_path)
        
        # Load CT
        ct_path = os.path.join(patient_path, f"{sample_id}.nii.gz")
        
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

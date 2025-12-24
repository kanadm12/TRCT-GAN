"""
Utility functions for TRCT-GAN
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename, checkpoint_dir='checkpoints'):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, device='cpu'):
    """Load model checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def visualize_slices(ct_volume, num_slices=5, save_path=None):
    """
    Visualize slices from a 3D CT volume
    
    Args:
        ct_volume: (1, D, H, W) or (D, H, W) torch tensor or numpy array
        num_slices: Number of slices to visualize
        save_path: Path to save the visualization
    """
    if isinstance(ct_volume, torch.Tensor):
        ct_volume = ct_volume.cpu().numpy()
    
    # Remove channel dimension if present
    if ct_volume.ndim == 4:
        ct_volume = ct_volume[0]
    
    D, H, W = ct_volume.shape
    
    # Select evenly spaced slices
    slice_indices = np.linspace(0, D-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
    
    for i, idx in enumerate(slice_indices):
        axes[i].imshow(ct_volume[idx], cmap='gray')
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_comparison(xray_frontal, xray_lateral, ct_pred, ct_real=None, save_path=None):
    """
    Visualize input X-rays and output CT (with optional ground truth)
    
    Args:
        xray_frontal: (1, H, W) frontal X-ray
        xray_lateral: (1, H, W) lateral X-ray
        ct_pred: (1, D, H, W) predicted CT volume
        ct_real: (1, D, H, W) ground truth CT volume (optional)
        save_path: Path to save the visualization
    """
    if isinstance(xray_frontal, torch.Tensor):
        xray_frontal = xray_frontal.cpu().numpy()[0]
        xray_lateral = xray_lateral.cpu().numpy()[0]
        ct_pred = ct_pred.cpu().numpy()[0]
        if ct_real is not None:
            ct_real = ct_real.cpu().numpy()[0]
    
    # Select middle slices
    D = ct_pred.shape[0]
    mid_slice = D // 2
    
    if ct_real is not None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Input X-rays
        axes[0, 0].imshow(xray_frontal, cmap='gray')
        axes[0, 0].set_title('Frontal X-ray')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(xray_lateral, cmap='gray')
        axes[0, 1].set_title('Lateral X-ray')
        axes[0, 1].axis('off')
        
        axes[0, 2].axis('off')
        
        # CT slices
        axes[1, 0].imshow(ct_pred[mid_slice], cmap='gray')
        axes[1, 0].set_title('Predicted CT (Axial)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ct_real[mid_slice], cmap='gray')
        axes[1, 1].set_title('Ground Truth CT (Axial)')
        axes[1, 1].axis('off')
        
        # Difference map
        diff = np.abs(ct_pred[mid_slice] - ct_real[mid_slice])
        axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('Absolute Difference')
        axes[1, 2].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(xray_frontal, cmap='gray')
        axes[0].set_title('Frontal X-ray')
        axes[0].axis('off')
        
        axes[1].imshow(xray_lateral, cmap='gray')
        axes[1].set_title('Lateral X-ray')
        axes[1].axis('off')
        
        axes[2].imshow(ct_pred[mid_slice], cmap='gray')
        axes[2].set_title('Predicted CT (Axial)')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_metrics(pred, target):
    """
    Compute evaluation metrics
    
    Args:
        pred: Predicted CT volume
        target: Ground truth CT volume
    
    Returns:
        metrics: Dictionary of metrics
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred - target))
    
    # Mean Squared Error
    mse = np.mean((pred - target) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Peak Signal-to-Noise Ratio
    max_val = np.max(target) - np.min(target)
    psnr = 20 * np.log10(max_val / rmse) if rmse > 0 else float('inf')
    
    # Structural Similarity Index (simplified)
    # For proper SSIM, use skimage.metrics.structural_similarity
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'PSNR': psnr
    }
    
    return metrics


def save_volume_as_nifti(volume, filepath, affine=None):
    """
    Save 3D volume as NIfTI file
    
    Args:
        volume: 3D numpy array or torch tensor
        filepath: Output file path
        affine: Affine transformation matrix (optional)
    """
    import nibabel as nib
    
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    
    # Remove channel dimension if present
    if volume.ndim == 4:
        volume = volume[0]
    
    # Create NIfTI image
    if affine is None:
        affine = np.eye(4)
    
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, filepath)
    print(f"Volume saved to {filepath}")


if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    # Test visualization
    dummy_volume = torch.randn(1, 128, 128, 128)
    visualize_slices(dummy_volume, num_slices=5)
    
    print("✓ Utility tests completed!")

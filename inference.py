"""
Inference Script for TRCT-GAN
Generate 3D CT volumes from biplane X-ray images
"""

import os
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
import nibabel as nib

from models import TRCTGenerator
from utils.utils import visualize_slices, visualize_comparison, save_volume_as_nifti, compute_metrics


class TRCTInference:
    """
    TRCT-GAN Inference Engine
    """
    
    def __init__(self, config_path, checkpoint_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(self.config['hardware']['device'] 
                                   if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Build model
        self.build_model()
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Set to evaluation mode
        self.generator.eval()
    
    def build_model(self):
        """Initialize generator"""
        gen_config = {
            'encoder_channels': self.config['model']['generator']['encoder_channels'],
            'decoder_channels': self.config['model']['generator']['decoder_channels'],
            'transformer': self.config['model']['generator']['transformer'],
            'use_aia_2d': self.config['model']['generator']['aia_2d']['enabled'],
            'use_aia_3d': self.config['model']['generator']['aia_3d']['enabled'],
            'use_trilinear': self.config['model']['generator']['aia_3d']['use_trilinear'],
            'aia_reduction': self.config['model']['generator']['aia_2d']['reduction_ratio']
        }
        
        self.generator = TRCTGenerator(gen_config).to(self.device)
        print("Generator model loaded")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Training epoch: {checkpoint.get('epoch', 'N/A')}")
    
    def load_xray(self, path):
        """Load and preprocess X-ray image"""
        img = Image.open(path).convert('L')
        img = img.resize((128, 128), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Normalize
        normalize = self.config['dataset']['normalize']
        xray_min = normalize.get('xray_min', -1.0)
        xray_max = normalize.get('xray_max', 1.0)
        img = img * (xray_max - xray_min) + xray_min
        
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return img.to(self.device)
    
    def load_ct(self, path):
        """Load ground truth CT for comparison"""
        nii = nib.load(path)
        volume = nii.get_fdata()
        
        # Resize to target size
        from scipy.ndimage import zoom
        target_shape = (128, 128, 128)
        zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
        volume = zoom(volume, zoom_factors, order=1)
        
        # Normalize
        volume = volume.astype(np.float32)
        volume = np.clip(volume, -1000, 3000)
        volume = (volume + 1000) / 4000
        
        normalize = self.config['dataset']['normalize']
        ct_min = normalize.get('ct_min', -1.0)
        ct_max = normalize.get('ct_max', 1.0)
        volume = volume * (ct_max - ct_min) + ct_min
        
        volume = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        return volume.to(self.device)
    
    @torch.no_grad()
    def predict(self, xray_frontal_path, xray_lateral_path):
        """
        Generate 3D CT volume from biplane X-rays
        
        Args:
            xray_frontal_path: Path to frontal X-ray image
            xray_lateral_path: Path to lateral X-ray image
        
        Returns:
            ct_volume: Generated 3D CT volume (1, 1, D, H, W)
        """
        # Load X-rays
        xray_frontal = self.load_xray(xray_frontal_path)
        xray_lateral = self.load_xray(xray_lateral_path)
        
        # Generate CT
        ct_volume = self.generator(xray_frontal, xray_lateral)
        
        return ct_volume, xray_frontal, xray_lateral
    
    def run_inference(self, frontal_path, lateral_path, output_dir, 
                     ground_truth_path=None, visualize=True):
        """
        Run complete inference pipeline
        
        Args:
            frontal_path: Path to frontal X-ray
            lateral_path: Path to lateral X-ray
            output_dir: Directory to save results
            ground_truth_path: Optional path to ground truth CT for comparison
            visualize: Whether to create visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nRunning inference...")
        print(f"  Frontal X-ray: {frontal_path}")
        print(f"  Lateral X-ray: {lateral_path}")
        
        # Predict
        ct_pred, xray_f, xray_l = self.predict(frontal_path, lateral_path)
        
        print(f"  Generated CT shape: {ct_pred.shape}")
        
        # Load ground truth if provided
        ct_real = None
        if ground_truth_path:
            ct_real = self.load_ct(ground_truth_path)
            print(f"  Ground truth CT shape: {ct_real.shape}")
            
            # Compute metrics
            metrics = compute_metrics(ct_pred, ct_real)
            print(f"\n  Evaluation Metrics:")
            for key, value in metrics.items():
                print(f"    {key}: {value:.4f}")
            
            # Save metrics
            metrics_path = os.path.join(output_dir, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
        
        # Save CT volume
        output_format = self.config['inference'].get('output_format', 'nifti')
        
        if output_format == 'nifti':
            ct_path = os.path.join(output_dir, 'predicted_ct.nii.gz')
            save_volume_as_nifti(ct_pred, ct_path)
        elif output_format == 'numpy':
            ct_path = os.path.join(output_dir, 'predicted_ct.npy')
            np.save(ct_path, ct_pred.cpu().numpy())
            print(f"Volume saved to {ct_path}")
        
        # Visualizations
        if visualize or self.config['inference'].get('save_visualizations', True):
            # Slice visualization
            slice_path = os.path.join(output_dir, 'ct_slices.png')
            visualize_slices(ct_pred, num_slices=5, save_path=slice_path)
            
            # Comparison visualization
            comp_path = os.path.join(output_dir, 'comparison.png')
            visualize_comparison(xray_f, xray_l, ct_pred, ct_real, save_path=comp_path)
            
            print(f"\n  Visualizations saved to {output_dir}")
        
        print(f"\n✓ Inference completed successfully!")
        
        return ct_pred


def main():
    parser = argparse.ArgumentParser(description='TRCT-GAN Inference')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--frontal', type=str, required=True,
                       help='Path to frontal X-ray image')
    parser.add_argument('--lateral', type=str, required=True,
                       help='Path to lateral X-ray image')
    parser.add_argument('--output', type=str, default='outputs/inference',
                       help='Output directory')
    parser.add_argument('--ground_truth', type=str, default=None,
                       help='Path to ground truth CT (optional, for evaluation)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = TRCTInference(args.config, args.checkpoint)
    
    # Run inference
    inference.run_inference(
        frontal_path=args.frontal,
        lateral_path=args.lateral,
        output_dir=args.output,
        ground_truth_path=args.ground_truth,
        visualize=args.visualize
    )


if __name__ == '__main__':
    main()

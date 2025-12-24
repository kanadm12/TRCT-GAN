"""
TRCT-GAN: Transformer and GAN for CT Reconstruction from Biplane X-rays

A PyTorch implementation of TRCT-GAN, a deep learning framework for reconstructing
3D CT volumes from 2D biplane X-ray images using Transformers and GANs.

Main Components:
- TRCTGenerator: Dual-encoder generator with transformer bridge
- PatchGANDiscriminator3D: 3D PatchGAN discriminator
- TRCTGANLoss: Combined loss function (adversarial, reconstruction, projection, perceptual)

Quick Start:
    # Training
    python train.py --config config/config.yaml
    
    # Inference
    python inference.py --config config/config.yaml --checkpoint best_model.pth \
                        --frontal frontal.png --lateral lateral.png --output results/

For detailed documentation, see README.md
"""

__version__ = "1.0.0"
__author__ = "TRCT-GAN Implementation"
__description__ = "Transformer and GAN for CT Reconstruction from Biplane X-rays"

# Import main components for easy access
from models import (
    TRCTGenerator,
    PatchGANDiscriminator3D,
    MultiScaleDiscriminator3D,
    ConditionalDiscriminator3D,
    TRCTGANLoss,
    LSGANLoss,
    ReconstructionLoss,
    ProjectionLoss,
    VGGPerceptualLoss
)

from utils import (
    XRayCTDataset,
    AverageMeter,
    visualize_slices,
    visualize_comparison,
    compute_metrics,
    save_volume_as_nifti
)

__all__ = [
    # Models
    'TRCTGenerator',
    'PatchGANDiscriminator3D',
    'MultiScaleDiscriminator3D',
    'ConditionalDiscriminator3D',
    
    # Losses
    'TRCTGANLoss',
    'LSGANLoss',
    'ReconstructionLoss',
    'ProjectionLoss',
    'VGGPerceptualLoss',
    
    # Utilities
    'XRayCTDataset',
    'AverageMeter',
    'visualize_slices',
    'visualize_comparison',
    'compute_metrics',
    'save_volume_as_nifti',
]

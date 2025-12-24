# TRCT-GAN: Transformer and GAN for CT Reconstruction from Biplane X-rays

A PyTorch implementation of **TRCT-GAN** (Transformer and Generative Adversarial Network for CT reconstruction), a deep learning framework designed to reconstruct **3D CT volumes from 2D biplane X-ray images**.

## 🏗️ Architecture Overview

TRCT-GAN combines the power of **Transformers** and **GANs** to bridge the gap between 2D projections and 3D spatial structures:

### Key Components

1. **Dual 2D Encoders** with Dense Connections
   - Process frontal and lateral X-rays independently
   - Extract multi-scale features with rich representations

2. **2D Attention In Attention (AIA) Module**
   - Three-branch architecture: attention, non-attention, and dynamic weighting
   - Focuses on valuable features while ignoring noise

3. **Transformer Bridge**
   - Converts 2D features to 3D representation
   - Multi-head self-attention captures global context
   - Preserves anatomical details across the entire volume

4. **3D Decoder** with Upsampling
   - Reconstructs full 3D CT volume
   - Uses trilinear interpolation for quality or nearest neighbor for speed

5. **3D AIA Module**
   - Refines 3D features in the decoder
   - Balances quality and computational efficiency

6. **PatchGAN Discriminator**
   - Evaluates local patch realism
   - Provides better generalization than whole-image discrimination

### Loss Functions

- **Adversarial Loss (LSGAN)**: Stable GAN training
- **Reconstruction Loss (L1)**: Pixel-wise accuracy
- **Projection Loss (DRR-based)**: Multi-view consistency
- **Perceptual Loss (VGG16)**: Anatomical structure preservation

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- nibabel >= 3.2.0 (for NIfTI file handling)
- pyyaml, tqdm, einops
- matplotlib, seaborn (for visualization)

## 🚀 Quick Start

### 1. Data Preparation

Organize your data in the following structure:

```
data/
├── train/
│   ├── xray_frontal/
│   │   ├── sample_001.png
│   │   ├── sample_002.png
│   │   └── ...
│   ├── xray_lateral/
│   │   ├── sample_001.png
│   │   ├── sample_002.png
│   │   └── ...
│   └── ct_volumes/
│       ├── sample_001.nii.gz
│       ├── sample_002.nii.gz
│       └── ...
├── val/
│   └── (same structure as train)
└── test/
    └── (same structure as train)
```

**Image Requirements:**
- X-rays: 128×128 pixels (grayscale)
- CT volumes: 128×128×128 voxels (NIfTI format)

### 2. Configuration

Edit [config/config.yaml](config/config.yaml) to customize:
- Model architecture parameters
- Training hyperparameters
- Loss function weights
- Data paths
- Hardware settings

### 3. Training

```bash
python train.py --config config/config.yaml
```

**Resume from checkpoint:**
```bash
python train.py --config config/config.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

**Training parameters:**
- Optimizer: Adam (lr=4e-4, betas=[0.5, 0.999])
- Scheduler: Linear decay starting at epoch 50
- Epochs: 100
- Batch size: 4 (adjustable based on GPU memory)
- Normalization: Instance normalization
- Mixed precision: Supported (AMP)

### 4. Inference

Generate 3D CT from biplane X-rays:

```bash
python inference.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --frontal data/test/xray_frontal/sample.png \
    --lateral data/test/xray_lateral/sample.png \
    --output outputs/inference \
    --visualize
```

**With ground truth for evaluation:**
```bash
python inference.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --frontal data/test/xray_frontal/sample.png \
    --lateral data/test/xray_lateral/sample.png \
    --ground_truth data/test/ct_volumes/sample.nii.gz \
    --output outputs/inference \
    --visualize
```

## 📊 Model Architecture Details

### Generator Architecture

```
Input: Frontal X-ray (1×128×128), Lateral X-ray (1×128×128)
    ↓
[Dual 2D Encoders]
    ├─ Dense Blocks (64→128→256→512 channels)
    ├─ 2D AIA Module (bottleneck)
    └─ View Fusion
    ↓
[Transformer Bridge]
    ├─ 2D → 1D feature embedding
    ├─ Positional encoding
    ├─ Multi-head self-attention (8 heads, 6 layers)
    └─ 1D → 3D reconstruction
    ↓
[3D Decoder]
    ├─ 3D upsampling blocks (512→256→128→64→32)
    ├─ Skip connections (optional)
    ├─ 3D AIA Module
    └─ Output convolution
    ↓
Output: CT Volume (1×128×128×128)
```

### Discriminator Architecture

```
PatchGAN Discriminator (3D)
    Input: CT Volume (1×128×128×128)
    ↓
    Conv3D layers (64→128→256→512)
    ↓
    Output: Patch-wise scores (B×1×8×8×8)
```

## 🎯 Key Features

### 1. Attention In Attention (AIA) Modules

**2D AIA (Encoder):**
- Channel attention + Spatial attention
- Dynamic weight generation for attention/non-attention balance
- Reduces noise while preserving important features

**3D AIA (Decoder):**
- 3D spatial and channel attention
- Flexible upsampling (trilinear for quality, nearest for speed)
- Refines 3D structural details

### 2. Transformer for Global Context

- Converts 2D features to 3D with global receptive field
- Captures long-range anatomical dependencies
- Preserves spatial information through positional encoding

### 3. Multi-Loss Training

| Loss Type | Weight (λ) | Purpose |
|-----------|-----------|---------|
| Adversarial | 1.0 | Realism |
| Reconstruction | 10.0 | Voxel accuracy |
| Projection | 5.0 | View consistency |
| Perceptual | 1.0 | Structure preservation |

### 4. Training Optimizations

- **Mixed Precision (AMP)**: Faster training with reduced memory
- **Gradient Clipping**: Stabilizes training
- **Linear LR Decay**: Smooth convergence
- **Instance Normalization**: Better for medical images
- **TensorBoard Logging**: Real-time monitoring

## 📈 Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

**Logged Metrics:**
- Generator/Discriminator losses
- Individual loss components (adversarial, reconstruction, projection, perceptual)
- Learning rates
- Validation metrics

## 🧪 Testing Components

Test individual modules:

```bash
# Test AIA modules
python models/aia_modules.py

# Test Transformer
python models/transformer.py

# Test Generator
python models/generator.py

# Test Discriminator
python models/discriminator.py

# Test Loss functions
python models/losses.py

# Test Dataset
python utils/dataset.py
```

## 📁 Project Structure

```
trct_gan/
├── config/
│   └── config.yaml           # Configuration file
├── models/
│   ├── __init__.py
│   ├── aia_modules.py        # 2D/3D AIA modules
│   ├── transformer.py        # Transformer module
│   ├── generator.py          # TRCT Generator
│   ├── discriminator.py      # PatchGAN Discriminator
│   └── losses.py             # Loss functions
├── utils/
│   ├── __init__.py
│   ├── dataset.py            # Dataset loader
│   └── utils.py              # Utility functions
├── train.py                  # Training script
├── inference.py              # Inference script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔧 Customization

### Modify Architecture

Edit [config/config.yaml](config/config.yaml):

```yaml
model:
  generator:
    encoder_channels: [64, 128, 256, 512]  # Encoder depth
    decoder_channels: [256, 128, 64, 32]   # Decoder depth
    transformer:
      embed_dim: 512      # Feature dimension
      num_heads: 8        # Attention heads
      num_layers: 6       # Transformer depth
```

### Adjust Loss Weights

```yaml
loss:
  lambda_adv: 1.0           # Adversarial loss weight
  lambda_recon: 10.0        # Reconstruction loss weight
  lambda_proj: 5.0          # Projection loss weight
  lambda_perceptual: 1.0    # Perceptual loss weight
```

### Hardware Configuration

```yaml
hardware:
  device: "cuda"              # or "cpu"
  gpu_ids: [0]                # Multi-GPU support
  mixed_precision: true       # Enable AMP
```

## 📊 Evaluation Metrics

The inference script computes:

- **MAE** (Mean Absolute Error): Average voxel-wise difference
- **MSE** (Mean Squared Error): Squared voxel-wise difference
- **RMSE** (Root Mean Squared Error): Standard deviation of errors
- **PSNR** (Peak Signal-to-Noise Ratio): Image quality metric

## 💡 Tips for Best Results

1. **Data Quality**: High-quality, aligned X-ray pairs are crucial
2. **Preprocessing**: Consistent normalization and windowing
3. **Batch Size**: Larger batches stabilize GAN training (if GPU memory allows)
4. **Learning Rate**: Start with 4e-4, adjust if training is unstable
5. **Loss Weights**: Balance adversarial and reconstruction losses
6. **Training Duration**: 100 epochs is recommended, but monitor validation loss
7. **GPU Memory**: Use mixed precision if training on GPUs with <16GB VRAM

## 🎓 Understanding TRCT-GAN

**Analogy:** Think of TRCT-GAN as a master sculptor creating a 3D statue from two shadows:

- **Encoders (Eyes)**: Observe the shadows from two angles
- **AIA Modules (Focus)**: Concentrate on important details
- **Transformer (Mental Map)**: Understand how parts connect in 3D space
- **Decoder (Hands)**: Shape the 3D reconstruction
- **Discriminator (Critic)**: Compare with real statues to ensure quality

## 🐛 Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size in config
2. Enable mixed precision training
3. Reduce model size (fewer channels/layers)

### Training Instability

1. Lower learning rate
2. Increase gradient clipping
3. Adjust loss weights (reduce λ_adv)
4. Use spectral normalization in discriminator

### Poor Reconstruction Quality

1. Increase λ_recon and λ_proj
2. Train for more epochs
3. Check data quality and alignment
4. Verify normalization ranges

## 📚 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{trct-gan,
  title={TRCT-GAN: Transformer and GAN for CT Reconstruction from Biplane X-rays},
  author={Your Name},
  year={2024}
}
```


**Hardware Requirements:**
- **Minimum**: NVIDIA GPU with 8GB VRAM, 16GB RAM
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (e.g., V100, A100), 32GB+ RAM
- **Training Time**: ~2-3 days on a V100 for 100 epochs (dataset dependent)

**Note**: This implementation is based on the TRCT-GAN architecture described in the research literature. Adjust hyperparameters and architecture based on your specific dataset and computational resources.

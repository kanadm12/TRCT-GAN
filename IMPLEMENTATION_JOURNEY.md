# TRCT-GAN Implementation Journey

**Project**: Transformer and GAN for CT Reconstruction from Biplane X-rays  
**Duration**: December 24-25, 2025  
**Dataset**: 100 patients (70 train, 15 val, 15 test)  
**Platform**: RunPod GPU (NVIDIA GPU with CUDA)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Implemented](#architecture-implemented)
3. [Implementation Timeline](#implementation-timeline)
4. [Dataset Setup](#dataset-setup)
5. [Issues Encountered & Solutions](#issues-encountered--solutions)
6. [Training Configuration Evolution](#training-configuration-evolution)
7. [Metrics & Results](#metrics--results)
8. [What Worked](#what-worked)
9. [What Failed](#what-failed)
10. [Current Status & Next Steps](#current-status--next-steps)

---

## Project Overview

**Objective**: Reconstruct 3D CT volumes (128×128×128) from biplane X-ray DRR images (PA and Lateral views) using a GAN-based deep learning approach.

**Key Innovation**: Combination of:
- Dual 2D encoders with dense connections
- Attention-in-Attention (AIA) modules (both 2D and 3D)
- Transformer bridge for 2D→3D feature lifting
- 3D decoder with upsampling
- PatchGAN discriminator
- Multi-component loss function

---

## Architecture Implemented

### Generator Architecture

```
Input: Frontal X-ray (1×128×128), Lateral X-ray (1×128×128)
    ↓
[Dual 2D Encoders] - Dense connections
    ├─ Encoder Channels: [32, 64, 128, 256] (reduced from [64, 128, 256, 512])
    ├─ 2D AIA Module at bottleneck
    └─ View Fusion
    ↓
[Transformer Bridge]
    ├─ Embed dim: 256 (reduced from 512)
    ├─ Num heads: 8
    ├─ Num layers: 4 (reduced from 6)
    └─ 2D → 3D feature lifting
    ↓
[3D Decoder]
    ├─ Decoder channels: [256, 128, 64] (reduced from [512, 256, 128, 64])
    ├─ 3D AIA Module (enabled)
    ├─ Upsampling: nearest neighbor (for memory efficiency)
    └─ Output convolution
    ↓
Output: CT Volume (1×128×128×128)
```

**Total Parameters**:
- Generator: 73,945,508 parameters
- Discriminator: 11,046,977 parameters

### Discriminator Architecture
- Type: PatchGAN 3D
- Channels: [64, 128, 256, 512]
- Output: Patch-wise scores (B×1×8×8×8)

### Loss Functions

| Component | Type | Weight (λ) | Purpose |
|-----------|------|-----------|---------|
| Adversarial | LSGAN | 0.5 (reduced from 1.0) | Realism |
| Reconstruction | L1 | 20.0 (increased from 10.0) | Voxel accuracy |
| Projection | L1 | 10.0 (increased from 5.0) | View consistency |
| Perceptual | VGG16 | 2.0 (increased from 1.0) | Structure preservation |

---

## Implementation Timeline

### Day 1: December 24, 2025

**Initial Setup**
- Created complete model architecture (generator, discriminator, losses)
- Implemented AIA modules (2D and 3D)
- Set up transformer for 2D→3D lifting
- Created dataset loader
- Set up training loop with TensorBoard logging

**Repository Setup**
- Initialized GitHub repository: `https://github.com/kanadm12/TRCT-GAN.git`
- Pushed initial implementation

### Day 2: December 25, 2025

**Morning: Dataset Configuration**
- **Issue**: Dataset was organized differently than expected
  - Expected: Separate folders (xray_frontal/, xray_lateral/, ct_volumes/)
  - Actual: Patient-based folders (patient_id/patient_id.nii.gz, patient_id_pa_drr.png, patient_id_lat_drr.png)
- **Solution**: Updated dataset loader to handle patient-based directory structure
- **Created**: `split_dataset.py` script to split 100 patients into train/val/test (70/15/15)

**Afternoon: Memory Issues**
- **Issue**: Out of Memory (OOM) errors
  - Tried to allocate 216 GB on 140 GB GPU
  - Batch size 4 was too large
- **Solutions Applied**:
  1. Reduced batch size: 4 → 1
  2. Reduced encoder channels: [64,128,256,512] → [32,64,128,256]
  3. Reduced transformer: embed_dim 512→256, layers 6→4
  4. Reduced decoder channels: [512,256,128,64] → [256,128,64]
  5. Initially disabled 3D AIA (later re-enabled as it's a key feature)
  6. Changed upsampling: trilinear → nearest neighbor

**Evening: Training Issues & Fixes**

1. **Autocast Deprecation Warning**
   - **Issue**: `torch.cuda.amp.autocast()` deprecated
   - **Fix**: Updated to `torch.amp.autocast('cuda')`

2. **Inplace Operation Error**
   - **Issue**: `projection[i] = ...` caused gradient computation error
   - **Fix**: Used list comprehension and torch.cat instead of inplace modification

3. **Perceptual Loss Dimension Mismatch**
   - **Issue**: Projection function returned wrong dimensions
   - **Fix**: Properly handled dimension squeezing and channel management

4. **DRR Orientation Issue**
   - **Issue**: DRR images were vertically flipped (shoulder at bottom, chest at top)
   - **Fix**: Added `np.flipud()` in dataset loader
   - **Impact**: Improved initial SSIM from 0.47 → 0.60

**Night: First Training Run**
- Successfully trained for 30 epochs (~30-40 minutes)
- No crashes, stable training

---

## Dataset Setup

### Original Structure
```
/workspace/drr_patient_data/
├── patient_id_1/
│   ├── patient_id_1.nii.gz
│   ├── patient_id_1_pa_drr.png
│   └── patient_id_1_lat_drr.png
├── patient_id_2/
...
```

### Data Split
- **Total**: 100 patients
- **Train**: 70 patients
- **Validation**: 15 patients
- **Test**: 15 patients
- **Random seed**: 42 (for reproducibility)

### Data Processing Pipeline
1. Load DRR images (PA and Lateral)
2. Resize to 128×128
3. **Flip vertically** (critical for correct orientation)
4. Normalize to [-1, 1]
5. Load CT volumes (NIfTI format)
6. Resize to 128×128×128
7. Clip HU values: [-1000, 3000]
8. Normalize to [-1, 1]

---

## Issues Encountered & Solutions

### 1. Out of Memory (OOM) Errors
**Problem**: GPU running out of memory during training  
**Root Cause**: Model too large for available memory  
**Solutions**:
- ✅ Reduced batch size to 1
- ✅ Reduced model capacity (encoder/decoder channels)
- ✅ Reduced transformer size
- ✅ Disabled pin_memory
- ✅ Reduced num_workers
- ✅ Used nearest neighbor upsampling instead of trilinear

### 2. Deprecated API Warnings
**Problem**: `torch.cuda.amp.autocast()` deprecated  
**Solution**: Updated to `torch.amp.autocast('cuda')`

### 3. Gradient Computation Errors
**Problem**: Inplace operations breaking autograd  
**Root Cause**: `projection[i] = ...` modified tensors needing gradients  
**Solution**: Used non-inplace operations with list concatenation

### 4. Dimension Mismatches
**Problem**: Various dimension mismatches in loss functions and visualization  
**Solutions**:
- Fixed projection function to maintain proper dimensions
- Added proper squeezing in visualization functions
- Individual isinstance checks for tensor→numpy conversion

### 5. DRR Orientation
**Problem**: DRR images were upside down  
**Impact**: Poor initial training (SSIM ~0.47)  
**Solution**: Added vertical flip in dataset loader  
**Result**: Improved SSIM to 0.60 immediately

### 6. Inference Output Issues
**Problem**: Predicted CT nearly blank (1.5 MB file, very low values)  
**Root Cause**: Model outputs in [-1, 1] but saved without denormalization  
**Solution**: Added `denormalize_ct()` function to convert back to HU units [-1000, 3000]  
**Result**: Proper CT volumes with visible anatomy

### 7. Poor Initial Results
**Problem**: Generated CT almost uniform (mode collapse indication)  
**Root Cause**: 
- Too few training epochs (30)
- Small dataset (70 patients)
- Adversarial loss too strong
**Solution**: 
- Increased epochs: 30 → 100
- Adjusted loss weights (prioritize reconstruction)
- Continuing training

---

## Training Configuration Evolution

### Initial Configuration
```yaml
Training:
  batch_size: 4
  num_epochs: 100
  
Model:
  encoder_channels: [64, 128, 256, 512]
  decoder_channels: [512, 256, 128, 64]
  transformer:
    embed_dim: 512
    num_layers: 6
  aia_3d:
    enabled: true
    use_trilinear: true

Loss Weights:
  lambda_adv: 1.0
  lambda_recon: 10.0
  lambda_proj: 5.0
  lambda_perceptual: 1.0
```

### After OOM Fixes
```yaml
Training:
  batch_size: 1
  num_epochs: 30
  
Model:
  encoder_channels: [32, 64, 128, 256]
  decoder_channels: [256, 128, 64]
  transformer:
    embed_dim: 256
    num_layers: 4
  aia_3d:
    enabled: false  # Initially disabled
    use_trilinear: false
```

### Final Configuration (Current)
```yaml
Training:
  batch_size: 1
  num_epochs: 100
  decay_start_epoch: 50
  
Model:
  encoder_channels: [32, 64, 128, 256]
  decoder_channels: [256, 128, 64]
  transformer:
    embed_dim: 256
    num_layers: 4
  aia_3d:
    enabled: true  # Re-enabled (key feature)
    use_trilinear: false

Loss Weights:
  lambda_adv: 0.5      # Reduced
  lambda_recon: 20.0   # Doubled
  lambda_proj: 10.0    # Doubled
  lambda_perceptual: 2.0  # Doubled
```

---

## Metrics & Results

### Training Progress (30 Epochs)

**Epoch 0:**
- Train Loss G: 243.50, D: 2.68
- Val Loss G: 98.58, D: 0.33
- PSNR: 15.09 dB
- SSIM: 0.5978

**Epoch 2:**
- Train Loss G: 71.67, D: 0.28
- Val Loss G: 60.34, D: 0.30
- PSNR: 19.70 dB
- SSIM: 0.6229

**Epoch 11 (Best Model):**
- Train Loss G: 61.79, D: 0.08
- Val Loss G: 58.83, D: 0.06
- PSNR: 19.71 dB
- SSIM: 0.6300

**Epoch 18 (Best Model - Saved):**
- Train Loss G: 61.37, D: 0.02
- Val Loss G: 57.99, D: 0.04
- PSNR: 19.72 dB
- SSIM: 0.6319

**Epoch 29 (Final):**
- Train Loss G: 60.19, D: 0.0000
- Val Loss G: 58.63, D: 0.0000
- PSNR: 19.76 dB
- SSIM: 0.6315

### Test Set Inference (Patient 0038fd5f09f5)
```
MAE: 0.1840
MSE: 0.0546
RMSE: 0.2336
PSNR: 18.6470 dB
SSIM: 0.6041
```

### Training Statistics
- **Training time**: ~1 minute per epoch
- **Total training time**: ~30-40 minutes for 30 epochs
- **Training speed**: ~1.15 it/s (70 batches)
- **Validation speed**: ~1.10 s/it (15 batches)

### Observations
1. **Rapid initial improvement**: Loss dropped 75% in first 3 epochs
2. **Stable convergence**: Plateaued around epoch 10-15
3. **Discriminator collapse**: D loss → 0.0000 by epoch 29 (concerning)
4. **PSNR improvement**: 15.09 → 19.76 dB (+31%)
5. **SSIM stabilization**: 0.598 → 0.632 (+6%)

---

## What Worked

### ✅ Successful Implementations

1. **Architecture Design**
   - AIA modules effectively implemented (2D and 3D)
   - Transformer successfully bridges 2D→3D
   - PatchGAN discriminator works well

2. **Memory Optimizations**
   - Model size reduction allowed training on available GPU
   - Nearest neighbor upsampling saved memory without major quality loss
   - Batch size 1 enabled training

3. **Data Processing**
   - Patient-based dataset organization works well
   - Vertical flip correction was crucial
   - Normalization pipeline effective

4. **Training Infrastructure**
   - TensorBoard logging comprehensive
   - Checkpointing works reliably
   - Resume training capability functional
   - Mixed precision training stable

5. **Loss Functions**
   - Multi-component loss balances different objectives
   - VGG perceptual loss adds structure awareness
   - Projection loss improves view consistency

6. **Code Quality**
   - Modular architecture easy to modify
   - Good separation of concerns
   - Well-documented

---

## What Failed

### ❌ Issues & Limitations

1. **Insufficient Training**
   - **Problem**: Only 30 epochs with 70 patients
   - **Evidence**: Generated CT nearly uniform/blank
   - **Impact**: Model hasn't learned anatomical structures properly
   - **Status**: Addressed by increasing to 100 epochs

2. **Mode Collapse Indicators**
   - **Problem**: Discriminator loss → 0, Generator producing uniform outputs
   - **Evidence**: Predicted CT almost entirely white/uniform
   - **Root Cause**: Adversarial loss too strong, discriminator overpowering generator
   - **Status**: Adjusted loss weights, continuing training

3. **Small Dataset**
   - **Problem**: Only 70 training samples
   - **Impact**: Limited generalization capability
   - **Typical Requirement**: 1000+ samples for robust medical imaging GANs
   - **Status**: No immediate solution (dataset limitation)

4. **Model Capacity Reduction**
   - **Problem**: Had to significantly reduce model size for memory
   - **Impact**: Reduced learning capacity
   - **Trade-off**: Necessary for training but limits potential performance

5. **Initial Configuration Issues**
   - **Problem**: DRR images flipped, loss weights not optimal
   - **Impact**: Wasted initial training epochs
   - **Status**: Fixed, but lost some training time

6. **Inference Output Quality**
   - **Problem**: Predicted CT lacks anatomical detail
   - **Evidence**: Nearly uniform output, high absolute difference
   - **Root Cause**: Insufficient training + small dataset
   - **Status**: Requires more training

7. **Discriminator Dominance**
   - **Problem**: Discriminator became too strong (loss → 0)
   - **Impact**: Generator may have stopped learning effectively
   - **Status**: Reduced adversarial loss weight to 0.5

---

## Current Status & Next Steps

### Current Status (End of Day 2)

**✅ Completed:**
- Full architecture implementation
- Training pipeline functional
- 30 epochs completed successfully
- Inference pipeline working
- Comprehensive logging and visualization
- All critical bugs fixed

**⚠️ In Progress:**
- Extended training to 100 epochs (recommended running now)
- Loss weight rebalancing applied

**❌ Needs Improvement:**
- Generated CT quality (too uniform)
- Dataset size (limited by available data)
- Model capacity (limited by GPU memory)

### Immediate Next Steps

1. **Continue Training** (Priority 1)
   ```bash
   cd /workspace/TRCT-GAN
   git pull
   python train.py --config config/config.yaml --resume checkpoints/best_model.pth
   ```
   - Target: 100 epochs total (70 more epochs)
   - Expected time: ~1.5 hours
   - Expected improvement: PSNR 22-25+ dB, SSIM 0.75+

2. **Monitor Training**
   ```bash
   tensorboard --logdir logs --bind_all --port 6006
   ```
   - Watch for mode collapse recovery
   - Check anatomical structure emergence
   - Monitor loss balance

3. **Evaluate at Checkpoints**
   - Run inference every 10 epochs
   - Compare visual quality progression
   - Check for overfitting

### Future Improvements

**Short-term (If continuing project):**
1. Increase dataset size (acquire more patients)
2. Data augmentation (random flips, rotations, brightness)
3. Progressive training (start with lower resolution)
4. Gradient penalty for discriminator stability
5. Feature matching loss

**Long-term (Research directions):**
1. Larger model on better hardware (A100/H100)
2. Multi-scale architecture
3. Attention mechanisms in decoder
4. Cycle consistency losses
5. Uncertainty quantification
6. Clinical validation with radiologists

---

## Lessons Learned

### Technical Insights

1. **Memory Management Critical**: For 3D medical imaging, memory is the primary constraint
2. **Orientation Matters**: Small preprocessing errors (vertical flip) have major impact
3. **Loss Balancing Crucial**: GAN training extremely sensitive to loss weights
4. **Small Datasets Challenging**: 70 samples insufficient for complex 3D generation
5. **Denormalization Essential**: Always verify output units match expected ranges

### Best Practices Established

1. **Incremental Development**: Build, test, fix one component at a time
2. **Comprehensive Logging**: TensorBoard + image visualization essential
3. **Checkpoint Frequently**: Resume capability saved time after issues
4. **Version Control**: Git commits tracked all changes effectively
5. **Documentation**: Inline comments and READMEs helpful for debugging

### GAN Training Specific

1. **Start Conservative**: Lower adversarial loss weight initially
2. **Monitor Both Losses**: If D → 0, generator may stop learning
3. **Visual Inspection**: Metrics don't tell whole story, look at outputs
4. **Patience Required**: GANs need many epochs, especially for complex tasks
5. **Hardware Matters**: 3D GANs extremely memory-hungry

---

## File Structure Summary

```
trct_gan/
├── models/
│   ├── generator.py          # TRCT Generator with AIA + Transformer
│   ├── discriminator.py      # PatchGAN 3D discriminator
│   ├── aia_modules.py        # 2D and 3D AIA implementations
│   ├── transformer.py        # 2D→3D transformer bridge
│   └── losses.py             # Multi-component loss functions
├── utils/
│   ├── dataset.py            # Patient-based dataset loader
│   └── utils.py              # Metrics, visualization, utilities
├── config/
│   └── config.yaml           # Training configuration
├── train.py                  # Main training script
├── inference.py              # Inference pipeline
├── split_dataset.py          # Dataset splitting utility
├── requirements.txt          # Dependencies
├── README.md                 # Project overview
└── IMPLEMENTATION_JOURNEY.md # This document
```

---

## Key Commits & Changes

1. **Initial commit** (Dec 24): Full architecture implementation
2. **Dataset structure update**: Patient-based organization
3. **Memory optimizations**: Reduced model size
4. **Vertical flip fix**: Corrected DRR orientation (+SSIM)
5. **Autocast update**: Fixed deprecation warnings
6. **Inplace operation fix**: Resolved gradient errors
7. **Denormalization**: Proper HU unit conversion
8. **Loss rebalancing**: Prioritize reconstruction (current)

---

## Conclusion

This implementation demonstrates a **functional but undertrained** TRCT-GAN system. The architecture is solid, the training infrastructure works well, and all major bugs have been resolved. However, the model requires significantly more training (70+ additional epochs) to generate clinically meaningful CT reconstructions.

The main limitation is the small dataset (70 training patients), which is far below the typical requirement for medical imaging GANs (1000+ samples). Despite this, the model shows learning capability with PSNR improving from 15 to 19.7 dB.

**Current recommendation**: Continue training to 100 epochs with adjusted loss weights and evaluate results. If quality remains insufficient, consider data augmentation or acquiring additional training data.

---

**Last Updated**: January 5, 2026  
**Status**: Training in progress (30/100 epochs completed)  
**Repository**: https://github.com/kanadm12/TRCT-GAN.git

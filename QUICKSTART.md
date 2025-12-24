# TRCT-GAN Quick Start Guide

Welcome to TRCT-GAN! This guide will help you get started quickly.

## 📦 Installation

### Step 1: Clone or Navigate to the Project

```bash
cd trct_gan
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Minimum Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA (for GPU training)

### Step 3: Verify Installation

```bash
python test_installation.py
```

This will test all components and verify your setup. ✓ All tests should pass!

## 🎯 Quick Training Example

### 1. Prepare Your Data

Create the following directory structure:

```
data/
├── train/
│   ├── xray_frontal/    # Frontal X-ray PNGs (128×128)
│   ├── xray_lateral/    # Lateral X-ray PNGs (128×128)
│   └── ct_volumes/      # CT NIfTI files (128³)
└── val/
    └── (same structure)
```

**Data Format:**
- X-rays: `.png` or `.jpg` (grayscale, 128×128 pixels)
- CT: `.nii.gz` or `.nii` (128×128×128 voxels)
- **Matching names**: `sample_001.png` ↔ `sample_001.nii.gz`

### 2. Start Training

```bash
python train.py --config config/config.yaml
```

Training will:
- Save checkpoints every 5 epochs to `checkpoints/`
- Log metrics to TensorBoard in `logs/`
- Save best model based on validation loss

### 3. Monitor Progress

```bash
tensorboard --logdir logs
```

Open http://localhost:6006 in your browser to view:
- Loss curves
- Learning rate schedules
- Validation metrics

## 🔮 Quick Inference Example

Generate CT from X-rays:

```bash
python inference.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --frontal data/test/xray_frontal/patient001.png \
    --lateral data/test/xray_lateral/patient001.png \
    --output outputs/patient001 \
    --visualize
```

**Output:**
- `predicted_ct.nii.gz` - 3D CT volume
- `ct_slices.png` - Visualization of CT slices
- `comparison.png` - Input X-rays vs output CT

## ⚙️ Basic Configuration

Edit `config/config.yaml` for quick adjustments:

### Reduce Memory Usage

```yaml
training:
  batch_size: 2  # Reduce from 4 to 2
  
hardware:
  mixed_precision: true  # Enable AMP
```

### Adjust Training Duration

```yaml
training:
  num_epochs: 50  # Reduce from 100
```

### Change Loss Weights

```yaml
loss:
  lambda_recon: 20.0  # Increase reconstruction importance
  lambda_adv: 0.5     # Decrease adversarial importance
```

## 🐛 Common Issues

### Issue: Out of Memory

**Solution:**
1. Reduce batch size in config (e.g., 2 or 1)
2. Enable mixed precision: `mixed_precision: true`
3. Use smaller model (reduce channels)

### Issue: Training is Unstable

**Solution:**
1. Lower learning rate: `lr: 0.0002` (from 0.0004)
2. Increase gradient clipping: `gradient_clip: 2.0`
3. Reduce adversarial loss weight: `lambda_adv: 0.5`

### Issue: Poor Reconstruction Quality

**Solution:**
1. Train longer (100+ epochs)
2. Increase reconstruction loss: `lambda_recon: 20.0`
3. Check data quality and alignment
4. Verify normalization ranges

### Issue: CUDA Out of Memory

**Solution:**
```yaml
training:
  batch_size: 1
  
hardware:
  mixed_precision: true
```

## 📊 Understanding Output

### Training Logs

```
Epoch 10:
  Train - G: 0.3245, D: 0.1234
  Val   - G: 0.3456, D: 0.1345
  ✓ Best model saved (val_loss: 0.3456)
```

- **G**: Generator loss (lower is better)
- **D**: Discriminator loss (should stabilize around 0.1-0.3)

### Inference Output

```
Evaluation Metrics:
  MAE: 0.0234   ← Average absolute error
  MSE: 0.0012   ← Mean squared error
  RMSE: 0.0346  ← Root mean squared error
  PSNR: 34.56   ← Image quality (higher is better)
```

## 🎓 Next Steps

### 1. Fine-tune Hyperparameters

Experiment with:
- Loss weights (`lambda_*`)
- Learning rates
- Model architecture (channels, layers)

### 2. Add Your Own Data Preprocessing

Modify `utils/dataset.py` to:
- Custom windowing
- Additional augmentations
- Different normalization

### 3. Customize Architecture

Edit `config/config.yaml`:
- Change encoder/decoder depths
- Adjust transformer parameters
- Modify attention mechanisms

### 4. Implement Custom Loss

Add your loss function in `models/losses.py` and integrate in `TRCTGANLoss`

## 📚 Additional Resources

- **Full Documentation**: See [README.md](README.md)
- **Architecture Details**: See [README.md#model-architecture-details](README.md#-model-architecture-details)
- **Test Components**: Run `python models/<module>.py` for any module

## 💡 Tips for Success

1. **Start Small**: Test with a few samples first
2. **Monitor Early**: Check TensorBoard after first epoch
3. **Save Often**: Checkpoints every 5 epochs by default
4. **Validate Often**: Run validation every epoch
5. **Be Patient**: Good results may take 50-100 epochs

## 🎯 Typical Workflow

```bash
# 1. Prepare data
# (organize your X-rays and CT volumes)

# 2. Test installation
python test_installation.py

# 3. Start training
python train.py --config config/config.yaml

# 4. Monitor (in another terminal)
tensorboard --logdir logs

# 5. After training completes, run inference
python inference.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --frontal test_frontal.png \
    --lateral test_lateral.png \
    --output results \
    --visualize
```

## ❓ Getting Help

1. **Check the logs**: Look for error messages
2. **Run tests**: `python test_installation.py`
3. **Test individual modules**: `python models/generator.py`
4. **Review configuration**: Ensure paths and parameters are correct

## 🚀 Ready to Go!

You're all set! Start with:

```bash
python train.py --config config/config.yaml
```

Good luck with your CT reconstruction! 🎉

---

**Quick Reference:**

| Task | Command |
|------|---------|
| Train | `python train.py --config config/config.yaml` |
| Resume | `python train.py --config config/config.yaml --resume checkpoints/xxx.pth` |
| Infer | `python inference.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --frontal x.png --lateral y.png --output out/` |
| Monitor | `tensorboard --logdir logs` |
| Test | `python test_installation.py` |

# TRCT-GAN Command Reference

Quick reference for common commands and workflows.

## 🚀 Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

## 🏋️ Training

### Basic Training

```bash
# Start training with default config
python train.py --config config/config.yaml
```

### Resume Training

```bash
# Resume from a checkpoint
python train.py --config config/config.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

### Monitor Training

```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir logs

# Then open in browser: http://localhost:6006
```

## 🔮 Inference

### Basic Inference

```bash
python inference.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --frontal path/to/frontal.png \
    --lateral path/to/lateral.png \
    --output outputs/result
```

### Inference with Visualization

```bash
python inference.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --frontal path/to/frontal.png \
    --lateral path/to/lateral.png \
    --output outputs/result \
    --visualize
```

### Inference with Ground Truth (for evaluation)

```bash
python inference.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --frontal path/to/frontal.png \
    --lateral path/to/lateral.png \
    --ground_truth path/to/ct.nii.gz \
    --output outputs/result \
    --visualize
```

## 🧪 Testing

### Test Installation

```bash
python test_installation.py
```

### Test Individual Components

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

## ⚙️ Configuration Quick Edits

### Reduce Memory Usage

Edit `config/config.yaml`:

```yaml
training:
  batch_size: 2  # or 1
  
hardware:
  mixed_precision: true
```

### Adjust Learning Rate

```yaml
training:
  optimizer:
    generator:
      lr: 0.0002  # reduce from 0.0004
    discriminator:
      lr: 0.0002
```

### Change Loss Weights

```yaml
loss:
  lambda_adv: 1.0
  lambda_recon: 20.0  # increase reconstruction importance
  lambda_proj: 5.0
  lambda_perceptual: 1.0
```

### Reduce Training Time

```yaml
training:
  num_epochs: 50  # reduce from 100
  save_freq: 10   # save less frequently
```

## 📊 Common Workflows

### Full Training Pipeline

```bash
# 1. Prepare data (organize into correct folders)

# 2. Test setup
python test_installation.py

# 3. Start training
python train.py --config config/config.yaml

# 4. Monitor (in another terminal)
tensorboard --logdir logs

# 5. Wait for training to complete...

# 6. Run inference on test data
python inference.py \
    --config config/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --frontal test_frontal.png \
    --lateral test_lateral.png \
    --output results \
    --visualize
```

### Hyperparameter Tuning Workflow

```bash
# 1. Create config variants
cp config/config.yaml config/config_v1.yaml
# Edit config_v1.yaml with different hyperparameters

# 2. Train with different configs
python train.py --config config/config_v1.yaml

# 3. Compare results in TensorBoard
tensorboard --logdir logs
```

### Evaluation Workflow

```bash
# Run inference on test set with ground truth
for sample in test_samples:
    python inference.py \
        --config config/config.yaml \
        --checkpoint checkpoints/best_model.pth \
        --frontal test/frontal/${sample}.png \
        --lateral test/lateral/${sample}.png \
        --ground_truth test/ct/${sample}.nii.gz \
        --output results/${sample} \
        --visualize
done

# Metrics will be saved in results/${sample}/metrics.txt
```

## 🐛 Troubleshooting Commands

### Check CUDA availability

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Check GPU memory

```python
python -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB' if torch.cuda.is_available() else 'No CUDA')"
```

### Test model on dummy data

```python
python -c "
import torch
from models import TRCTGenerator

gen = TRCTGenerator()
x_f = torch.randn(1, 1, 128, 128)
x_l = torch.randn(1, 1, 128, 128)

with torch.no_grad():
    out = gen(x_f, x_l)
    
print(f'Input: {x_f.shape}, Output: {out.shape}')
print('✓ Generator working!')
"
```

### Check checkpoint contents

```python
python -c "
import torch
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
print('Checkpoint keys:', checkpoint.keys())
print('Epoch:', checkpoint.get('epoch', 'N/A'))
"
```

## 📁 Directory Structure

```
trct_gan/
├── config/config.yaml          # Main configuration
├── train.py                    # Training script
├── inference.py                # Inference script
├── test_installation.py        # Test suite
├── models/                     # Model implementations
├── utils/                      # Utilities
├── checkpoints/                # Saved checkpoints (created during training)
├── logs/                       # TensorBoard logs (created during training)
└── outputs/                    # Inference outputs (created during inference)
```

## 🎯 Quick Reference Table

| Task | Command |
|------|---------|
| **Install** | `pip install -r requirements.txt` |
| **Test** | `python test_installation.py` |
| **Train** | `python train.py --config config/config.yaml` |
| **Resume** | `python train.py --config config/config.yaml --resume checkpoints/xxx.pth` |
| **Monitor** | `tensorboard --logdir logs` |
| **Infer** | `python inference.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --frontal x.png --lateral y.png --output out/` |
| **Infer+Viz** | Add `--visualize` to inference command |
| **Infer+Eval** | Add `--ground_truth ct.nii.gz` to inference command |

## 💡 Pro Tips

### Speed up training
```yaml
hardware:
  mixed_precision: true  # Use AMP
  
training:
  num_workers: 4         # Parallel data loading
  pin_memory: true       # Faster GPU transfer
```

### Improve stability
```yaml
training:
  gradient_clip: 1.0     # Prevent exploding gradients
  
loss:
  lambda_adv: 0.5        # Reduce adversarial weight if unstable
```

### Better reconstruction
```yaml
loss:
  lambda_recon: 20.0     # Increase reconstruction weight
  lambda_proj: 10.0      # Increase projection weight
```

### Debug mode (fast iteration)
```yaml
training:
  batch_size: 1
  num_epochs: 5
  save_freq: 1
  val_freq: 1
```

## 🔗 File Paths

**Training outputs:**
- Checkpoints: `checkpoints/checkpoint_epoch_*.pth`
- Best model: `checkpoints/best_model.pth`
- Logs: `logs/events.out.tfevents.*`

**Inference outputs:**
- CT volume: `outputs/<name>/predicted_ct.nii.gz`
- Visualizations: `outputs/<name>/ct_slices.png`, `comparison.png`
- Metrics: `outputs/<name>/metrics.txt`

## 📞 Getting Help

1. Check documentation: `README.md`, `QUICKSTART.md`
2. Run tests: `python test_installation.py`
3. Test specific component: `python models/<module>.py`
4. Check logs: `logs/` directory
5. Review configuration: `config/config.yaml`

---

**Last Updated**: 2024
**Version**: 1.0.0

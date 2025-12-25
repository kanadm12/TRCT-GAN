"""
Training Script for TRCT-GAN
Implements complete training loop with:
- Adam optimizer with learning rate scheduling
- Mixed precision training (optional)
- Checkpoint saving and loading
- TensorBoard logging
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models import TRCTGenerator, PatchGANDiscriminator3D, TRCTGANLoss
from utils.dataset import XRayCTDataset
from utils.utils import save_checkpoint, load_checkpoint, AverageMeter


class LinearLRScheduler:
    """
    Linear learning rate scheduler that decays to zero
    """
    def __init__(self, optimizer, total_epochs, decay_start_epoch):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.decay_start_epoch = decay_start_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.decay_start_epoch:
            return
        
        # Linear decay to zero
        decay_ratio = (epoch - self.decay_start_epoch) / (self.total_epochs - self.decay_start_epoch)
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * (1 - decay_ratio)


class Trainer:
    """
    TRCT-GAN Trainer
    """
    
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(self.config['hardware']['device'] 
                                   if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(self.config['logging']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['output_dir'], exist_ok=True)
        
        # Initialize models
        self.build_models()
        
        # Initialize optimizers
        self.build_optimizers()
        
        # Initialize loss
        self.criterion = TRCTGANLoss(self.config['loss']).to(self.device)
        
        # Initialize data loaders
        self.build_dataloaders()
        
        # Initialize logging
        if self.config['logging']['tensorboard']:
            self.writer = SummaryWriter(log_dir=self.config['logging']['log_dir'])
        else:
            self.writer = None
        
        # Mixed precision training
        self.use_amp = self.config['hardware'].get('mixed_precision', False)
        if self.use_amp:
            self.scaler_G = GradScaler()
            self.scaler_D = GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def build_models(self):
        """Initialize generator and discriminator"""
        print("Building models...")
        
        # Generator
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
        
        # Discriminator
        self.discriminator = PatchGANDiscriminator3D(
            in_channels=1,
            channels=self.config['model']['discriminator']['channels'],
            num_layers=self.config['model']['discriminator']['num_layers'],
            use_spectral_norm=self.config['model']['discriminator']['use_spectral_norm']
        ).to(self.device)
        
        # Count parameters
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Generator parameters: {gen_params:,}")
        print(f"Discriminator parameters: {disc_params:,}")
    
    def build_optimizers(self):
        """Initialize optimizers and schedulers"""
        print("Building optimizers...")
        
        # Generator optimizer
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config['training']['optimizer']['generator']['lr'],
            betas=self.config['training']['optimizer']['generator']['betas'],
            weight_decay=self.config['training']['optimizer']['generator']['weight_decay']
        )
        
        # Discriminator optimizer
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['training']['optimizer']['discriminator']['lr'],
            betas=self.config['training']['optimizer']['discriminator']['betas'],
            weight_decay=self.config['training']['optimizer']['discriminator']['weight_decay']
        )
        
        # Learning rate schedulers
        total_epochs = self.config['training']['num_epochs']
        decay_start = self.config['training']['scheduler']['decay_start_epoch']
        
        self.scheduler_G = LinearLRScheduler(self.optimizer_G, total_epochs, decay_start)
        self.scheduler_D = LinearLRScheduler(self.optimizer_D, total_epochs, decay_start)
    
    def build_dataloaders(self):
        """Initialize data loaders"""
        print("Building data loaders...")
        
        # Training dataset
        train_dataset = XRayCTDataset(
            data_path=self.config['dataset']['train_data_path'],
            augmentation=self.config['dataset']['augmentation'],
            normalize=self.config['dataset']['normalize']
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory']
        )
        
        # Validation dataset
        val_dataset = XRayCTDataset(
            data_path=self.config['dataset']['val_data_path'],
            augmentation={'enabled': False},
            normalize=self.config['dataset']['normalize']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory']
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        # Metrics
        losses_G = AverageMeter()
        losses_D = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            xray_frontal = batch['xray_frontal'].to(self.device)
            xray_lateral = batch['xray_lateral'].to(self.device)
            ct_real = batch['ct_volume'].to(self.device)
            
            batch_size = xray_frontal.size(0)
            
            # ==================== Train Discriminator ====================
            self.optimizer_D.zero_grad()
            
            if self.use_amp:
                with autocast():
                    # Generate fake CT
                    ct_fake = self.generator(xray_frontal, xray_lateral)
                    
                    # Discriminator predictions
                    pred_real = self.discriminator(ct_real)
                    pred_fake = self.discriminator(ct_fake.detach())
                    
                    # Discriminator loss
                    loss_D = self.criterion.discriminator_loss(pred_real, pred_fake)
                
                self.scaler_D.scale(loss_D).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler_D.unscale_(self.optimizer_D)
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler_D.step(self.optimizer_D)
                self.scaler_D.update()
            else:
                # Generate fake CT
                ct_fake = self.generator(xray_frontal, xray_lateral)
                
                # Discriminator predictions
                pred_real = self.discriminator(ct_real)
                pred_fake = self.discriminator(ct_fake.detach())
                
                # Discriminator loss
                loss_D = self.criterion.discriminator_loss(pred_real, pred_fake)
                loss_D.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer_D.step()
            
            # ==================== Train Generator ====================
            self.optimizer_G.zero_grad()
            
            if self.use_amp:
                with autocast():
                    # Generate fake CT
                    ct_fake = self.generator(xray_frontal, xray_lateral)
                    
                    # Discriminator prediction on fake
                    pred_fake = self.discriminator(ct_fake)
                    
                    # Generator loss
                    loss_G, loss_dict = self.criterion.generator_loss(
                        ct_fake, ct_real, pred_fake
                    )
                
                self.scaler_G.scale(loss_G).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler_G.unscale_(self.optimizer_G)
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler_G.step(self.optimizer_G)
                self.scaler_G.update()
            else:
                # Generate fake CT
                ct_fake = self.generator(xray_frontal, xray_lateral)
                
                # Discriminator prediction on fake
                pred_fake = self.discriminator(ct_fake)
                
                # Generator loss
                loss_G, loss_dict = self.criterion.generator_loss(
                    ct_fake, ct_real, pred_fake
                )
                loss_G.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer_G.step()
            
            # Update metrics
            losses_G.update(loss_G.item(), batch_size)
            losses_D.update(loss_D.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{losses_G.avg:.4f}',
                'D': f'{losses_D.avg:.4f}'
            })
            
            # Logging
            if self.writer and batch_idx % self.config['logging']['log_freq'] == 0:
                self.writer.add_scalar('Train/Loss_G', loss_G.item(), self.global_step)
                self.writer.add_scalar('Train/Loss_D', loss_D.item(), self.global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train/Loss_G_{key}', value, self.global_step)
            
            self.global_step += 1
        
        return losses_G.avg, losses_D.avg
    
    @torch.no_grad()
    def validate(self):
        """Validation with PSNR and SSIM metrics"""
        from utils.utils import compute_metrics
        
        self.generator.eval()
        self.discriminator.eval()
        
        losses_G = AverageMeter()
        losses_D = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                xray_frontal = batch['xray_frontal'].to(self.device)
                xray_lateral = batch['xray_lateral'].to(self.device)
                ct_real = batch['ct_volume'].to(self.device)
                
                batch_size = xray_frontal.size(0)
                
                # Generate fake CT
                ct_fake = self.generator(xray_frontal, xray_lateral)
                
                # Discriminator predictions
                pred_real = self.discriminator(ct_real)
                pred_fake = self.discriminator(ct_fake)
                
                # Losses
                loss_D = self.criterion.discriminator_loss(pred_real, pred_fake)
                loss_G, loss_dict = self.criterion.generator_loss(ct_fake, ct_real, pred_fake)
                
                losses_G.update(loss_G.item(), batch_size)
                losses_D.update(loss_D.item(), batch_size)
                
                # Compute PSNR and SSIM for each sample in batch
                for i in range(batch_size):
                    metrics = compute_metrics(ct_fake[i], ct_real[i])
                    psnr_meter.update(metrics['PSNR'], 1)
                    ssim_meter.update(metrics['SSIM'], 1)
        
        # Log metrics to tensorboard
        if self.writer:
            self.writer.add_scalar('Val/PSNR', psnr_meter.avg, self.current_epoch)
            self.writer.add_scalar('Val/SSIM', ssim_meter.avg, self.current_epoch)
        
        print(f"\n  Val Loss G: {losses_G.avg:.4f}, Val Loss D: {losses_D.avg:.4f}")
        print(f"  PSNR: {psnr_meter.avg:.2f} dB, SSIM: {ssim_meter.avg:.4f}")
        
        return losses_G.avg, losses_D.avg
    
    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        print(f"Total epochs: {self.config['training']['num_epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss_G, train_loss_D = self.train_epoch()
            
            # Update learning rates
            self.scheduler_G.step(epoch)
            self.scheduler_D.step(epoch)
            
            # Validate
            if epoch % self.config['training']['val_freq'] == 0:
                val_loss_G, val_loss_D = self.validate()
                
                print(f"\nEpoch {epoch}:")
                print(f"  Train - G: {train_loss_G:.4f}, D: {train_loss_D:.4f}")
                print(f"  Val   - G: {val_loss_G:.4f}, D: {val_loss_D:.4f}")
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Val/Loss_G', val_loss_G, epoch)
                    self.writer.add_scalar('Val/Loss_D', val_loss_D, epoch)
                    self.writer.add_scalar('LR/Generator', 
                                         self.optimizer_G.param_groups[0]['lr'], epoch)
                    self.writer.add_scalar('LR/Discriminator', 
                                         self.optimizer_D.param_groups[0]['lr'], epoch)
                
                # Save best model
                if val_loss_G < best_val_loss:
                    best_val_loss = val_loss_G
                    self.save_checkpoint('best_model.pth')
                    print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
            
            # Save checkpoint
            if epoch % self.config['training']['save_freq'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("="*50)
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, filename):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_G_state_dict'] = self.scaler_G.state_dict()
            checkpoint['scaler_D_state_dict'] = self.scaler_D.state_dict()
        
        save_path = os.path.join(self.config['logging']['checkpoint_dir'], filename)
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        if self.use_amp and 'scaler_G_state_dict' in checkpoint:
            self.scaler_G.load_state_dict(checkpoint['scaler_G_state_dict'])
            self.scaler_D.load_state_dict(checkpoint['scaler_D_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")


def main():
    parser = argparse.ArgumentParser(description='Train TRCT-GAN')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()

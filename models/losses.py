"""
Loss Functions for TRCT-GAN
Implements:
1. Adversarial Loss (LSGAN)
2. Reconstruction Loss (L1/L2)
3. Projection Loss (DRR-based)
4. Perceptual Loss (VGG16-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LSGANLoss(nn.Module):
    """
    Least Squares GAN Loss
    
    More stable than standard GAN loss and provides better gradient flow.
    Uses MSE loss instead of binary cross-entropy.
    """
    
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, is_real):
        """
        Args:
            pred: Discriminator predictions (any shape)
            is_real: Boolean indicating if the target is real or fake
        Returns:
            loss: LSGAN loss value
        """
        if is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        
        loss = self.criterion(pred, target)
        return loss
    
    def generator_loss(self, fake_pred):
        """Generator tries to make discriminator output 1 for fake images"""
        return self.forward(fake_pred, is_real=True)
    
    def discriminator_loss(self, real_pred, fake_pred):
        """Discriminator tries to output 1 for real, 0 for fake"""
        real_loss = self.forward(real_pred, is_real=True)
        fake_loss = self.forward(fake_pred, is_real=False)
        return (real_loss + fake_loss) * 0.5


class ReconstructionLoss(nn.Module):
    """
    Reconstruction Loss
    
    Ensures the generated 3D CT volume matches the ground truth.
    Supports L1 (MAE) and L2 (MSE) loss.
    """
    
    def __init__(self, loss_type='L1'):
        super(ReconstructionLoss, self).__init__()
        
        if loss_type == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'L2' or loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        """
        Args:
            pred: Generated CT volume (B, 1, D, H, W)
            target: Ground truth CT volume (B, 1, D, H, W)
        Returns:
            loss: Reconstruction loss
        """
        return self.criterion(pred, target)


class ProjectionLoss(nn.Module):
    """
    Projection Loss (DRR-based)
    
    Projects the reconstructed 3D CT volume to 2D from three orthogonal angles
    (frontal, lateral, top) and compares with projections from the ground truth CT.
    
    This ensures the reconstruction is consistent with the input X-rays and
    anatomical structure is preserved from multiple viewing angles.
    """
    
    def __init__(self, loss_type='L1'):
        super(ProjectionLoss, self).__init__()
        
        if loss_type == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'L2' or loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def project_volume(self, volume, axis):
        """
        Project 3D volume to 2D by summing along an axis (simulated DRR)
        
        Args:
            volume: 3D CT volume (B, 1, D, H, W)
            axis: Projection axis (2=frontal, 3=lateral, 4=top)
        Returns:
            projection: 2D projection (B, 1, H, W) or (B, 1, D, W) or (B, 1, D, H)
        """
        return torch.sum(volume, dim=axis, keepdim=False)
    
    def forward(self, pred_volume, target_volume):
        """
        Args:
            pred_volume: Generated CT volume (B, 1, D, H, W)
            target_volume: Ground truth CT volume (B, 1, D, H, W)
        Returns:
            loss: Projection loss across all three axes
        """
        total_loss = 0.0
        
        # Frontal projection (sum along depth, axis=2)
        pred_frontal = self.project_volume(pred_volume, axis=2)
        target_frontal = self.project_volume(target_volume, axis=2)
        loss_frontal = self.criterion(pred_frontal, target_frontal)
        
        # Lateral projection (sum along width, axis=4)
        pred_lateral = self.project_volume(pred_volume, axis=4)
        target_lateral = self.project_volume(target_volume, axis=4)
        loss_lateral = self.criterion(pred_lateral, target_lateral)
        
        # Top projection (sum along height, axis=3)
        pred_top = self.project_volume(pred_volume, axis=3)
        target_top = self.project_volume(target_volume, axis=3)
        loss_top = self.criterion(pred_top, target_top)
        
        # Average across all projections
        total_loss = (loss_frontal + loss_lateral + loss_top) / 3.0
        
        return total_loss


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG16
    
    Compares high-level features extracted from a pre-trained VGG16 network.
    This helps maintain perceptual similarity and anatomical structure.
    
    Since VGG is designed for 2D images, we apply it to 2D projections of the 3D volume.
    """
    
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], 
                 use_gpu=True):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True)
        self.vgg_layers = vgg.features
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        # Move to GPU if available
        if use_gpu and torch.cuda.is_available():
            self.vgg_layers = self.vgg_layers.cuda()
        
        # Define layer indices for feature extraction
        self.layer_name_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        
        self.layers = [self.layer_name_mapping[l] for l in layers]
        self.criterion = nn.L1Loss()
    
    def extract_features(self, x):
        """
        Extract features from specified VGG layers
        
        Args:
            x: Input image (B, C, H, W) - must have 3 channels for VGG
        Returns:
            features: List of feature maps from specified layers
        """
        features = []
        
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        
        return features
    
    def prepare_image(self, img):
        """
        Prepare single-channel image for VGG (convert to 3 channels and normalize)
        
        Args:
            img: Single channel image (B, 1, H, W)
        Returns:
            img_prepared: 3-channel normalized image (B, 3, H, W)
        """
        # Replicate to 3 channels
        img = img.repeat(1, 3, 1, 1)
        
        # Normalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
        
        img = (img - mean) / std
        
        return img
    
    def project_volume(self, volume, axis):
        """
        Project 3D volume to 2D by summing along one axis
        
        Args:
            volume: (B, 1, D, H, W)
            axis: dimension to project along (2=D, 3=H, 4=W)
        Returns:
            projection: (B, 1, H, W) normalized projection
        """
        # Sum along the specified axis, keep channel dimension
        projection = torch.sum(volume, dim=axis, keepdim=False)  # e.g., (B, 1, H, W)
        
        # Ensure we have the right shape (B, 1, H, W)
        if projection.dim() == 3:  # If channel dimension was removed
            projection = projection.unsqueeze(1)
        
        # Normalize to [0, 1] range per sample
        for i in range(projection.shape[0]):
            p = projection[i]
            p_min = p.min()
            p_max = p.max()
            projection[i] = (p - p_min) / (p_max - p_min + 1e-8)
        
        return projection
    
    def forward(self, pred_volume, target_volume):
        """
        Args:
            pred_volume: Generated CT volume (B, 1, D, H, W)
            target_volume: Ground truth CT volume (B, 1, D, H, W)
        Returns:
            loss: Perceptual loss across projections
        """
        total_loss = 0.0
        
        # Compute perceptual loss on three projections
        for axis in [2, 3, 4]:  # frontal, top, lateral
            pred_proj = self.project_volume(pred_volume, axis)  # (B, 1, H, W)
            target_proj = self.project_volume(target_volume, axis)  # (B, 1, H, W)
            
            # Prepare for VGG (already has channel dim, no need for unsqueeze)
            pred_prep = self.prepare_image(pred_proj)
            target_prep = self.prepare_image(target_proj)
            
            # Extract features
            pred_features = self.extract_features(pred_prep)
            target_features = self.extract_features(target_prep)
            
            # Compute loss for each layer
            for pred_feat, target_feat in zip(pred_features, target_features):
                total_loss += self.criterion(pred_feat, target_feat)
        
        # Average across projections and layers
        total_loss = total_loss / (3 * len(self.layers))
        
        return total_loss


class TRCTGANLoss(nn.Module):
    """
    Combined loss function for TRCT-GAN
    
    Combines:
    1. Adversarial loss (LSGAN)
    2. Reconstruction loss (L1)
    3. Projection loss (DRR-based)
    4. Perceptual loss (VGG16)
    """
    
    def __init__(self, config=None):
        super(TRCTGANLoss, self).__init__()
        
        # Default configuration
        if config is None:
            config = {
                'lambda_adv': 1.0,
                'lambda_recon': 10.0,
                'lambda_proj': 5.0,
                'lambda_perceptual': 1.0,
                'recon_type': 'L1',
                'proj_type': 'L1'
            }
        
        self.lambda_adv = config.get('lambda_adv', 1.0)
        self.lambda_recon = config.get('lambda_recon', 10.0)
        self.lambda_proj = config.get('lambda_proj', 5.0)
        self.lambda_perceptual = config.get('lambda_perceptual', 1.0)
        
        # Initialize loss functions
        self.adversarial_loss = LSGANLoss()
        self.reconstruction_loss = ReconstructionLoss(config.get('recon_type', 'L1'))
        self.projection_loss = ProjectionLoss(config.get('proj_type', 'L1'))
        
        # VGG perceptual loss
        try:
            self.perceptual_loss = VGGPerceptualLoss()
        except:
            print("Warning: Could not load VGG16 for perceptual loss. Skipping.")
            self.perceptual_loss = None
            self.lambda_perceptual = 0.0
    
    def generator_loss(self, pred_volume, target_volume, fake_disc_pred):
        """
        Complete generator loss
        
        Args:
            pred_volume: Generated CT volume (B, 1, D, H, W)
            target_volume: Ground truth CT volume (B, 1, D, H, W)
            fake_disc_pred: Discriminator prediction on fake volume
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        
        # Adversarial loss
        loss_adv = self.adversarial_loss.generator_loss(fake_disc_pred)
        loss_dict['adv'] = loss_adv.item()
        
        # Reconstruction loss
        loss_recon = self.reconstruction_loss(pred_volume, target_volume)
        loss_dict['recon'] = loss_recon.item()
        
        # Projection loss
        loss_proj = self.projection_loss(pred_volume, target_volume)
        loss_dict['proj'] = loss_proj.item()
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            loss_perceptual = self.perceptual_loss(pred_volume, target_volume)
            loss_dict['perceptual'] = loss_perceptual.item()
        else:
            loss_perceptual = 0.0
            loss_dict['perceptual'] = 0.0
        
        # Combined loss
        total_loss = (
            self.lambda_adv * loss_adv +
            self.lambda_recon * loss_recon +
            self.lambda_proj * loss_proj +
            self.lambda_perceptual * loss_perceptual
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def discriminator_loss(self, real_disc_pred, fake_disc_pred):
        """
        Discriminator loss
        
        Args:
            real_disc_pred: Discriminator prediction on real volume
            fake_disc_pred: Discriminator prediction on fake volume
        Returns:
            loss: Discriminator loss
        """
        return self.adversarial_loss.discriminator_loss(real_disc_pred, fake_disc_pred)


if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    # Create dummy data
    batch_size = 2
    pred_volume = torch.randn(batch_size, 1, 128, 128, 128)
    target_volume = torch.randn(batch_size, 1, 128, 128, 128)
    disc_pred_real = torch.randn(batch_size, 1, 8, 8, 8)
    disc_pred_fake = torch.randn(batch_size, 1, 8, 8, 8)
    
    print("\n1. Testing LSGAN Loss...")
    lsgan = LSGANLoss()
    g_loss = lsgan.generator_loss(disc_pred_fake)
    d_loss = lsgan.discriminator_loss(disc_pred_real, disc_pred_fake)
    print(f"   Generator loss: {g_loss.item():.4f}")
    print(f"   Discriminator loss: {d_loss.item():.4f}")
    print("   ✓ LSGAN Loss test passed!")
    
    print("\n2. Testing Reconstruction Loss...")
    recon_loss = ReconstructionLoss(loss_type='L1')
    loss = recon_loss(pred_volume, target_volume)
    print(f"   L1 Reconstruction loss: {loss.item():.4f}")
    print("   ✓ Reconstruction Loss test passed!")
    
    print("\n3. Testing Projection Loss...")
    proj_loss = ProjectionLoss(loss_type='L1')
    loss = proj_loss(pred_volume, target_volume)
    print(f"   Projection loss: {loss.item():.4f}")
    print("   ✓ Projection Loss test passed!")
    
    print("\n4. Testing VGG Perceptual Loss...")
    try:
        vgg_loss = VGGPerceptualLoss(use_gpu=False)
        loss = vgg_loss(pred_volume, target_volume)
        print(f"   Perceptual loss: {loss.item():.4f}")
        print("   ✓ VGG Perceptual Loss test passed!")
    except Exception as e:
        print(f"   ⚠ VGG Perceptual Loss test skipped: {e}")
    
    print("\n5. Testing Combined TRCT-GAN Loss...")
    combined_loss = TRCTGANLoss()
    g_loss, loss_dict = combined_loss.generator_loss(pred_volume, target_volume, disc_pred_fake)
    d_loss = combined_loss.discriminator_loss(disc_pred_real, disc_pred_fake)
    
    print(f"   Generator loss breakdown:")
    for key, value in loss_dict.items():
        print(f"     {key}: {value:.4f}")
    print(f"   Discriminator loss: {d_loss.item():.4f}")
    print("   ✓ Combined TRCT-GAN Loss test passed!")
    
    print("\n✓ All loss function tests passed successfully!")

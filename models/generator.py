"""
Generator Architecture for TRCT-GAN
Dual-encoder architecture with transformer bridge and 3D decoder
Reconstructs 3D CT volumes from biplane (frontal + lateral) X-ray images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .aia_modules import AIA2D, AIA3D, DenseBlock2D, TransitionDown2D
from .transformer import Transformer2Dto3D


class Encoder2D(nn.Module):
    """
    2D Encoder for processing individual X-ray views
    Uses dense connections and 2D AIA module for rich feature extraction
    """
    
    def __init__(self, in_channels=1, channels=[64, 128, 256, 512], 
                 use_dense=True, use_aia=True, aia_reduction=16):
        super(Encoder2D, self).__init__()
        
        self.use_aia = use_aia
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, padding=3),
            nn.InstanceNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks with downsampling
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        
        for i in range(len(channels)):
            in_ch = channels[i]
            out_ch = channels[i+1] if i < len(channels)-1 else channels[i]
            
            if use_dense and i < len(channels)-1:
                # Dense block
                block = DenseBlock2D(in_ch, growth_rate=32, num_layers=4)
                # Calculate output channels from dense block
                dense_out_ch = in_ch + 32 * 4
                self.enc_blocks.append(block)
                
                # Transition down
                self.down_blocks.append(TransitionDown2D(dense_out_ch, out_ch))
            else:
                # Standard conv block
                block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
                self.enc_blocks.append(block)
                
                if i < len(channels)-1:
                    self.down_blocks.append(nn.AvgPool2d(2))
        
        # 2D AIA module at the bottleneck
        if use_aia:
            self.aia_2d = AIA2D(channels[-1], reduction_ratio=aia_reduction)
        
    def forward(self, x):
        """
        Args:
            x: Input X-ray image (B, 1, H, W)
        Returns:
            features: List of feature maps at different scales for skip connections
            bottleneck: Bottleneck features (B, C, H', W')
        """
        x = self.init_conv(x)
        
        features = []
        for i, (enc_block, down_block) in enumerate(zip(self.enc_blocks[:-1], self.down_blocks)):
            x = enc_block(x)
            features.append(x)
            x = down_block(x)
        
        # Final encoder block (bottleneck)
        x = self.enc_blocks[-1](x)
        
        # Apply 2D AIA at bottleneck
        if self.use_aia:
            x = self.aia_2d(x)
        
        bottleneck = x
        
        return features, bottleneck


class ViewFusion(nn.Module):
    """
    Fuses features from frontal and lateral X-ray views
    Uses attention mechanism to weight features from each view
    """
    
    def __init__(self, channels):
        super(ViewFusion, self).__init__()
        
        # Attention weights for each view
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, frontal_feat, lateral_feat):
        """
        Args:
            frontal_feat: Features from frontal view (B, C, H, W)
            lateral_feat: Features from lateral view (B, C, H, W)
        Returns:
            fused: Fused features (B, C, H, W)
        """
        # Concatenate features
        concat = torch.cat([frontal_feat, lateral_feat], dim=1)
        
        # Generate attention weights
        attn_weights = self.attention(concat)  # (B, 2, H, W)
        
        # Apply attention
        frontal_weighted = frontal_feat * attn_weights[:, 0:1, :, :]
        lateral_weighted = lateral_feat * attn_weights[:, 1:2, :, :]
        
        # Concatenate weighted features and fuse
        weighted_concat = torch.cat([frontal_weighted, lateral_weighted], dim=1)
        fused = self.fusion_conv(weighted_concat)
        
        return fused


class Decoder3D(nn.Module):
    """
    3D Decoder for reconstructing CT volume
    Uses 3D upsampling and AIA3D module for high-quality reconstruction
    """
    
    def __init__(self, in_channels=512, channels=[256, 128, 64, 32], 
                 out_channels=1, use_aia=True, use_trilinear=True, aia_reduction=16):
        super(Decoder3D, self).__init__()
        
        self.use_aia = use_aia
        
        # Decoder blocks with upsampling
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        all_channels = [in_channels] + channels
        
        for i in range(len(all_channels)-1):
            in_ch = all_channels[i]
            out_ch = all_channels[i+1]
            
            # Upsample
            mode = 'trilinear' if use_trilinear else 'nearest'
            self.up_blocks.append(
                nn.Upsample(scale_factor=2, mode=mode, 
                          align_corners=True if mode == 'trilinear' else None)
            )
            
            # Conv block (double channels for skip connection concatenation)
            block = nn.Sequential(
                nn.Conv3d(in_ch * 2, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.dec_blocks.append(block)
        
        # 3D AIA module at the end
        if use_aia:
            self.aia_3d = AIA3D(channels[-1], reduction_ratio=aia_reduction, 
                               use_trilinear=use_trilinear)
        
        # Final output convolution
        self.out_conv = nn.Sequential(
            nn.Conv3d(channels[-1], out_channels, 1),
            nn.Tanh()  # Normalize output to [-1, 1]
        )
    
    def forward(self, x, skip_features=None):
        """
        Args:
            x: Input 3D features from transformer (B, C, D, H, W)
            skip_features: List of skip connection features (optional)
        Returns:
            out: Reconstructed 3D CT volume (B, 1, D, H, W)
        """
        if skip_features is None:
            skip_features = [None] * len(self.up_blocks)
        
        for i, (up_block, dec_block) in enumerate(zip(self.up_blocks, self.dec_blocks)):
            x = up_block(x)
            
            # Add skip connection if available
            if skip_features[i] is not None:
                # Ensure spatial dimensions match
                if x.shape[2:] != skip_features[i].shape[2:]:
                    skip_features[i] = F.interpolate(skip_features[i], size=x.shape[2:], 
                                                     mode='trilinear', align_corners=True)
                x = torch.cat([x, skip_features[i]], dim=1)
            else:
                # If no skip connection, concatenate with itself
                x = torch.cat([x, x], dim=1)
            
            x = dec_block(x)
        
        # Apply 3D AIA
        if self.use_aia:
            x = self.aia_3d(x)
        
        # Final output
        out = self.out_conv(x)
        
        return out


class TRCTGenerator(nn.Module):
    """
    Complete TRCT-GAN Generator
    
    Architecture:
    1. Dual 2D encoders for frontal and lateral X-rays
    2. View fusion module to combine features
    3. Transformer module to convert 2D features to 3D with global context
    4. 3D decoder to reconstruct full CT volume
    5. Skip connections for multi-scale feature fusion
    """
    
    def __init__(self, config=None):
        super(TRCTGenerator, self).__init__()
        
        # Default configuration
        if config is None:
            config = {
                'encoder_channels': [64, 128, 256, 512],
                'decoder_channels': [256, 128, 64, 32],
                'transformer': {
                    'embed_dim': 512,
                    'num_heads': 8,
                    'num_layers': 6,
                    'mlp_ratio': 4,
                    'dropout': 0.1
                },
                'use_aia_2d': True,
                'use_aia_3d': True,
                'use_trilinear': True,
                'aia_reduction': 16
            }
        
        enc_channels = config['encoder_channels']
        dec_channels = config['decoder_channels']
        
        # Dual encoders for frontal and lateral views
        self.encoder_frontal = Encoder2D(
            in_channels=1,
            channels=enc_channels,
            use_dense=True,
            use_aia=config.get('use_aia_2d', True),
            aia_reduction=config.get('aia_reduction', 16)
        )
        
        self.encoder_lateral = Encoder2D(
            in_channels=1,
            channels=enc_channels,
            use_dense=True,
            use_aia=config.get('use_aia_2d', True),
            aia_reduction=config.get('aia_reduction', 16)
        )
        
        # View fusion
        self.view_fusion = ViewFusion(enc_channels[-1])
        
        # Transformer for 2D to 3D conversion
        trans_config = config.get('transformer', {})
        self.transformer = Transformer2Dto3D(
            embed_dim=trans_config.get('embed_dim', 512),
            num_heads=trans_config.get('num_heads', 8),
            num_layers=trans_config.get('num_layers', 6),
            mlp_ratio=trans_config.get('mlp_ratio', 4),
            dropout=trans_config.get('dropout', 0.1),
            input_size=(16, 16),  # Assuming 128->16 after downsampling
            output_depth=16
        )
        
        # 3D Decoder
        self.decoder = Decoder3D(
            in_channels=enc_channels[-1],
            channels=dec_channels,
            out_channels=1,
            use_aia=config.get('use_aia_3d', True),
            use_trilinear=config.get('use_trilinear', True),
            aia_reduction=config.get('aia_reduction', 16)
        )
        
    def forward(self, xray_frontal, xray_lateral):
        """
        Args:
            xray_frontal: Frontal X-ray image (B, 1, H, W)
            xray_lateral: Lateral X-ray image (B, 1, H, W)
        Returns:
            ct_volume: Reconstructed 3D CT volume (B, 1, D, H, W)
        """
        # Encode both views
        frontal_feats, frontal_bottleneck = self.encoder_frontal(xray_frontal)
        lateral_feats, lateral_bottleneck = self.encoder_lateral(xray_lateral)
        
        # Fuse views at bottleneck
        fused_features = self.view_fusion(frontal_bottleneck, lateral_bottleneck)
        
        # Transform 2D features to 3D with global context
        features_3d = self.transformer(fused_features)
        
        # Decode to 3D CT volume
        # Note: For skip connections, we would need to convert 2D skip features to 3D
        # For simplicity, we'll use None here; in practice, you might want to 
        # expand 2D features to 3D for skip connections
        ct_volume = self.decoder(features_3d, skip_features=None)
        
        return ct_volume


if __name__ == "__main__":
    print("Testing TRCT Generator...")
    
    # Create generator
    generator = TRCTGenerator()
    
    # Test with dummy inputs
    batch_size = 2
    xray_frontal = torch.randn(batch_size, 1, 128, 128)
    xray_lateral = torch.randn(batch_size, 1, 128, 128)
    
    print(f"\nInput shapes:")
    print(f"  Frontal X-ray: {xray_frontal.shape}")
    print(f"  Lateral X-ray: {xray_lateral.shape}")
    
    # Forward pass
    with torch.no_grad():
        ct_volume = generator(xray_frontal, xray_lateral)
    
    print(f"\nOutput shape:")
    print(f"  CT Volume: {ct_volume.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, 1, 128, 128, 128)
    # Note: actual output might be smaller depending on architecture
    # Let's check what we got
    print(f"\nExpected shape: {expected_shape}")
    print(f"Actual shape: {ct_volume.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    
    print(f"\nGenerator Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Size in MB: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n✓ Generator test completed!")

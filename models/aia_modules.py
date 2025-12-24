"""
Attention In Attention (AIA) Modules for TRCT-GAN
Implements both 2D and 3D AIA modules with dynamic attention weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AIA2D(nn.Module):
    """
    2D Attention In Attention Module
    
    Uses three branches:
    1. Attention branch - generates attention maps
    2. Non-attention branch - standard convolution path
    3. Dynamic weight branch - learns to balance between attention and non-attention
    
    This helps the network focus on valuable features while ignoring noise.
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(AIA2D, self).__init__()
        self.in_channels = in_channels
        
        # Attention branch - spatial and channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, 1),
            nn.Sigmoid()
        )
        
        # Non-attention branch
        self.non_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dynamic weight generation branch
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 2, 1),  # 2 weights: attention vs non-attention
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Attention branch
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        attention_out = x * channel_att * spatial_att
        
        # Non-attention branch
        non_attention_out = self.non_attention(x)
        
        # Dynamic weight generation
        weights = self.weight_generator(x)  # (B, 2, 1, 1)
        weight_att = weights[:, 0:1, :, :]
        weight_non_att = weights[:, 1:2, :, :]
        
        # Weighted combination
        out = weight_att * attention_out + weight_non_att * non_attention_out
        
        return out


class AIA3D(nn.Module):
    """
    3D Attention In Attention Module
    
    Similar to 2D AIA but operates on 3D volumes.
    Uses both trilinear interpolation (quality) and nearest neighbor (speed)
    for upsampling based on configuration.
    """
    
    def __init__(self, in_channels, reduction_ratio=16, use_trilinear=True):
        super(AIA3D, self).__init__()
        self.in_channels = in_channels
        self.use_trilinear = use_trilinear
        
        # Attention branch - 3D spatial and channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, 1, 1),
            nn.Sigmoid()
        )
        
        # Non-attention branch
        self.non_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dynamic weight generation branch
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Attention branch
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        attention_out = x * channel_att * spatial_att
        
        # Non-attention branch
        non_attention_out = self.non_attention(x)
        
        # Dynamic weight generation
        weights = self.weight_generator(x)  # (B, 2, 1, 1, 1)
        weight_att = weights[:, 0:1, :, :, :]
        weight_non_att = weights[:, 1:2, :, :, :]
        
        # Weighted combination
        out = weight_att * attention_out + weight_non_att * non_attention_out
        
        return out
    
    def upsample(self, x, scale_factor=2):
        """
        Upsample 3D feature maps using either trilinear or nearest neighbor interpolation
        """
        mode = 'trilinear' if self.use_trilinear else 'nearest'
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, 
                           align_corners=True if mode == 'trilinear' else None)


class DenseBlock2D(nn.Module):
    """
    Dense Block for 2D feature extraction in the encoder
    Implements densely connected convolutions for rich feature representation
    """
    
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock2D, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.InstanceNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1)
                )
            )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class TransitionDown2D(nn.Module):
    """
    Transition layer for downsampling in dense connections
    """
    
    def __init__(self, in_channels, out_channels):
        super(TransitionDown2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


if __name__ == "__main__":
    # Test 2D AIA module
    print("Testing 2D AIA Module...")
    aia2d = AIA2D(in_channels=256, reduction_ratio=16)
    x2d = torch.randn(2, 256, 64, 64)
    out2d = aia2d(x2d)
    print(f"Input shape: {x2d.shape}, Output shape: {out2d.shape}")
    assert out2d.shape == x2d.shape, "2D AIA shape mismatch!"
    print("✓ 2D AIA Module test passed!")
    
    # Test 3D AIA module
    print("\nTesting 3D AIA Module...")
    aia3d = AIA3D(in_channels=64, reduction_ratio=16, use_trilinear=True)
    x3d = torch.randn(2, 64, 32, 32, 32)
    out3d = aia3d(x3d)
    print(f"Input shape: {x3d.shape}, Output shape: {out3d.shape}")
    assert out3d.shape == x3d.shape, "3D AIA shape mismatch!"
    
    # Test upsampling
    upsampled = aia3d.upsample(x3d, scale_factor=2)
    print(f"Upsampled shape: {upsampled.shape}")
    assert upsampled.shape == (2, 64, 64, 64, 64), "3D upsampling shape mismatch!"
    print("✓ 3D AIA Module test passed!")
    
    # Test Dense Block
    print("\nTesting Dense Block...")
    dense = DenseBlock2D(in_channels=64, growth_rate=32, num_layers=4)
    xd = torch.randn(2, 64, 32, 32)
    outd = dense(xd)
    print(f"Dense Block - Input shape: {xd.shape}, Output shape: {outd.shape}")
    print("✓ Dense Block test passed!")
    
    print("\n✓ All AIA module tests passed successfully!")

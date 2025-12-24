"""
PatchGAN Discriminator for TRCT-GAN
Evaluates the quality of reconstructed 3D CT volumes by examining local patches
"""

import torch
import torch.nn as nn


class SpectralNorm(nn.Module):
    """
    Spectral Normalization wrapper (optional, for training stability)
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)
        
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height,-1).data, v.data))
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))


class PatchGANDiscriminator3D(nn.Module):
    """
    3D PatchGAN Discriminator
    
    Evaluates the realism of 3D CT volumes by examining local patches rather than
    the entire volume. This provides better generalization and focuses on local
    texture and structure consistency.
    
    The discriminator outputs a matrix of values (one per patch) which are averaged
    for the final discrimination score.
    """
    
    def __init__(self, in_channels=1, channels=[64, 128, 256, 512], 
                 num_layers=4, use_spectral_norm=False):
        super(PatchGANDiscriminator3D, self).__init__()
        
        self.use_spectral_norm = use_spectral_norm
        
        # Build discriminator layers
        layers = []
        
        # First layer (no normalization)
        layers.append(nn.Sequential(
            self._conv_block(in_channels, channels[0], normalize=False)
        ))
        
        # Intermediate layers
        for i in range(1, min(num_layers, len(channels))):
            layers.append(
                self._conv_block(channels[i-1], channels[i], normalize=True)
            )
        
        # Final layers
        if num_layers > len(channels):
            for _ in range(num_layers - len(channels)):
                layers.append(
                    self._conv_block(channels[-1], channels[-1], normalize=True, stride=1)
                )
        
        self.model = nn.Sequential(*layers)
        
        # Output layer - maps to patch discrimination scores
        self.output = nn.Conv3d(channels[min(num_layers-1, len(channels)-1)], 1, 
                               kernel_size=4, stride=1, padding=1)
    
    def _conv_block(self, in_ch, out_ch, normalize=True, stride=2):
        """
        Create a convolutional block for the discriminator
        """
        layers = []
        
        conv = nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not normalize)
        
        if self.use_spectral_norm:
            conv = SpectralNorm(conv)
        
        layers.append(conv)
        
        if normalize:
            layers.append(nn.InstanceNorm3d(out_ch))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input 3D CT volume (B, 1, D, H, W)
        Returns:
            out: Patch-wise discrimination scores (B, 1, D', H', W')
                 Each value represents the "realness" score for a local patch
        """
        x = self.model(x)
        out = self.output(x)
        return out


class MultiScaleDiscriminator3D(nn.Module):
    """
    Multi-scale PatchGAN Discriminator
    
    Uses multiple discriminators at different scales to evaluate both
    fine-grained details and coarse structure.
    """
    
    def __init__(self, in_channels=1, num_discriminators=2, channels=[64, 128, 256, 512],
                 num_layers=4, use_spectral_norm=False):
        super(MultiScaleDiscriminator3D, self).__init__()
        
        self.num_discriminators = num_discriminators
        self.discriminators = nn.ModuleList()
        
        for i in range(num_discriminators):
            self.discriminators.append(
                PatchGANDiscriminator3D(in_channels, channels, num_layers, use_spectral_norm)
            )
        
        # Downsample for multi-scale
        self.downsample = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        """
        Args:
            x: Input 3D CT volume (B, 1, D, H, W)
        Returns:
            outputs: List of discrimination outputs at different scales
        """
        outputs = []
        
        for i, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(x))
            if i < self.num_discriminators - 1:
                x = self.downsample(x)
        
        return outputs


class ConditionalDiscriminator3D(nn.Module):
    """
    Conditional PatchGAN Discriminator
    
    Takes both the generated/real CT volume and the input X-ray images
    to provide conditional discrimination (ensures the output matches the input)
    """
    
    def __init__(self, ct_channels=1, xray_channels=2, channels=[64, 128, 256, 512],
                 num_layers=4, use_spectral_norm=False):
        super(ConditionalDiscriminator3D, self).__init__()
        
        # Project 2D X-rays to 3D for concatenation with CT
        self.xray_projection = nn.Sequential(
            nn.Conv2d(xray_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, ct_channels, 3, padding=1)
        )
        
        # Main discriminator (input is CT + projected X-rays)
        self.discriminator = PatchGANDiscriminator3D(
            in_channels=ct_channels * 2,  # CT + projected X-rays
            channels=channels,
            num_layers=num_layers,
            use_spectral_norm=use_spectral_norm
        )
    
    def forward(self, ct, xray_frontal, xray_lateral):
        """
        Args:
            ct: CT volume (B, 1, D, H, W)
            xray_frontal: Frontal X-ray (B, 1, H, W)
            xray_lateral: Lateral X-ray (B, 1, H, W)
        Returns:
            out: Discrimination scores
        """
        # Concatenate X-rays
        xrays = torch.cat([xray_frontal, xray_lateral], dim=1)  # (B, 2, H, W)
        
        # Project to 3D
        xray_proj = self.xray_projection(xrays)  # (B, 1, H, W)
        
        # Expand to match CT depth
        D = ct.shape[2]
        xray_3d = xray_proj.unsqueeze(2).expand(-1, -1, D, -1, -1)  # (B, 1, D, H, W)
        
        # Concatenate with CT
        combined = torch.cat([ct, xray_3d], dim=1)  # (B, 2, D, H, W)
        
        # Discriminate
        out = self.discriminator(combined)
        
        return out


if __name__ == "__main__":
    print("Testing Discriminator Architectures...")
    
    # Test basic PatchGAN Discriminator
    print("\n1. Testing PatchGAN Discriminator...")
    disc = PatchGANDiscriminator3D(
        in_channels=1,
        channels=[64, 128, 256, 512],
        num_layers=4,
        use_spectral_norm=False
    )
    
    x = torch.randn(2, 1, 128, 128, 128)
    print(f"   Input shape: {x.shape}")
    
    with torch.no_grad():
        out = disc(x)
    print(f"   Output shape: {out.shape}")
    print(f"   Output is a matrix of patch scores")
    
    # Count parameters
    total_params = sum(p.numel() for p in disc.parameters())
    print(f"   Parameters: {total_params:,}")
    print("   ✓ PatchGAN Discriminator test passed!")
    
    # Test Multi-Scale Discriminator
    print("\n2. Testing Multi-Scale Discriminator...")
    multi_disc = MultiScaleDiscriminator3D(
        in_channels=1,
        num_discriminators=2,
        channels=[64, 128, 256, 512],
        num_layers=4
    )
    
    with torch.no_grad():
        outputs = multi_disc(x)
    
    print(f"   Number of scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"   Scale {i} output shape: {out.shape}")
    
    total_params = sum(p.numel() for p in multi_disc.parameters())
    print(f"   Parameters: {total_params:,}")
    print("   ✓ Multi-Scale Discriminator test passed!")
    
    # Test Conditional Discriminator
    print("\n3. Testing Conditional Discriminator...")
    cond_disc = ConditionalDiscriminator3D(
        ct_channels=1,
        xray_channels=2,
        channels=[64, 128, 256, 512],
        num_layers=4
    )
    
    ct = torch.randn(2, 1, 128, 128, 128)
    xray_f = torch.randn(2, 1, 128, 128)
    xray_l = torch.randn(2, 1, 128, 128)
    
    print(f"   CT shape: {ct.shape}")
    print(f"   X-ray frontal shape: {xray_f.shape}")
    print(f"   X-ray lateral shape: {xray_l.shape}")
    
    with torch.no_grad():
        out = cond_disc(ct, xray_f, xray_l)
    
    print(f"   Output shape: {out.shape}")
    
    total_params = sum(p.numel() for p in cond_disc.parameters())
    print(f"   Parameters: {total_params:,}")
    print("   ✓ Conditional Discriminator test passed!")
    
    print("\n✓ All discriminator tests passed successfully!")

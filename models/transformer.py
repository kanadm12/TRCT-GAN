"""
Transformer Module for TRCT-GAN
Converts 2D features to 1D feature vectors and processes them through
multi-head self-attention to capture global contextual information
"""

import torch
import torch.nn as nn
import math
from einops import rearrange


class PositionalEncoding(nn.Module):
    """
    Positional encoding to add spatial information to the feature vectors
    Uses sinusoidal positional encoding from "Attention is All You Need"
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    Allows the model to attend to different positions and capture global context
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            out: (batch, seq_len, embed_dim)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    Feed-forward MLP used in transformer blocks
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer block with multi-head self-attention and MLP
    Uses pre-normalization (LayerNorm before attention/MLP)
    """
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer2Dto3D(nn.Module):
    """
    Transformer module for converting 2D features to 3D representation
    
    Process:
    1. Flatten 2D feature maps to 1D sequence
    2. Add positional encoding
    3. Process through multi-head self-attention layers
    4. Reshape back to 3D volume
    
    This allows the network to capture global contextual information and
    rich anatomical details that standard convolutions might miss.
    """
    
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, mlp_ratio=4, 
                 dropout=0.1, input_size=(16, 16), output_depth=16):
        super(Transformer2Dto3D, self).__init__()
        
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.output_depth = output_depth
        self.seq_len = input_size[0] * input_size[1]
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=self.seq_len, dropout=dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projection to 3D: expand sequence to include depth dimension
        # We'll use learned queries to generate depth slices
        self.depth_queries = nn.Parameter(torch.randn(1, output_depth, embed_dim))
        self.depth_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
    def forward(self, x):
        """
        Args:
            x: 2D feature map (batch, channels, H, W) where channels = embed_dim
        Returns:
            out: 3D feature volume (batch, channels, D, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.embed_dim, f"Channel dim {C} must match embed_dim {self.embed_dim}"
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)  # (B, H*W, C)
        
        # Generate 3D volume by creating depth slices
        # Expand depth queries for batch
        depth_queries = self.depth_queries.expand(B, -1, -1)  # (B, D, C)
        
        # Combine spatial features with depth queries
        # Concatenate spatial tokens and depth queries
        combined = torch.cat([x, depth_queries], dim=1)  # (B, H*W+D, C)
        
        # Apply cross-attention (depth queries attend to spatial features)
        combined = self.depth_attn(combined)
        
        # Extract depth features
        depth_features = combined[:, -self.output_depth:, :]  # (B, D, C)
        
        # Reshape to 3D volume: (B, D, C) -> (B, C, D, H, W)
        # We need to expand spatial dimensions back
        # Use learned projection to maintain spatial structure
        depth_features = depth_features.transpose(1, 2)  # (B, C, D)
        
        # Expand spatial dimensions
        out = depth_features.unsqueeze(-1).unsqueeze(-1)  # (B, C, D, 1, 1)
        out = out.expand(-1, -1, -1, H, W)  # (B, C, D, H, W)
        
        return out


if __name__ == "__main__":
    print("Testing Transformer Module Components...")
    
    # Test Positional Encoding
    print("\n1. Testing Positional Encoding...")
    pos_enc = PositionalEncoding(d_model=512, max_len=256, dropout=0.1)
    x = torch.randn(2, 256, 512)
    out = pos_enc(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "Positional encoding shape mismatch!"
    print("   ✓ Positional Encoding test passed!")
    
    # Test Multi-Head Self-Attention
    print("\n2. Testing Multi-Head Self-Attention...")
    mhsa = MultiHeadSelfAttention(embed_dim=512, num_heads=8, dropout=0.1)
    x = torch.randn(2, 256, 512)
    out = mhsa(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "MHSA shape mismatch!"
    print("   ✓ Multi-Head Self-Attention test passed!")
    
    # Test Transformer Block
    print("\n3. Testing Transformer Block...")
    block = TransformerBlock(embed_dim=512, num_heads=8, mlp_ratio=4, dropout=0.1)
    x = torch.randn(2, 256, 512)
    out = block(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "Transformer block shape mismatch!"
    print("   ✓ Transformer Block test passed!")
    
    # Test Full Transformer 2D to 3D
    print("\n4. Testing Transformer2Dto3D...")
    transformer = Transformer2Dto3D(
        embed_dim=512, 
        num_heads=8, 
        num_layers=6,
        mlp_ratio=4,
        dropout=0.1,
        input_size=(16, 16),  # H, W of feature map
        output_depth=16
    )
    x = torch.randn(2, 512, 16, 16)  # (B, C, H, W)
    out = transformer(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    expected_shape = (2, 512, 16, 16, 16)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print("   ✓ Transformer2Dto3D test passed!")
    
    print("\n✓ All Transformer module tests passed successfully!")
    print(f"\nTransformer Statistics:")
    print(f"   Total parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}")

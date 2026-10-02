"""
Complete Optimal Asymmetric Audio Encoder
Integrates all components: Frequency Encoder, TCN, Sophisticated Forgery Branch, Simple Irrelevant Branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import all audio encoder components
from .audio_encoder import (
    FrequencyEncoder,
    TemporalConvNet,
    SophisticatedForgeryBranch,
    SimpleIrrelevantBranch
)


class AudioEncoderTCN(nn.Module):
    """
    Complete Optimal Asymmetric Audio Encoder
    
    Architecture Flow:
    Input: [B, 4, 1, 128, 32] (mel-spectrograms)
    ↓
    Frequency Encoder: [B*4, 256, 16, 32]
    ↓
    Reshape: [B*4, 4096, 32]
    ↓
    TCN: [B*4, 256, 32]
    ↓
    Temporal Downsample: [B*4, 256, 25]
    ↓
    Asymmetric Branches:
    ├─ Sophisticated Forgery Branch: [B*4, 512, 25]
    └─ Simple Irrelevant Branch: [B*4, 512, 25]
    ↓
    Output: [B, 4, 25, 512] (matches visual encoder)
    """
    def __init__(self, input_dim=1, output_dim=512, freq_channels=256, tcn_channels=256):
        super(AudioEncoderTCN, self).__init__()
        
        # print(f"[DEBUG] AudioEncoderTCN init: input_dim={input_dim}, output_dim={output_dim}, "
        #   f"freq_channels={freq_channels}, tcn_channels={tcn_channels}")
        
        # Shared temporal processing components
        self.freq_encoder = FrequencyEncoder(
            input_channels=input_dim, 
            output_channels=freq_channels
        )
        # print(f"[DEBUG] Frequency encoder created: {input_dim} -> {freq_channels}")
        
        # Calculate TCN input dimension (freq_channels * reduced_freq_bins)
        # Frequency encoder reduces 128 -> 16 bins, so input to TCN = freq_channels * 16
        tcn_input_dim = freq_channels * 16  # 256 * 16 = 4096
        # print(f"[DEBUG] TCN input dimension: {tcn_input_dim}")
        
        self.tcn = TemporalConvNet(
            num_inputs=tcn_input_dim,
            num_channels=[tcn_channels, tcn_channels, tcn_channels, tcn_channels, tcn_channels, tcn_channels],
            kernel_size=3,
            dropout=0.1
        )
        # print(f"[DEBUG] TCN created: {tcn_input_dim} -> {tcn_channels}")
        
        # Temporal downsampling (32 -> 25) using adaptive pooling
        self.temporal_downsample = nn.AdaptiveAvgPool1d(25)
        # print(f"[DEBUG] Temporal downsample: AdaptiveAvgPool1d(25)")
        
        # Asymmetric disentanglement branches
        self.forgery_branch = SophisticatedForgeryBranch(
            input_dim=tcn_channels, 
            output_dim=output_dim
        )
        # print(f"[DEBUG] Sophisticated forgery branch created: {tcn_channels} -> {output_dim}")
        
        self.irrelevant_branch = SimpleIrrelevantBranch(
            input_dim=tcn_channels, 
            output_dim=output_dim
        )
        # print(f"[DEBUG] Simple irrelevant branch created: {tcn_channels} -> {output_dim}")
        
        # Orthogonality loss computation
        self.orthogonality_weight = 0.1
        # print(f"[DEBUG] Orthogonality weight: {self.orthogonality_weight}")
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Forward pass through the complete audio encoder
        
        Args:
            x: Input tensor [B, 4, 1, 128, 32] (mel-spectrograms) or dict with 'mels' key
            
        Returns:
            dict: {
                'afs': forgery_features,      # [B, 4, 25, 512] - Sophisticated artifact features
                'air': irrelevant_features,   # [B, 4, 25, 512] - Simple content features
                'orthogonality_loss': orthogonality_loss
            }
        """
        # Handle dictionary input (from MMD_NET)
        if isinstance(x, dict):
            x = x['mels']
        
        # print(f"[DEBUG] AudioEncoderTCN forward input: {x.shape}")
        
        B, V, C, H, W = x.shape
        # print(f"[DEBUG] Input dimensions: B={B}, V={V}, C={C}, H={H}, W={W}")
        
        # Reshape for processing: [B*4, 1, 128, 32]
        x_reshaped = x.view(B*V, C, H, W)
        # print(f"[DEBUG] Reshaped input: {x_reshaped.shape}")
        
        # Frequency processing
        freq_features = self.freq_encoder(x_reshaped)  # [B*4, 256, 16, 32]
        # print(f"[DEBUG] After frequency encoder: {freq_features.shape}")
        
        # Reshape for temporal processing
        B_freq, C_freq, F_freq, T_freq = freq_features.shape
        temporal_input = freq_features.view(B_freq, C_freq*F_freq, T_freq)  # [B*4, 4096, 32]
        # print(f"[DEBUG] Reshaped for TCN: {temporal_input.shape}")
        
        # Temporal processing
        temporal_features = self.tcn(temporal_input)  # [B*4, 256, 32]
        # print(f"[DEBUG] After TCN: {temporal_features.shape}")
        
        # Temporal downsampling (32 -> 25)
        temporal_features = self.temporal_downsample(temporal_features)  # [B*4, 256, 25]
        # print(f"[DEBUG] After temporal downsample: {temporal_features.shape}")
        
        # Asymmetric disentanglement
        # print(f"[DEBUG] Starting asymmetric disentanglement...")
        
        forgery_features = self.forgery_branch(temporal_features)  # [B*4, 512, 25]
        # print(f"[DEBUG] After forgery branch: {forgery_features.shape}")
        
        irrelevant_features = self.irrelevant_branch(temporal_features)  # [B*4, 512, 25]
        # print(f"[DEBUG] After irrelevant branch: {irrelevant_features.shape}")
        
        # Reshape back to batch format
        forgery_features = forgery_features.view(B, V, 512, 25).permute(0, 1, 3, 2)  # [B, 4, 25, 512]
        irrelevant_features = irrelevant_features.view(B, V, 512, 25).permute(0, 1, 3, 2)  # [B, 4, 25, 512]
        
        #print(f"[DEBUG] Final forgery features: {forgery_features.shape}")
        #print(f"[DEBUG] Final irrelevant features: {irrelevant_features.shape}")
        
        # Compute orthogonality loss for disentanglement
        orthogonality_loss = self.compute_orthogonality_loss(forgery_features, irrelevant_features)
        #print(f"[DEBUG] Orthogonality loss: {orthogonality_loss.item():.6f}")
        
        return {
            'afs': forgery_features,      # [B, 4, 25, 512] - Sophisticated artifact features
            'air': irrelevant_features,   # [B, 4, 25, 512] - Simple content features
            'orthogonality_loss': orthogonality_loss
        }
    
    def compute_orthogonality_loss(self, forgery, irrelevant, alpha=0.5):
        """
        Compute orthogonality loss to encourage decorrelation between forgery and irrelevant features
        FIXED: Now uses per-video computation like other losses
        
        Args:
            forgery: Forgery features [B, 4, 25, 512]
            irrelevant: Irrelevant features [B, 4, 25, 512]
            
        Returns:
            torch.Tensor: Orthogonality loss scalar
        """
        # print(f"[DEBUG] Computing orthogonality loss...")
        # print(f"[DEBUG] Forgery features shape: {forgery.shape}")
        # print(f"[DEBUG] Irrelevant features shape: {irrelevant.shape}")
        
        B, V, T, D = forgery.shape  # [B, 4, 25, 512]
        
        # FIXED: Per-video flattening (like other losses)
        # Flatten each video separately: [B, V, T*D] then [B*V, T*D]
        # forgery_per_video = forgery.reshape(B, V, T*D)  # [B, 4, 25*512]
        # irrelevant_per_video = irrelevant.reshape(B, V, T*D)  # [B, 4, 25*512]
        
        # Flatten to per-video samples (consistent with binary/multi losses)
        forgery_flat = forgery.reshape(B*V, T*D)  # [B*4, 25*512] = [4, 12800]
        irrelevant_flat = irrelevant.reshape(B*V, T*D)  # [B*4, 25*512] = [4, 12800]
        
        # print(f"[DEBUG] Per-video flattened forgery: {forgery_flat.shape}")
        # print(f"[DEBUG] Per-video flattened irrelevant: {irrelevant_flat.shape}")
        
        # Normalize features before computing orthogonality to make it more comparable to other losses
        f_norm = F.normalize(forgery_flat, dim=1)  # L2 normalize along feature dimension
        r_norm = F.normalize(irrelevant_flat, dim=1)  # L2 normalize along feature dimension
        
        # print(f"[DEBUG] Normalized forgery features (L2): {f_norm.shape}")
        # print(f"[DEBUG] Normalized irrelevant features (L2): {r_norm.shape}")
        
        # Compute per-video orthogonality (element-wise dot product of normalized features)
        # per_video_correlation = torch.sum(f_norm * r_norm, dim=1)  # [B*4]
        raw_corr = torch.mean(torch.abs(torch.sum(forgery_flat * irrelevant_flat, dim=1))) / (T*D)
        # print(f"[DEBUG] Per-video correlations: {raw_corr.shape}")
        # print(f"[DEBUG] Correlation range: [{raw_corr.min().item():.6f}, {raw_corr.max().item():.6f}]")
        
        # Cosine correlation (scale-insensitive)
        cos_corr = torch.mean(torch.abs(torch.sum(f_norm * r_norm, dim=1)))
        # print(f"[DEBUG] Per-video correlations: {cos_corr.shape}")
        # print(f"[DEBUG] Correlation range: [{cos_corr.min().item():.6f}, {cos_corr.max().item():.6f}]")
        
        # Hybrid
        orthogonality_loss = alpha * raw_corr + (1 - alpha) * cos_corr

        # FIXED: Per-video orthogonality loss (no double normalization)
        # .mean() already normalizes by B*V, so no need to divide by V again
        # print(f"[DEBUG] hybrid orthogonality loss: {orthogonality_loss.item():.6f}")
        
        # Apply weight
        # weighted_loss = self.orthogonality_weight * orthogonality_loss
        # print(f"[DEBUG] Weighted orthogonality loss: {weighted_loss.item():.6f}")
        
        return orthogonality_loss
    
    def get_model_info(self):
        """Get model information and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frequency_encoder_params": sum(p.numel() for p in self.freq_encoder.parameters()),
            "tcn_params": sum(p.numel() for p in self.tcn.parameters()),
            "forgery_branch_params": sum(p.numel() for p in self.forgery_branch.parameters()),
            "irrelevant_branch_params": sum(p.numel() for p in self.irrelevant_branch.parameters()),
            "orthogonality_weight": self.orthogonality_weight
        }
        
        return info


def test_audio_encoder_tcn():
    """Test function to verify complete audio encoder implementation"""
    print("\n=== Testing Complete AudioEncoderTCN Implementation ===")
    
    # Test parameters
    batch_size = 1
    num_videos = 4
    input_channels = 1
    freq_bins = 128
    time_steps = 32
    
    # Create test input (mel-spectrograms)
    x = torch.randn(batch_size, num_videos, input_channels, freq_bins, time_steps)
    print(f"[TEST] Input shape: {x.shape}")
    
    # Create AudioEncoderTCN
    audio_encoder = AudioEncoderTCN(
        input_dim=input_channels,
        output_dim=512,
        freq_channels=256,
        tcn_channels=256
    )
    
    # Get model info
    model_info = audio_encoder.get_model_info()
    print(f"[TEST] Model info: {model_info}")
    
    # Forward pass
    output = audio_encoder(x)
    
    # Verify outputs
    print(f"[TEST] Output keys: {output.keys()}")
    print(f"[TEST] Forgery features shape: {output['afs'].shape}")
    print(f"[TEST] Irrelevant features shape: {output['air'].shape}")
    print(f"[TEST] Orthogonality loss: {output['orthogonality_loss'].item():.6f}")
    
    # Verify output shapes
    expected_shape = (batch_size, num_videos, 25, 512)
    assert output['afs'].shape == expected_shape, f"Expected {expected_shape}, got {output['afs'].shape}"
    assert output['air'].shape == expected_shape, f"Expected {expected_shape}, got {output['air'].shape}"
    
    # Verify orthogonality loss is a scalar tensor
    assert output['orthogonality_loss'].dim() == 0, f"Expected scalar tensor, got {output['orthogonality_loss'].shape}"
    
    print("✅ Complete AudioEncoderTCN test passed!")
    return output


if __name__ == "__main__":
    test_audio_encoder_tcn()

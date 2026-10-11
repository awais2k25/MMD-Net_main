import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .consts import MODEL_CONFIGS

class FeatureFusionModule(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, 8)
        self.norm = nn.LayerNorm(feature_dim)
        
    def swap_irrelevant_features(self, static, temporal, irr):
        """Enhanced swapping including temporal features
        Args:
            static: Forgery features [B, 4, T, D]
            temporal: Temporal features dict with 'forgery' and 'irrelevant'
            irr: Irrelevant features [B, 4, T, D]
        """
        B, V, T, D = irr.shape
        # Create swap indices for better tracking
        swap_indices = torch.arange(V).roll(1)
        
        # Swap static irrelevant features
        irr_swapped = irr[:, swap_indices]
        
        return static, irr_swapped, swap_indices

    def forward(self, static, irr, enable_swap=True):
        """
        Args:
            static: Static forgery features [B, 4, 25, 512]
            irr: Irrelevant features [B, 4, 25, 512]
        """
        if enable_swap and self.training:
            # Don't swap static features, only swap irrelevant features
            irr_swapped = irr.clone().roll(1, dims=1)  # Roll along video dimension
            
            # Concatenate original static with itself to treat each as separate video
            total_static = torch.cat([static, static], dim=1)  # [B, 8, 25, 512]
            
            # Concatenate original and swapped irrelevant features
            total_irr = torch.cat([irr, irr_swapped], dim=1)  # [B, 8, 25, 512]
            
            # Get fusion for concatenated features
            fused = self._fuse_without_temporal(total_static, total_irr)
            
            swap_indices = torch.arange(irr.size(1)).roll(1)
            return fused, irr_swapped, swap_indices
            
        # For evaluation, apply fusion without swapping
        fused = self._fuse_without_temporal(static, irr)
        return fused, irr, None
    
    def _fuse_without_temporal(self, static, irr):
        """Modified fusion for concatenated features (Visual)
        Args:
            static: Static forgery features [B, 8, T, D] or [B, 4, T, D]
            irr: Irrelevant features [B, 8, T, D] or [B, 4, T, D]
        """
        # Process all videos together through attention
        B, V, T, D = static.shape
        queries = static.reshape(1, B*V*T, D)  # [1, B*V*T, D]
        keys = irr.reshape(-1, B*V*T, D)       # [1, B*V*T, D]
        values = keys
        
        # Apply attention
        attn_output, _ = self.attention(queries, keys, values)
        
        # Reshape back to original dimensions
        fused = self.norm(attn_output.squeeze(0).reshape(B, V, T, D) + static)
        
        return fused
    
    def _fuse_without_temporal_audio(self, static, irr):
        """Audio-specific fusion for features with temporal dimension
        Args:
            static: Static forgery features [B, V, T, D] or [B, V, D]
            irr: Irrelevant features [B, V, T, D] or [B, V, D]
        """
        # Handle both temporal and non-temporal inputs
        if len(static.shape) == 4:  # [B, V, T, D] - new format with temporal dimension
            B, V, T, D = static.shape
            # Reshape to process temporal dimension
            static_flat = static.reshape(B*V*T, D)  # [B*V*T, D]
            irr_flat = irr.reshape(B*V*T, D)        # [B*V*T, D]
            
            # Apply attention across all temporal features
            queries = static_flat.unsqueeze(0)  # [1, B*V*T, D]
            keys = irr_flat.unsqueeze(0)        # [1, B*V*T, D]
            values = keys
            
            attn_output, _ = self.attention(queries, keys, values)
            
            # Reshape back to original dimensions
            fused = self.norm(attn_output.squeeze(0).reshape(B, V, T, D) + static)
            
        else:  # [B, V, D] - old format without temporal dimension
            B, V, D = static.shape
            queries = static.reshape(1, B*V, D)  # [1, B*V, D]
            keys = irr.reshape(-1, B*V, D)       # [1, B*V, D]
            values = keys
            
            # Apply attention
            attn_output, _ = self.attention(queries, keys, values)
            
            # Reshape back to original dimensions
            fused = self.norm(attn_output.squeeze(0).reshape(B, V, D) + static)
        
        return fused

class VisualDecoder(nn.Module):
    def __init__(self, feature_dim=512, efficientnet_version='b2'):
        super().__init__()
        self.fusion = FeatureFusionModule(feature_dim)
        
        # Get target size from configs (Override to 140 based on dataset)
        self.target_size = 140 # MODEL_CONFIGS[efficientnet_version]['input_size'][0]  # 250 for b2
        self.feature_map = MODEL_CONFIGS[efficientnet_version]['feature_map'][0]  # 8 for b2
        initial_size = self.feature_map * 2  # 16 for b2

        channels = 64
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, initial_size * initial_size * channels)
        )
        
        # Simplified upsampling layers
        self.conv_transpose = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, vf_static, virr, enable_swap=True):
        
        # Get fusion outputs including swap indices
        fused_features, virr_swapped, swap_indices = self.fusion(
            vf_static, virr, enable_swap
        )
        
        # Process each frame independently
        B, V, T, D = fused_features.shape
        x = self.mlp(fused_features)  # [B, V, T, initial_size*initial_size*64]
        
        
        # Reshape to image dimensions
        initial_size = self.feature_map * 2
        x = x.reshape(-1, 64, initial_size, initial_size)  # [B*V*T, 64, 16, 16]
        
        # Upsample to intermediate size
        x = self.conv_transpose(x)  # [B*V*T, 3, 128, 128]

        # Ensure output size matches target size
        if x.shape[-1] != self.target_size:
            x = F.interpolate(
                x, size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            )

        # Reshape back to [B, V, T, C, H, W] format
        x = x.reshape(B, V, T, 3, self.target_size, self.target_size)

        return x,swap_indices  # torch.Size([1, 8, 25, 3, 250, 250]) reconstructed videos

class AudioDecoder(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.fusion = FeatureFusionModule(feature_dim)
        
        # Temporal upsampling: 25 -> 32 time steps (single step)
        self.temporal_upsample = nn.Upsample(size=32, mode='linear', align_corners=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64*4*8)  # 2048 for [64, 4, 8] reshape
        )
        self.conv_transpose = nn.Sequential(
            # Upsample time dimension 8->16, reduce channels 64->32, keep frequency at 4
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            # Upsample time dimension 16->32, reduce channels 32->16, keep frequency at 4
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            # Upsample frequency dimension 4->64, keep time at 32, reduce channels 16->1
            nn.ConvTranspose2d(16, 1, kernel_size=(16, 1), stride=(16, 1), padding=0),
            nn.Tanh()
        )
        
    def forward(self, af_static, airr, enable_swap=True):
        # Get fusion outputs including swap indices
        
        # Use audio-specific fusion method
        if enable_swap and self.training:
            # Don't swap static features, only swap irrelevant features
            airr_swapped = airr.clone().roll(1, dims=1)  # Roll along video dimension
            
            # Concatenate original static with itself to treat each as separate video
            total_static = torch.cat([af_static, af_static], dim=1)  # [B, 8, T, D] or [B, 8, D]
            
            # Concatenate original and swapped irrelevant features
            total_airr = torch.cat([airr, airr_swapped], dim=1)  # [B, 8, T, D] or [B, 8, D]
            
            # Get fusion for concatenated features using unified method
            fused = self.fusion._fuse_without_temporal(total_static, total_airr)
            swap_indices = torch.arange(airr.size(1)).roll(1)
            fused_features = fused
        else:
            # For evaluation, apply fusion without swapping using unified method
            fused_features = self.fusion._fuse_without_temporal(af_static, airr)
            airr_swapped = airr
            swap_indices = None
        
        # Handle temporal dimension for upsampling
        if len(fused_features.shape) == 4:  # [B, V, T, D] - new format with temporal dimension
            B, V, T, D = fused_features.shape
            
            # Reshape for temporal upsampling: [B*V, D, T]
            fused_temporal = fused_features.permute(0, 1, 3, 2).reshape(B*V, D, T)
            
            # Temporal upsampling: 25 -> 32 (single step)
            upsampled_temporal = self.temporal_upsample(fused_temporal)  # [B*V, D, 32]
            
            # Reshape back: [B, V, 32, D]
            fused_features = upsampled_temporal.reshape(B, V, D, 32).permute(0, 1, 3, 2)
            
            # Average pool temporal dimension to get single feature vector per video
            fused_features = fused_features.mean(dim=2)  # [B, V, D]
        
        # Process fused features through decoder
        x = self.mlp(fused_features)
        x = x.reshape(-1, 64, 4, 8)  # Reshape to [B*V, 64, 4, 8]
        x = self.conv_transpose(x)
        # Reshape to [B, V, 1, 128, 32] to maintain batch size and add channel dimension
        B, V = fused_features.shape[:2]
        x = x.reshape(B, V, 1, 64, 32)
        
        # Return only reconstructed audio
        return x,swap_indices  # torch.Size([1, 8, 1, 64, 32])

class Reconstruction(nn.Module):
    def __init__(self, feature_dim=512, efficientnet_version='b2'):
        super().__init__()
        self.visual_decoder = VisualDecoder(feature_dim, efficientnet_version)
        self.audio_decoder = AudioDecoder(feature_dim)
        
    def forward(self, vf_static, virr, af_static, airr, enable_swap=True):
        """Process only static features through decoders
        Args:
            vf_static: Visual forgery static features [B, 4, 25, 512]
            virr: Visual irrelevant features [B, 4, 25, 512]
            af_static: Audio forgery static features [B, 4, 25, 512]
            airr: Audio irrelevant features [B, 4, 25, 512]
            enable_swap: Enable feature swapping during training
        """
        # Process visual features
        visual_fused, v_indices = self.visual_decoder(
            vf_static, virr, enable_swap
        )
        
        # Process audio features
        audio_fused, a_indices = self.audio_decoder(
            af_static, airr, enable_swap
        )
        
        outputs = {
            'visual': visual_fused,  # [B, 8, 25, 3, H, W] reconstructed videos
            'audio': audio_fused,    # [B, 8, 1, 64, 32] reconstructed audio
        }
        
        # Include swap indices for correct label mapping during training
        if self.training and enable_swap:
            outputs['swap_info'] = {
                'visual_swap': v_indices,
                'audio_swap': a_indices
            }
        
        return outputs


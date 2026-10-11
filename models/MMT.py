# Multimodal Transformer (MMT)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class MultiLevelCrossAttention(nn.Module):
    """Multi-Level Cross Attention with Hierarchical Pyramid Architecture"""
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        
        # Temporal downsampling for different levels
        self.fine_pool = nn.Identity()  # 25 steps (no downsampling)
        self.medium_pool = nn.AdaptiveAvgPool1d(15)  # 15 steps (25→15)
        self.coarse_pool = nn.AdaptiveAvgPool1d(8)  # 8 steps (25→8)
        
        # Cross-attention at each level (increased heads from 8 to 12)
        self.attention_levels = nn.ModuleDict({
            'fine': nn.MultiheadAttention(dim, 16, batch_first=True),  # 8→16
            'medium': nn.MultiheadAttention(dim, 16, batch_first=True),  # 8→16
            'coarse': nn.MultiheadAttention(dim, 16, batch_first=True)  # 8→16
        })
        
        # Hierarchical pyramid connections (coarse → medium → fine)
        self.coarse_to_medium = nn.Linear(dim, dim)
        self.medium_to_fine = nn.Linear(dim, dim)
        
        # Level fusion with dropout
        self.level_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(dim, dim),
            nn.Dropout(0.3)  # Add dropout before output
        )
        

    def forward(self, visual_features, audio_features):
        # visual_features, audio_features: [B, 25, 512]
        B, T, D = visual_features.shape
        
        # Step 1: Create multi-scale representations
        levels = {}
        
        # Fine level (25 steps)
        levels['fine'] = {
            'visual': self.fine_pool(visual_features.transpose(1,2)).transpose(1,2),
            'audio': self.fine_pool(audio_features.transpose(1,2)).transpose(1,2)
        }
        
        # Medium level (15 steps)  
        levels['medium'] = {
            'visual': self.medium_pool(visual_features.transpose(1,2)).transpose(1,2),
            'audio': self.medium_pool(audio_features.transpose(1,2)).transpose(1,2)
        }
        
        # Coarse level (8 steps)
        levels['coarse'] = {
            'visual': self.coarse_pool(visual_features.transpose(1,2)).transpose(1,2),
            'audio': self.coarse_pool(audio_features.transpose(1,2)).transpose(1,2)
        }
        
        # Step 2: Hierarchical pyramid processing (coarse → medium → fine)
        attended_features = {}
        
        # Coarse level processing
        coarse_attended, _ = self.attention_levels['coarse'](
            levels['coarse']['visual'], 
            levels['coarse']['audio'], 
            levels['coarse']['audio']
        )
        attended_features['coarse'] = coarse_attended
        
        # Medium level with coarse guidance
        coarse_guidance = F.interpolate(
            coarse_attended.transpose(1,2), 
            size=15, mode='linear', align_corners=False
        ).transpose(1,2)
        medium_enhanced = levels['medium']['visual'] + self.coarse_to_medium(coarse_guidance)
        
        medium_attended, _ = self.attention_levels['medium'](
            medium_enhanced,
            levels['medium']['audio'],
            levels['medium']['audio']
        )
        attended_features['medium'] = medium_attended
        
        # Fine level with medium guidance
        medium_guidance = F.interpolate(
            medium_attended.transpose(1,2),
            size=25, mode='linear', align_corners=False  
        ).transpose(1,2)
        fine_enhanced = levels['fine']['visual'] + self.medium_to_fine(medium_guidance)
        
        fine_attended, _ = self.attention_levels['fine'](
            fine_enhanced,
            levels['fine']['audio'], 
            levels['fine']['audio']
        )
        attended_features['fine'] = fine_attended
        
        # Step 3: Combine multi-level features
        # Upsample all to fine resolution (25 steps)
        coarse_upsampled = F.interpolate(
            attended_features['coarse'].transpose(1,2),
            size=25, mode='linear', align_corners=False
        ).transpose(1,2)
        
        medium_upsampled = F.interpolate(
            attended_features['medium'].transpose(1,2), 
            size=25, mode='linear', align_corners=False
        ).transpose(1,2)
        
        # Concatenate and fuse
        multi_level_features = torch.cat([
            attended_features['fine'],
            medium_upsampled, 
            coarse_upsampled
        ], dim=-1)  # [B, 25, 3*512]
        
        fused_features = self.level_fusion(multi_level_features)  # [B, 25, 512]
        # Add residual connection for better gradient flow
        fused_features = fused_features + attended_features['fine']  # Residual connection
        
        return fused_features

class ModalitySeparation(nn.Module):
    """Gating mechanism to separate modality-common vs modality-specific features"""
    def __init__(self, dim=512):
        super().__init__()
        
       
        
        # Gating networks with dropout
        self.common_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        
        self.specific_gate = nn.Sequential(
            nn.Linear(dim, dim // 2), 
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        
        # Feature projections with dropout
        self.common_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.3)
        )
        self.specific_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.3)
        )
        
        # Layer normalization for stable training
        self.norm_common = nn.LayerNorm(dim)
        self.norm_specific = nn.LayerNorm(dim)
        
        
    def forward(self, fused_features):
        # fused_features: [B, 25, 512] from multi-level attention
        
        # Generate gates
        common_gate = self.common_gate(fused_features)  # [B, 25, 512]
        specific_gate = self.specific_gate(fused_features)  # [B, 25, 512]
        
        # Apply gating (ensure they sum to 1)
        gate_sum = common_gate + specific_gate + 1e-8
        common_gate = common_gate / gate_sum
        specific_gate = specific_gate / gate_sum
    
        
        # Separate features with layer normalization for stable training
        common_features = self.norm_common(self.common_projection(fused_features * common_gate))
        specific_features = self.norm_specific(self.specific_projection(fused_features * specific_gate))
    
        
        return common_features, specific_features

class EfficientTemporalPooling(nn.Module):
    """Multi-scale temporal convolutions for pattern-preserving pooling"""
    def __init__(self, dim=512):
        super().__init__()
        
        # Multi-scale temporal convolutions
        # Use dim//4 to ensure proper concatenation: 128*4 = 512
        conv_dim = dim // 4  # 512 // 4 = 128
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(dim, conv_dim, kernel_size=1, stride=1),   # Fine: 25→25
            nn.Conv1d(dim, conv_dim, kernel_size=3, stride=3),   # Medium: 25→8
            nn.Conv1d(dim, conv_dim, kernel_size=5, stride=5),   # Coarse: 25→5  
            nn.Conv1d(dim, conv_dim, kernel_size=25, stride=25), # Extra Coarse: 25→1
        ])
        
        # Adaptive pooling to same size
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),  # All → 1 time step
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveAvgPool1d(1)
        ])
        
        # Final fusion
        self.fusion = nn.Linear(dim, dim)
        
    def forward(self, x):  # [B, 25, 512]

        x = x.transpose(1, 2)  # [B, 512, 25]
        
        # Multi-scale processing
        scales = []
        for i, (conv, pool) in enumerate(zip(self.temporal_convs, self.adaptive_pools)):
            scale_out = conv(x)      # Different temporal scales
            pooled = pool(scale_out) # [B, dim//3, 1]
            scales.append(pooled.squeeze(-1))  # [B, conv_dim]
        
        # Combine scales
        combined = torch.cat(scales, dim=1)  # [B, dim]
        output = self.fusion(combined)       # [B, 512]

        
        return output

class CrossModalityMatchingLayer(nn.Module):
    """Cross-Modality Matching Layer for feature alignment"""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_a = nn.Linear(dim, dim)
        self.fusion = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, v_feat: torch.Tensor, a_feat: torch.Tensor) -> torch.Tensor:
        # Project features
        v_proj = self.proj_v(v_feat)
        a_proj = self.proj_a(a_feat)
        
        # Compute cross-attention
        attn = torch.matmul(v_proj, a_proj.transpose(-2, -1))
        attn = attn / self.temperature
        attn = torch.softmax(attn, dim=-1)
        
        # Cross-modal fusion
        v_aligned = torch.matmul(attn, a_feat)
        fused = self.fusion(torch.cat([v_feat, v_aligned], dim=-1))
        return fused

class MultiModalTransformer(nn.Module):
    """Multimodal Transformer for audio-visual fusion"""
    def __init__(
        self, 
        dim: int = 512, 
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Add new components
        self.multi_level_attention = MultiLevelCrossAttention(dim, num_heads)
        self.modality_separation = ModalitySeparation(dim)
        
        # Efficient temporal pooling (separate for common and specific features)
        self.temporal_pooling_common = EfficientTemporalPooling(dim)
        self.temporal_pooling_specific = EfficientTemporalPooling(dim)
        
        # Keep existing components
        self.proj_v = nn.Linear(dim, dim)
        self.proj_a = nn.Linear(dim, dim)
        

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Enhanced MMT Forward with Multi-Level Cross Attention and Modality Separation
        
        Args:
            batch_data: Dictionary containing:
                vfs: Video forgery features [B, 4, 25, 512]
                afs: Audio forgery features [B, 4, 25, 512]
        """
        # Extract features
        vfs = batch_data['vfs']  # [B, 4, 25, 512]
        afs = batch_data['afs']  # [B, 4, 25, 512]
        


        B, V, T, D = vfs.size()
        
        # Process each video in the set
        av_common_features = []
        av_specific_features = []
        
        
        for v in range(V):
            
            # Extract temporal slices
            vfs_v = vfs[:, v]  # [B, 25, 512]
            afs_v = afs[:, v]  # [B, 25, 512]
            

            
            # Project features
            vfs_v = self.proj_v(vfs_v)
            afs_v = self.proj_a(afs_v)
            
            
            # Multi-level cross attention
            multi_level_features = self.multi_level_attention(vfs_v, afs_v)  # [B, 25, 512]

            
            # Separate modality-common vs modality-specific
            common_features, specific_features = self.modality_separation(multi_level_features)
            
            # Use EfficientTemporalPooling instead of mean pooling to preserve temporal patterns
            common_pooled = self.temporal_pooling_common(common_features)  # [B, 512]
            specific_pooled = self.temporal_pooling_specific(specific_features)  # [B, 512]
            
            av_common_features.append(common_pooled)
            av_specific_features.append(specific_pooled)
        
        # Stack all videos
        av_common = torch.stack(av_common_features, dim=1)  # [B, 4, 512]
        av_specific = torch.stack(av_specific_features, dim=1)  # [B, 4, 512]
        

        
        return {
            'modality_common': av_common,     # AV_cf - for binary classification
            'modality_specific': av_specific, # AV_sf - for multi-class classification
        }

class AudioVisualFusion(nn.Module):
    """Audio-Visual Fusion Module"""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.static_mmt = MultiModalTransformer(dim=dim)
        
        # Remove temporal components
        self.temporal_mmt = None
        self.temporal_fusion = None
        
        self.static_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, features: Dict[str, Dict]) -> Dict[str, Dict]:

        outputs = {}
        
        orig_static = features['static']['original']
        
        original_static = self.static_mmt(orig_static)
        
            
        outputs['static'] = {
            'original': original_static,
            'combined': {
                'modality_common': original_static['modality_common'],      # [B, 4, 512]
                'modality_specific': original_static['modality_specific'],  # [B, 4, 512]
            }
        }
        
        return outputs
def get_av_fusion(dim: int = 512) -> AudioVisualFusion:
    """Helper function to create audio-visual fusion module"""
    return AudioVisualFusion(dim=dim)

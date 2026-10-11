"""
Sophisticated Forgery Branch Module
Multi-scale temporal analysis + attention for subtle artifact detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SophisticatedForgeryBranch(nn.Module):
    """
    Multi-scale temporal analysis + attention for subtle artifact detection
    Expected input: [B*4, 256, 75] (shared temporal features)
    Expected output: [B*4, 512, 75] (sophisticated artifact features)
    """
    def __init__(self, input_dim=256, output_dim=512):
        super(SophisticatedForgeryBranch, self).__init__()
        
        
        # Multi-scale temporal analysis for subtle artifacts
        self.multi_scale_conv = nn.ModuleDict({
            'fine': nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),    # Local glitches
            'medium': nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),  # Rhythm breaks
            'coarse': nn.Conv1d(input_dim, 128, kernel_size=11, padding=5)  # Global inconsistencies
        })
        
        # Artifact attention for localization
        self.artifact_attention = nn.MultiheadAttention(
            embed_dim=384,  # 128*3
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # High-frequency artifact detector
        self.hf_detector = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Final forgery projection
        self.forgery_projection = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
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
            elif isinstance(m, nn.MultiheadAttention):
                # Initialize attention weights
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.constant_(m.in_proj_bias, 0)
                nn.init.xavier_uniform_(m.out_proj.weight)
                nn.init.constant_(m.out_proj.bias, 0)
        
    def forward(self, x):
        
        # Multi-scale temporal analysis
        scale_features = []
        for scale_name, conv in self.multi_scale_conv.items():
            scale_feat = conv(x)  # [B*4, 128, 75]
            scale_features.append(scale_feat)
        
        # Combine multi-scale features
        combined = torch.cat(scale_features, dim=1)  # [B*4, 384, 75]
        
        # Artifact attention for localization
        B, C, T = combined.shape
        combined_att = combined.permute(0, 2, 1)  # [B*4, 75, 384]
        
        
        attended, attention_weights = self.artifact_attention(
            combined_att, combined_att, combined_att
        )  # [B*4, 75, 384]
        
        attended = attended.permute(0, 2, 1)  # [B*4, 384, 75]
        
        # High-frequency artifact detection
        hf_features = self.hf_detector(attended)  # [B*4, 128, 75]
        
        # Final forgery projection
        forgery_features = self.forgery_projection(hf_features)  # [B*4, 512, 75]
        
        # Apply dropout
        forgery_features = self.dropout(forgery_features)
        
        return forgery_features


class EnhancedForgeryBranch(nn.Module):
    """
    Enhanced version with additional temporal modeling and artifact-specific processing
    Expected input: [B*4, 256, 75] (shared temporal features)
    Expected output: [B*4, 512, 75] (enhanced artifact features)
    """
    def __init__(self, input_dim=256, output_dim=512):
        super(EnhancedForgeryBranch, self).__init__()
        
        # Temporal artifact detector (focuses on temporal inconsistencies)
        self.temporal_detector = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=7, padding=3),  # Larger kernel for temporal patterns
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Spectral artifact detector (focuses on frequency domain artifacts)
        self.spectral_detector = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Cross-attention between temporal and spectral features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256,  # 128*2
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Artifact fusion and projection
        self.artifact_fusion = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
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
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.constant_(m.in_proj_bias, 0)
                nn.init.xavier_uniform_(m.out_proj.weight)
                nn.init.constant_(m.out_proj.bias, 0)
        
    def forward(self, x):
        
        # Temporal artifact detection
        temporal_features = self.temporal_detector(x)  # [B*4, 128, 75]
        
        # Spectral artifact detection
        spectral_features = self.spectral_detector(x)  # [B*4, 128, 75]

        
        # Combine temporal and spectral features
        combined_features = torch.cat([temporal_features, spectral_features], dim=1)  # [B*4, 256, 75]

        
        # Cross-attention between temporal and spectral
        B, C, T = combined_features.shape
        features_att = combined_features.permute(0, 2, 1)  # [B*4, 75, 256]
        
        
        attended_features, attention_weights = self.cross_attention(
            features_att, features_att, features_att
        )  # [B*4, 75, 256]
        
        attended_features = attended_features.permute(0, 2, 1)  # [B*4, 256, 75]

        
        # Artifact fusion and projection
        forgery_features = self.artifact_fusion(attended_features)  # [B*4, 512, 75]

        
        # Apply dropout
        forgery_features = self.dropout(forgery_features)
        return forgery_features


def test_forgery_branch():
    """Test function to verify forgery branch implementation"""
    
    # Test parameters
    batch_size = 2
    input_dim = 256
    output_dim = 512
    sequence_length = 75
    
    # Create test input
    x = torch.randn(batch_size, input_dim, sequence_length)
    
    # Test SophisticatedForgeryBranch
    forgery_branch = SophisticatedForgeryBranch(input_dim=input_dim, output_dim=output_dim)
    output1 = forgery_branch(x)
    
    # Test EnhancedForgeryBranch
    enhanced_branch = EnhancedForgeryBranch(input_dim=input_dim, output_dim=output_dim)
    output2 = enhanced_branch(x)
    
    # Verify output shapes
    expected_shape = (batch_size, output_dim, sequence_length)
    assert output1.shape == expected_shape, f"Expected {expected_shape}, got {output1.shape}"
    assert output2.shape == expected_shape, f"Expected {expected_shape}, got {output2.shape}"
    
    return output1, output2


if __name__ == "__main__":
    test_forgery_branch()

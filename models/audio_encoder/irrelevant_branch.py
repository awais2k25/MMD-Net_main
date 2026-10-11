"""
Simple Irrelevant Branch Module
Simple content extraction for identity and speaker characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleIrrelevantBranch(nn.Module):
    """
    Simple content extraction for identity and speaker characteristics
    Expected input: [B*4, 256, 25] (shared temporal features)
    Expected output: [B*4, 512, 25] (simple content features)
    """
    def __init__(self, input_dim=256, output_dim=512):
        super(SimpleIrrelevantBranch, self).__init__()
        
        
        # Step 1: Extract content features with larger kernels (captures speaker characteristics)
        self.content_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=7, padding=3),  # Larger kernel for content
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Step 2: Simple temporal modeling (preserves temporal variations)
        # Uses smaller kernels to capture natural speech rhythm
        self.temporal_modeling = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
        )
        
        # Step 3: Final projection to output dimension
        self.output_projection = nn.Sequential(
            nn.Conv1d(256, output_dim, kernel_size=1),
            nn.Dropout(0.1)  # Add dropout before output
        )
        
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
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):  # [B*4, 256, 25]
        
        # Extract content features (speaker characteristics)
        content_features = self.content_extractor(x)  # [B*4, 256, 25]
        
        
        # Apply simple temporal modeling (natural speech rhythm)
        temporal_features = self.temporal_modeling(content_features)  # [B*4, 256, 25]
        
        # Project to output dimension
        # Note: output_projection is now a Sequential, so dropout is included
        irrelevant_features = self.output_projection(temporal_features)  # [B*4, 512, 25]
        return irrelevant_features  # [B*4, 512, 25]


class PoolingIrrelevantBranch(nn.Module):
    """
    Alternative irrelevant branch using different pooling strategies
    Expected input: [B*4, 256, 25] (shared temporal features)
    Expected output: [B*4, 512, 25] (pooled content features)
    """
    def __init__(self, input_dim=256, output_dim=512):
        super(PoolingIrrelevantBranch, self).__init__()
        
        
        # Multiple pooling strategies for content extraction
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Content fusion
        self.content_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, 512),  # 2 * 256 = 512
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        # Temporal broadcast
        self.temporal_broadcast = nn.Conv1d(512, output_dim, kernel_size=1)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Max pooling
        max_pooled = self.max_pool(x)  # [B*4, 256, 1]
        max_pooled = max_pooled.squeeze(-1)  # [B*4, 256]
        
        
        # Average pooling
        avg_pooled = self.avg_pool(x)  # [B*4, 256, 1]
        avg_pooled = avg_pooled.squeeze(-1)  # [B*4, 256]
        
        
        # Combine pooling results
        combined_pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # [B*4, 512]
        
        
        # Content fusion
        fused_content = self.content_fusion(combined_pooled)  # [B*4, 512]
        
        
        # Broadcast back to temporal dimension
        fused_content = fused_content.unsqueeze(-1)  # [B*4, 512, 1]
        
        irrelevant_features = self.temporal_broadcast(fused_content)  # [B*4, 512, 75]
        
        return irrelevant_features


class MinimalIrrelevantBranch(nn.Module):
    """
    Minimal irrelevant branch with simplest possible architecture
    Expected input: [B*4, 256, 75] (shared temporal features)
    Expected output: [B*4, 512, 75] (minimal content features)
    """
    def __init__(self, input_dim=256, output_dim=512):
        super(MinimalIrrelevantBranch, self).__init__()
        
        
        # Single global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        
        # Single linear projection
        self.content_projection = nn.Linear(input_dim, output_dim)
        
        
        # Broadcast to temporal dimension
        self.temporal_broadcast = nn.Conv1d(output_dim, output_dim, kernel_size=1)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        
        # Global pooling
        pooled = self.global_pool(x)  # [B*4, 256, 1]
        pooled = pooled.squeeze(-1)  # [B*4, 256]
        
        # Content projection
        projected = self.content_projection(pooled)  # [B*4, 512]
        
        
        # Broadcast to temporal dimension
        projected = projected.unsqueeze(-1)  # [B*4, 512, 1]
        irrelevant_features = self.temporal_broadcast(projected)  # [B*4, 512, 75]
        
        return irrelevant_features


def test_irrelevant_branch():
    """Test function to verify irrelevant branch implementation"""
    
    # Test parameters
    batch_size = 2
    input_dim = 256
    output_dim = 512
    sequence_length = 75
    
    # Create test input
    x = torch.randn(batch_size, input_dim, sequence_length)
    
    irrelevant_branch = SimpleIrrelevantBranch(input_dim=input_dim, output_dim=output_dim)
    output1 = irrelevant_branch(x)

    pooling_branch = PoolingIrrelevantBranch(input_dim=input_dim, output_dim=output_dim)
    output2 = pooling_branch(x)

    minimal_branch = MinimalIrrelevantBranch(input_dim=input_dim, output_dim=output_dim)
    output3 = minimal_branch(x)

    expected_shape = (batch_size, output_dim, sequence_length)
    assert output1.shape == expected_shape, f"Expected {expected_shape}, got {output1.shape}"
    assert output2.shape == expected_shape, f"Expected {expected_shape}, got {output2.shape}"
    assert output3.shape == expected_shape, f"Expected {expected_shape}, got {output3.shape}"
    
    return output1, output2, output3


if __name__ == "__main__":
    test_irrelevant_branch()

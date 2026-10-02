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
        
        #print(f"[DEBUG] SimpleIrrelevantBranch init: input_dim={input_dim}, "
        #      f"output_dim={output_dim}")
        
        # Step 1: Extract content features with larger kernels (captures speaker characteristics)
        self.content_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=7, padding=3),  # Larger kernel for content
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        #print(f"[DEBUG] Content extractor: {input_dim} -> 256 -> 256, kernel=7")
        
        # Step 2: Simple temporal modeling (preserves temporal variations)
        # Uses smaller kernels to capture natural speech rhythm
        self.temporal_modeling = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
        )
        #print(f"[DEBUG] Temporal modeling: 256 -> 256, kernel=3, dropout=0.1")
        
        # Step 3: Final projection to output dimension
        self.output_projection = nn.Sequential(
            nn.Conv1d(256, output_dim, kernel_size=1),
            nn.Dropout(0.1)  # Add dropout before output
        )
        #print(f"[DEBUG] Output projection: 256 -> {output_dim}, kernel=1, dropout=0.1")
        
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
        #print(f"[DEBUG] SimpleIrrelevantBranch forward input: {x.shape}")
        
        # Extract content features (speaker characteristics)
        content_features = self.content_extractor(x)  # [B*4, 256, 25]
        #print(f"[DEBUG] After content extractor: {content_features.shape}")
        
        # Apply simple temporal modeling (natural speech rhythm)
        temporal_features = self.temporal_modeling(content_features)  # [B*4, 256, 25]
        #print(f"[DEBUG] After temporal modeling: {temporal_features.shape}")
        
        # Project to output dimension
        # Note: output_projection is now a Sequential, so dropout is included
        irrelevant_features = self.output_projection(temporal_features)  # [B*4, 512, 25]
        #print(f"[DEBUG] After output projection: {irrelevant_features.shape}")
        
        #print(f"[DEBUG] SimpleIrrelevantBranch final output: {irrelevant_features.shape}")
        return irrelevant_features  # [B*4, 512, 25]


class PoolingIrrelevantBranch(nn.Module):
    """
    Alternative irrelevant branch using different pooling strategies
    Expected input: [B*4, 256, 25] (shared temporal features)
    Expected output: [B*4, 512, 25] (pooled content features)
    """
    def __init__(self, input_dim=256, output_dim=512):
        super(PoolingIrrelevantBranch, self).__init__()
        
        #print(f"[DEBUG] PoolingIrrelevantBranch init: input_dim={input_dim}, "
        #   f"output_dim={output_dim}")
        
        # Multiple pooling strategies for content extraction
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        #print(f"[DEBUG] Pooling strategies: MaxPool + AvgPool")
        
        # Content fusion
        self.content_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, 512),  # 2 * 256 = 512
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        #print(f"[DEBUG] Content fusion: {input_dim * 2} -> 512 -> 512")
        
        # Temporal broadcast
        self.temporal_broadcast = nn.Conv1d(512, output_dim, kernel_size=1)
        #print(f"[DEBUG] Temporal broadcast: 512 -> {output_dim}, kernel=1")
        
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
        #print(f"[DEBUG] PoolingIrrelevantBranch forward input: {x.shape}")
        
        # Max pooling
        max_pooled = self.max_pool(x)  # [B*4, 256, 1]
        #print(f"[DEBUG] After max pool: {max_pooled.shape}")
        max_pooled = max_pooled.squeeze(-1)  # [B*4, 256]
        #print(f"[DEBUG] After max pool squeeze: {max_pooled.shape}")
        
        # Average pooling
        avg_pooled = self.avg_pool(x)  # [B*4, 256, 1]
        #print(f"[DEBUG] After avg pool: {avg_pooled.shape}")
        avg_pooled = avg_pooled.squeeze(-1)  # [B*4, 256]
        #print(f"[DEBUG] After avg pool squeeze: {avg_pooled.shape}")
        
        # Combine pooling results
        combined_pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # [B*4, 512]
        #print(f"[DEBUG] Combined pooled: {combined_pooled.shape}")
        
        # Content fusion
        fused_content = self.content_fusion(combined_pooled)  # [B*4, 512]
        #print(f"[DEBUG] After content fusion: {fused_content.shape}")
        
        # Broadcast back to temporal dimension
        fused_content = fused_content.unsqueeze(-1)  # [B*4, 512, 1]
        #print(f"[DEBUG] After unsqueeze: {fused_content.shape}")
        irrelevant_features = self.temporal_broadcast(fused_content)  # [B*4, 512, 75]
        #print(f"[DEBUG] After temporal broadcast: {irrelevant_features.shape}")
        
        #print(f"[DEBUG] PoolingIrrelevantBranch final output: {irrelevant_features.shape}")
        return irrelevant_features


class MinimalIrrelevantBranch(nn.Module):
    """
    Minimal irrelevant branch with simplest possible architecture
    Expected input: [B*4, 256, 75] (shared temporal features)
    Expected output: [B*4, 512, 75] (minimal content features)
    """
    def __init__(self, input_dim=256, output_dim=512):
        super(MinimalIrrelevantBranch, self).__init__()
        
        #print(f"[DEBUG] MinimalIrrelevantBranch init: input_dim={input_dim}, "
        # f"output_dim={output_dim}")
        
        # Single global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        #print(f"[DEBUG] Global pool: AdaptiveAvgPool1d(1)")
        
        # Single linear projection
        self.content_projection = nn.Linear(input_dim, output_dim)
        #print(f"[DEBUG] Content projection: {input_dim} -> {output_dim}")
        
        # Broadcast to temporal dimension
        self.temporal_broadcast = nn.Conv1d(output_dim, output_dim, kernel_size=1)
        #print(f"[DEBUG] Temporal broadcast: {output_dim} -> {output_dim}, kernel=1")
        
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
        #print(f"[DEBUG] MinimalIrrelevantBranch forward input: {x.shape}")
        
        # Global pooling
        pooled = self.global_pool(x)  # [B*4, 256, 1]
        #print(f"[DEBUG] After global pool: {pooled.shape}")
        pooled = pooled.squeeze(-1)  # [B*4, 256]
        #print(f"[DEBUG] After squeeze: {pooled.shape}")
        
        # Content projection
        projected = self.content_projection(pooled)  # [B*4, 512]
        #print(f"[DEBUG] After content projection: {projected.shape}")
        
        # Broadcast to temporal dimension
        projected = projected.unsqueeze(-1)  # [B*4, 512, 1]
        #print(f"[DEBUG] After unsqueeze: {projected.shape}")
        irrelevant_features = self.temporal_broadcast(projected)  # [B*4, 512, 75]
        #print(f"[DEBUG] After temporal broadcast: {irrelevant_features.shape}")
        
        #print(f"[DEBUG] MinimalIrrelevantBranch final output: {irrelevant_features.shape}")
        return irrelevant_features


def test_irrelevant_branch():
    """Test function to verify irrelevant branch implementation"""
    #print("\n=== Testing Irrelevant Branch Implementation ===")
    
    # Test parameters
    batch_size = 2
    input_dim = 256
    output_dim = 512
    sequence_length = 75
    
    # Create test input
    x = torch.randn(batch_size, input_dim, sequence_length)
    #print(f"[TEST] Input shape: {x.shape}")
    
    # Test SimpleIrrelevantBranch
    #print("\n--- Testing SimpleIrrelevantBranch ---")
    irrelevant_branch = SimpleIrrelevantBranch(input_dim=input_dim, output_dim=output_dim)
    output1 = irrelevant_branch(x)
    #print(f"[TEST] SimpleIrrelevantBranch output shape: {output1.shape}")
    
    # Test PoolingIrrelevantBranch
    #print("\n--- Testing PoolingIrrelevantBranch ---")
    pooling_branch = PoolingIrrelevantBranch(input_dim=input_dim, output_dim=output_dim)
    output2 = pooling_branch(x)
    #print(f"[TEST] PoolingIrrelevantBranch output shape: {output2.shape}")
    
    # Test MinimalIrrelevantBranch
    #print("\n--- Testing MinimalIrrelevantBranch ---")
    minimal_branch = MinimalIrrelevantBranch(input_dim=input_dim, output_dim=output_dim)
    output3 = minimal_branch(x)
    #print(f"[TEST] MinimalIrrelevantBranch output shape: {output3.shape}")
    
    # Verify output shapes
    expected_shape = (batch_size, output_dim, sequence_length)
    assert output1.shape == expected_shape, f"Expected {expected_shape}, got {output1.shape}"
    assert output2.shape == expected_shape, f"Expected {expected_shape}, got {output2.shape}"
    assert output3.shape == expected_shape, f"Expected {expected_shape}, got {output3.shape}"
    
    #print("✅ Irrelevant branch tests passed!")
    return output1, output2, output3


if __name__ == "__main__":
    test_irrelevant_branch()

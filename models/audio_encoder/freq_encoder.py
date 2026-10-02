"""
Frequency Encoder Module
Processes 2D mel-spectrograms to extract frequency-domain features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyEncoder(nn.Module):
    """
    2D CNN for frequency dimension processing
    Expected input: [B*4, 1, 128, 32] (mel-spectrograms)
    Expected output: [B*4, 256, freq_reduced, 32] (frequency features)
    """
    def __init__(self, input_channels=1, output_channels=256):
        super(FrequencyEncoder, self).__init__()
        
        #print(f"[DEBUG] FrequencyEncoder init: input_channels={input_channels}, "
        #      f"output_channels={output_channels}")
        
        # First layer: initial frequency processing
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        #print(f"[DEBUG] Conv1: {input_channels} -> 64, kernel=(3,3), padding=1")
        
        # Second layer: further frequency reduction
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        #print(f"[DEBUG] Conv2: 64 -> 128, kernel=(3,3), padding=1")
        
        # Third layer: final frequency encoding
        self.conv3 = nn.Conv2d(128, output_channels, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu3 = nn.ReLU()
        #print(f"[DEBUG] Conv3: 128 -> {output_channels}, kernel=(3,3), padding=1")
        
        # Optional: Add frequency pooling to reduce frequency dimension
        self.freq_pool = nn.AdaptiveAvgPool2d((16, None))  # Reduce frequency to 16 bins
        #print(f"[DEBUG] Frequency pooling: AdaptiveAvgPool2d((16, None))")
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        #print(f"[DEBUG] FrequencyEncoder forward input: {x.shape}")
        
        # First convolution block
        x = self.conv1(x)
        #print(f"[DEBUG] After conv1: {x.shape}")
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second convolution block
        x = self.conv2(x)
        #print(f"[DEBUG] After conv2: {x.shape}")
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Third convolution block
        x = self.conv3(x)
        #print(f"[DEBUG] After conv3: {x.shape}")
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Frequency pooling to reduce frequency dimension
        x = self.freq_pool(x)
        #print(f"[DEBUG] After freq_pool: {x.shape}")
        
        #print(f"[DEBUG] FrequencyEncoder final output: {x.shape}")
        return x


class FrequencyEncoderWithStride(nn.Module):
    """
    Alternative frequency encoder using stride for dimension reduction
    Expected input: [B*4, 1, 128, 32] (mel-spectrograms)
    Expected output: [B*4, 256, freq_reduced, 32] (frequency features)
    """
    def __init__(self, input_channels=1, output_channels=256):
        super(FrequencyEncoderWithStride, self).__init__()
        
        #print(f"[DEBUG] FrequencyEncoderWithStride init: input_channels={input_channels}, "
        #      f"output_channels={output_channels}")
        
        # First layer: reduce frequency dimension with stride
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(4, 3), stride=(2, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        #print(f"[DEBUG] Conv1: {input_channels} -> 64, kernel=(4,3), stride=(2,1), padding=1")
        
        # Second layer: further frequency reduction
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 3), stride=(2, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        #print(f"[DEBUG] Conv2: 64 -> 128, kernel=(4,3), stride=(2,1), padding=1")
        
        # Third layer: final frequency encoding
        self.conv3 = nn.Conv2d(128, output_channels, kernel_size=(4, 3), stride=(2, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu3 = nn.ReLU()
        #print(f"[DEBUG] Conv3: 128 -> {output_channels}, kernel=(4,3), stride=(2,1), padding=1")
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        #print(f"[DEBUG] FrequencyEncoderWithStride forward input: {x.shape}")
        
        # First convolution block
        x = self.conv1(x)
        #print(f"[DEBUG] After conv1: {x.shape}")
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second convolution block
        x = self.conv2(x)
        #print(f"[DEBUG] After conv2: {x.shape}")
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Third convolution block
        x = self.conv3(x)
        #print(f"[DEBUG] After conv3: {x.shape}")
        x = self.bn3(x)
        x = self.relu3(x)
        
        #print(f"[DEBUG] FrequencyEncoderWithStride final output: {x.shape}")
        return x


def test_frequency_encoder():
    """Test function to verify frequency encoder implementation"""
    #print("\n=== Testing Frequency Encoder Implementation ===")
    
    # Test parameters
    batch_size = 2
    input_channels = 1
    freq_bins = 128
    time_steps = 32
    
    # Create test input (mel-spectrograms)
    x = torch.randn(batch_size, input_channels, freq_bins, time_steps)
    #print(f"[TEST] Input shape: {x.shape}")
    
    # Test FrequencyEncoder (with pooling)
    #print("\n--- Testing FrequencyEncoder (with pooling) ---")
    freq_encoder = FrequencyEncoder(input_channels=input_channels, output_channels=256)
    output1 = freq_encoder(x)
    #print(f"[TEST] FrequencyEncoder output shape: {output1.shape}")
    
    # Test FrequencyEncoderWithStride
    #print("\n--- Testing FrequencyEncoderWithStride ---")
    freq_encoder_stride = FrequencyEncoderWithStride(input_channels=input_channels, output_channels=256)
    output2 = freq_encoder_stride(x)
    #print(f"[TEST] FrequencyEncoderWithStride output shape: {output2.shape}")
    
    # Verify output shapes
    assert output1.shape[0] == batch_size, f"Batch size mismatch: {output1.shape[0]} != {batch_size}"
    assert output1.shape[1] == 256, f"Channel mismatch: {output1.shape[1]} != 256"
    assert output1.shape[3] == time_steps, f"Time dimension mismatch: {output1.shape[3]} != {time_steps}"
    
    assert output2.shape[0] == batch_size, f"Batch size mismatch: {output2.shape[0]} != {batch_size}"
    assert output2.shape[1] == 256, f"Channel mismatch: {output2.shape[1]} != 256"
    assert output2.shape[3] == time_steps, f"Time dimension mismatch: {output2.shape[3]} != {time_steps}"
    
    #print("✅ Frequency encoder tests passed!")
    return output1, output2


if __name__ == "__main__":
    test_frequency_encoder()

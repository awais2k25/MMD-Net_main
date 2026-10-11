"""
Temporal Convolutional Network (TCN) Module
Implements dilated causal convolutions for temporal sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    Remove padding from causal convolution to maintain causality
    Expected input: [B, C, T] with padding
    Expected output: [B, C, T] without padding
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        # Remove padding from the end to maintain causality
        output = x[:, :, :-self.chomp_size].contiguous()
        return output


class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated convolution + residual connection
    Expected input: [B, n_inputs, T]
    Expected output: [B, n_outputs, T]
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        
        # First dilated convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Combine layers
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Downsample for residual connection if needed
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better training stability"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            
    def forward(self, x):
        # Apply temporal convolutions
        out = self.net(x)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        # Final output with residual
        output = self.relu(out + res)
        
        return output


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with exponentially increasing dilation
    Expected input: [B, num_inputs, T]
    Expected output: [B, num_channels[-1], T]
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        

        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation: 1, 2, 4, 8, 16, 32
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size  # Causal padding
            
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size, 
                padding=padding, dropout=dropout
            )]
        
        self.network = nn.Sequential(*layers)
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(kernel_size, num_levels)

        
    def _calculate_receptive_field(self, kernel_size, num_levels):
        """Calculate the total receptive field of the TCN"""
        rf = 1
        for i in range(num_levels):
            dilation = 2 ** i
            rf += (kernel_size - 1) * dilation
        return rf
        
    def forward(self, x):
        # Apply all temporal blocks
        output = self.network(x)
        
        return output


def test_tcn():
    """Test function to verify TCN implementation"""

    
    # Test parameters
    batch_size = 2
    num_inputs = 256
    num_channels = [256, 256, 256, 256, 256, 256]
    sequence_length = 94
    
    # Create test input
    x = torch.randn(batch_size, num_inputs, sequence_length)
    
    # Create TCN
    tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=3, dropout=0.1)
    
    # Forward pass
    output = tcn(x)
    
    # Verify output shape
    expected_shape = (batch_size, num_channels[-1], sequence_length)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    return output


if __name__ == "__main__":
    test_tcn()

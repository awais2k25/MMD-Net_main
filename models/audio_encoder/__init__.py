"""
Audio Encoder Package
Optimal Asymmetric TCN-based Audio Encoder for Deepfake Detection

This package contains all components for the Optimal Asymmetric Audio Encoder:
- TCN: Temporal Convolutional Network for temporal sequence modeling
- Frequency Encoder: 2D CNN for frequency dimension processing
- Forgery Branch: Sophisticated multi-scale artifact detection
- Irrelevant Branch: Simple content extraction for identity/speaker characteristics
"""

# Import all components for easy access
from .tcn import (
    TemporalBlock,
    TemporalConvNet,
    Chomp1d
)

from .freq_encoder import (
    FrequencyEncoder,
    FrequencyEncoderWithStride
)

from .forgery_branch import (
    SophisticatedForgeryBranch,
    EnhancedForgeryBranch
)

from .irrelevant_branch import (
    SimpleIrrelevantBranch,
    PoolingIrrelevantBranch,
    MinimalIrrelevantBranch
)

# Package version
__version__ = "1.0.0"

# Package description
__description__ = "Optimal Asymmetric TCN-based Audio Encoder for Deepfake Detection"

# Available components
__all__ = [
    # TCN components
    "TemporalBlock",
    "TemporalConvNet", 
    "Chomp1d",
    
    # Frequency encoder components
    "FrequencyEncoder",
    "FrequencyEncoderWithStride",
    
    # Forgery branch components
    "SophisticatedForgeryBranch",
    "EnhancedForgeryBranch",
    
    # Irrelevant branch components
    "SimpleIrrelevantBranch",
    "PoolingIrrelevantBranch",
    "MinimalIrrelevantBranch",
]

# Default configurations
DEFAULT_CONFIG = {
    "tcn": {
        "num_channels": [256, 256, 256, 256, 256, 256],
        "kernel_size": 3,
        "dropout": 0.1
    },
    "freq_encoder": {
        "input_channels": 1,
        "output_channels": 256
    },
    "forgery_branch": {
        "input_dim": 256,
        "output_dim": 512
    },
    "irrelevant_branch": {
        "input_dim": 256,
        "output_dim": 512
    }
}

def get_default_config():
    """Get default configuration for all components"""
    return DEFAULT_CONFIG.copy()

def create_audio_encoder_components(config=None):
    """
    Create all audio encoder components with given configuration
    
    Args:
        config (dict): Configuration dictionary. If None, uses default config.
        
    Returns:
        dict: Dictionary containing all components
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    components = {
        "freq_encoder": FrequencyEncoder(
            input_channels=config["freq_encoder"]["input_channels"],
            output_channels=config["freq_encoder"]["output_channels"]
        ),
        "tcn": TemporalConvNet(
            num_inputs=config["freq_encoder"]["output_channels"],
            num_channels=config["tcn"]["num_channels"],
            kernel_size=config["tcn"]["kernel_size"],
            dropout=config["tcn"]["dropout"]
        ),
        "forgery_branch": SophisticatedForgeryBranch(
            input_dim=config["forgery_branch"]["input_dim"],
            output_dim=config["forgery_branch"]["output_dim"]
        ),
        "irrelevant_branch": SimpleIrrelevantBranch(
            input_dim=config["irrelevant_branch"]["input_dim"],
            output_dim=config["irrelevant_branch"]["output_dim"]
        )
    }
    
    return components

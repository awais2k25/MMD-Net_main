"""
Model constants and configurations
"""

# Dataset constants
FRAMES_PER_VIDEO = 25
SEQUENCE_LENGTH = 5  # 5 frames per sequence
NUM_SEQUENCES = 5     # 25/5 = 5 sequences
# TEMPORAL_SCALES = [15, 8, 4, 2]  # Temporal downsampling levels
MEL_FEATURE_SIZE = 64

TEMPORAL_SCALES = [25, 15, 8]  # Better scales for 25 frames
NUM_SEQUENCES_PER_SCALE = {
    25: 1,  # 1 sequence of 25 frames
    15: 1,  # 1 sequence of 15 frames
    8: 3    # 3 sequences of 8 frames (approx 25)
}

# Augmentation parameters
HEAVY_AUG_PROB = 0.4
TEMPORAL_AUG_PROB = 0.3
MASK_RATIO = 0.15

# Audio normalization
MEL_MEAN = -39.798222
MEL_STD = 17.590710
# Image normalization (RGB order)
# IMG_MEAN = [0.327, 0.369, 0.542] 
IMG_MEAN = [0.326424, 0.369014, 0.541994] # Rounded values
# IMG_STD = [0.182, 0.184, 0.232]  
IMG_STD = [0.182600, 0.184478, 0.232022]  # Rounded values



# EfficientNet output dimensions for different versions
EFFICIENT_NET_DIMS = {
    'b0': (1280, 7, 7),
    'b1': (1280, 7, 7),
    'b2': (1408, 9, 9),  # 9,9 is working for b2
    'b3': (1536, 8, 8),
    'b4': (1792, 8, 8),
}

# Model specific constants
MODEL_CONFIGS = {
    'b0': {'dim': 1280, 'size': (224, 224), 'feature_map': (7, 7)},
    'b1': {'dim': 1280, 'size': (240, 240), 'feature_map': (7, 7)},
    'b2': {
        'dim': 1408, 
        'size': (260, 260),  # Keep original size for model
        'feature_map': (8, 8),  # 8x8 for 140x140 input (EfficientNet-B2 stride pattern)
        'input_size': (250, 250)  # Updated input size
    },
    'b3': {'dim': 1536, 'size': (300, 300), 'feature_map': (8, 8)},
    'b4': {'dim': 1792, 'size': (380, 380), 'feature_map': (8, 8)}
}

# Feature extraction constants
FORGERY_FEATURE_DIM = 512
IRRELEVANT_FEATURE_DIM = 512
TEMPORAL_FEATURE_DIM = 512

# Processing constants
CHUNK_SIZE = 32  # Maximum number of samples to process at once
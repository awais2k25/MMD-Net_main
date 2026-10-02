# Dataset constants
FRAMES_PER_VIDEO = 25
SEQUENCE_LENGTH = 5  # 5 frames per sequence
NUM_SEQUENCES = 5     # 25/5 = 5 sequences
# TEMPORAL_SCALES = [15]  # Remove coarse scale (5)
TEMPORAL_SCALES = [25, 15, 8]  # Better scales for 25 frames

NUM_SEQUENCES_PER_SCALE = {
    25: 1,  # 1 sequence of 25 frames
    15: 1,  # 1 sequence of 15 frames
    8: 3    # 3 sequences of 8 frames (approx 25)
}

MEL_FEATURE_SIZE = 128

# Augmentation parameters
HEAVY_AUG_PROB = 0.4
TEMPORAL_AUG_PROB = 0.3
MASK_RATIO = 0.15

# Normalization constants
# IMG_MEAN = [0.485, 0.456, 0.406]
# IMG_STD = [0.229, 0.224, 0.225]
# MEL_MEAN = 0.5
# MEL_STD = 0.5

# # actual dataset statistics for FakeAVCeleb_balanced_V6_500 dataset (6/8/25)
# # Audio normalization
# MEL_MEAN = -43.646339
# MEL_STD = 17.266540

# # Image normalization (RGB order)
# IMG_MEAN = [0.322, 0.367, 0.534]  # Rounded values
# IMG_STD = [0.186, 0.189, 0.235]    # Rounded values

# #end

# actual dataset statistics for FakeAVCeleb_balanced_V7.1_3025 dataset (28/02/26)
# Audio normalization
MEL_MEAN = -39.798222
MEL_STD = 17.590710
# Image normalization (RGB order)
# IMG_MEAN = [0.327, 0.369, 0.542] 
IMG_MEAN = [0.326424, 0.369014, 0.541994] # Rounded values
# IMG_STD = [0.182, 0.184, 0.232]  
IMG_STD = [0.182600, 0.184478, 0.232022]  # Rounded values

# # actual dataset statistics for DFDC test_e2_1000samples dataset (07/04/26)
# # Audio normalization
# MEL_MEAN = -37.020779
# MEL_STD = 13.824956
# IMG_MEAN = [0.303503, 0.334661, 0.452881] #RGB
# IMG_STD = [0.181450, 0.192654, 0.221241]


# EfficientNet output dimensions for different versions
EFFICIENT_NET_DIMS = {
    'b0': (1280, 7, 7),
    'b1': (1280, 7, 7),
    'b2': (1408, 9, 9), # 9,9 is working for b2
    'b3': (1536, 8, 8),
    'b4': (1792, 8, 8),
}

# Feature dimensions
FEATURE_DIM = 512
# Model specific constants
MODEL_CONFIGS = {
    'b0': {'dim': 1280, 'size': (224, 224), 'feature_map': (7, 7)},
    'b1': {'dim': 1280, 'size': (240, 240), 'feature_map': (7, 7)},
    'b2': {
        'dim': 1408, 
        'size': (260, 260),  # Keep original size for model
        'feature_map': (8, 8),  # 8x8 for 250x250 input
        'input_size': (250, 250)  # Updated input size
        },
    'b3': {'dim': 1536, 'size': (300, 300), 'feature_map': (8, 8)},
    'b4': {'dim': 1792, 'size': (380, 380), 'feature_map': (8, 8)}
}

# Feature extraction constants
FORGERY_FEATURE_DIM = 512
IRRELEVANT_FEATURE_DIM = 512
TEMPORAL_FEATURE_DIM = 512

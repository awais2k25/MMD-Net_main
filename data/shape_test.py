# not usable gives errors

import os
import sys
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.MMD_dataset_v2 import MMDDatasetV2
import torch
import numpy as np
from prettytable import PrettyTable

def analyze_dataset_shapes():
    # Create dummy dataset
    B, V = 2, 4  # Batch size, Videos per batch
    
    # Create dummy batch data with proper numpy arrays for frames
    frames = np.random.randint(0, 255, (25, 3, 250, 250), dtype=np.uint8)
    frames = frames.transpose(0, 2, 3, 1)  # Convert to [25, H, W, C] for OpenCV
    
    batch = {
        'frames': torch.randn(B, V, 25, 3, 250, 250),
        'mels': torch.randn(B, V, 128, 32),
        'labels': {
            'multi': torch.randint(0, 4, (B, V)),
            'binary': torch.randint(0, 2, (B, V))
        }
    }
    
    def print_tensor_shape(d, prefix=''):
        """Recursively print tensor shapes in nested dictionary"""
        table = PrettyTable(['Tensor Name', 'Shape', 'Type'])
        
        def _add_to_table(name, tensor, indent=''):
            if isinstance(tensor, (torch.Tensor, np.ndarray)):
                table.add_row([
                    indent + name,
                    str(tuple(tensor.shape)),
                    type(tensor).__name__
                ])
            elif isinstance(tensor, dict):
                table.add_row([indent + name, '<dict>', ''])
                for k, v in tensor.items():
                    _add_to_table(k, v, indent + '  ')
        
        for k, v in d.items():
            _add_to_table(k, v)
        
        print("\nDataset Output Shapes:")
        print(table)
    
    # Print input shapes
    print("\nInput Batch Structure:")
    print_tensor_shape(batch)
    
    # Create dataset instance
    dataset = MMDDatasetV2(
        root_dir="D:/FAST NUCES/Semester 5/Thesis i237606/Project_Code/Fake_AV_sample/FakeAVCeleb_balanced_V6_500",
        split='train',
        enable_temporal=True
    )
    
    # Process batch through dataset transforms
    processed = dataset._process_static_features(
        frames,  # Pass numpy array frames directly
        batch['mels'][0, 0].numpy(),  # Convert mel to numpy
        needs_heavy_aug=False
    )
    
    print("\nProcessed Static Features:")
    print_tensor_shape(processed)
    
    # Process temporal features
    temporal = dataset._process_temporal_features(
        frames,  # Pass numpy array frames
        batch['mels'][0, 0].numpy()  # Convert mel to numpy
    )
    
    print("\nProcessed Temporal Features:")
    print_tensor_shape(temporal)
    
    # Print detailed sequence information
    print("\nTemporal Sequence Details:")
    for scale in [15, 5]:
        print(f"\nScale {scale}:")
        print(f"- Number of sequences: {len(temporal['sequences'])}")
        if temporal['sequences']:
            seq_shape = temporal['sequences'][0].shape if isinstance(temporal['sequences'][0], np.ndarray) else temporal['sequences'][0][0].shape
            mel_shape = temporal['mel_sequences'][0].shape
            print(f"- Sequence shape: {seq_shape}")
            print(f"- Mel segment shape: {mel_shape}")
            print(f"- Frames per sequence: {seq_shape[0]}")
            print(f"- Mel frames per segment: {mel_shape[1]}")

if __name__ == '__main__':
    print("Starting Dataset Shape Analysis...")
    analyze_dataset_shapes()

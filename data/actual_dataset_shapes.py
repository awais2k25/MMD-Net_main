"""
6/8/25
prints shape of MMD_datset_vX.py files, one set to check dimensions.

Actual sample output

==================================================
Inspecting set: id00032_v0
==================================================

Labels:
Multi-class: [0, 1, 2, 3]
Binary: [0, 1, 1, 1]

Video 1:

Raw data shapes:
Frames: (25, 250, 250, 3) (dtype: uint8, range: [0.00, 253.00])
Mel: (128, 32) (dtype: float32, range: [-80.00, 0.00])

Normalized tensors:
Frames tensor: torch.Size([25, 3, 250, 250]) (dtype: torch.float32, range: [-2.27, 2.87])
  Normalized - mean: 0.03, std: 0.75
Mel tensor: torch.Size([1, 128, 32]) (dtype: torch.float32, range: [2.51, 2.53])
  Normalized - mean: 2.52, std: 0.00

Video 2:

Raw data shapes:
Frames: (25, 250, 250, 3) (dtype: uint8, range: [0.00, 253.00])
Mel: (128, 32) (dtype: float32, range: [-80.00, 0.00])

Normalized tensors:
Frames tensor: torch.Size([25, 3, 250, 250]) (dtype: torch.float32, range: [-2.27, 2.87])
  Normalized - mean: 0.01, std: 0.75
Mel tensor: torch.Size([1, 128, 32]) (dtype: torch.float32, range: [2.51, 2.53])
  Normalized - mean: 2.52, std: 0.00

video 3, 4 data removed as it it is same as above.

==================================================
Testing DataLoader Output Shapes
==================================================

Batch shapes:
frames: torch.Size([2, 4, 25, 3, 250, 250])
mels: torch.Size([2, 4, 1, 128, 32])
labels_multi: torch.Size([2, 4])
labels_binary: torch.Size([2, 4])
set_ids: ['id00032_v0', 'id00126_v0']

"""
import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from MMD_dataset_v4 import MMDDatasetV4

# Configuration
ROOT_DIR = "D:/FAST NUCES/Semester 5/Thesis i237606/Project_Code/Fake_AV_sample/FakeAVCeleb_balanced_V6_500"
SPLIT = 'val'  # Can change to 'train' or 'test'
SET_INDEX = 0  # Index of the set to inspect

def print_stats(data, name):
    """Print statistics for a numpy array or tensor"""
    if isinstance(data, (torch.Tensor, np.ndarray)):
        print(f"{name}: {data.shape} (dtype: {data.dtype}, range: [{data.min():.2f}, {data.max():.2f}])")
        if isinstance(data, torch.Tensor) and data.dtype == torch.float32:
            print(f"  Normalized - mean: {data.mean():.2f}, std: {data.std():.2f}")
    else:
        print(f"{name} is type: {type(data)}")

def inspect_set(set_data, dataset):
    """Inspect one set from the dataset"""
    print("\n" + "="*50)
    print(f"Inspecting set: {set_data['set_id']}")
    print("="*50)
    
    # Print labels
    print("\nLabels:")
    print(f"Multi-class: {set_data['labels_multi'].tolist()}")
    print(f"Binary: {set_data['labels_binary'].tolist()}")
    
    # Print stats for each video in the set
    for i in range(len(set_data['frames'])):
        print(f"\nVideo {i+1}:")
        
        # Get the corresponding metadata row
        row = dataset.metadata[dataset.metadata['set_id'] == set_data['set_id']].iloc[i]
        base_path = os.path.splitext(row['new_path'])[0]
        
        try:
            # Raw data inspection (before transforms)
            raw_frames = np.load(os.path.join(base_path, 'frames', 'faces.npy'))
            raw_mel = np.load(os.path.join(base_path, 'mels', 'mel.npy'))
            
            print("\nRaw data shapes:")
            print_stats(raw_frames, "Frames")
            print_stats(raw_mel, "Mel")
        except FileNotFoundError as e:
            print(f"\nError loading raw data: {e}")
        
        # Processed data inspection
        print("\nNormalized tensors:")
        print_stats(set_data['frames'][i], "Frames tensor")
        print_stats(set_data['mels'][i], "Mel tensor")

def custom_collate(batch):
    """Custom collate function to handle our data structure"""
    return {
        'frames': torch.stack([item['frames'] for item in batch]),
        'mels': torch.stack([item['mels'] for item in batch]),
        'labels_multi': torch.stack([item['labels_multi'] for item in batch]),
        'labels_binary': torch.stack([item['labels_binary'] for item in batch]),
        'set_id': [item['set_id'] for item in batch]
    }

if __name__ == "__main__":
    # Initialize dataset
    dataset = MMDDatasetV4(root_dir=ROOT_DIR, split=SPLIT)
    
    # Get one set
    sample_set = dataset[SET_INDEX]
    
    # Inspect the set
    inspect_set(sample_set, dataset)
    
    # Test dataloader with custom collate
    print("\n" + "="*50)
    print("Testing DataLoader Output Shapes")
    print("="*50)
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate
    )
    batch = next(iter(dataloader))
    
    print("\nBatch shapes:")
    print(f"frames: {batch['frames'].shape}")  # [batch_size, 4, 25, 3, H, W]
    print(f"mels: {batch['mels'].shape}")      # [batch_size, 4, 1, 128, 32]
    print(f"labels_multi: {batch['labels_multi'].shape}")  # [batch_size, 4]
    print(f"labels_binary: {batch['labels_binary'].shape}")  # [batch_size, 4]
    print(f"set_ids: {batch['set_id']}")  # list of length batch_size
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import random
import pandas as pd

class MMDDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory of FakeAVCeleb_balanced_V4
            split: 'train' or 'val'
            transform: Optional transform to be applied on frames
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Read metadata
        metadata_path = os.path.join(root_dir, split, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # Group by set_id to get sets of 4 videos
        self.video_sets = []
        for set_id, group in self.metadata.groupby('set_id'):
            # Ensure we have all 4 categories
            if len(group) == 4:
                self.video_sets.append(set_id)

        # Standard augmentation for all videos
        self.standard_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ])

        # Heavy augmentation for v1+ versions
        self.heavy_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(var_limit=(10.0, 150.0), p=0.4),
            A.OneOf([
                A.MotionBlur(p=0.4),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.4),
            A.OneOf([
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p=0.2),
                A.ElasticTransform(p=0.2),
            ], p=0.3),
            A.RandomRotate90(p=0.3),
            A.HueSaturationValue(p=0.4),
        ])

        # Mel spectrogram augmentation
        self.mel_transform = A.Compose([
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ])

    def __len__(self):
        return len(self.video_sets)

    def augment_frames(self, frames):
        """Apply heavy augmentation to frames"""
        augmented_frames = []
        for frame in frames:
            augmented = self.heavy_transform(image=frame)['image']
            augmented_frames.append(augmented)
        return np.array(augmented_frames)

    def augment_mel(self, mel):
        """Apply augmentation to mel spectrogram"""
        # Convert to uint8 for albumentations
        mel_norm = ((mel - mel.min()) * 255 / (mel.max() - mel.min())).astype(np.uint8)
        mel_aug = self.mel_transform(image=mel_norm)['image']
        # Convert back to original range
        mel_aug = mel_aug.astype(np.float32) * (mel.max() - mel.min()) / 255 + mel.min()
        return mel_aug

    def __getitem__(self, idx):
        set_id = self.video_sets[idx]
        set_data = self.metadata[self.metadata['set_id'] == set_id]
        
        # Initialize containers for the set
        frames_set = []
        mels_set = []
        labels_multi = []    # A:0, B:1, C:2, D:3
        labels_binary = []   # Real:0, Fake:1
        
        # Check if this is a version higher than v0
        version = int(set_id.split('_v')[-1]) if '_v' in set_id else 0
        needs_heavy_aug = version > 0
        
        for _, row in set_data.iterrows():
            video_path = row['new_path']
            category = row['category']  # A, B, C, or D
            base_path = os.path.splitext(video_path)[0]
            
            # Load frames and apply standard transforms
            frames_path = os.path.join(base_path, 'frames', 'faces.npy')
            frames = np.load(frames_path)
            
            # Apply standard augmentation to all frames
            augmented_frames = []
            for frame in frames:
                aug_frame = self.standard_transform(image=frame)['image']
                augmented_frames.append(aug_frame)
            frames = np.array(augmented_frames)
            
            # Apply heavy augmentation if needed
            if needs_heavy_aug:
                frames = self.augment_frames(frames)
            
            # Load and process mel spectrogram
            mel_path = os.path.join(base_path, 'mels', 'mel.npy')
            mel = np.load(mel_path)
            
            # Apply standard mel augmentation
            mel = self.augment_mel(mel)
            
            # Apply heavy mel augmentation if needed
            if needs_heavy_aug:
                mel = self.augment_mel(mel)
            
            # Regular transform if specified
            if self.transform:
                frames = torch.stack([self.transform(frame) for frame in frames])
            
            # Create labels
            label_multi = ord(category) - ord('A')  # A:0, B:1, C:2, D:3
            label_binary = 0 if category == 'A' else 1  # Real:0, Fake:1
            
            frames_set.append(torch.FloatTensor(frames))
            mels_set.append(torch.FloatTensor(mel))
            labels_multi.append(label_multi)
            labels_binary.append(label_binary)

        return {
            'frames': frames_set,          # List of 4 frame sequences
            'mels': mels_set,             # List of 4 mel spectrograms
            'labels_multi': labels_multi,  # List of 4 multiclass labels
            'labels_binary': labels_binary,# List of 4 binary labels
            'set_id': set_id
        }

def get_dataloader(root_dir, split='train', batch_size=1, num_workers=4, transform=None):
    """Helper function to create dataloader"""
    dataset = MMDDataset(root_dir=root_dir, split=split, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

"""
Simplified version of MMD dataset that only processes static features.
No temporal sequences, simplified processing pipeline.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torchaudio.transforms as TA
import cv2
import pandas as pd
from .consts import * #MEL_MEAN, MEL_STD, IMG_MEAN, IMG_STD,MODEL_CONFIGS

# print(f"MEL_MEAN: {MEL_MEAN}, MEL_STD: {MEL_STD}")
# print(f"IMG_MEAN: {IMG_MEAN}, IMG_STD: {IMG_STD}")
class MMDDatasetV4(Dataset):
    def __init__(self, root_dir, split='train', model_version='b2'):
        """
        Args:
            root_dir: Root directory of FakeAVCeleb_balanced_Vx
            split: 'train', 'val', or 'test'
            model_version: EfficientNet version (b0-b4)
        """
        self.root_dir = root_dir
        self.split = split
        self.model_config = MODEL_CONFIGS[model_version]
        self.input_size = self.model_config['input_size'][0]
        
        # Read metadata
        metadata_path = os.path.join(root_dir, split, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # Group by set_id and ensure complete sets
        # Each set_id should have exactly 4 rows (one for each category: A, B, C, D)
        self.video_sets = []
        skipped = 0
        for set_id, group in self.metadata.groupby('set_id'):
            if len(group) != 4:
                continue  # incomplete set (not 4 videos)

            all_ok = True
            for _, row in group.iterrows():
                base_path   = os.path.splitext(row['new_path'])[0]
                frames_path = os.path.join(base_path, 'frames', 'faces.npy')
                mel_path    = os.path.join(base_path, 'mels',   'mel.npy')
                if not os.path.exists(frames_path) or not os.path.exists(mel_path):
                    all_ok = False
                    break

            if all_ok:
                self.video_sets.append(set_id)
            else:
                skipped += 1

        if skipped:
            print(f"⚠️  Skipped {skipped} sets (missing files) from {split} split")
        self._init_transforms()


    def _init_transforms(self):
        """Initialize transforms for frames and mel spectrograms"""
        # Frame transforms (Base + Normalization)
        self.frame_transform = v2.Compose([
            v2.ToImage(),  # convert numpy.ndarray (H, W, C) to torch.Tensor (C, H, W)
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((self.input_size, self.input_size), antialias=True),
            v2.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

        # "Safe" Deepfake Augmentation (Applied during Train mode only)
        self.safe_aug_transform = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2), # Mild blur
            v2.RandomApply([v2.JPEG(quality=(70, 100))], p=0.3), # standard Social Media compression
            v2.Resize((self.input_size, self.input_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.2), value=0) # Face cutout
        ])

        # Mel spectrogram transforms (Restoring explicit transform block for Mel)
        self.mel_transform = v2.Compose([
            v2.Lambda(lambda mel: torch.from_numpy(mel).float()), # Convert to tensor
            v2.Lambda(lambda x: (x - MEL_MEAN) / MEL_STD)         # Normalize
        ])
        
        # Audio specific masking
        self.freq_mask = TA.FrequencyMasking(freq_mask_param=15)
        self.time_mask = TA.TimeMasking(time_mask_param=5)

    def _apply_spec_augment(self, mel, freq_mask_param=12, time_mask_param=4):
        """Native implementation of Frequency and Time Masking (SpecAugment)"""
        mel = mel.copy()
        num_mels, num_frames = mel.shape
        
        # Frequency masking
        if np.random.rand() > 0.5:
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, num_mels - f)
            mel[f0:f0+f, :] = 0
            
        # Time masking
        if np.random.rand() > 0.5:
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, num_frames - t)
            mel[:, t0:t0+t] = 0
            
        return mel

    def _process_features(self, frames, mel, is_train=False):
        """Process static features with safe augmentations if training"""
        # Process frames
        processed_frames = []
        for frame in frames: # frame is numpy array
            if is_train:
                transformed = self.safe_aug_transform(frame)
            else:
                transformed = self.frame_transform(frame)
            processed_frames.append(transformed)
        
        # Stack frames
        processed_frames = torch.stack(processed_frames)  # [25, 3, H, W]
        
        # Process audio (Mel Spectrogram)
        # Apply the mel transform (which converts to tensor and normalizes)
        processed_mel = self.mel_transform(mel)
        
        if is_train:
            # Safe audio augmentation: Add mild Gaussian noise (SNR variation)
            if np.random.rand() > 0.5:
                snr_db = np.random.uniform(15, 30)
                snr_linear = 10 ** (snr_db / 10)
                signal_power = processed_mel.norm() ** 2 / processed_mel.numel()
                noise_power = signal_power / snr_linear
                noise = torch.randn_like(processed_mel) * torch.sqrt(noise_power)
                processed_mel = processed_mel + noise
                
            # Safe audio augmentation: SpecAugment (Time/Frequency Masking)
            if np.random.rand() > 0.5:
                processed_mel = self.freq_mask(processed_mel)
            if np.random.rand() > 0.5:
                processed_mel = self.time_mask(processed_mel)
                
        if processed_mel.ndim == 2:
            processed_mel = processed_mel.unsqueeze(0)  # Add channel dimension [1, 64, 32]
            
        return processed_frames, processed_mel

  

    def __getitem__(self, idx):
        set_id = self.video_sets[idx]
        set_data = self.metadata[self.metadata['set_id'] == set_id]
        
        # Initialize with empty lists
        frames_list = []
        mels_list = []
        labels_multi = []
        labels_binary = []
        
        # Heavy augmentation is disabled
        for _, row in set_data.iterrows():
            # Load data
            # new_path format: /path/to/root/split/set_id/video_filename.mp4
            # base_path will be: /path/to/root/split/set_id/video_filename
            base_path = os.path.splitext(row['new_path'])[0]
            
            frames_path = os.path.join(base_path, 'frames', 'faces.npy')
            mel_path = os.path.join(base_path, 'mels', 'mel.npy')
            
            # Verify paths exist
            if not os.path.exists(frames_path):
                raise FileNotFoundError(f"Frames not found: {frames_path}")
            if not os.path.exists(mel_path):
                raise FileNotFoundError(f"Mel spectrogram not found: {mel_path}")
            
            frames = np.load(frames_path)
            mel = np.load(mel_path)
            
            # Process features (heavy augmentation disabled)
            is_train = (self.split == 'train')
            proc_frames, proc_mel = self._process_features(frames, mel, is_train=is_train)
            
            # Collect features
            frames_list.append(proc_frames)
            mels_list.append(proc_mel)
            
            # Add labels
            category = str(row['category'])  # Ensure string type
            labels_multi.append(ord(category) - ord('A'))
            labels_binary.append(0 if category == 'A' else 1)
        
        batch_data = {
            'frames': torch.stack(frames_list),  # [4, 25, 3, H, W]
            'mels': torch.stack(mels_list),      # [4, 1, 128, 32]
            'labels_multi': torch.tensor(labels_multi),
            'labels_binary': torch.tensor(labels_binary),
            'set_id': set_id
        }
        
        # Verify shapes
        # print("Dataset shapes:")
        # print(f"frames: {batch_data['frames'].shape}")
        # print(f"mels: {batch_data['mels'].shape}")
        
        return batch_data

    def __len__(self):
        return len(self.video_sets)

def get_dataloader_v4(root_dir, split='train', batch_size=2, num_workers=4, worker_init_fn=None):
    dataset = MMDDatasetV4(
        root_dir=root_dir,
        split=split
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

class MMDDatasetV4Single(Dataset):
    """Dataset for single videos (for testing) - returns one video per sample instead of sets of 4"""
    def __init__(self, root_dir, split='test', model_version='b2'):
        """
        Args:
            root_dir: Root directory of FakeAVCeleb_balanced_Vx
            split: 'test', 'val' or 'train'
            model_version: EfficientNet version (b0-b4)
        """
        self.root_dir = root_dir
        self.split = split
        self.model_config = MODEL_CONFIGS[model_version]
        self.input_size = self.model_config['input_size'][0]
        
        # Read metadata
        metadata_path = os.path.join(root_dir, split, 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Test metadata not found at {metadata_path}")
        self.metadata = pd.read_csv(metadata_path)
        
        # Use all videos (no need to group by set_id in single mode)
        # Filter to only videos that actually exist
        self.video_list = []
        skipped = 0
        for _, row in self.metadata.iterrows():
            base_path = os.path.splitext(row['new_path'])[0]
            frames_path = os.path.join(base_path, 'frames', 'faces.npy')
            mel_path = os.path.join(base_path, 'mels', 'mel.npy')
            if os.path.exists(frames_path) and os.path.exists(mel_path):
                self.video_list.append(row.to_dict())
            else:
                skipped += 1
                
        if skipped:
            print(f"⚠️  Skipped {skipped} single videos (missing files) from {split} split")
        
        self._init_transforms()

    def _init_transforms(self):
        """Initialize transforms for frames and mel spectrograms"""
        # Frame transforms (Base + Normalization)
        self.frame_transform = v2.Compose([
            v2.ToImage(),  # convert numpy.ndarray (H, W, C) to torch.Tensor (C, H, W)
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((self.input_size, self.input_size), antialias=True),
            v2.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

        # "Safe" Deepfake Augmentation (Applied during Train mode only)
        self.safe_aug_transform = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2), # Mild blur
            v2.RandomApply([v2.JPEG(quality=(70, 100))], p=0.3), # standard Social Media compression
            v2.Resize((self.input_size, self.input_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.2), value=0) # Face cutout
        ])

        # Mel spectrogram transforms
        self.mel_transform = v2.Compose([
            v2.Lambda(lambda mel: torch.from_numpy(mel).float()), # Convert to tensor
            v2.Lambda(lambda x: (x - MEL_MEAN) / MEL_STD)         # Normalize
        ])
        
        # Audio specific masking
        self.freq_mask = TA.FrequencyMasking(freq_mask_param=15)
        self.time_mask = TA.TimeMasking(time_mask_param=5)

        # Mel Audio augmentations
        self.freq_mask = TA.FrequencyMasking(freq_mask_param=15)
        self.time_mask = TA.TimeMasking(time_mask_param=5)

    def _process_features(self, frames, mel, is_train=False):
        """Process static features with safe augmentations if training"""
        # Process frames
        processed_frames = []
        for frame in frames: # frame is numpy array
            if is_train:
                transformed = self.safe_aug_transform(frame)
            else:
                transformed = self.frame_transform(frame)
            processed_frames.append(transformed)
        
        # Stack frames
        processed_frames = torch.stack(processed_frames)  # [25, 3, H, W]
        
        # Process audio (Mel Spectrogram)
        # Apply the explicit mel transform
        processed_mel = self.mel_transform(mel)
        
        if is_train:
            # Safe audio augmentation: Add Gaussian noise (SNR variation 15-30 dB range)
            if np.random.rand() > 0.5:
                snr_db = np.random.uniform(15, 30)
                snr_linear = 10 ** (snr_db / 10)
                signal_power = processed_mel.norm() ** 2 / processed_mel.numel()
                noise_power = signal_power / snr_linear
                noise = torch.randn_like(processed_mel) * torch.sqrt(noise_power)
                processed_mel = processed_mel + noise
                
            # Safe audio augmentation: SpecAugment (Time/Frequency Masking)
            if np.random.rand() > 0.5:
                processed_mel = self.freq_mask(processed_mel)
            if np.random.rand() > 0.5:
                processed_mel = self.time_mask(processed_mel)
        
        if processed_mel.ndim == 2:
            processed_mel = processed_mel.unsqueeze(0)  # Add channel dimension [1, 64, 32]
            
        return processed_frames, processed_mel

    def _init_transforms(self):
        """Initialize transforms for frames and mel spectrograms"""
        # Frame transforms (no augmentation for testing)
        self.frame_transform = v2.Compose([
            v2.ToImage(),  # convert numpy.ndarray (H, W, C) to torch.Tensor (C, H, W)
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((self.input_size, self.input_size), antialias=True),
            v2.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

        # Mel spectrogram transforms
        self.mel_transform = v2.Compose([
            v2.Lambda(lambda mel: torch.from_numpy(mel).float()), # Convert to tensor
            v2.Lambda(lambda x: (x - MEL_MEAN) / MEL_STD)         # Normalize
        ])

    def _process_features(self, frames, mel):
        """Process static features (no augmentation for testing)"""
        # Process frames
        processed_frames = []
        for frame in frames: # frame is numpy array
            transformed = self.frame_transform(frame)
            processed_frames.append(transformed)
        
        # Stack frames
        processed_frames = torch.stack(processed_frames)  # [25, 3, H, W]
        
        # Process mel spectrogram
        # Apply the explicit mel transform
        processed_mel = self.mel_transform(mel)
        
        if processed_mel.ndim == 2:
            processed_mel = processed_mel.unsqueeze(0)  # Add channel dimension [1, 64, 32]
            
        return processed_frames, processed_mel

    def __getitem__(self, idx):
        row = self.video_list[idx]
        
        # Load data
        base_path = os.path.splitext(row['new_path'])[0]
        frames = np.load(os.path.join(base_path, 'frames', 'faces.npy'))
        mel = np.load(os.path.join(base_path, 'mels', 'mel.npy'))
        
        # Process features
        proc_frames, proc_mel = self._process_features(frames, mel)
        
        # Get labels
        category = str(row['category'])
        label_multi = ord(category) - ord('A')
        label_binary = 0 if category == 'A' else 1
        
        # Return single video
        batch_data = {
            'frames': proc_frames,  # [25, 3, H, W]
            'mels': proc_mel,       # [1, 64, 32]
            'labels_multi': torch.tensor([label_multi]),  # [1]
            'labels_binary': torch.tensor([label_binary], dtype=torch.float32),  # [1]
            'set_id': row.get('set_id', f'single_{idx}'),
            'video_path': row.get('new_path', '')
        }
        
        return batch_data

    def __len__(self):
        return len(self.video_list)


def get_dataloader_v4_single(root_dir, split='test', batch_size=1, num_workers=4, worker_init_fn=None):
    """Get dataloader for single videos (testing mode)"""
    dataset = MMDDatasetV4Single(
        root_dir=root_dir,
        split=split
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for testing
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )


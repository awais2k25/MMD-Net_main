#!/usr/bin/env python3
"""
Script to find all frame files with incorrect shapes.
Expected shape: (25, 250, 250, 3) for saved .npy files
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
ROOT_DIR = Path("/home/i237606/FakeAV_sample/FakeAVCeleb_balanced_V7.1_3025")
EXPECTED_FRAME_SHAPE = (25, 140, 140, 3)  # Expected frame shape: (num_frames, height, width, channels)

def check_frame_file(frames_path):
    """Check if frames file has correct shape"""
    try:
        frames = np.load(frames_path)
        if frames.ndim == 4:
            # Should be (25, 250, 250, 3)
            if frames.shape != EXPECTED_FRAME_SHAPE:
                return False, frames.shape
        elif frames.ndim == 3:
            # Might be (25, 250, 250) - missing channel dimension
            return False, frames.shape
        elif frames.ndim == 2:
            # 2D - definitely wrong
            return False, frames.shape
        elif frames.ndim == 1:
            # 1D - definitely wrong
            return False, frames.shape
        else:
            # 5D or higher - unexpected
            return False, frames.shape
        
        # Additional validation: check dtype and value range
        if frames.dtype != np.uint8:
            return False, f"Wrong dtype: {frames.dtype} (expected uint8)"
        
        if frames.min() < 0 or frames.max() > 255:
            return False, f"Value range: [{frames.min()}, {frames.max()}] (expected [0, 255])"
        
        return True, frames.shape
    except Exception as e:
        return False, f"Error: {str(e)}"

def find_corrupted_frames(root_dir, split='train'):
    """Find all corrupted frame files in a split"""
    metadata_path = root_dir / split / 'metadata.csv'
    
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return []
    
    metadata = pd.read_csv(metadata_path)
    corrupted = []
    
    print(f"\nChecking {split} split...")
    print(f"Total videos: {len(metadata)}")
    
    for idx, row in metadata.iterrows():
        base_path = Path(os.path.splitext(row['new_path'])[0])
        frames_path = base_path / 'frames' / 'faces.npy'
        
        if not frames_path.exists():
            corrupted.append({
                'split': split,
                'set_id': row['set_id'],
                'path': str(frames_path),
                'category': row['category'],
                'video_type': row.get('video_type', 'unknown'),
                'issue': 'File not found'
            })
            continue
        
        is_valid, shape_or_error = check_frame_file(frames_path)
        if not is_valid:
            corrupted.append({
                'split': split,
                'set_id': row['set_id'],
                'path': str(frames_path),
                'category': row['category'],
                'video_type': row.get('video_type', 'unknown'),
                'issue': f'Invalid shape: {shape_or_error}'
            })
    
    return corrupted

def main():
    print("="*80)
    print("Frame Shape Checker")
    print("="*80)
    print(f"Expected shape: {EXPECTED_FRAME_SHAPE}")
    print("="*80)
    
    all_corrupted = []
    
    for split in ['train', 'val', 'test']:
        corrupted = find_corrupted_frames(ROOT_DIR, split)
        all_corrupted.extend(corrupted)
        print(f"  Found {len(corrupted)} corrupted files in {split}")
    
    print("\n" + "="*80)
    print(f"Total corrupted files: {len(all_corrupted)}")
    print("="*80)
    
    if all_corrupted:
        # Group by issue type
        by_issue = defaultdict(list)
        by_set = defaultdict(list)
        
        for item in all_corrupted:
            by_issue[item['issue']].append(item)
            by_set[item['set_id']].append(item)
        
        print("\nIssues by type:")
        for issue, items in sorted(by_issue.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {issue}: {len(items)} files")
        
        print("\nSets with corrupted files:")
        for set_id, items in sorted(by_set.items()):
            print(f"  {set_id}: {len(items)} corrupted files")
            for item in items[:3]:  # Show first 3
                print(f"    - {item['category']} ({item['video_type']}): {item['issue']}")
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more")
        
        # Save to file
        output_file = ROOT_DIR / 'corrupted_frames.txt'
        with open(output_file, 'w') as f:
            f.write("Corrupted Frame Files\n")
            f.write("="*80 + "\n")
            f.write(f"Expected shape: {EXPECTED_FRAME_SHAPE}\n")
            f.write("="*80 + "\n\n")
            
            for item in all_corrupted:
                f.write(f"Set: {item['set_id']}\n")
                f.write(f"  Split: {item['split']}\n")
                f.write(f"  Category: {item['category']}, Type: {item['video_type']}\n")
                f.write(f"  Path: {item['path']}\n")
                f.write(f"  Issue: {item['issue']}\n")
                f.write("\n")
        
        print(f"\nDetailed report saved to: {output_file}")
        
        # Also create a summary by set_id
        print("\n" + "="*80)
        print("Summary: Sets with corrupted frames")
        print("="*80)
        for set_id, items in sorted(by_set.items()):
            print(f"{set_id}: {len(items)} corrupted files")
    else:
        print("\n✓ All frame files have correct shapes!")

if __name__ == "__main__":
    main()
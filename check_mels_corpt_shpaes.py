#!/usr/bin/env python3
"""
Script to find all mel spectrograms with incorrect shapes.
Expected shape: (128, 32) or (1, 128, 32) after processing
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
ROOT_DIR = Path("/home/i237606/FakeAV_sample/FakeAVCeleb_balanced_V7_3025")
EXPECTED_MEL_SHAPE = (64, 32)  # Raw mel shape before processing

def check_mel_file(mel_path):
    """Check if mel file has correct shape"""
    try:
        mel = np.load(mel_path)
        if mel.ndim == 2:
            # Should be (128, 32)
            if mel.shape != EXPECTED_MEL_SHAPE:
                return False, mel.shape
        elif mel.ndim == 1:
            return False, mel.shape
        else:
            # 3D or higher - unexpected
            return False, mel.shape
        return True, mel.shape
    except Exception as e:
        return False, f"Error: {str(e)}"

def find_corrupted_mels(root_dir, split='train'):
    """Find all corrupted mel files in a split"""
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
        mel_path = base_path / 'mels' / 'mel.npy'
        
        if not mel_path.exists():
            corrupted.append({
                'split': split,
                'set_id': row['set_id'],
                'path': str(mel_path),
                'category': row['category'],
                'video_type': row.get('video_type', 'unknown'),
                'issue': 'File not found'
            })
            continue
        
        is_valid, shape_or_error = check_mel_file(mel_path)
        if not is_valid:
            corrupted.append({
                'split': split,
                'set_id': row['set_id'],
                'path': str(mel_path),
                'category': row['category'],
                'video_type': row.get('video_type', 'unknown'),
                'issue': f'Invalid shape: {shape_or_error}'
            })
    
    return corrupted

def main():
    print("="*80)
    print("Mel Spectrogram Shape Checker")
    print("="*80)
    
    all_corrupted = []
    
    for split in ['train', 'val', 'test']:
        corrupted = find_corrupted_mels(ROOT_DIR, split)
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
        for issue, items in by_issue.items():
            print(f"  {issue}: {len(items)} files")
        
        print("\nSets with corrupted files:")
        for set_id, items in sorted(by_set.items()):
            print(f"  {set_id}: {len(items)} corrupted files")
            for item in items[:3]:  # Show first 3
                print(f"    - {item['category']} ({item['video_type']}): {item['issue']}")
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more")
        
        # Save to file
        output_file = ROOT_DIR / 'corrupted_mels.txt'
        with open(output_file, 'w') as f:
            f.write("Corrupted Mel Spectrogram Files\n")
            f.write("="*80 + "\n\n")
            for item in all_corrupted:
                f.write(f"Set: {item['set_id']}\n")
                f.write(f"  Split: {item['split']}\n")
                f.write(f"  Category: {item['category']}, Type: {item['video_type']}\n")
                f.write(f"  Path: {item['path']}\n")
                f.write(f"  Issue: {item['issue']}\n")
                f.write("\n")
        
        print(f"\nDetailed report saved to: {output_file}")
    else:
        print("\n✓ All mel spectrograms have correct shapes!")

if __name__ == "__main__":
    main()
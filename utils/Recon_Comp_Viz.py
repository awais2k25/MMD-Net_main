"""
Recon_Comp_Viz.py - Reconstruction Comparison Visualization
============================================================
Utilities for visualizing original vs reconstructed visual and audio data
from the MMD-Net deepfake detection model.

Features:
- Individual image saving (Set 0 only)
- Side-by-side comparisons (all sets)
- Grid layouts (2x4 and combined)
- Both visual (faces) and audio (mel-spectrograms)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ReconstructionVisualizer:
    """
    Handles visualization of reconstruction comparisons for MMD-Net

    Args:
        output_dir: Base directory for saving visualizations
        num_sets: Number of video sets to visualize (default: 3)
        frame_idx: Which frame to extract from video sequence (default: 37, middle frame)
    """
    
    def __init__(self, output_dir: str = 'reconstruction_visualizations', 
                 num_sets: int = 3, frame_idx: int = 37):
        self.output_dir = Path(output_dir)
        self.num_sets = num_sets
        self.frame_idx = frame_idx
        
        # Video type labels
        self.video_types = [
            'VideoA_RealReal',
            'VideoB_RealFakeAudio',
            'VideoC_FakeRealAudio',
            'VideoD_FakeFake'
        ]
        
        self.video_labels = [
            'A (Real-Real)',
            'B (Real-FakeAud)',
            'C (Fake-RealAud)',
            'D (Fake-Fake)'
        ]
        
        # Create directory structure
        self._create_directories()
        
        print(f"✅ ReconstructionVisualizer initialized")
        print(f"   Output: {self.output_dir}")
        print(f"   Sets: {self.num_sets}, Frame: {self.frame_idx}")
    
    def _create_directories(self):
        """Create output directory structure"""
        dirs = [
            self.output_dir / 'visual' / 'individual',
            self.output_dir / 'visual' / 'comparison_sidebyside',
            self.output_dir / 'visual' / 'comparison_grid',
            self.output_dir / 'visual' / 'comparison_combined',
            self.output_dir / 'audio' / 'individual',
            self.output_dir / 'audio' / 'comparison_sidebyside',
            self.output_dir / 'audio' / 'comparison_grid',
            self.output_dir / 'audio' / 'comparison_combined',
            self.output_dir / 'summary'
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def tensor_to_numpy_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy image for visualization
        
        Args:
            tensor: [3, H, W] tensor in range [0, 1] or [-1, 1]
        
        Returns:
            numpy array [H, W, 3] in range [0, 255] uint8, BGR format
        """
        img = tensor.detach().cpu().numpy()
        
        # Handle different tensor formats
        if img.ndim == 3 and img.shape[0] == 3:  # [3, H, W]
            img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
        
        # Normalize to [0, 1] if needed
        if img.min() < 0:
            img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        
        # Clip and convert to uint8
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def create_melspec_image(self, melspec: torch.Tensor) -> np.ndarray:
        """
        Convert mel-spectrogram tensor to heatmap image
        
        Args:
            melspec: [64, 94] mel-spectrogram tensor
        
        Returns:
            numpy array representing mel-spectrogram visualization
        """
        mel = melspec.detach().cpu().numpy()
        
        # Create figure without displaying
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel ('Mel Bins')
        plt.colorbar(im, ax=ax, label='Magnitude')
        
        # Convert to numpy array using modern matplotlib API
        fig.canvas.draw()
        
        # Get the RGBA buffer
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        
        plt.close(fig)
        
        # Convert RGBA to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        return img
    
    def add_text_label(self, img: np.ndarray, text: str, position: str = 'top') -> np.ndarray:
        """
        Add text label to image
        
        Args:
            img: Input image
            text: Text to add
            position: 'top' or 'bottom'
        
        Returns:
            Image with text label
        """
        img_labeled = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.2
        thickness = 1
        color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position
        if position == 'top':
            text_y = 20
        else:
            text_y = img.shape[0] - 10
        
        text_x = (img.shape[1] - text_width) // 2
        
        # Draw background rectangle
        cv2.rectangle(img_labeled, 
                     (text_x - 5, text_y - text_height - 5),
                     (text_x + text_width + 5, text_y + 5),
                     bg_color, -1)
        
        # Draw text
        cv2.putText(img_labeled, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return img_labeled
    
    def save_individual_images(self, vis_orig: torch.Tensor, vis_recon: torch.Tensor,
                               aud_orig: torch.Tensor, aud_recon: torch.Tensor, set_idx: int):
        """
        Save individual original and reconstructed images for Set 0 only
        
        Args:
            vis_orig: [4, 3, H, W] visual originals for 4 videos
            vis_recon: [4, 3, H, W] visual reconstructions
            aud_orig: [4, 64, 94] audio mel-spectrograms originals
            aud_recon: [4, 64, 94] audio mel-spectrogram reconstructions
            set_idx: Set index (0, 1, 2)
        """
        if set_idx != 0:
            return  # Only save for Set 0
        
        print(f"\n📸 Saving individual images for Set {set_idx}...")
        
        for vid_idx, vid_type in enumerate(self.video_types):
            # Visual original
            vis_orig_img = self.tensor_to_numpy_image(vis_orig[vid_idx])
            vis_orig_path = self.output_dir / 'visual' / 'individual' / f'Set{set_idx}_{vid_type}_Original.png'
            cv2.imwrite(str(vis_orig_path), vis_orig_img)
            
            # Visual reconstructed
            vis_recon_img = self.tensor_to_numpy_image(vis_recon[vid_idx])
            vis_recon_path = self.output_dir / 'visual' / 'individual' / f'Set{set_idx}_{vid_type}_Reconstructed.png'
            cv2.imwrite(str(vis_recon_path), vis_recon_img)
            
            # Audio original (mel-spectrogram)
            aud_orig_img = self.create_melspec_image(aud_orig[vid_idx])
            aud_orig_path = self.output_dir / 'audio' / 'individual' / f'Set{set_idx}_{vid_type}_Mel_Original.png'
            cv2.imwrite(str(aud_orig_path), aud_orig_img)
            
            # Audio reconstructed
            aud_recon_img = self.create_melspec_image(aud_recon[vid_idx])
            aud_recon_path = self.output_dir / 'audio' / 'individual' / f'Set{set_idx}_{vid_type}_Mel_Reconstructed.png'
            cv2.imwrite(str(aud_recon_path), aud_recon_img)
            
            print(f"   ✓ Saved: {vid_type}")
        
        print(f"   Total: 8 visual + 8 audio = 16 individual images")
    
    def save_sidebyside_comparisons(self, vis_orig: torch.Tensor, vis_recon: torch.Tensor,
                                     aud_orig: torch.Tensor, aud_recon: torch.Tensor, set_idx: int):
        """
        Save side-by-side comparisons for all videos in a set
        
        Args:
            vis_orig: [4, 3, H, W] visual originals
            vis_recon: [4, 3, H, W] visual reconstructions
            aud_orig: [4, 64, 94] audio originals
            aud_recon: [4, 64, 94] audio reconstructions
            set_idx: Set index
        """
        print(f"\n🔀 Saving side-by-side comparisons for Set {set_idx}...")
        
        for vid_idx, vid_type in enumerate(self.video_types):
            # Visual side-by-side
            orig_img = self.tensor_to_numpy_image(vis_orig[vid_idx])
            recon_img = self.tensor_to_numpy_image(vis_recon[vid_idx])
            
            # Add labels
            orig_labeled = self.add_text_label(orig_img, 'Original', 'top')
            recon_labeled = self.add_text_label(recon_img, 'Reconstructed', 'top')
            
            # Concatenate horizontally
            combined_vis = np.hstack([orig_labeled, recon_labeled])
            
            vis_path = self.output_dir / 'visual' / 'comparison_sidebyside' / f'Set{set_idx}_{vid_type}_SideBySide.png'
            cv2.imwrite(str(vis_path), combined_vis)
            
            # Audio side-by-side
            aud_orig_img = self.create_melspec_image(aud_orig[vid_idx])
            aud_recon_img = self.create_melspec_image(aud_recon[vid_idx])
            
            combined_aud = np.hstack([aud_orig_img, aud_recon_img])
            
            # Add title to combined audio
            combined_aud = cv2.copyMakeBorder(combined_aud, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(combined_aud, 'Original', (combined_aud.shape[1]//4 - 40, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(combined_aud, 'Reconstructed', (3*combined_aud.shape[1]//4 - 80, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            aud_path = self.output_dir / 'audio' / 'comparison_sidebyside' / f'Set{set_idx}_{vid_type}_Mel_SideBySide.png'
            cv2.imwrite(str(aud_path), combined_aud)
            
            print(f"   ✓ Saved: {vid_type}")
        
        print(f"   Total: 4 visual + 4 audio = 8 side-by-side images")
    
    def save_grid_comparisons(self, vis_orig: torch.Tensor, vis_recon: torch.Tensor,
                               aud_orig: torch.Tensor, aud_recon: torch.Tensor, set_idx: int):
        """
        Save 2x4 grid layout for a set
        
        Args:
            vis_orig: [4, 3, H, W] visual originals
            vis_recon: [4, 3, H, W] visual reconstructions
            aud_orig: [4, 64, 94] audio originals
            aud_recon: [4, 64, 94] audio reconstructions
            set_idx: Set index
        """
        print(f"\n📊 Saving grid layout for Set {set_idx}...")
        
        # Visual grid (2x4)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for col in range(4):
            # Original row
            orig_img = self.tensor_to_numpy_image(vis_orig[col])
            orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            axes[0, col].imshow(orig_img_rgb)
            axes[0, col].set_title(f'Original {self.video_labels[col]}', fontsize=10, fontweight='bold')
            axes[0, col].axis('off')
            
            # Reconstructed row
            recon_img = self.tensor_to_numpy_image(vis_recon[col])
            recon_img_rgb = cv2.cvtColor(recon_img, cv2.COLOR_BGR2RGB)
            axes[1, col].imshow(recon_img_rgb)
            axes[1, col].set_title(f'Reconstructed {self.video_labels[col]}', fontsize=10, fontweight='bold')
            axes[1, col].axis('off')
        
        plt.suptitle(f'Set {set_idx} - Visual Reconstruction Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        vis_grid_path = self.output_dir / 'visual' / 'comparison_grid' / f'Set{set_idx}_AllVideos_Grid2x4.png'
        plt.savefig(str(vis_grid_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Audio grid (2x4)
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        
        for col in range(4):
            # Original row
            mel_orig = aud_orig[col].detach().cpu().numpy()
            im1 = axes[0, col].imshow(mel_orig, aspect='auto', origin='lower', cmap='viridis')
            axes[0, col].set_title(f'Original {self.video_labels[col]}', fontsize=10, fontweight='bold')
            axes[0, col].set_ylabel('Mel Bins')
            
            # Reconstructed row
            mel_recon = aud_recon[col].detach().cpu().numpy()
            im2 = axes[1, col].imshow(mel_recon, aspect='auto', origin='lower', cmap='viridis')
            axes[1, col].set_title(f'Reconstructed {self.video_labels[col]}', fontsize=10, fontweight='bold')
            axes[1, col].set_xlabel('Time Steps')
            axes[1, col].set_ylabel('Mel Bins')
        
        plt.suptitle(f'Set {set_idx} - Audio Mel-Spectrogram Reconstruction Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        aud_grid_path = self.output_dir / 'audio' / 'comparison_grid' / f'Set{set_idx}_AllVideos_Grid2x4.png'
        plt.savefig(str(aud_grid_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved grid: 1 visual + 1 audio = 2 grid images")
    
    def process_batch_reconstructions(self, batch_data: Dict, model_outputs: Dict, set_idx: int):
        """
        Process a batch and save all visualizations
        
        Args:
            batch_data: Dictionary containing 'frames' and 'mels'
            model_outputs: Dictionary containing 'internal' and 'external' outputs
            set_idx: Current set index (0, 1, 2)
        """
        print(f"\n{'='*60}")
        print(f"📦 Processing Set {set_idx}")
        print(f"{'='*60}")
        
        # Extract data
        # Visual: [B, 4, 75, 3, H, W] -> select frame_idx -> [B, 4, 3, H, W]
        visual_orig = batch_data['frames'][0, :, self.frame_idx]  # [4, 3, H, W]
        
        # Reconstructed visual: model_outputs['internal']['reconstructed']['visual']
        # Shape: [B, 4 or 8, 75, 3, H, W] - take first 4 videos (originals without swap)
        visual_recon_full = model_outputs['internal']['reconstructed']['visual'][0, :4, self.frame_idx]  # [4, 3, H, W]
        
        # Audio: [B, 4, 1, 64, 94] -> [B, 4, 64, 94]
        audio_orig = batch_data['mels'][0, :, 0]  # [4, 64, 94]
        
        # Reconstructed audio: model_outputs['internal']['reconstructed']['audio']
        # Shape: [B, 4 or 8, 1, 64, 96] - take first 4 videos, remove channel dim
        audio_recon_full = model_outputs['internal']['reconstructed']['audio'][0, :4, 0]  # [4, 64, 96]
        
        # Crop audio reconstruction from 96 to 94 time steps to match original
        audio_recon = audio_recon_full[:, :, :94]  # [4, 64, 94]
        
        # Method 1: Individual images (Set 0 only)
        self.save_individual_images(visual_orig, visual_recon_full, audio_orig, audio_recon, set_idx)
        
        # Method 2: Side-by-side comparisons
        self.save_sidebyside_comparisons(visual_orig, visual_recon_full, audio_orig, audio_recon, set_idx)
        
        # Method 3: Grid layouts
        self.save_grid_comparisons(visual_orig, visual_recon_full, audio_orig, audio_recon, set_idx)
    
    def create_summary_report(self):
        """Create a text summary of all generated visualizations"""
        summary_path = self.output_dir / 'summary' / 'visualization_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RECONSTRUCTION VISUALIZATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Number of sets processed: {self.num_sets}\n")
            f.write(f"Frame extracted: {self.frame_idx} (out of 75)\n\n")
            
            f.write("DIRECTORY STRUCTURE:\n")
            f.write("-"*70 + "\n")
            f.write(f"{self.output_dir}/\n")
            f.write("├── visual/\n")
            f.write("│   ├── individual/              (Set 0 only: 8 images)\n")
            f.write("│   ├── comparison_sidebyside/   (All sets: 12 images)\n")
            f.write("│   ├── comparison_grid/         (All sets: 3 images)\n")
            f.write("│   └── comparison_combined/     (Combined grid: upcoming)\n")
            f.write("├── audio/\n")
            f.write("│   ├── individual/              (Set 0 only: 8 images)\n")
            f.write("│   ├── comparison_sidebyside/   (All sets: 12 images)\n")
            f.write("│   ├── comparison_grid/         (All sets: 3 images)\n")
            f.write("│   └── comparison_combined/     (Combined grid: upcoming)\n")
            f.write("└── summary/\n")
            f.write("    └── visualization_summary.txt\n\n")
            
            f.write("FILE NAME CONVENTION:\n")
            f.write("-"*70 + "\n")
            f.write("Set{N}_Video{X}_{Category}_{Stage}.png\n\n")
            f.write("Where:\n")
            f.write("  N: Set number (0, 1, 2)\n")
            f.write("  X: Video type (A, B, C, D)\n")
            f.write("  Category: RealReal, RealFakeAudio, FakeRealAudio, FakeFake\n")
            f.write("  Stage: Original, Reconstructed, SideBySide, or Grid2x4\n\n")
            
            f.write("TOTAL IMAGE COUNT:\n")
            f.write("-"*70 + "\n")
            
            # Calculate totals
            set0_individual = 8 + 8  # visual + audio
            all_sets_sidebyside = self.num_sets * 4 * 2  # sets * videos * (visual+audio)
            all_sets_grids = self.num_sets * 2  # sets * (visual+audio)
            
            total = set0_individual + all_sets_sidebyside + all_sets_grids
            
            f.write(f"Individual images (Set 0 only):     {set0_individual:3d}\n")
            f.write(f"Side-by-side comparisons (all sets): {all_sets_sidebyside:3d}\n")
            f.write(f"Grid layouts (all sets):             {all_sets_grids:3d}\n")
            f.write(f"{'='*70}\n")
            f.write(f"TOTAL IMAGES:                        {total:3d}\n")
            f.write(f"{'='*70}\n")
        
        print(f"\n📝 Summary report saved: {summary_path}")


def run_reconstruction_visualization(trainer, num_sets: int = 3, frame_idx: int = 37):
    """
    Main function to run reconstruction visualization during test mode
    
    Args:
        trainer: Trainer instance with loaded model and val_loader
        num_sets: Number of sets to process (default: 3)
        frame_idx: Frame index to extract (default: 37, middle of 75 frames)
    
    Returns:
        Path to output directory
    """
    print("\n" + "="*70)
    print("🎨 RECONSTRUCTION VISUALIZATION")
    print("="*70)
    
    # Check if we have validation loader (needed for sets of 4 videos)
    if not hasattr(trainer, 'val_loader') or trainer.val_loader is None:
        print("❌ ERROR: Validation loader not available!")
        print("   Reconstruction visualization requires sets of 4 videos (A, B, C, D).")
        print("   Test loader only provides single videos.")
        print("\n💡 Solution: The trainer needs to be initialized with val_loader.")
        print("   This is only available when test_mode=False in Config.")
        return None
    
    # Initialize visualizer
    visualizer = ReconstructionVisualizer(
        output_dir='reconstruction_visualizations',
        num_sets=num_sets,
        frame_idx=frame_idx
    )
    
    total_batches = len(trainer.val_loader)
    print(f"📊 Using TEST dataset in 4-video set format for visualization")
    print(f"   Total batches available: {total_batches}")
    
    # Randomly select num_sets indices from available batches
    import random
    if num_sets >= total_batches:
        # If requesting more sets than available, use all
        selected_indices = list(range(total_batches))
        print(f"   ⚠️  Requested {num_sets} sets but only {total_batches} available - using all")
    else:
        selected_indices = sorted(random.sample(range(total_batches), num_sets))
        print(f"   🎲 Randomly selected {num_sets} sets: {selected_indices}")
    
    # Set model to eval mode
    trainer.model.eval()
    
    # Process randomly selected batches from validation set
    processed_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(trainer.val_loader):
            # Only process if this batch index is in our random selection
            if batch_idx in selected_indices:
                # Process batch (NOT single_video_mode - we need all 4 videos)
                processed_batch = trainer._process_batch(batch, single_video_mode=False)
                
                # Forward pass
                model_outputs = trainer.model(processed_batch)
                
                # Visualize reconstructions (use processed_count for naming, not batch_idx)
                visualizer.process_batch_reconstructions(processed_batch, model_outputs, processed_count)
                processed_count += 1
                
                # Stop if we've processed enough sets
                if processed_count >= num_sets:
                    break
    
    # Create summary
    visualizer.create_summary_report()
    
    print("\n" + "="*70)
    print("✅ RECONSTRUCTION VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"📁 Output directory: {visualizer.output_dir.absolute()}")
    print(f"📊 Total sets processed: {processed_count}")
    print(f"🎲 Batch indices used: {selected_indices[:processed_count]}")
    print("\nCheck the 'summary' folder for a complete visualization report!")
    
    return visualizer.output_dir

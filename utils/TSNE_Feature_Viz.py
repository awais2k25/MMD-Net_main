"""
TSNE_Feature_Viz.py - t-SNE Feature Visualization
==================================================
Visualizes latent features from MMD-Net using t-SNE dimensionality reduction.

Creates 3 visualizations:
1. Specific Features (modality_specific from CMAF)
2. Common Features (modality_common from CMAF)  
3. Content Features (Visual/Audio irrelevant features)

Following the style from the reference paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns


class TSNEFeatureVisualizer:
    """
    Visualizes MMD-Net features using t-SNE
    
    Creates scatter plots showing how different forgery types cluster in latent space
    """
    
    def __init__(self, output_dir: str = 'tsne_visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Video type labels matching FakeAVCeleb dataset (A, B, C, D)
        self.video_labels = {
            0: 'A (RealVideo-RealAudio)',           # Category A: Real video + Real audio
            1: 'B (RealVideo-FakeAudio)',        # Category B: Real video + Fake audio
            2: 'C (FakeVideo-RealAudio)',        # Category C: Fake video + Real audio
            3: 'D (FakeVideo-FakeAudio)'            # Category D: Fake video + Fake audio
        }
        
        # Bright, distinct color palette for maximum visibility
        self.colors = {
            'A (RealVideo-RealAudio)': '#0066FF',        # Bright Blue
            'B (RealVideo-FakeAudio)': '#FF6600',     # Bright Orange
            'C (FakeVideo-RealAudio)': '#00CC00',     #    Bright Green
            'D (FakeVideo-FakeAudio)': '#FF0033'         # Bright Red
        }
        
        print(f"✅ TSNEFeatureVisualizer initialized")
        print(f"   Output directory: {self.output_dir}")
    
    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        """Clean features and add jitter to avoid t-SNE reshape errors"""
        # Ensure it's a numpy array
        features = np.array(features)
        
        # Remove NaNs/Infs
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add tiny jitter to avoid issues with duplicate features 
        # which can cause t-SNE/KNN errors in sklearn
        # Use a fixed seed for reproducibility within a run
        rng = np.random.RandomState(42)
        jitter = rng.normal(0, 1e-6, features.shape)
        
        return features + jitter
    
    def collect_features(self, trainer, max_samples: int = None):
        """
        Collect features from entire test set
        
        Args:
            trainer: Trainer instance with loaded model and test_loader
            max_samples: Maximum samples to collect (None = all)
        
        Returns:
            Dictionary containing collected features and labels
        """
        print("\n" + "="*70)
        print("📊 COLLECTING FEATURES FOR t-SNE")
        print("="*70)
        
        trainer.model.eval()
        
        # Storage for features
        specific_features = []
        common_features = []
        content_features_visual = []
        content_features_audio = []
        labels = []
        
        total_batches = len(trainer.test_loader)
        samples_collected = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(trainer.test_loader):
                if max_samples and samples_collected >= max_samples:
                    break
                
                # Process batch
                processed_batch = trainer._process_batch(batch, single_video_mode=True)
                
                # Forward pass
                model_outputs = trainer.model(processed_batch)
                
                # Extract features from CMAF output
                av_features = model_outputs['internal']['av_features']
                modality_common = av_features['static']['combined']['modality_common']  # [B, 4, 512]
                modality_specific = av_features['static']['combined']['modality_specific']  # [B, 4, 512]
                
                # Extract content features (irrelevant features from encoders)
                encoder_features = model_outputs['internal']['encoder_features']
                vir = encoder_features['visual']['vir']  # [B, 4, 75, 512]
                air = encoder_features['audio']['air']   # [B, 4, 75, 512]
                
                # Average over temporal dimension for content features
                vir_avg = vir.mean(dim=2)  # [B, 4, 512]
                air_avg = air.mean(dim=2)  # [B, 4, 512]
                
                # Get labels (multi-class labels)
                batch_labels = processed_batch['labels_multi']  # [B, V] where V=1 or 4
                
                # Get actual number of videos in this batch (1 for test mode, 4 for train/val mode)
                B, num_videos = modality_common.shape[0], modality_common.shape[1]
                
                # Store features (flatten batch and video dimensions)
                for b in range(B):
                    for v in range(num_videos):  # Dynamically handle 1 or 4 videos
                        specific_features.append(modality_specific[b, v].cpu().numpy())
                        common_features.append(modality_common[b, v].cpu().numpy())
                        content_features_visual.append(vir_avg[b, v].cpu().numpy())
                        content_features_audio.append(air_avg[b, v].cpu().numpy())
                        labels.append(batch_labels[b, v].item())
                        samples_collected += 1
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"   Processed {batch_idx + 1}/{total_batches} batches, {samples_collected} samples collected")
        
        print(f"\n✅ Feature collection complete!")
        print(f"   Total samples: {samples_collected}")
        print(f"   Feature dimension: 512")
        
        # Convert to numpy arrays
        features_dict = {
            'specific': np.array(specific_features),      # [N, 512]
            'common': np.array(common_features),          # [N, 512]
            'content_visual': np.array(content_features_visual),  # [N, 512]
            'content_audio': np.array(content_features_audio),    # [N, 512]
            'labels': np.array(labels)                    # [N]
        }
        
        return features_dict
    
    def collect_features_from_splits(self, trainer, splits: List[str], max_samples_per_split: int = None):
        """
        Collect features from multiple dataset splits
        
        Args:
            trainer: Trainer instance with loaded model
            splits: List of splits to use: ['train', 'val', 'test']
            max_samples_per_split: Max samples per split (None = all)
        
        Returns:
            Dictionary containing combined features from all splits
        """
        print("\n" + "="*70)
        print(f"📊 COLLECTING FEATURES FROM SPLITS: {', '.join(splits)}")
        print("="*70)
        
        trainer.model.eval()
        
        # Storage for features
        all_specific = []
        all_common = []
        all_content_visual = []
        all_content_audio = []
        all_labels = []
        all_split_names = []  # Track which split each sample came from
        
        # Process each requested split
        for split_name in splits:
            print(f"\n📂 Processing {split_name.upper()} split...")
            
            # Get appropriate dataloader
            if split_name == 'test':
                dataloader = trainer.test_loader
            elif split_name == 'val':
                # Need to create val loader if not exists
                if not hasattr(trainer, 'val_loader') or trainer.val_loader is None:
                    from data import get_dataloader_v4
                    from functools import partial
                    def worker_init_fn(worker_id):
                        import numpy as np
                        import random
                        seed = trainer.config.seed + worker_id
                        np.random.seed(seed)
                        random.seed(seed)
                    
                    dataloader = get_dataloader_v4(
                        root_dir=trainer.config.data_dir,
                        split='val',
                        batch_size=trainer.config.batch_size,
                        num_workers=trainer.config.num_workers,
                        worker_init_fn=worker_init_fn if trainer.config.num_workers > 0 else None
                    )
                else:
                    dataloader = trainer.val_loader
            elif split_name == 'train':
                # Create train loader
                from data import get_dataloader_v4
                def worker_init_fn(worker_id):
                    import numpy as np
                    import random
                    seed = trainer.config.seed + worker_id
                    np.random.seed(seed)
                    random.seed(seed)
                
                dataloader = get_dataloader_v4(
                    root_dir=trainer.config.data_dir,
                    split='train',
                    batch_size=trainer.config.batch_size,
                    num_workers=trainer.config.num_workers,
                    worker_init_fn=worker_init_fn if trainer.config.num_workers > 0 else None
                )
            else:
                print(f"   ⚠️  Unknown split '{split_name}', skipping...")
                continue
            
            split_samples = 0
            total_batches = len(dataloader)
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if max_samples_per_split and split_samples >= max_samples_per_split:
                        break
                    
                    # Determine if single video mode based on split
                    single_video = (split_name == 'test')
                    
                    # Process batch
                    processed_batch = trainer._process_batch(batch, single_video_mode=single_video)
                    
                    # Forward pass
                    model_outputs = trainer.model(processed_batch)
                    
                    # Extract features
                    av_features = model_outputs['internal']['av_features']
                    modality_common = av_features['static']['combined']['modality_common']
                    modality_specific = av_features['static']['combined']['modality_specific']
                    
                    encoder_features = model_outputs['internal']['encoder_features']
                    vir = encoder_features['visual']['vir']
                    air = encoder_features['audio']['air']
                    vir_avg = vir.mean(dim=2)
                    air_avg = air.mean(dim=2)
                    
                    batch_labels = processed_batch['labels_multi']
                    B, num_videos = modality_common.shape[0], modality_common.shape[1]
                    
                    # Store features
                    for b in range(B):
                        for v in range(num_videos):
                            all_specific.append(modality_specific[b, v].cpu().numpy())
                            all_common.append(modality_common[b, v].cpu().numpy())
                            all_content_visual.append(vir_avg[b, v].cpu().numpy())
                            all_content_audio.append(air_avg[b, v].cpu().numpy())
                            all_labels.append(batch_labels[b, v].item())
                            all_split_names.append(split_name)
                            split_samples += 1
                    
                    if (batch_idx + 1) % 20 == 0:
                        print(f"   Processed {batch_idx + 1}/{total_batches} batches, {split_samples} samples from {split_name}")
            
            print(f"   ✓ Collected {split_samples} samples from {split_name.upper()}")
        
        total_samples = len(all_labels)
        print(f"\n✅ Multi-split feature collection complete!")
        print(f"   Total samples across all splits: {total_samples}")
        
        # Convert to numpy arrays
        features_dict = {
            'specific': np.array(all_specific),
            'common': np.array(all_common),
            'content_visual': np.array(all_content_visual),
            'content_audio': np.array(all_content_audio),
            'labels': np.array(all_labels),
            'splits': all_split_names  # Keep as list for flexibility
        }
        
        return features_dict
    
    def create_tsne_visualization(self, features: np.ndarray, labels: np.ndarray, 
                                   title: str, filename: str):
        """
        Create and save t-SNE visualization
        
        Args:
            features: Feature array [N, D]
            labels: Label array [N]
            title: Plot title
            filename: Output filename
        """
        features = self._prepare_features(features)
        print(f"\n🔄 Computing t-SNE for {title}...")
        
        # Apply t-SNE - use method='exact' for better stability on smaller datasets
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate='auto',
            n_iter=1000,
            random_state=42,
            method='exact',
            init='random',
            verbose=0
        )
        
        features_2d = tsne.fit_transform(features)
        
        print(f"   ✓ t-SNE computed: {features.shape} → {features_2d.shape}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each class separately for legend
        for label_idx in np.unique(labels):
            mask = labels == label_idx
            label_name = self.video_labels.get(label_idx, f'Class_{label_idx}')
            color = self.colors.get(label_name, '#333333')
            
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=color,
                label=label_name,
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        
        # Styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved: {output_path}")
    
    def create_combined_visualization(self, features_dict: Dict):
        """
        Create combined visualization with all three feature types side-by-side
        
        Args:
            features_dict: Dictionary containing specific, common, and content features
        """
        print(f"\n🎨 Creating combined t-SNE visualization...")
        
        # Prepare all features
        specific_f = self._prepare_features(features_dict['specific'])
        common_f = self._prepare_features(features_dict['common'])
        content_f = self._prepare_features(features_dict['content_visual'])
        
        # Compute t-SNE for all three feature types
        # Use method='exact' for robustness against duplicate points
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', 
                   n_iter=1000, random_state=42, method='exact', init='random', verbose=0)
        
        specific_2d = tsne.fit_transform(specific_f)
        
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto',
                   n_iter=1000, random_state=42, method='exact', init='random', verbose=0)
        common_2d = tsne.fit_transform(common_f)
        
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto',
                   n_iter=1000, random_state=42, method='exact', init='random', verbose=0)
        content_2d = tsne.fit_transform(content_f)
        
        labels = features_dict['labels']
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Specific Features
        for label_idx in np.unique(labels):
            mask = labels == label_idx
            label_name = self.video_labels.get(label_idx, f'Class_{label_idx}')
            color = self.colors.get(label_name, '#333333')
            
            axes[0].scatter(specific_2d[mask, 0], specific_2d[mask, 1],
                          c=color, label=label_name, alpha=0.6, s=15, edgecolors='none')
        
        axes[0].set_title('Specific Features', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Plot 2: Common Features
        for label_idx in np.unique(labels):
            mask = labels == label_idx
            label_name = self.video_labels.get(label_idx, f'Class_{label_idx}')
            color = self.colors.get(label_name, '#333333')
            
            axes[1].scatter(common_2d[mask, 0], common_2d[mask, 1],
                          c=color, label=label_name, alpha=0.6, s=15, edgecolors='none')
        
        axes[1].set_title('Common Features', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        # Plot 3: Content Features
        for label_idx in np.unique(labels):
            mask = labels == label_idx
            label_name = self.video_labels.get(label_idx, f'Class_{label_idx}')
            color = self.colors.get(label_name, '#333333')
            
            axes[2].scatter(content_2d[mask, 0], content_2d[mask, 1],
                          c=color, label=label_name, alpha=0.6, s=15, edgecolors='none')
        
        axes[2].set_title('Content Features', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        
        # Add common legend
        handles, labels_text = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_text, loc='lower center', ncol=5, 
                  frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.05))
        
        plt.suptitle('t-SNE Visualization of MMD-Net Features', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'combined_tsne_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved combined visualization: {output_path}")
    
    def run_visualization(self, trainer, max_samples: int = None):
        """
        Main function to run complete t-SNE visualization pipeline
        
        Args:
            trainer: Trainer instance with loaded model and test_loader
            max_samples: Maximum samples to use (None = all test set)
        """
        # Collect features
        features_dict = self.collect_features(trainer, max_samples)
        
        print("\n" + "="*70)
        print("🎨 GENERATING t-SNE VISUALIZATIONS")
        print("="*70)
        
        # Create individual visualizations
        self.create_tsne_visualization(
            features_dict['specific'],
            features_dict['labels'],
            'Specific Features (Modality-Specific)',
            'tsne_specific_features.png'
        )
        
        self.create_tsne_visualization(
            features_dict['common'],
            features_dict['labels'],
            'Common Features (Modality-Common)',
            'tsne_common_features.png'
        )
        
        self.create_tsne_visualization(
            features_dict['content_visual'],
            features_dict['labels'],
            'Content Features (Visual Irrelevant)',
            'tsne_content_visual.png'
        )
        
        self.create_tsne_visualization(
            features_dict['content_audio'],
            features_dict['labels'],
            'Content Features (Audio Irrelevant)',
            'tsne_content_audio.png'
        )
        
        # Create combined visualization
        self.create_combined_visualization(features_dict)
        
        print("\n" + "="*70)
        print("✅ t-SNE VISUALIZATION COMPLETE!")
        print("="*70)
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"📊 Total samples visualized: {len(features_dict['labels'])}")
        print("\nGenerated files:")
        print("  • tsne_specific_features.png")
        print("  • tsne_common_features.png")
        print("  • tsne_content_visual.png")
        print("  • tsne_content_audio.png")
        print("  • combined_tsne_visualization.png")
        
        return self.output_dir


def run_multi_split_visualizations(trainer, split_combinations: List[List[str]] = None, 
                                   max_samples_per_split: int = None):
    """
    Run t-SNE visualizations for multiple split combinations
    
    Args:
        trainer: Trainer instance
        split_combinations: List of split combinations to visualize
                           e.g. [['test'], ['val'], ['train'], ['test', 'val'], ['test', 'val', 'train']]
        max_samples_per_split: Max samples per split (None = all)
    
    Returns:
        Dictionary mapping split names to output directories
    """
    if split_combinations is None:
        # Default: only test split as requested
        split_combinations = [['test']]
    
    print("\n" + "="*70)
    print("🎨 MULTI-SPLIT t-SNE VISUALIZATION")
    print("="*70)
    print(f"Will generate visualizations for {len(split_combinations)} split combinations:")
    for i, splits in enumerate(split_combinations, 1):
        print(f"  {i}. {' + '.join(splits).upper()}")
    
    output_dirs = {}
    
    for splits in split_combinations:
        # Create descriptive name
        split_name = '_'.join(splits)
        output_dir_name = f'tsne_visualizations_{split_name}'
        
        print("\n" + "="*70)
        print(f"📊 PROCESSING: {' + '.join(splits).upper()}")
        print("="*70)
        
        # Create visualizer with specific output directory
        visualizer = TSNEFeatureVisualizer(output_dir=output_dir_name)
        
        # Collect features from specified splits
        features_dict = visualizer.collect_features_from_splits(
            trainer, splits, max_samples_per_split
        )
        
        # Generate visualizations
        print("\n" + "="*70)
        print(f"🎨 GENERATING t-SNE VISUALIZATIONS FOR: {' + '.join(splits).upper()}")
        print("="*70)
        
        visualizer.create_tsne_visualization(
            features_dict['specific'],
            features_dict['labels'],
            f'Specific Features ({split_name.upper()})',
            'tsne_specific_features.png'
        )
        
        visualizer.create_tsne_visualization(
            features_dict['common'],
            features_dict['labels'],
            f'Common Features ({split_name.upper()})',
            'tsne_common_features.png'
        )
        
        visualizer.create_tsne_visualization(
            features_dict['content_visual'],
            features_dict['labels'],
            f'Content Features - Visual ({split_name.upper()})',
            'tsne_content_visual.png'
        )
        
        visualizer.create_tsne_visualization(
            features_dict['content_audio'],
            features_dict['labels'],
            f'Content Features - Audio ({split_name.upper()})',
            'tsne_content_audio.png'
        )
        
        visualizer.create_combined_visualization(features_dict)
        
        output_dirs[split_name] = visualizer.output_dir
        
        print(f"\n✅ Completed: {split_name.upper()}")
        print(f"   Output: {visualizer.output_dir.absolute()}")
    
    print("\n" + "="*70)
    print("✅ ALL t-SNE VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\nGenerated visualization sets:")
    for split_name, output_dir in output_dirs.items():
        print(f"  • {split_name.upper()}: {output_dir.absolute()}")
    
    return output_dirs


def run_tsne_visualization(trainer, max_samples: int = None):
    """
    Helper function to run t-SNE visualization (single split - test only)
    
    Args:
        trainer: Trainer instance
        max_samples: Max samples to use (None = entire test set)
    
    Returns:
        Path to output directory
    """
    visualizer = TSNEFeatureVisualizer()
    return visualizer.run_visualization(trainer, max_samples)

"""
Main_v3.py - Safer and Cleaner Training Script
==============================================
Simplified version of Main_v2.py that removes:
- AMP/GradScaler (causes instability)
- OneCycleLR scheduler (requires fine-tuning)
- WandB logging (adds overhead)
- Verbose debug prints

Preserves:
- Core training/validation loops
- Uncertainty-Weighted Multi-Loss System
- Multi-video batch processing
- Comprehensive metrics (binary + multi-class, AUC, F1)
- Rolling-average CSV logging
- Gradient clipping + NaN guards
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import csv
import logging
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# Local imports
from data import get_dataloader_v4, get_dataloader_v4_single
from models.MMD_NET import get_mmdnet
from utils.Losses_v2 import get_loss_fn
from utils.Recon_Comp_Viz import run_reconstruction_visualization
from utils.TSNE_Feature_Viz import run_tsne_visualization, run_multi_split_visualizations


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Initialize worker seeds for DataLoader reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_confusion_matrix_metrics(y_true, y_pred):
    """Compute TPR, FPR, TNR, FNR, and Balanced Accuracy from confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        return {'tpr': TPR, 'fpr': FPR, 'tnr': TNR, 'fnr': FNR, 'balanced_acc': (TPR + TNR) / 2.0}
    return {'tpr': 0.0, 'fpr': 0.0, 'tnr': 0.0, 'fnr': 0.0, 'balanced_acc': 0.0}


def plot_classification_metrics(
    bin_true_concat=None, bin_proba_concat=None,
    multi_true_concat=None, multi_proba_concat=None,
    class_names=None, save_prefix="metrics"
):
    """Generate ROC, PR, and Confusion Matrix plots for binary and multi-class results."""
    if bin_true_concat is not None and bin_proba_concat is not None:
        print("\n=== Binary Classification Metrics ===")
        fpr, tpr, _ = roc_curve(bin_true_concat, bin_proba_concat)
        roc_auc_val = auc(fpr, tpr)
        pr_auc_val = average_precision_score(bin_true_concat, bin_proba_concat)
        print(f"ROC AUC: {roc_auc_val:.4f}  PR AUC: {pr_auc_val:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='b', label=f'ROC AUC = {roc_auc_val:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Binary ROC Curve'); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(f"{save_prefix}_binary_ROC.png", dpi=150); plt.close()

        prec, rec, _ = precision_recall_curve(bin_true_concat, bin_proba_concat)
        plt.figure(figsize=(8, 6))
        plt.plot(rec, prec, color='g', label=f'PR AUC = {pr_auc_val:.3f}')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('Binary Precision-Recall Curve'); plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(f"{save_prefix}_binary_PR.png", dpi=150); plt.close()

        bin_pred = (bin_proba_concat > 0.5).astype(int)
        cm_val = confusion_matrix(bin_true_concat, bin_pred, labels=[0, 1])
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay(cm_val, display_labels=['Real', 'Fake']).plot(cmap='Blues', values_format='d')
        plt.title('Binary Confusion Matrix'); plt.tight_layout()
        plt.savefig(f"{save_prefix}_binary_CM.png", dpi=150); plt.close()

    if multi_true_concat is not None and multi_proba_concat is not None:
        print("\n=== Multi-class Classification Metrics ===")
        n_classes = multi_proba_concat.shape[1]
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(n_classes)]
        roc_auc_list, pr_auc_list = [], []

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            true_bin = (multi_true_concat == i).astype(int)
            fpr, tpr, _ = roc_curve(true_bin, multi_proba_concat[:, i])
            roc_auc_val = auc(fpr, tpr); roc_auc_list.append(roc_auc_val)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc_val:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Multi-class ROC (OvR)')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(f"{save_prefix}_multi_ROC.png", dpi=150); plt.close()

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            true_bin = (multi_true_concat == i).astype(int)
            prec, rec, _ = precision_recall_curve(true_bin, multi_proba_concat[:, i])
            pr_auc_val = average_precision_score(true_bin, multi_proba_concat[:, i])
            pr_auc_list.append(pr_auc_val)
            plt.plot(rec, prec, label=f"{class_names[i]} (PR AUC={pr_auc_val:.3f})")
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Multi-class PR Curves (OvR)')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(f"{save_prefix}_multi_PR.png", dpi=150); plt.close()

        multi_pred = np.argmax(multi_proba_concat, axis=1)
        cm_multi = confusion_matrix(multi_true_concat, multi_pred, labels=range(n_classes))
        plt.figure(figsize=(10, 8))
        ConfusionMatrixDisplay(cm_multi, display_labels=class_names).plot(cmap='Blues', values_format='d')
        plt.title('Multi-class Confusion Matrix')
        plt.tight_layout(); plt.savefig(f"{save_prefix}_multi_CM.png", dpi=150); plt.close()
        print(f"Macro ROC AUC={np.mean(roc_auc_list):.4f}  Macro PR AUC={np.mean(pr_auc_list):.4f}")


class Trainer:
    def __init__(self, config):
        """
        Initialize Trainer with simplified configuration
        
        Args:
            config: Configuration object
            loss_weights: Dict with keys 'reconstruction', 'orthogonality', 'binary', 'multi', 'triplet'
                         Values should sum to 1.0. If None, uses default weights.
        """
        self.config = config
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model initialization
        self.model = get_mmdnet(
            visual_encoder=config.visual_encoder,
            audio_encoder=config.audio_encoder
        ).to(self.device).float()
        
        # Loss and optimizer (skipped in test mode)
        self.criterion = get_loss_fn().to(self.device)
        if not getattr(config, 'test_mode', False):
            # FIXED: Include criterion parameters so that uncertainty log_vars are optimized
            self.optimizer = AdamW(
                list(self.model.parameters()) + list(self.criterion.parameters()),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

            # Freeze EfficientNet-B2 backbone (all its weights).
            frozen_params = 0
            for param in self.model.visual_encoder.backbone.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            print(f"\U0001f512 Froze EfficientNet-B2 backbone: {frozen_params:,} parameters frozen")

            # LR Scheduler: Cosine annealing with warm restarts. T_0=30 per round.
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=30, T_mult=1, eta_min=1e-6
            )
        else:
            self.optimizer = None
            self.scheduler = None
            print("\U0001f9ea Test mode: optimizer/scheduler disabled.")

        # Load trained model if provided (for test mode)
        if getattr(config, 'trained_model_path', None):
            model_path = Path(config.trained_model_path)
            if model_path.exists():
                print(f"\U0001f4e6 Loading trained model from {model_path}")
                checkpoint = torch.load(str(model_path), map_location=self.device)
                state = checkpoint.get('model_state_dict', checkpoint)
                # strict=False: ignore reconstruction module keys from v2_6 checkpoint
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                if unexpected:
                    print(f"   ℹ️  Ignored {len(unexpected)} unexpected keys (reconstruction ablation)")
                print("✅ Trained model loaded.")
            else:
                raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        # Data loaders
        if not getattr(config, 'test_mode', False):
            self.train_loader = get_dataloader_v4(
                root_dir=config.data_dir, split='train',
                batch_size=config.batch_size, num_workers=config.num_workers,
                worker_init_fn=worker_init_fn if config.num_workers > 0 else None
            )
            self.val_loader = get_dataloader_v4(
                root_dir=config.data_dir, split='val',
                batch_size=config.batch_size, num_workers=config.num_workers,
                worker_init_fn=worker_init_fn if config.num_workers > 0 else None
            )
            self.test_loader = None
        else:
            # Test mode: only test/val loaders, no train loader
            self.train_loader = None
            self.val_loader = get_dataloader_v4(
                root_dir=config.data_dir, split='val',
                batch_size=config.batch_size, num_workers=config.num_workers,
                worker_init_fn=worker_init_fn if config.num_workers > 0 else None
            ) if getattr(config, 'visualize_reconstructions', False) else None
            self.test_loader = get_dataloader_v4_single(
                root_dir=config.data_dir, split='test',
                batch_size=config.batch_size, num_workers=config.num_workers,
                worker_init_fn=worker_init_fn if config.num_workers > 0 else None
            )
        
        # Directory setup
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # CSV logging paths
        self.train_csv = self.log_dir / 'train_metrics.csv'
        self.val_csv = self.log_dir / 'val_metrics.csv'
        self.test_csv = self.log_dir / 'test_metrics.csv'
        if not getattr(config, 'test_mode', False):
            self._init_csv(self.train_csv)
            self._init_csv(self.val_csv)
        else:
            self._init_csv(self.test_csv)
        
        # Best metrics tracking (in-memory; overridden by persistent file below)
        self.best_binary_acc = 0.0
        self.best_auc = 0.0
        
        # Path for the persistent global-best file
        self.best_metrics_file = self.checkpoint_dir / 'best_metrics.txt'
        
        # Load global bests from file (if present) so a fresh start respects prior runs
        self._load_global_best()
        
        # Training configuration
        self.max_loss_threshold = config.max_loss_threshold
        self.grad_clip = config.grad_clip
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        
        # Logging setup
        self.logger = self._setup_logger()
        
        # Resume support
        self.start_epoch = 1
        if getattr(config, 'resume_checkpoint', None):
            resume_path = Path(config.resume_checkpoint)
            if resume_path.exists():
                checkpoint = torch.load(str(resume_path), map_location=self.device)
                # strict=False: ignore reconstruction module keys from v2_6 checkpoint
                missing, unexpected = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if unexpected:
                    print(f"   ℹ️  Ignored {len(unexpected)} unexpected keys (reconstruction ablation)")
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception:
                        print("⚠️ Optimizer state could not be loaded; continuing with fresh optimizer.")
                # Determine start epoch: prefer explicit config, else checkpoint epoch + 1
                self.start_epoch = getattr(config, 'resume_start_epoch', None) or int(checkpoint.get('epoch', 0)) + 1
                # Restore best metrics from checkpoint as a fallback
                ckpt_metrics = checkpoint.get('metrics', {}) or {}
                self.best_binary_acc = float(ckpt_metrics.get('binary_acc', self.best_binary_acc))
                self.best_auc = float(ckpt_metrics.get('binary_auc', self.best_auc))
                # Always let the persistent file override the checkpoint values
                self._load_global_best()
                print(f"🔁 Resumed from '{resume_path.name}' → starting at epoch {self.start_epoch}")
                print(f"   Global best (from file): Acc={self.best_binary_acc:.4f}, AUC={self.best_auc:.4f}")
            else:
                print(f"⚠️ Resume checkpoint not found at {resume_path}. Starting fresh.")

        print(f"✅ Trainer initialized on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    # ------------------------------------------------------------------
    # Persistent global-best helpers
    # ------------------------------------------------------------------
    def _load_global_best(self):
        """Read best_metrics.txt and update in-memory bests if the file values are higher.

        The parser is intentionally lenient:
        - Values may have trailing commas or extra whitespace (e.g. "0.8076, ")
        - Supports the canonical keys written by _save_global_best:
              best_binary_acc = ...
              best_auc        = ...
        - Also supports short-form keys a human might type manually:
              Acc = ...    (case-insensitive)
              AUC = ...    (case-insensitive)
        - Lines that cannot be parsed are logged but do NOT raise an exception.
        """
        if not self.best_metrics_file.exists():
            return
        try:
            with open(self.best_metrics_file, 'r') as f:
                lines = f.read().strip().splitlines()
        except Exception as e:
            print(f"⚠️ Could not open best_metrics.txt: {e}")
            return

        data = {}
        for line in lines:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key_raw, val_raw = line.split('=', 1)
            key = key_raw.strip().lower()  # normalise to lower-case
            # Strip trailing commas, spaces, and any junk after the number
            val_clean = val_raw.strip().rstrip(',').strip()
            try:
                val = float(val_clean)
            except ValueError:
                print(f"⚠️ best_metrics.txt: cannot parse value on line: '{line}' — skipping")
                continue
            data[key] = val

        # Resolve the two metrics, accepting both canonical and short-form key names
        file_acc = data.get('best_binary_acc', data.get('acc', 0.0))
        file_auc = data.get('best_auc',        data.get('auc', 0.0))

        # Take the maximum so in-memory bests never regress
        self.best_binary_acc = max(self.best_binary_acc, file_acc)
        self.best_auc        = max(self.best_auc,        file_auc)
        print(f"📂 Loaded global best from file → Acc={self.best_binary_acc:.4f}, AUC={self.best_auc:.4f}")

    def _save_global_best(self):
        """Overwrite best_metrics.txt with current in-memory best values."""
        try:
            with open(self.best_metrics_file, 'w') as f:
                f.write(f"best_binary_acc = {self.best_binary_acc:.6f}\n")
                f.write(f"best_auc        = {self.best_auc:.6f}\n")
        except Exception as e:
            print(f"⚠️ Could not write best_metrics.txt: {e}")
    # ------------------------------------------------------------------

    def _init_csv(self, path):
        """Initialize CSV with header"""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'total_batches', 'epoch_progress', 'split', 'loss',
                'loss_binary', 'loss_multi', 'loss_rec_video', 'loss_rec_audio', 'loss_triplet', 'loss_orthogonality',
                'uw_binary', 'uw_multi', 'uw_rec_video', 'uw_rec_audio', 'uw_triplet', 'uw_orthogonality',
                'optimal_threshold', 'optimal_acc',
                'binary_acc', 'multi_acc',
                'binary_auc', 'multi_auc', 'binary_precision', 'binary_recall', 'binary_f1',
                'multi_precision', 'multi_recall', 'multi_f1'
            ])
    
    def _setup_logger(self):
        """Setup logging for debugging"""
        log_path = self.log_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'
        logging.basicConfig(
            filename=str(log_path),
            level=logging.WARNING,
            format='%(asctime)s - %(message)s'
        )
        return logging.getLogger('trainer')
    
    def train_epoch(self, epoch):
        """Training epoch with gradient accumulation and rolling-average CSV logging"""
        self.model.train()
        running_loss = 0.0
        running_metrics = {}
        running_losses = {'loss_binary': 0.0, 'loss_multi': 0.0, 'loss_rec_video': 0.0, 'loss_rec_audio': 0.0, 'loss_triplet': 0.0, 'loss_orthogonality': 0.0}
        running_uw = {'uw_binary': 0.0, 'uw_multi': 0.0, 'uw_rec_video': 0.0, 'uw_rec_audio': 0.0, 'uw_triplet': 0.0, 'uw_orthogonality': 0.0}
        log_interval = 20  # batch-wise logging interval
        
        # Epoch-level accumulators
        epoch_loss_total = 0.0
        num_batches = 0
        
        # For epoch-wide sklearn metrics
        bin_proba_all = []   # list of np arrays shape [N]
        bin_true_all = []    # list of np arrays shape [N]
        multi_proba_all = [] # list of np arrays shape [N, 4]
        multi_true_all = []  # list of np arrays shape [N]
        
        # For epoch-wide accuracies
        bin_correct = 0
        bin_total = 0
        multi_correct = 0
        multi_total = 0
        
        # Gradient accumulation: zero gradients at start of accumulation cycle
        self.optimizer.zero_grad()
        accum_step_count = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f'Train Epoch {epoch}')):
            try:
                # Process batch
                processed_batch = self._process_batch(batch)
                
                # Forward pass
                model_outputs = self.model(processed_batch)
                loss_dict = self.criterion(model_outputs, processed_batch)
                loss = loss_dict['total_loss']
                
                # NaN guard
                if torch.isnan(loss) or loss.item() > self.max_loss_threshold:
                    self.logger.warning(f"Skipping batch {batch_idx} - invalid loss: {loss.item():.4f}")
                    continue
                
                # Scale loss by accumulation steps (to average over accumulated batches)
                scaled_loss = loss / self.gradient_accumulation_steps
                
                # Backward pass (accumulates gradients)
                scaled_loss.backward()
                accum_step_count += 1
                
                # Accumulate metrics (use unscaled loss for logging)
                running_loss += loss.item()
                # Accumulate loss components if present
                for k_csv, k_src in [('loss_binary','binary'), ('loss_multi','multi'), ('loss_rec_video','rec_video'), ('loss_rec_audio','rec_audio'), ('loss_triplet','triplet'), ('loss_orthogonality','orthogonality')]:
                    if k_src in loss_dict and torch.is_tensor(loss_dict[k_src]):
                        running_losses[k_csv] += float(loss_dict[k_src].detach().cpu().item())
                # Accumulate uncertainty weights — 6 floats: [binary,multi,rec_vid,rec_aud,triplet,ortho]
                uw = loss_dict.get('uncertainties', [1.0]*6)
                uw_keys = ['uw_binary','uw_multi','uw_rec_video','uw_rec_audio','uw_triplet','uw_orthogonality']
                for j, uw_key in enumerate(uw_keys):
                    running_uw[uw_key] = running_uw.get(uw_key, 0.0) + uw[j]
                epoch_loss_total += loss.item()
                num_batches += 1
                outputs = model_outputs['external']
                
                # Batch-wise: only accuracies
                accs = self.compute_accuracy(outputs, processed_batch)
                running_metrics['binary_acc'] = running_metrics.get('binary_acc', 0) + (accs['binary'].item() if hasattr(accs['binary'], 'item') else float(accs['binary']))
                running_metrics['multi_acc'] = running_metrics.get('multi_acc', 0) + (accs['multi'].item() if hasattr(accs['multi'], 'item') else float(accs['multi']))
                
                # Debug on first batch
                if batch_idx == 0:
                    print(f"\n[DEBUG TRAIN] Outputs shape: binary={outputs['classifications']['binary'].shape}, multi={outputs['classifications']['multi'].shape}")
                    print(f"[DEBUG TRAIN] Targets shape: binary={processed_batch['labels_binary'].shape}, multi={processed_batch['labels_multi'].shape}")
                    print(f"[DEBUG TRAIN] First batch acc: {{'binary_acc': {accs['binary']}, 'multi_acc': {accs['multi']}}}")
                
                # Epoch-wide accumulation (binary)
                pred_binary_proba = outputs['classifications']['binary'].sigmoid().squeeze(-1)  # [B,4]
                target_binary = processed_batch['labels_binary']  # [B,4]
                bin_proba_flat = pred_binary_proba.view(-1).detach().cpu().numpy()
                bin_true_flat = target_binary.view(-1).detach().cpu().numpy()
                bin_proba_all.append(bin_proba_flat)
                bin_true_all.append(bin_true_flat)
                bin_pred_flat = (bin_proba_flat > 0.5).astype(np.int64)
                bin_correct += int((bin_pred_flat == bin_true_flat).sum())
                bin_total += int(bin_true_flat.size)
                
                # Epoch-wide accumulation (multi)
                logits_multi = outputs['classifications']['multi']  # [B,4,4]
                multi_proba = logits_multi.softmax(dim=-1).view(-1, 4).detach().cpu().numpy()  # [N,4]
                target_multi = processed_batch['labels_multi'].view(-1).detach().cpu().numpy()  # [N]
                multi_proba_all.append(multi_proba)
                multi_true_all.append(target_multi)
                multi_pred_flat = multi_proba.argmax(axis=1)
                multi_correct += int((multi_pred_flat == target_multi).sum())
                multi_total += int(target_multi.size)
                
                # Perform optimizer step once enough gradients have accumulated
                if accum_step_count >= self.gradient_accumulation_steps:
                    # Clip gradients before update (include criterion params since log_vars can have gradients)
                    all_params = list(self.model.parameters()) + list(self.criterion.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.grad_clip)
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Zero gradients for next accumulation cycle
                    self.optimizer.zero_grad()
                    accum_step_count = 0
                
                # Log every N batches or at end
                should_log = (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(self.train_loader)
                
                if should_log:
                    # Calculate window size for rolling average
                    if (batch_idx + 1) % log_interval == 0:
                        window = log_interval
                    else:
                        window = (batch_idx + 1) % log_interval
                    
                    # Average over window
                    avg_loss = running_loss / window
                    # Include accuracies and averaged loss components in batch logs
                    avg_metrics = {
                        'binary_acc': running_metrics.get('binary_acc', 0) / window,
                        'multi_acc': running_metrics.get('multi_acc', 0) / window,
                    }
                    avg_metrics.update({
                        'loss_binary': running_losses['loss_binary'] / window,
                        'loss_multi': running_losses['loss_multi'] / window,
                        'loss_rec_video': running_losses['loss_rec_video'] / window,
                        'loss_rec_audio': running_losses['loss_rec_audio'] / window,
                        'loss_triplet': running_losses['loss_triplet'] / window,
                        'loss_orthogonality': running_losses['loss_orthogonality'] / window,
                    })
                    # Uncertainty weights averaged over window
                    avg_metrics.update({
                        k: running_uw.get(k, 0.0) / window
                        for k in ['uw_binary','uw_multi','uw_rec_video','uw_rec_audio','uw_triplet','uw_orthogonality']
                    })
                    
                    # Log to CSV
                    self._log_to_csv('train', epoch, batch_idx + 1, avg_loss, avg_metrics)
                    
                    # Print summary with loss components
                    print(
                        f"[Train] Batch {batch_idx+1}/{len(self.train_loader)} → "
                        f"Loss: {avg_loss:.4f} | "
                        f"L_bin: {avg_metrics.get('loss_binary', 0):.4f} | "
                        f"L_multi: {avg_metrics.get('loss_multi', 0):.4f} | "
                        f"L_rec_v: {avg_metrics.get('loss_rec_video', 0):.4f} | "
                        f"L_rec_a: {avg_metrics.get('loss_rec_audio', 0):.4f} | "
                        f"L_trip: {avg_metrics.get('loss_triplet', 0):.4f} | "
                        f"L_ortho: {avg_metrics.get('loss_orthogonality', 0):.4f} | "
                        f"Binary Acc: {avg_metrics.get('binary_acc', 0):.4f} | "
                        f"Multi Acc: {avg_metrics.get('multi_acc', 0):.4f}"
                    )
                    
                    # Reset accumulators (rolling window)
                    running_loss = 0.0
                    running_metrics = {}
                    running_losses = {k: 0.0 for k in running_losses}
                    running_uw = {k: 0.0 for k in running_uw}
                    
            except Exception as e:
                import traceback
                error_msg = f"Error in train batch {batch_idx}: {str(e)}\n{traceback.format_exc()}"
                self.logger.error(error_msg)
                print(f"❌ [Training Error] Batch {batch_idx}: {str(e)}")
                continue
        
        # Apply remaining gradients if the epoch ended mid-accumulation
        if accum_step_count > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Compute epoch-wide metrics
        if bin_proba_all:
            bin_proba_concat = np.concatenate(bin_proba_all)
            bin_true_concat = np.concatenate(bin_true_all)
        else:
            bin_proba_concat = np.array([])
            bin_true_concat = np.array([])
        
        if multi_proba_all:
            multi_proba_concat = np.concatenate(multi_proba_all, axis=0)
            multi_true_concat = np.concatenate(multi_true_all)
        else:
            multi_proba_concat = np.empty((0, 4))
            multi_true_concat = np.array([])
        
        epoch_metrics = {}
        # Accuracies
        epoch_metrics['binary_acc'] = (bin_correct / bin_total) if bin_total > 0 else 0.0
        epoch_metrics['multi_acc'] = (multi_correct / multi_total) if multi_total > 0 else 0.0
        
        print(f"\n[DEBUG TRAIN EPOCH-WIDE] Binary metrics:")
        print(f"  bin_true shape={bin_true_concat.shape}, unique={np.unique(bin_true_concat, return_counts=True)}")
        print(f"  bin_proba shape={bin_proba_concat.shape}, range=[{bin_proba_concat.min():.3f}, {bin_proba_concat.max():.3f}]")
        
        # Binary AUC/PRF
        try:
            if bin_true_concat.size > 0 and len(np.unique(bin_true_concat)) > 1:
                epoch_metrics['binary_auc'] = roc_auc_score(bin_true_concat, bin_proba_concat)
                print(f"  ✓ AUC: {epoch_metrics['binary_auc']:.4f}")
            else:
                epoch_metrics['binary_auc'] = 0.0
                print(f"  ✗ AUC skipped")
                
            bin_pred_concat = (bin_proba_concat > 0.5).astype(np.int64)
            print(f"  bin_pred unique={np.unique(bin_pred_concat, return_counts=True)}")
            
            # Youden's J Optimization
            fpr_out, tpr_out, thresholds_out = roc_curve(bin_true_concat, bin_proba_concat)
            j_scores = tpr_out - fpr_out
            best_idx = np.argmax(j_scores)
            optimal_threshold = thresholds_out[best_idx]
            opt_pred = (bin_proba_concat > optimal_threshold).astype(np.int64)
            epoch_metrics['optimal_threshold'] = optimal_threshold
            epoch_metrics['optimal_acc'] = (opt_pred == bin_true_concat).mean()
            print(f"  ✓ Opt Threshold: {optimal_threshold:.4f} (Acc: {epoch_metrics['optimal_acc']:.4f})")
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                bin_true_concat, bin_pred_concat, average='binary', zero_division=0.0
            )
            epoch_metrics.update({
                'binary_precision': precision,
                'binary_recall': recall,
                'binary_f1': f1
            })
            print(f"  ✓ P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            epoch_metrics.update({
                'binary_auc': 0.0,
                'binary_precision': 0.0,
                'binary_recall': 0.0,
                'binary_f1': 0.0,
                'optimal_threshold': 0.5,
                'optimal_acc': 0.0
            })
        
        print(f"\n[DEBUG TRAIN EPOCH-WIDE] Multi-class metrics:")
        print(f"  multi_true shape={multi_true_concat.shape}, unique={np.unique(multi_true_concat, return_counts=True)}")
        print(f"  multi_proba shape={multi_proba_concat.shape}")
        
        # Multi-class metrics
        try:
            if multi_true_concat.size > 0 and len(np.unique(multi_true_concat)) > 1:
                epoch_metrics['multi_auc'] = roc_auc_score(
                    multi_true_concat, multi_proba_concat, multi_class='ovr', average='macro'
                )
                print(f"  ✓ AUC: {epoch_metrics['multi_auc']:.4f}")
            else:
                epoch_metrics['multi_auc'] = 0.0
                print(f"  ✗ AUC skipped")
                
            multi_pred_concat = multi_proba_concat.argmax(axis=1) if multi_proba_concat.size else np.array([])
            print(f"  multi_pred unique={np.unique(multi_pred_concat, return_counts=True)}")
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                multi_true_concat, multi_pred_concat, average='macro', zero_division=0.0
            )
            epoch_metrics.update({
                'multi_precision': precision,
                'multi_recall': recall,
                'multi_f1': f1
            })
            print(f"  ✓ P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            epoch_metrics.update({
                'multi_auc': 0.0,
                'multi_precision': 0.0,
                'multi_recall': 0.0,
                'multi_f1': 0.0
            })
        
        # Epoch-average loss
        final_avg_loss = (epoch_loss_total / num_batches) if num_batches > 0 else 0.0
        
        # Write epoch-wide metrics row to CSV (train)
        self._log_to_csv('train', epoch, len(self.train_loader), final_avg_loss, epoch_metrics)
        
        # Return epoch-wide metrics
        return final_avg_loss, epoch_metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validation with rolling-average CSV logging (batch: acc only), epoch-wide metrics at end"""
        self.model.eval()
        running_loss = 0.0
        running_metrics = {}
        log_interval = 2  # Changed back to 5
        running_losses = {'loss_binary': 0.0, 'loss_multi': 0.0, 'loss_rec_video': 0.0, 'loss_rec_audio': 0.0, 'loss_triplet': 0.0, 'loss_orthogonality': 0.0}
        running_uw = {'uw_binary': 0.0, 'uw_multi': 0.0, 'uw_rec_video': 0.0, 'uw_rec_audio': 0.0, 'uw_triplet': 0.0, 'uw_orthogonality': 0.0}
        
        # Epoch-level accumulators
        epoch_loss_total = 0.0
        num_batches = 0
        
        # For epoch-wide sklearn metrics
        bin_proba_all = []   # list of np arrays shape [N]
        bin_true_all = []    # list of np arrays shape [N]
        multi_proba_all = [] # list of np arrays shape [N, 4]
        multi_true_all = []  # list of np arrays shape [N]
        
        # For epoch-wide accuracies
        bin_correct = 0
        bin_total = 0
        multi_correct = 0
        multi_total = 0
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
            try:
                # Process batch
                processed_batch = self._process_batch(batch)
                
                # Forward pass
                model_outputs = self.model(processed_batch)
                loss_dict = self.criterion(model_outputs, processed_batch)
                loss = loss_dict['total_loss']
                
                # Accumulate loss
                running_loss += loss.item()
                # Accumulate loss components if present
                for k_csv, k_src in [('loss_binary','binary'), ('loss_multi','multi'), ('loss_rec_video','rec_video'), ('loss_rec_audio','rec_audio'), ('loss_triplet','triplet'), ('loss_orthogonality','orthogonality')]:
                    if k_src in loss_dict and torch.is_tensor(loss_dict[k_src]):
                        running_losses[k_csv] += float(loss_dict[k_src].detach().cpu().item())
                # Accumulate uncertainty weights
                uw = loss_dict.get('uncertainties', [1.0]*6)
                uw_keys = ['uw_binary','uw_multi','uw_rec_video','uw_rec_audio','uw_triplet','uw_orthogonality']
                for j, uw_key in enumerate(uw_keys):
                    running_uw[uw_key] = running_uw.get(uw_key, 0.0) + uw[j]
                epoch_loss_total += loss.item()
                num_batches += 1
                
                outputs = model_outputs['external']
                # Batch-wise: only accuracies
                accs = self.compute_accuracy(outputs, processed_batch)
                
                # Debug: Check if metrics are being computed
                if batch_idx == 0:
                    print(f"\n[DEBUG VAL] Outputs shape: binary={outputs['classifications']['binary'].shape}, multi={outputs['classifications']['multi'].shape}")
                    print(f"[DEBUG VAL] Targets shape: binary={processed_batch['labels_binary'].shape}, multi={processed_batch['labels_multi'].shape}")
                    print(f"[DEBUG VAL] First batch acc: {{'binary_acc': {accs['binary']}, 'multi_acc': {accs['multi']}}}")
                
                running_metrics['binary_acc'] = running_metrics.get('binary_acc', 0) + (accs['binary'].item() if hasattr(accs['binary'], 'item') else float(accs['binary']))
                running_metrics['multi_acc'] = running_metrics.get('multi_acc', 0) + (accs['multi'].item() if hasattr(accs['multi'], 'item') else float(accs['multi']))
                
                # Epoch-wide accumulation (binary)
                pred_binary_proba = outputs['classifications']['binary'].sigmoid().squeeze(-1)  # [B,4]
                target_binary = processed_batch['labels_binary']  # [B,4]
                bin_proba_flat = pred_binary_proba.view(-1).detach().cpu().numpy()
                bin_true_flat = target_binary.view(-1).detach().cpu().numpy()
                bin_proba_all.append(bin_proba_flat)
                bin_true_all.append(bin_true_flat)
                bin_pred_flat = (bin_proba_flat > 0.5).astype(np.int64)
                bin_correct += int((bin_pred_flat == bin_true_flat).sum())
                bin_total += int(bin_true_flat.size)
                
                # Epoch-wide accumulation (multi)
                logits_multi = outputs['classifications']['multi']  # [B,4,4]
                multi_proba = logits_multi.softmax(dim=-1).view(-1, 4).detach().cpu().numpy()  # [N,4]
                target_multi = processed_batch['labels_multi'].view(-1).detach().cpu().numpy()  # [N]
                multi_proba_all.append(multi_proba)
                multi_true_all.append(target_multi)
                multi_pred_flat = multi_proba.argmax(axis=1)
                multi_correct += int((multi_pred_flat == target_multi).sum())
                multi_total += int(target_multi.size)
                
                # Log every N batches or at end
                should_log = (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(self.val_loader)
                
                if should_log:
                    # Calculate window size
                    if (batch_idx + 1) % log_interval == 0:
                        window = log_interval
                    else:
                        window = (batch_idx + 1) % log_interval
                    
                    # Average over window
                    avg_loss = running_loss / window
                    # Include accuracies and averaged loss components in batch logs
                    avg_metrics = {
                        'binary_acc': running_metrics.get('binary_acc', 0) / window,
                        'multi_acc': running_metrics.get('multi_acc', 0) / window,
                    }
                    avg_metrics.update({
                        'loss_binary': running_losses['loss_binary'] / window,
                        'loss_multi': running_losses['loss_multi'] / window,
                        'loss_rec_video': running_losses['loss_rec_video'] / window,
                        'loss_rec_audio': running_losses['loss_rec_audio'] / window,
                        'loss_triplet': running_losses['loss_triplet'] / window,
                        'loss_orthogonality': running_losses['loss_orthogonality'] / window,
                    })
                    # Uncertainty weights averaged over window
                    avg_metrics.update({
                        k: running_uw.get(k, 0.0) / window
                        for k in ['uw_binary','uw_multi','uw_rec_video','uw_rec_audio','uw_triplet','uw_orthogonality']
                    })
                    
                    # Log to CSV
                    self._log_to_csv('val', epoch, batch_idx + 1, avg_loss, avg_metrics)
                    
                    # Print summary with loss components
                    print(
                        f"[Val] Batch {batch_idx+1}/{len(self.val_loader)} → "
                        f"Loss: {avg_loss:.4f} | "
                        f"L_bin: {avg_metrics.get('loss_binary', 0):.4f} | "
                        f"L_multi: {avg_metrics.get('loss_multi', 0):.4f} | "
                        f"L_rec_v: {avg_metrics.get('loss_rec_video', 0):.4f} | "
                        f"L_rec_a: {avg_metrics.get('loss_rec_audio', 0):.4f} | "
                        f"L_trip: {avg_metrics.get('loss_triplet', 0):.4f} | "
                        f"L_ortho: {avg_metrics.get('loss_orthogonality', 0):.4f} | "
                        f"Binary Acc: {avg_metrics.get('binary_acc', 0):.4f} | "
                        f"Multi Acc: {avg_metrics.get('multi_acc', 0):.4f}"
                    )
                    
                    
                    # Reset accumulators (rolling window)
                    running_loss = 0.0
                    running_metrics = {}
                    running_losses = {k: 0.0 for k in running_losses}
                    running_uw = {k: 0.0 for k in running_uw}
                    
            except Exception as e:
                import traceback
                error_msg = f"Error in val batch {batch_idx}: {str(e)}\n{traceback.format_exc()}"
                self.logger.error(error_msg)
                print(f"❌ [Validation Error] Batch {batch_idx}: {str(e)}")
                continue
        
        # Compute epoch-wide metrics
        if bin_proba_all:
            bin_proba_concat = np.concatenate(bin_proba_all)
            bin_true_concat = np.concatenate(bin_true_all)
        else:
            bin_proba_concat = np.array([])
            bin_true_concat = np.array([])
        
        if multi_proba_all:
            multi_proba_concat = np.concatenate(multi_proba_all, axis=0)
            multi_true_concat = np.concatenate(multi_true_all)
        else:
            multi_proba_concat = np.empty((0, 4))
            multi_true_concat = np.array([])
        
        epoch_metrics = {}
        # Accuracies
        epoch_metrics['binary_acc'] = (bin_correct / bin_total) if bin_total > 0 else 0.0
        epoch_metrics['multi_acc'] = (multi_correct / multi_total) if multi_total > 0 else 0.0
        
        print(f"\n[DEBUG VAL EPOCH-WIDE] Binary metrics:")
        print(f"  bin_true shape={bin_true_concat.shape}, unique={np.unique(bin_true_concat, return_counts=True)}")
        print(f"  bin_proba shape={bin_proba_concat.shape}, range=[{bin_proba_concat.min():.3f}, {bin_proba_concat.max():.3f}]")
        
        # Binary AUC/PRF
        try:
            if bin_true_concat.size > 0 and len(np.unique(bin_true_concat)) > 1:
                epoch_metrics['binary_auc'] = roc_auc_score(bin_true_concat, bin_proba_concat)
                print(f"  ✓ AUC: {epoch_metrics['binary_auc']:.4f}")
            else:
                epoch_metrics['binary_auc'] = 0.0
                print(f"  ✗ AUC skipped")
                
            bin_pred_concat = (bin_proba_concat > 0.5).astype(np.int64)
            print(f"  bin_pred unique={np.unique(bin_pred_concat, return_counts=True)}")
            
            # Youden's J Optimization
            fpr_out, tpr_out, thresholds_out = roc_curve(bin_true_concat, bin_proba_concat)
            j_scores = tpr_out - fpr_out
            best_idx = np.argmax(j_scores)
            optimal_threshold = thresholds_out[best_idx]
            opt_pred = (bin_proba_concat > optimal_threshold).astype(np.int64)
            epoch_metrics['optimal_threshold'] = optimal_threshold
            epoch_metrics['optimal_acc'] = (opt_pred == bin_true_concat).mean()
            print(f"  ✓ Opt Threshold: {optimal_threshold:.4f} (Acc: {epoch_metrics['optimal_acc']:.4f})")
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                bin_true_concat, bin_pred_concat, average='binary', zero_division=0.0
            )
            epoch_metrics.update({
                'binary_precision': precision,
                'binary_recall': recall,
                'binary_f1': f1
            })
            print(f"  ✓ P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            epoch_metrics.update({
                'binary_auc': 0.0,
                'binary_precision': 0.0,
                'binary_recall': 0.0,
                'binary_f1': 0.0,
                'optimal_threshold': 0.5,
                'optimal_acc': 0.0
            })
        
        print(f"\n[DEBUG VAL EPOCH-WIDE] Multi-class metrics:")
        print(f"  multi_true shape={multi_true_concat.shape}, unique={np.unique(multi_true_concat, return_counts=True)}")
        print(f"  multi_proba shape={multi_proba_concat.shape}")
        
        # Multi-class metrics
        try:
            if multi_true_concat.size > 0 and len(np.unique(multi_true_concat)) > 1:
                epoch_metrics['multi_auc'] = roc_auc_score(
                    multi_true_concat, multi_proba_concat, multi_class='ovr', average='macro'
                )
                print(f"  ✓ AUC: {epoch_metrics['multi_auc']:.4f}")
            else:
                epoch_metrics['multi_auc'] = 0.0
                print(f"  ✗ AUC skipped")
                
            multi_pred_concat = multi_proba_concat.argmax(axis=1) if multi_proba_concat.size else np.array([])
            print(f"  multi_pred unique={np.unique(multi_pred_concat, return_counts=True)}")
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                multi_true_concat, multi_pred_concat, average='macro', zero_division=0.0
            )
            epoch_metrics.update({
                'multi_precision': precision,
                'multi_recall': recall,
                'multi_f1': f1
            })
            print(f"  ✓ P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            epoch_metrics.update({
                'multi_auc': 0.0,
                'multi_precision': 0.0,
                'multi_recall': 0.0,
                'multi_f1': 0.0
            })
        
        # Epoch-average loss
        final_avg_loss = (epoch_loss_total / num_batches) if num_batches > 0 else 0.0
        
        # Write an additional CSV row for epoch-wide metrics
        self._log_to_csv('val', epoch, len(self.val_loader), final_avg_loss, epoch_metrics)
        
        # Save checkpoint if best metrics improved
        # self.save_checkpoint(epoch, epoch_metrics)
        
        # Return epoch-wide metrics
        return final_avg_loss, epoch_metrics
    
    def compute_accuracy(self, outputs, targets):
        """Compute binary and multi-class accuracy"""
        accuracies = {}
        
        # FIXED: Correct shape handling (from Main_v2)
        pred_binary = outputs['classifications']['binary'].sigmoid() > 0.5  # [B, 4, 1]
        accuracies['binary'] = (pred_binary.squeeze(-1) == targets['labels_binary']).float().mean()  # [B, 4] == [B, 4]
        
        pred_multi = outputs['classifications']['multi'].argmax(dim=-1)  # [B, 4]
        accuracies['multi'] = (pred_multi == targets['labels_multi']).float().mean()  # [B, 4] == [B, 4]
        
        return accuracies

    def compute_comprehensive_metrics(self, outputs, targets):
        """Compute comprehensive metrics including AUC, F1, precision, recall"""
        metrics = {}
        
        # Binary metrics (Real/Fake) — follow Main_v2 implementations
        pred_binary_proba = outputs['classifications']['binary'].sigmoid().squeeze(-1)  # [B, 4]
        pred_binary = (pred_binary_proba > 0.5).float()  # [B, 4]
        target_binary = targets['labels_binary']  # [B, 4]
        
        # Accuracies using tensor comparisons (no sklearn)
        accuracies = self.compute_accuracy(outputs, targets)
        metrics['binary_acc'] = accuracies['binary'].item() if hasattr(accuracies['binary'], 'item') else float(accuracies['binary'])
        metrics['multi_acc'] = accuracies['multi'].item() if hasattr(accuracies['multi'], 'item') else float(accuracies['multi'])
        
        # Flatten for sklearn metrics (AUC, PRF)
        pred_binary_flat = pred_binary.view(-1).detach().cpu().numpy()
        pred_binary_proba_flat = pred_binary_proba.view(-1).detach().cpu().numpy()
        target_binary_flat = target_binary.view(-1).detach().cpu().numpy()
        
        try:
            metrics['binary_auc'] = roc_auc_score(target_binary_flat, pred_binary_proba_flat)
            
            # Youden's J Optimization
            fpr_out, tpr_out, thresholds_out = roc_curve(target_binary_flat, pred_binary_proba_flat)
            j_scores = tpr_out - fpr_out
            best_idx = np.argmax(j_scores)
            optimal_threshold = thresholds_out[best_idx]
            opt_pred = (pred_binary_proba_flat > optimal_threshold).astype(np.float32)
            metrics['optimal_threshold'] = optimal_threshold
            metrics['optimal_acc'] = (opt_pred == target_binary_flat).mean()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_binary_flat, pred_binary_flat, average='binary', zero_division=0.0
            )
            metrics.update({
                'binary_precision': precision,
                'binary_recall': recall,
                'binary_f1': f1
            })
        except ValueError:
            # Keep metrics absent or default to zeros if needed by downstream consumers
            # metrics.update({
            #     'binary_auc': 0.0,
            #     'binary_precision': 0.0,
            #     'binary_recall': 0.0,
            #     'binary_f1': 0.0
            # })
            print("ValueError in binary metrics")
        
        # Multi-class metrics (A/B/C/D)
        pred_multi = outputs['classifications']['multi'].argmax(dim=-1)  # [B, 4]
        target_multi = targets['labels_multi']  # [B, 4]
        
        # Flatten for sklearn metrics
        pred_multi_flat = pred_multi.view(-1).detach().cpu().numpy()
        target_multi_flat = target_multi.view(-1).detach().cpu().numpy()
        
        try:
            metrics['multi_auc'] = roc_auc_score(
                target_multi_flat,
                outputs['classifications']['multi'].softmax(dim=-1).view(-1, 4).detach().cpu().numpy(),
                multi_class='ovr',
                average='macro'
            )
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_multi_flat, pred_multi_flat, average='macro', zero_division=0.0
            )
            metrics.update({
                'multi_precision': precision,
                'multi_recall': recall,
                'multi_f1': f1
            })
        except ValueError:
            print("ValueError in multi-class metrics")
        
        return metrics
    
    def _log_to_csv(self, split, epoch, batch, loss, metrics):
        """Log metrics to CSV with rolling averages"""
        if split == 'test':
            csv_path = self.test_csv
            total_batches = len(self.test_loader) if self.test_loader else 0
        elif split == 'train':
            csv_path = self.train_csv
            total_batches = len(self.train_loader) if self.train_loader else 0
        else:
            csv_path = self.val_csv
            total_batches = len(self.val_loader) if self.val_loader else 0
        
        if csv_path is None:
            return
        
        # Calculate epoch progress fraction
        epoch_progress = batch / total_batches if total_batches > 0 else 0.0
        
        row = [
            epoch, batch, total_batches, f"{epoch_progress:.4f}", split, f"{loss:.6f}",
            f"{metrics.get('loss_binary', 0):.6f}",
            f"{metrics.get('loss_multi', 0):.6f}",
            f"{metrics.get('loss_rec_video', 0):.6f}",
            f"{metrics.get('loss_rec_audio', 0):.6f}",
            f"{metrics.get('loss_triplet', 0):.6f}",
            f"{metrics.get('loss_orthogonality', 0):.6f}",
            # Uncertainty weights (exp(log_var) for each loss component)
            f"{metrics.get('uw_binary', 1.0):.6f}",
            f"{metrics.get('uw_multi', 1.0):.6f}",
            f"{metrics.get('uw_rec_video', 1.0):.6f}",
            f"{metrics.get('uw_rec_audio', 1.0):.6f}",
            f"{metrics.get('uw_triplet', 1.0):.6f}",
            f"{metrics.get('uw_orthogonality', 1.0):.6f}",
            f"{metrics.get('optimal_threshold', 0.5):.6f}",
            f"{metrics.get('optimal_acc', 0):.6f}",
            f"{metrics.get('binary_acc', 0):.6f}",
            f"{metrics.get('multi_acc', 0):.6f}",
            f"{metrics.get('binary_auc', 0):.6f}",
            f"{metrics.get('multi_auc', 0):.6f}",
            f"{metrics.get('binary_precision', 0):.6f}",
            f"{metrics.get('binary_recall', 0):.6f}",
            f"{metrics.get('binary_f1', 0):.6f}",
            f"{metrics.get('multi_precision', 0):.6f}",
            f"{metrics.get('multi_recall', 0):.6f}",
            f"{metrics.get('multi_f1', 0):.6f}"
        ]
        
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
    
    def _process_batch(self, batch, single_video_mode=False):
        """Process batch data - handles both 4-video sets and single videos"""
        if single_video_mode:
            # Single video mode: expand to [B, 1, ...] format
            frames = batch['frames'].float().to(self.device).unsqueeze(1)  # [B,T,C,H,W]->[B,1,T,C,H,W]
            mels = batch['mels'].float().to(self.device).unsqueeze(1)      # [B,1,F,T]->[B,1,1,F,T]
            labels_multi = batch['labels_multi'].long().to(self.device).unsqueeze(1)
            labels_binary = batch['labels_binary'].float().to(self.device).unsqueeze(1)
            return {
                'frames': frames, 'mels': mels,
                'labels_multi': labels_multi, 'labels_binary': labels_binary,
                'set_id': batch.get('set_id', ['unknown']),
                'video_path': batch.get('video_path', [''])
            }
        else:
            return {
                'frames': batch['frames'].float().to(self.device),
                'mels': batch['mels'].float().to(self.device),
                'labels_multi': batch['labels_multi'].long().to(self.device),
                'labels_binary': batch['labels_binary'].float().to(self.device),
                'set_id': batch['set_id']
            }
    
    def save_checkpoint(self, epoch, metrics):
        """Save checkpoints with best binary accuracy and AUC.
        
        Before every comparison the persistent best_metrics.txt is re-read so
        that the global best always reflects the true maximum across all runs,
        not just the values seen since the last startup.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_latest.pth')
        
        # Re-read the persistent file so we always compare against the true
        # global best (handles resume, multiple runs, or interrupted sessions).
        self._load_global_best()
        
        # Get current epoch's metrics
        val_binary_acc = metrics.get('val_binary_acc', metrics.get('binary_acc', 0))
        val_auc = metrics.get('val_binary_auc', metrics.get('binary_auc', 0))
        
        updated = False
        
        # Save best by binary accuracy (primary metric)
        if val_binary_acc > self.best_binary_acc:
            self.best_binary_acc = val_binary_acc
            updated = True
            filename = f'best_acc_e{epoch}_acc{val_binary_acc:.4f}_auc{val_auc:.4f}.pth'
            save_path = self.checkpoint_dir / filename
            torch.save(checkpoint, save_path)
            
            # Clean up old best_acc checkpoints
            for old_file in self.checkpoint_dir.glob('best_acc_e*.pth'):
                if old_file.name != filename:
                    old_file.unlink()
            
            print(f"✅ Saved best accuracy checkpoint: {filename}")
        
        # Save best by AUC (secondary metric)
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            updated = True
            filename = f'best_auc_e{epoch}_auc{val_auc:.4f}_acc{val_binary_acc:.4f}.pth'
            save_path = self.checkpoint_dir / filename
            torch.save(checkpoint, save_path)
            
            # Clean up old best_auc checkpoints
            for old_file in self.checkpoint_dir.glob('best_auc_e*.pth'):
                if old_file.name != filename:
                    old_file.unlink()
            
            print(f"✅ Saved best AUC checkpoint: {filename}")
        
        # Always persist the (possibly updated) global bests to disk
        if updated:
            self._save_global_best()
            print(f"💾 Updated best_metrics.txt → Acc={self.best_binary_acc:.4f}, AUC={self.best_auc:.4f}")
        
        # Save milestone checkpoints every 10 epochs
        if epoch % 10 == 0:
            milestone_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, milestone_path)
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting training up to epoch {self.config.epochs}...")
        print(f"📊 Training batches: {len(self.train_loader)}, Validation batches: {len(self.val_loader)}")

        # Iterate using true epoch numbering, supporting resume
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            print(f"\n{'='*60}")
            print(f"📅 Epoch [{epoch}/{self.config.epochs}]")
            print(f"{'='*60}")
            
            # Training
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate(epoch)

            # Step LR scheduler after each epoch
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"  📉 LR after epoch {epoch}: {current_lr:.2e}")
            
            # Save checkpoints
            # self.save_checkpoint(epoch, val_metrics)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"  [Train] Loss: {train_loss:.4f} | Bin Acc: {train_metrics.get('binary_acc', 0):.4f} | Multi Acc: {train_metrics.get('multi_acc', 0):.4f} | "
                  f"Bin AUC: {train_metrics.get('binary_auc', 0):.4f} | Bin P/R/F1: {train_metrics.get('binary_precision', 0):.4f}/{train_metrics.get('binary_recall', 0):.4f}/{train_metrics.get('binary_f1', 0):.4f} | "
                  f"Multi AUC: {train_metrics.get('multi_auc', 0):.4f} | Multi P/R/F1: {train_metrics.get('multi_precision', 0):.4f}/{train_metrics.get('multi_recall', 0):.4f}/{train_metrics.get('multi_f1', 0):.4f}")
            print(f"  [Val]   Loss: {val_loss:.4f} | Bin Acc: {val_metrics.get('binary_acc', 0):.4f} | Multi Acc: {val_metrics.get('multi_acc', 0):.4f} | "
                  f"Bin AUC: {val_metrics.get('binary_auc', 0):.4f} | Bin P/R/F1: {val_metrics.get('binary_precision', 0):.4f}/{val_metrics.get('binary_recall', 0):.4f}/{val_metrics.get('binary_f1', 0):.4f} | "
                  f"Multi AUC: {val_metrics.get('multi_auc', 0):.4f} | Multi P/R/F1: {val_metrics.get('multi_precision', 0):.4f}/{val_metrics.get('multi_recall', 0):.4f}/{val_metrics.get('multi_f1', 0):.4f}")


    @torch.no_grad()
    def test(self, num_runs=10):
        """Test mode with single videos per batch - runs multiple times and only prints epoch-wise metrics"""
        print(f"\n☣️ Starting testing ({num_runs} runs)...")
        print(f"📊 Test batches: {len(self.test_loader)} (batch_size={self.config.batch_size})")

        all_run_metrics = []
        final_bin_proba_concat = final_bin_true_concat = None
        final_multi_proba_concat = final_multi_true_concat = None

        for run in range(1, num_runs + 1):
            print(f"\n{'='*60}")
            print(f"📅 Test Run [{run}/{num_runs}]")
            print(f"{'='*60}")

            self.model.eval()
            epoch_loss_total, num_batches = 0.0, 0
            bin_proba_all, bin_true_all = [], []
            multi_proba_all, multi_true_all = [], []
            bin_correct, bin_total, multi_correct, multi_total = 0, 0, 0, 0

            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc=f'Test Run {run}', leave=False)):
                try:
                    processed_batch = self._process_batch(batch, single_video_mode=True)
                    
                    # Original pass
                    model_outputs = self.model(processed_batch)
                    outputs = model_outputs['external']
                    
                    # Store original prediction (sigmoid for binary, softmax for multi)
                    bin_proba_tensor = outputs['classifications']['binary'].sigmoid() # [B, 1]
                    multi_proba_tensor = outputs['classifications']['multi'].softmax(dim=-1) # [B, 1, 4]

                    # Test-Time Augmentation (TTA)
                    if getattr(self.config, 'use_tta', False):
                        v_frames = processed_batch['frames'] # [B, 1, T, C, H, W]
                        B, N, T, C, H, W = v_frames.shape
                        
                        tta_variants = []
                        # 1. Horizontal Flip
                        tta_variants.append(TF.hflip(v_frames))
                        
                        # 2. Zooms (0.7, 0.8, 0.9, 1.1, 1.2, 1.3)
                        frames_4d = v_frames.view(-1, C, H, W)
                        for z in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
                            if z > 1.0:
                                zoomed = TF.resize(frames_4d, [int(H*z), int(W*z)], antialias=True)
                                variant = TF.center_crop(zoomed, [H, W])
                            else:
                                resized = TF.resize(frames_4d, [int(H*z), int(W*z)], antialias=True)
                                pad_h, pad_w = (H - int(H*z)) // 2, (W - int(W*z)) // 2
                                variant = TF.pad(resized, [pad_w, pad_h, W - int(W*z) - pad_w, H - int(H*z) - pad_h])
                            tta_variants.append(variant.view(B, N, T, C, H, W))
                        
                        # 3. Brightness Adjustments (0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4)
                        for b in [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]:
                            tta_variants.append(TF.adjust_brightness(v_frames, b))

                        for variant_frames in tta_variants:
                            var_batch = processed_batch.copy()
                            var_batch['frames'] = variant_frames
                            var_out = self.model(var_batch)['external']
                            bin_proba_tensor += var_out['classifications']['binary'].sigmoid()
                            multi_proba_tensor += var_out['classifications']['multi'].softmax(dim=-1)
                        
                        # Average all predictions (Original + 4 Variants)
                        denominator = 1 + len(tta_variants)
                        bin_proba_tensor /= denominator
                        multi_proba_tensor /= denominator

                    if self.criterion is not None:
                        loss_dict = self.criterion(model_outputs, processed_batch)
                        epoch_loss_total += loss_dict['total_loss'].item()
                    num_batches += 1
                    
                    # Collect averaged probabilities for metric calculation
                    pred_binary_proba = bin_proba_tensor.view(-1)
                    target_binary = processed_batch['labels_binary']
                    bin_proba_flat = pred_binary_proba.detach().cpu().numpy()
                    bin_true_flat = target_binary.view(-1).detach().cpu().numpy()
                    bin_proba_all.append(bin_proba_flat); bin_true_all.append(bin_true_flat)
                    bin_pred_flat = (bin_proba_flat > 0.5).astype(np.int64)
                    bin_correct += int((bin_pred_flat == bin_true_flat).sum())
                    bin_total += int(bin_true_flat.size)

                    multi_proba = multi_proba_tensor.view(-1, 4).detach().cpu().numpy()
                    target_multi = processed_batch['labels_multi'].view(-1).detach().cpu().numpy()
                    multi_proba_all.append(multi_proba); multi_true_all.append(target_multi)
                    multi_pred_flat = multi_proba.argmax(axis=1)
                    multi_correct += int((multi_pred_flat == target_multi).sum())
                    multi_total += int(target_multi.size)

                except Exception as e:
                    import traceback
                    if self.logger:
                        self.logger.error(f"Test batch {batch_idx}: {traceback.format_exc()}")
                    print(f"❌ [Test Error] Run {run}, Batch {batch_idx}: {e}")
                    continue

            bin_proba_concat = np.concatenate(bin_proba_all) if bin_proba_all else np.array([])
            bin_true_concat = np.concatenate(bin_true_all) if bin_true_all else np.array([])
            multi_proba_concat = np.concatenate(multi_proba_all, axis=0) if multi_proba_all else np.empty((0, 4))
            multi_true_concat = np.concatenate(multi_true_all) if multi_true_all else np.array([])

            epoch_metrics = {}
            epoch_metrics['binary_acc'] = (bin_correct / bin_total) if bin_total > 0 else 0.0
            epoch_metrics['multi_acc'] = (multi_correct / multi_total) if multi_total > 0 else 0.0

            try:
                if bin_true_concat.size > 0 and len(np.unique(bin_true_concat)) > 1:
                    epoch_metrics['binary_auc'] = roc_auc_score(bin_true_concat, bin_proba_concat)
                    epoch_metrics['binary_pr_auc'] = average_precision_score(bin_true_concat, bin_proba_concat)
                else:
                    epoch_metrics['binary_auc'] = epoch_metrics['binary_pr_auc'] = 0.0

                bin_pred_concat = (bin_proba_concat > 0.5).astype(np.int64)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    bin_true_concat, bin_pred_concat, average='binary', zero_division=0.0
                )
                
                # Calculate optimal threshold using Youden's J statistic (TPR - FPR)
                fpr_out, tpr_out, thresholds_out = roc_curve(bin_true_concat, bin_proba_concat)
                j_scores = tpr_out - fpr_out
                best_idx = np.argmax(j_scores)
                optimal_threshold = thresholds_out[best_idx]
                opt_pred = (bin_proba_concat > optimal_threshold).astype(np.int64)
                opt_acc = (opt_pred == bin_true_concat).mean()
                epoch_metrics['optimal_threshold'] = optimal_threshold
                epoch_metrics['optimal_acc'] = opt_acc
                
                # Calculate default threshold (0.5) metrics
                cm_m = compute_confusion_matrix_metrics(bin_true_concat, bin_pred_concat)
                
                # Calculate optimal threshold metrics
                opt_prec, opt_rec, opt_f1, _ = precision_recall_fscore_support(
                    bin_true_concat, opt_pred, average='binary', zero_division=0.0
                )
                cm_opt = compute_confusion_matrix_metrics(bin_true_concat, opt_pred)
                
                epoch_metrics.update({
                    'binary_precision': prec, 'binary_recall': rec, 'binary_f1': f1,
                    'binary_tpr': cm_m['tpr'], 'binary_fpr': cm_m['fpr'],
                    'binary_tnr': cm_m['tnr'], 'binary_fnr': cm_m['fnr'],
                    'binary_balanced_acc': cm_m['balanced_acc'],
                    # Optimal threshold metrics
                    'opt_precision': opt_prec, 'opt_recall': opt_rec, 'opt_f1': opt_f1,
                    'opt_balanced_acc': cm_opt['balanced_acc'],
                    'opt_tpr': cm_opt['tpr'], 'opt_fpr': cm_opt['fpr']
                })
            except Exception:
                epoch_metrics.update({
                    'binary_auc': 0.0, 'binary_pr_auc': 0.0, 'binary_precision': 0.0,
                    'binary_recall': 0.0, 'binary_f1': 0.0, 'binary_tpr': 0.0,
                    'binary_fpr': 0.0, 'binary_tnr': 0.0, 'binary_fnr': 0.0, 'binary_balanced_acc': 0.0,
                    'opt_precision': 0.0, 'opt_recall': 0.0, 'opt_f1': 0.0, 'opt_balanced_acc': 0.0,
                    'opt_tpr': 0.0, 'opt_fpr': 0.0
                })

            try:
                if multi_true_concat.size > 0 and len(np.unique(multi_true_concat)) > 1:
                    epoch_metrics['multi_auc'] = roc_auc_score(
                        multi_true_concat, multi_proba_concat, multi_class='ovr', average='macro'
                    )
                else:
                    epoch_metrics['multi_auc'] = 0.0
                multi_pred_concat = multi_proba_concat.argmax(axis=1) if multi_proba_concat.size else np.array([])
                prec, rec, f1, _ = precision_recall_fscore_support(
                    multi_true_concat, multi_pred_concat, average='macro', zero_division=0.0
                )
                epoch_metrics.update({'multi_precision': prec, 'multi_recall': rec, 'multi_f1': f1})
            except Exception:
                epoch_metrics.update({'multi_auc': 0.0, 'multi_precision': 0.0, 'multi_recall': 0.0, 'multi_f1': 0.0})

            final_avg_loss = (epoch_loss_total / num_batches) if num_batches > 0 else 0.0
            epoch_metrics['loss'] = final_avg_loss
            all_run_metrics.append(epoch_metrics)

            if run == num_runs:
                final_bin_proba_concat, final_bin_true_concat = bin_proba_concat, bin_true_concat
                final_multi_proba_concat, final_multi_true_concat = multi_proba_concat, multi_true_concat

            self._log_to_csv('test', run, len(self.test_loader), final_avg_loss, epoch_metrics)

            # --- Diagnostic Checks ---
            print(f"\n🔬 DIAGNOSTIC CHECKS (Run {run})")
            
            # 1. Mean Predicted Probability Per Class
            if bin_true_concat.size > 0:
                real_mask = (bin_true_concat == 0)
                fake_mask = (bin_true_concat == 1)
                
                mean_prob_real = bin_proba_concat[real_mask].mean() if real_mask.sum() > 0 else 0.0
                mean_prob_fake = bin_proba_concat[fake_mask].mean() if fake_mask.sum() > 0 else 0.0
                
                print("  [Mean Probabilities]")
                print(f"    Mean P(Fake | Real Video) = {mean_prob_real:.4f}  (Should be closer to 0)")
                print(f"    Mean P(Fake | Fake Video) = {mean_prob_fake:.4f}  (Should be closer to 1)")
            
            # 2. Threshold Sensitivity Analysis & Youden's J Optimization
            if bin_true_concat.size > 0:
                print("  [Threshold Sensitivity Analysis (Accuracy)]")
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                for t in thresholds:
                    t_pred = (bin_proba_concat > t).astype(np.int64)
                    t_acc = (t_pred == bin_true_concat).mean()
                    print(f"    Threshold = {t:.1f}  -->  Acc = {t_acc:.4f}")

                # Use precomputed optimal threshold
                optimal_threshold = epoch_metrics.get('optimal_threshold', 0.5)
                opt_acc = epoch_metrics.get('optimal_acc', 0.0)
                print(f"  [Youden's J Optimization]")
                print(f"    Best Threshold = {optimal_threshold:.4f} -->  Acc = {opt_acc:.4f}")

            print(f"\n📊 Test Run {run} Summary:")
            print(f"  Loss: {final_avg_loss:.4f}")
            print(f"  Binary Acc: {epoch_metrics.get('binary_acc',0):.4f} | Balanced Acc: {epoch_metrics.get('binary_balanced_acc',0):.4f} | AUC: {epoch_metrics.get('binary_auc',0):.4f} | PR AUC: {epoch_metrics.get('binary_pr_auc',0):.4f}")
            print(f"    P/R/F1 (0.5): {epoch_metrics.get('binary_precision',0):.4f}/{epoch_metrics.get('binary_recall',0):.4f}/{epoch_metrics.get('binary_f1',0):.4f}")
            print(f"    [Post-Optimization at Threshold {optimal_threshold:.4f}]:")
            print(f"       Acc: {epoch_metrics.get('optimal_acc',0):.4f} | Balanced: {epoch_metrics.get('opt_balanced_acc',0):.4f}")
            print(f"       P/R/F1: {epoch_metrics.get('opt_precision',0):.4f}/{epoch_metrics.get('opt_recall',0):.4f}/{epoch_metrics.get('opt_f1',0):.4f}")
            print(f"       TPR/FPR: {epoch_metrics.get('opt_tpr',0):.4f}/{epoch_metrics.get('opt_fpr',0):.4f}")
            print(f"    TPR/FPR/TNR/FNR (0.5): {epoch_metrics.get('binary_tpr',0):.4f}/{epoch_metrics.get('binary_fpr',0):.4f}/{epoch_metrics.get('binary_tnr',0):.4f}/{epoch_metrics.get('binary_fnr',0):.4f}")
            print(f"  Multi Acc: {epoch_metrics.get('multi_acc',0):.4f} | AUC: {epoch_metrics.get('multi_auc',0):.4f} | P/R/F1: {epoch_metrics.get('multi_precision',0):.4f}/{epoch_metrics.get('multi_recall',0):.4f}/{epoch_metrics.get('multi_f1',0):.4f}")

        if all_run_metrics:
            avg_m = {k: np.mean([m.get(k, 0) for m in all_run_metrics]) for k in all_run_metrics[0]}
            avg_std = {k: np.std([m.get(k, 0) for m in all_run_metrics]) for k in all_run_metrics[0]}
            print(f"\n{'='*60}")
            print(f"📊 Final Test Results (Avg over {num_runs} runs):")
            print(f"{'='*60}")
            print(f"  Binary Acc: {avg_m.get('binary_acc',0):.4f}±{avg_std.get('binary_acc',0):.3f} | Balanced: {avg_m.get('binary_balanced_acc',0):.4f}±{avg_std.get('binary_balanced_acc',0):.3f} | AUC: {avg_m.get('binary_auc',0):.4f}±{avg_std.get('binary_auc',0):.3f}")
            print(f"  Multi  Acc: {avg_m.get('multi_acc',0):.4f}±{avg_std.get('multi_acc',0):.3f} | AUC: {avg_m.get('multi_auc',0):.4f}±{avg_std.get('multi_auc',0):.3f}")
            print(f"{'='*60}")

            if final_bin_proba_concat is not None:
                plot_save_prefix = str(self.log_dir / 'test_results') if self.log_dir else 'test_results'
                plot_classification_metrics(
                    bin_true_concat=final_bin_true_concat, bin_proba_concat=final_bin_proba_concat,
                    multi_true_concat=final_multi_true_concat, multi_proba_concat=final_multi_proba_concat,
                    class_names=['RealVideo-RealAudio', 'RealVideo-FakeAudio', 'FakeVideo-RealAudio', 'FakeVideo-FakeAudio'],
                    save_prefix=plot_save_prefix
                )
                print(f"\n📈 Test plots saved to: {plot_save_prefix}_*.png")

        return all_run_metrics


def main():
    """Main entry point"""
    class Config:
        # Data settings
        # data_dir = "/home/i237606/FakeAV_sample/FakeAVCeleb_balanced_V7.1_3025"
        data_dir = "/home/i212809/externaldrive/awais/Kodf_dataset/test_e2_1000samples/"
        # data_dir = "/home/i212809/externaldrive/awais/DFDC/test_e2_1000samples"
        checkpoint_dir = './checkpoints_test_140_v7.1_64_e1_ablation_recon_cross_dataset_kodf'
        log_dir = './logs_test_140_v7.1_64_e1_ablation_recon_cross_dataset_kodf'
        # log_dir = './logs_test_dfdc_cross_dataset_tta'

        # Model settings
        visual_encoder = 'b2'
        audio_encoder = '18'

        # Training settings
        batch_size = 20
        num_cpus = os.cpu_count() or 1
        num_workers = num_cpus // 2
        # learning_rate = 5.6115164153345e-05 #got 93.21 AUC on val , ACc 80.5 whilw separet bets is 85 is #1e-4
        learning_rate = 8.168455894760161e-05 #1e-4 # got best_auc_e191_auc0.9719_acc0.9450 in MMD-Net_main_v2_4
        # learning_rate = 6.168455894760161e-05 #1e-4 # got best_auc_e191_auc0.9719_acc0.9450 in MMD-Net_main_v2_4
        weight_decay = 0.1
        epochs = 150

        # Safety settings
        grad_clip = 1.0
        max_loss_threshold = 100.0
        gradient_accumulation_steps = 5

        # Reproducibility
        seed = 42

        # Resume settings (training only)
        resume_checkpoint = None# '/home/i237606/FakeAV_sample/MMD-Net_main_v2_7/checkpoints_test_140_v7.1_64_e1_ablation_recon/checkpoint_latest.pth'
        resume_start_epoch = None

        # ── TEST MODE SETTINGS ────────────────────────────────────────────────
        # Set test_mode = True to run inference on the test split instead of training.
        test_mode = True
        use_tta = False     # Activate Test-Time Augmentation (Flip, Zoom, Brightness)
        # Path to a trained checkpoint to load in test mode.
        # Example: './checkpoints_test_140_v7.1_64/best_auc_eXX_aucY_accZ.pth'
        # trained_model_path = None #'/home/i237606/FakeAV_sample/MMD-Net_main_v2_6/checkpoints_test_140_v7.1_64/best_auc_e34_auc0.8386_acc0.7500.pth'
        trained_model_path = '/home/i237606/FakeAV_sample/MMD-Net_main_v2_7/checkpoints_test_140_v7.1_64_e1_ablation_recon_full/best_auc_e125_auc0.9948_acc0.9192.pth'
        # ── VISUALIZATION SETTINGS ────────────────────────────────────────────
        # Reconstruction visualization (original vs reconstructed frames + mels)
        visualize_reconstructions = False   # Set True in test mode to generate comparison images
        viz_num_sets = 3                   # Number of video sets to visualize
        viz_frame_idx = 12                 # Frame index to extract (0-24 for 25-frame clips)

        # t-SNE feature space visualization
        visualize_tsne = False              # Set True to generate t-SNE embedding plots
        tsne_max_samples = None             # Max samples per split (None = all)

    config = Config()
    set_seed(config.seed)
    trainer = Trainer(config)

    if getattr(config, 'test_mode', False):
        print("\n🧪 Starting testing...")
        trainer.test(num_runs=1)

        if getattr(config, 'visualize_tsne', False):
            print("\n" + "="*70)
            print("🎨 GENERATING t-SNE FEATURE VISUALIZATIONS (TEST SPLIT)")
            print("="*70)
            output_dirs = run_multi_split_visualizations(
                trainer=trainer, 
                split_combinations=[['test']],
                max_samples_per_split=config.tsne_max_samples
            )
            print(f"\n✅ t-SNE visualizations complete! ({len(output_dirs)} sets)")

        if getattr(config, 'visualize_reconstructions', False):
            print("\n" + "="*70)
            print("🎨 GENERATING RECONSTRUCTION VISUALIZATIONS")
            print("="*70)
            viz_output_dir = run_reconstruction_visualization(
                trainer=trainer,
                num_sets=config.viz_num_sets,
                frame_idx=config.viz_frame_idx
            )
            print(f"\n✅ Reconstructions saved to: {viz_output_dir.absolute()}")
    else:
        trainer.train()


if __name__ == '__main__':
    main()

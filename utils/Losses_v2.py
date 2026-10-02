import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.debug_utils import print_shape_info
from sklearn.metrics import roc_auc_score, f1_score

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, av_cf, labels):
        """
        Args:
            av_cf: [B, T, 1] or [B, T, D] predictions
            labels: [B] binary labels
        """
        # Expand labels to match prediction shape
        labels = labels.view(-1, 1, 1).expand(-1, av_cf.size(1), av_cf.size(2))
        return self.bce(av_cf, labels).mean()


class DynamicBCEWithLogitsLoss(nn.Module):
    """BCE Loss with dynamic positive weighting and label smoothing.
    
    Dynamically scales the weight of the positive class (Real) based on the 
    batch composition to correct for class imbalance.
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits, any shape
            targets: Binary labels (0=fake, 1=real), same shape as logits
        """
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).float()
        
        # Calculate dynamic positive weight
        num_pos = targets_flat.sum()
        num_neg = targets_flat.size(0) - num_pos
        
        # Avoid division by zero
        if num_pos > 0:
            pos_weight = num_neg / num_pos
        else:
            pos_weight = torch.tensor(1.0, device=targets.device)
            
        # Apply label smoothing (reduces overconfidence)
        # 1.0 -> 0.95, 0.0 -> 0.05
        smoothed_targets = targets_flat * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # Standard BCE per element with positive weighting
        bce = F.binary_cross_entropy_with_logits(
            logits_flat, smoothed_targets, 
            pos_weight=pos_weight, reduction='mean'
        )

        return bce

class MultiClassCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, av_cf, labels):
        """
        Args:
            av_cf: [B, T, C] predictions
            labels: [B] multi-class labels
        """
        # Ensure labels are long type and properly shaped
        labels = labels.long().view(-1, 1).expand(-1, av_cf.size(1))  # Convert to long
        return self.ce(av_cf.transpose(1, 2), labels).mean()

class ReconstructionLoss(nn.Module):
    def __init__(self, mode='video'):
        super().__init__()
        self.mode = mode
        self.mse = nn.MSELoss()
        self.pool = nn.AdaptiveAvgPool2d((250, 250))
    
    def forward(self, recon, orig, swap_indices=None):
        """
        Args:
            recon: Reconstructed video/audio [B, 8, T, C, H, W] (original + swapped)
            orig: Original video/audio [B, 4, T, C, H, W] for video or [B, 4, T, F] for audio
            swap_indices: Swap indices from reconstruction module (e.g., [3,0,1,2])
        """
        #print(f"[DEBUG] ReconstructionLoss {self.mode}:")
        #print(f"[DEBUG] Input recon shape: {recon.shape}")
        #print(f"[DEBUG] Input orig shape: {orig.shape}")
        #print(f"[DEBUG] Swap indices: {swap_indices}")
        
        # Store original shape for swap handling
        orig_shape = orig.shape
        B, V, T = orig.shape[:3]
        
        # Reshape orig to match recon batch dimension
        if len(orig.shape) == 6:  # Video: [B, V, T, C, H, W]
            orig = orig.reshape(-1, *orig.shape[3:])  # [B*V*T, C, H, W]
        elif len(orig.shape) == 5:  # Audio: [B, V, C, F, T]
            orig = orig.reshape(-1, *orig.shape[3:])  # [B*V, F, T]
        else:
            orig = orig.reshape(-1, *orig.shape[3:])  # Fallback
        
        #print(f"[DEBUG] Reshaped orig: {orig.shape}")
        
        # Split reconstruction into original and swapped parts
        if len(recon.shape) == 6:  # Video: [B, 8, T, C, H, W]
            recon_orig = recon[:, :4]  # First 4 videos (original)
            recon_swapped = recon[:, 4:]  # Last 4 videos (swapped)
        else:  # Audio: [B, 8, C, F, T] 
            recon_orig = recon[:, :4]  # First 4 videos (original)
            recon_swapped = recon[:, 4:]  # Last 4 videos (swapped)
        
        #print(f"[DEBUG] Recon original: {recon_orig.shape}")
        #print(f"[DEBUG] Recon swapped: {recon_swapped.shape}")
        
        # Check if swapped videos exist (validation mode may not produce them)
        has_swapped = recon_swapped.shape[1] > 0
        #print(f"[DEBUG] Has swapped videos: {has_swapped}")
        
        # Reshape reconstructions to match orig format
        recon_orig = recon_orig.reshape(-1, *recon_orig.shape[3:])
        if has_swapped:
            recon_swapped = recon_swapped.reshape(-1, *recon_swapped.shape[3:])
        
        #print(f"[DEBUG] Recon orig reshaped: {recon_orig.shape}")
        if has_swapped:
            #print(f"[DEBUG] Recon swapped reshaped: {recon_swapped.shape}")
            pass
        
        # Compute losses for both original and swapped reconstructions
        loss_orig = 0
        loss_swapped = 0
        
        # Process original reconstruction loss
        if self.mode == 'video':
            recon_orig_processed = self.pool(recon_orig)  # [B*V*T, 3, 250, 250]
            orig_processed = self.pool(orig)  # [B*V*T, 3, 250, 250]
            loss_orig = self.mse(recon_orig_processed, orig_processed)
            #print(f"[DEBUG] Original video loss: {loss_orig.item():.6f}")
            
            # Only compute swapped loss if swapped videos exist
            if has_swapped:
                # FIXED: Swapped reconstruction should be compared to reordered ground truth
                recon_swapped_processed = self.pool(recon_swapped)  # [B*V*T, 3, 250, 250]
                
                if swap_indices is not None:
                    # Reshape orig_processed back to [B, V, T, C, H, W] for reordering
                    # orig_processed is [B*V*T, 3, 250, 250], so we need to reshape it properly
                    B, V, T = orig_shape[:3]  # Use original shape before pooling
                    orig_reshaped = orig_processed.view(B, V, T, 3, 250, 250)
                    
                    # Apply swap indices to reorder ground truth
                    # swap_indices: [3,0,1,2] means: new[0]=old[3], new[1]=old[0], new[2]=old[1], new[3]=old[2]
                    orig_swapped = orig_reshaped[:, swap_indices]  # [B, V, T, C, H, W]
                    orig_swapped_processed = orig_swapped.view(-1, 3, 250, 250)  # [B*V*T, 3, 250, 250]
                    
                    #print(f"[DEBUG] Using swap indices: {swap_indices}")
                    #print(f"[DEBUG] Orig swapped processed shape: {orig_swapped_processed.shape}")
                else:
                    # No swap indices provided, use original ground truth
                    orig_swapped_processed = orig_processed
                    #print(f"[DEBUG] No swap indices provided, using original ground truth")
                
                loss_swapped = self.mse(recon_swapped_processed, orig_swapped_processed)
                #print(f"[DEBUG] Swapped video loss: {loss_swapped.item():.6f}")
            else:
                #print(f"[DEBUG] No swapped videos in reconstruction - skipping swapped loss")
                pass
            
        else:  # audio
            # Handle audio tensors correctly
            if len(orig.shape) == 2:  # [B*V*T, F]
                orig = orig.unsqueeze(1)  # Add channel dim [B*V*T, 1, F]
            
            # Process original reconstruction
            if len(recon_orig.shape) == 4:  # [B*V*T, 1, F, T]
                recon_orig = recon_orig.squeeze(1)  # Remove extra channel dim
            
            # Get target sizes
            freq_dim = orig.size(-1)
            time_dim = orig.size(-2) if len(orig.shape) > 2 else 1
            
            # Interpolate original reconstruction
            recon_orig_interp = recon_orig.unsqueeze(1)  # Add channel dimension
            recon_orig_interp = torch.nn.functional.interpolate(
                recon_orig_interp, 
                size=(time_dim, freq_dim),
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)  # Remove channel dimension after interpolation
            
            loss_orig = self.mse(recon_orig_interp, orig.squeeze(1) if len(orig.shape) > 2 else orig)
            #print(f"[DEBUG] Original audio loss: {loss_orig.item():.6f}")
            
            # Only compute swapped loss if swapped audio exists
            if has_swapped:
                # FIXED: Swapped reconstruction should be compared to reordered ground truth
                if len(recon_swapped.shape) == 4:  # [B*V*T, 1, F, T]
                    recon_swapped = recon_swapped.squeeze(1)  # Remove extra channel dim
                
                recon_swapped_interp = recon_swapped.unsqueeze(1)  # Add channel dimension
                recon_swapped_interp = torch.nn.functional.interpolate(
                    recon_swapped_interp, 
                    size=(time_dim, freq_dim),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)  # Remove channel dimension after interpolation
            
                if swap_indices is not None:
                    # FIXED: Correct for audio shape [B, 4, 1, F, T]
                    # Work with original shape before flattening
                    if len(orig_shape) == 5:  # Audio: [B, V, C, F, T]
                        B, V, C, F, T = orig_shape
                        # Reshape orig back to original shape for swapping (keep C dim)
                        orig_reshaped = orig.view(B, V, C, F, T)  # [B, V, C, F, T] = [1, 4, 1, 128, 32]
                        
                        # Apply swap indices to reorder ground truth along V dimension
                        orig_swapped = orig_reshaped[:, swap_indices]  # [B, V, C, F, T] - swap along V
                        orig_swapped_processed = orig_swapped.view(-1, C, F, T)  # [B*V, C, F, T] = [4, 1, 128, 32]
                    else:
                        # Fallback for other shapes
                        #print(f"[DEBUG] Original audio shape, swapping failed: {orig_shape}")
                        orig_swapped_processed = orig
                    
                    #print(f"[DEBUG] Using swap indices for audio: {swap_indices}")
                    #print(f"[DEBUG] Original audio shape: {orig_shape}")
                    #print(f"[DEBUG] Orig swapped processed shape: {orig_swapped_processed.shape}")
                    
                    # Squeeze C dimension to match recon_swapped_interp shape [4, 128, 32]
                    if orig_swapped_processed.dim() == 4:  # [B*V, C, F, T]
                        orig_swapped_processed = orig_swapped_processed.squeeze(1)  # [B*V, F, T] = [4, 128, 32]
                else:
                    # No swap indices provided, use original ground truth
                    # orig is already squeezed from line 125-130
                    orig_swapped_processed = orig
                    #print(f"[DEBUG] No swap indices provided for audio, using original ground truth")
                
                #print(f"[DEBUG] Final shapes before MSE:")
                #print(f"[DEBUG] recon_swapped_interp: {recon_swapped_interp.shape}")
                #print(f"[DEBUG] orig_swapped_processed: {orig_swapped_processed.shape}")
                
                loss_swapped = self.mse(recon_swapped_interp, orig_swapped_processed)
                #print(f"[DEBUG] Swapped audio loss: {loss_swapped.item():.6f}")
            else:
                #print(f"[DEBUG] No swapped audio in reconstruction - skipping swapped loss")
                pass
        
        # Combine both losses (both original and swapped contribute to learning)
        # FIXED: Now correctly uses swap_indices to reorder ground truth for swapped reconstructions
        # - Original reconstructions: compared to original ground truth
        # - Swapped reconstructions: compared to reordered ground truth according to swap_indices
        # 
        # Video case: [B, V, T, C, H, W] -> swap along V dimension
        # Audio case: [B, V, C, F, T] -> swap along V dimension (correctly handles [B, 4, 1, 128, 32])
        
        # If no swapped videos (validation mode), only use original loss
        if has_swapped and loss_swapped != 0:
            total_loss = (loss_orig + loss_swapped) / 2
            #print(f"[DEBUG] Total {self.mode} reconstruction loss: {total_loss.item():.6f} (orig + swapped)")
        else:
            total_loss = loss_orig
            #print(f"[DEBUG] Total {self.mode} reconstruction loss: {total_loss.item():.6f} (orig only - no swapped)")
            pass
        return total_loss

'''
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, feat1, feat2, label):
        """
        Args:
            feat1: [B, V, T, D] - First feature set
            feat2: [B, V, T, D] - Second feature set (rolled)
            label: [B, V] - Binary labels
        """
        # Compute pairwise distances between corresponding time steps
        B, V, T, D = feat1.shape
        
        # Reshape to [B*V*T, D]
        feat1_flat = feat1.reshape(-1, D)
        feat2_flat = feat2.reshape(-1, D)
        
        # Compute pairwise distances
        dist = F.pairwise_distance(feat1_flat, feat2_flat)  # [B*V*T]
        dist = dist.reshape(B, V, T)  # Restore batch and video dimensions
        
        # Expand label to match distance tensor
        label = label.unsqueeze(-1).expand(-1, -1, T)  # [B, V, T]
        
        # Compute loss with broadcasting
        loss = (1-label) * torch.pow(dist, 2) + \
               label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        
        return loss.mean()
'''
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    
    def _compute_loss(self, anchor, positive, negative):
        """Compute triplet loss for given triplets.
        Args:
            anchor: [B, T, D] anchor features
            positive: [B, T, D] positive features
            negative: [B, T, D] negative features
        Returns:
            torch.Tensor: triplet loss value
        """
        # Compute distances
        # First reshape to [B*T, D] to handle temporal dimension
        B, T, D = anchor.shape
        anchor_flat = anchor.reshape(-1, D)
        positive_flat = positive.reshape(-1, D)
        negative_flat = negative.reshape(-1, D)
        
        # Compute distances using F.pairwise_distance
        pos_dist = F.pairwise_distance(anchor_flat, positive_flat)
        neg_dist = F.pairwise_distance(anchor_flat, negative_flat)
        
        # Reshape back to [B, T]
        pos_dist = pos_dist.reshape(B, T)
        neg_dist = neg_dist.reshape(B, T)
        
        # Compute triplet loss with margin
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        # Average over both batch and temporal dimensions
        return losses.mean()
    
    def _compute_hard_triplet_loss(self, anchors, positives, negatives):
        if anchors.size(0) == 0 or positives.size(0) == 0 or negatives.size(0) == 0:
            return torch.zeros((), device=anchors.device, requires_grad=True)
            
        # Compute distances across all pairs by averaging over Temporal dim
        a_mean = anchors.mean(dim=1)  # [N_a, D]
        p_mean = positives.mean(dim=1)  # [N_p, D]
        n_mean = negatives.mean(dim=1)  # [N_n, D]
        
        dist_ap = torch.cdist(a_mean, p_mean)  # [N_a, N_p]
        dist_an = torch.cdist(a_mean, n_mean)  # [N_a, N_n]
        
        # Hard mining: For each anchor, find the furthest positive and closest negative
        hard_p_indices = dist_ap.argmax(dim=1)
        hard_n_indices = dist_an.argmin(dim=1)
        
        hard_positives = positives[hard_p_indices]
        hard_negatives = negatives[hard_n_indices]
        
        return self._compute_loss(anchors, hard_positives, hard_negatives)

    def forward(self, features, labels):
        """
        Enhanced Triplet Loss for Deepfake Detection (with Hard Mining)
        
        Strategy:
        - B & C & D (Fakes) should cluster together
        - A (Real) should be pushed away from B,C,D
        """
        losses = []
        B, V = labels.shape[:2]
        
        # Find indices for each class (per batch and video)
        class_indices = {}
        for class_id, class_name in enumerate(['A', 'B', 'C', 'D']):
            indices = (labels == class_id).nonzero(as_tuple=False)  # [num_matches, 2]
            class_indices[class_name] = indices
        
        # Process static features only (as requested)
        if 'static' in features and 'avfs' in features['static']:
            avfs = features['static']['avfs']  # [B, V, T, D]
            
            # Extract features for each class
            a_features = self._extract_class_features(avfs, class_indices['A'])  # [num_A, T, D]
            b_features = self._extract_class_features(avfs, class_indices['B'])  # [num_B, T, D]
            c_features = self._extract_class_features(avfs, class_indices['C'])  # [num_C, T, D]
            d_features = self._extract_class_features(avfs, class_indices['D'])  # [num_D, T, D]
            
            # Triplet 1: B(anchor) - C(positive) - A(negative)
            if b_features.size(0) > 0 and c_features.size(0) > 0 and a_features.size(0) > 0:
                loss_bca = self._compute_hard_triplet_loss(b_features, c_features, a_features)
                if loss_bca > 0: losses.append(loss_bca)
            
            # Triplet 2: D(anchor) - B(positive) - A(negative)
            if d_features.size(0) > 0 and b_features.size(0) > 0 and a_features.size(0) > 0:
                loss_dba = self._compute_hard_triplet_loss(d_features, b_features, a_features)
                if loss_dba > 0: losses.append(loss_dba)
            
            # Triplet 3: C(anchor) - D(positive) - A(negative)
            if c_features.size(0) > 0 and d_features.size(0) > 0 and a_features.size(0) > 0:
                loss_cda = self._compute_hard_triplet_loss(c_features, d_features, a_features)
                if loss_cda > 0: losses.append(loss_cda)
        
        if losses:
            final_loss = torch.stack(losses).mean()
        else:
            final_loss = torch.zeros((), device=labels.device, requires_grad=True)

        # final_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=labels.device)
        #print(f"[DEBUG] Total triplet loss: {final_loss.item():.6f} (from {len(losses)} triplets)")
        
        return final_loss
    
    def _extract_class_features(self, avfs, class_indices):
        """Extract features for a specific class
        Args:
            avfs: [B, V, T, D] feature tensor
            class_indices: [num_matches, 2] indices (batch_idx, video_idx)
        Returns:
            [num_matches, T, D] features for the class
        """
        if len(class_indices) == 0:
            return torch.empty(0, avfs.size(2), avfs.size(3), device=avfs.device)
        
        # Extract features using advanced indexing
        batch_indices = class_indices[:, 0]  # [num_matches]
        video_indices = class_indices[:, 1]  # [num_matches]
        
        class_features = avfs[batch_indices, video_indices]  # [num_matches, T, D]
        return class_features

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Add log var for orthogonality loss (6 total losses)
        self.log_vars = nn.Parameter(torch.zeros(6))
        
        # Initialize individual losses
        self.bce_loss = DynamicBCEWithLogitsLoss(smoothing=0.1)  # Dynamic Pos Weight + Label Smoothing
        self.mce_loss = MultiClassCrossEntropyLoss()
        self.rec_video_loss = ReconstructionLoss('video')
        self.rec_audio_loss = ReconstructionLoss('audio')
        # self.contrastive_loss = ContrastiveLoss()  # Commented out - redundant with orthogonality loss
        self.triplet_loss = TripletLoss()

        self.last_metrics = {}  # Store metrics for logging

    def _compute_classification_metrics(self, pred_binary, pred_multi, target_binary, target_multi):
        """Compute detailed classification metrics"""
        # Binary metrics - reshape properly for batch and video dimensions
        B, V = target_binary.shape[:2]
        
        # Reshape predictions to match targets
        pred_binary = pred_binary.view(B, V, -1).mean(dim=-1)  # Average over temporal dimension
        pred_binary_flat = pred_binary.sigmoid().reshape(-1).detach().cpu().numpy()
        pred_binary_cls = (pred_binary.sigmoid() > 0.5).reshape(-1).detach().cpu().numpy()
        target_binary_flat = target_binary.view(B, V).reshape(-1).detach().cpu().numpy()
        
        # Multi-class metrics - reshape similarly
        pred_multi = pred_multi.view(B, V, -1, pred_multi.size(-1)).mean(dim=2)  # Average over temporal
        pred_multi_cls = pred_multi.argmax(dim=-1).reshape(-1).detach().cpu().numpy()
        target_multi_flat = target_multi.view(B, V).reshape(-1).detach().cpu().numpy()
        
        # Ensure all arrays have same length
        assert len(pred_binary_flat) == len(target_binary_flat)
        assert len(pred_multi_cls) == len(target_multi_flat)
        
        return {
            'binary_auc': roc_auc_score(target_binary_flat, pred_binary_flat),
            'binary_f1': f1_score(target_binary_flat, pred_binary_cls),
            'multi_f1': f1_score(target_multi_flat, pred_multi_cls, average='macro')
        }

    def _compute_per_video_loss(self, pred, target, loss_fn):
        """
        Compute loss for each video in the set
        Args:
            pred: [B, V, T, D] predictions
            target: [B, V] labels
        """
        B, V = pred.shape[:2]
        total_loss = 0
        
        for v in range(V):
            # Calculate loss for each video
            video_pred = pred[:, v]  # [B, T, D]
            video_target = target[:, v].long() if isinstance(loss_fn, MultiClassCrossEntropyLoss) else target[:, v].float()
            video_loss = loss_fn(video_pred, video_target)
            total_loss += video_loss
        
        return total_loss / V

    def forward(self, outputs, targets):
        # print(f"\n[DEBUG] ===== LOSS COMPUTATION START =====")
        # print(f"[DEBUG] Output keys: {outputs.keys()}")
        # print(f"[DEBUG] LOSS FUNCTION CALLED - THIS SHOULD APPEAR!")
        
        # Access internal outputs from MMD_NET for loss calculation
        internal_outputs = outputs.get('internal', {})
        
        # Per-video losses (enhanced classifier with separated features)
        external_outputs = outputs.get('external', {})
        cls_outputs = external_outputs.get('classifications', {})
        
        # print(f"[DEBUG] External output keys: {external_outputs.keys()}")
        # print(f"[DEBUG] Internal output keys: {internal_outputs.keys()}")
        # print(f"[DEBUG] Target keys: {targets.keys()}")
        
        losses = {}
        
        # print(f"[DEBUG] Loss function - cls_outputs keys: {cls_outputs.keys()}")
        
        # Per-video binary loss (Real/Fake) - trained on common features
        if 'binary' in cls_outputs:
            # targets['labels_binary']: [B, 4] - binary label for each video
            binary_pred = cls_outputs['binary']  # [B, 4, 1]
            binary_targets = targets['labels_binary']  # [B, 4]
            
            # print(f"[DEBUG] Binary loss - pred: {binary_pred.shape}, targets: {binary_targets.shape}")
            
            # Dynamic BCE+Logits handles class imbalance (1 Real, 3 Fake per set) dynamically
            binary_loss = self.bce_loss(
                binary_pred.view(-1),            # [B*4]
                binary_targets.view(-1).float()  # [B*4]
            )
            losses['binary'] = binary_loss
            # print(f"[DEBUG] Binary BCE loss computed: {binary_loss.item():.6f}")
        
        # Per-video multi-class loss (A,B,C,D) - trained on specific features  
        if 'multi' in cls_outputs:
            # targets['labels_multi']: [B, 4] - class label for each video
            multi_pred = cls_outputs['multi']    # [B, 4, 4]
            multi_targets = targets['labels_multi']  # [B, 4]
            
            # print(f"[DEBUG] Multi-class loss - pred: {multi_pred.shape}, targets: {multi_targets.shape}")
            
            # Compute loss for each video separately
            multi_loss = F.cross_entropy(
                multi_pred.view(-1, 4),    # [B*4, 4]
                multi_targets.view(-1),     # [B*4]
                label_smoothing=0.1
            )
            losses['multi'] = multi_loss
            # print(f"[DEBUG] Multi-class loss computed: {multi_loss.item():.6f}")
        
        # Benefits of per-video losses:
        # 1. Each video contributes individual gradient signal
        # 2. 4x more training samples per batch (B*4 instead of B)
        # 3. Better feature learning - no information loss from pooling
        # 4. Model learns video-level patterns, not set-level averages
        

    
        # Reconstruction losses with swap indices
        if 'reconstructed' in internal_outputs:
            rec = internal_outputs['reconstructed']
            swap_info = rec.get('swap_info', {})
            
            # print(f"[DEBUG] Reconstruction keys: {rec.keys()}")
            # if swap_info:
            #     print(f"[DEBUG] Swap info keys: {swap_info.keys()}")
            
            # Video reconstruction
            if 'visual' in rec:
                target_frames = targets['frames']  # [B, 4, 25, 3, 250, 250]
                visual_swap_indices = swap_info.get('visual_swap', None)
                
                # print(f"[DEBUG] Video reconstruction:")
                # print(f"[DEBUG] Target frames: {target_frames.shape}")
                # print(f"[DEBUG] Reconstructed visual: {rec['visual'].shape}")
                # print(f"[DEBUG] Visual swap indices: {visual_swap_indices}")
                
                losses['rec_video'] = self.rec_video_loss(
                    rec['visual'],
                    target_frames,
                    visual_swap_indices
                )
            
            # Audio reconstruction
            if 'audio' in rec:
                target_mels = targets['mels']  # Keep original shape [B, 4, 1, 128, 32]
                audio_swap_indices = swap_info.get('audio_swap', None)
                
                # print(f"[DEBUG] Audio reconstruction:")
                # print(f"[DEBUG] Target mels: {target_mels.shape}")
                # print(f"[DEBUG] Reconstructed audio: {rec['audio'].shape}")
                # print(f"[DEBUG] Audio swap indices: {audio_swap_indices}")
                
                losses['rec_audio'] = self.rec_audio_loss(
                    rec['audio'],
                    target_mels,
                    audio_swap_indices
                )
        
        # REMOVED: Contrastive Loss - Redundant with Orthogonality Loss
        # Orthogonality loss already enforces vfs ⊥ vir separation
        # No need for additional contrastive loss on encoder features
        # print(f"[DEBUG] Contrastive loss skipped - redundant with orthogonality loss")
    
        # REMOVED: Temporal Contrastive Loss - Also redundant
        # print(f"[DEBUG] Temporal contrastive loss skipped - redundant with orthogonality loss")
    
        # Enhanced Triplet Loss on Static Combined Forgery Features
        av_features = internal_outputs.get('av_features', {})
        
        if 'static' in av_features and 'combined' in av_features['static']:
            if 'modality_common' in av_features['static']['combined']:
                # Use modality_common features (these contain forgery information after fusion)
                static_forgery_features = av_features['static']['combined']['modality_common']  # [B, V, 512]
                
                # print(f"[DEBUG] Triplet Loss on Static Combined Forgery Features:")
                # print(f"[DEBUG] Static forgery features: {static_forgery_features.shape}")
                # print(f"[DEBUG] Multi-class labels: {targets['labels_multi'].shape}")
                
                # Expand temporal dimension for triplet loss (it expects [B, V, T, D])
                # Since these are pooled features, we treat them as single time step
                static_forgery_expanded = static_forgery_features.unsqueeze(2)  # [B, V, 1, 512]
                
                triplet_features = {
                    'static': {'avfs': static_forgery_expanded}
                }
                
                triplet_loss = self.triplet_loss(triplet_features, targets['labels_multi'])
                losses['triplet'] = triplet_loss
                # print(f"[DEBUG] Triplet loss (forgery clustering): {triplet_loss.item():.6f}")
            else:
                pass # print(f"[DEBUG] No modality_common features available for triplet loss")
        else:
            # print(f"[DEBUG] No static combined features available for triplet loss")
            pass
        
        # Orthogonality Loss from AudioEncoderTCN and Visual Encoder
        if 'orthogonality_loss' in internal_outputs:
            orthogonality_loss = internal_outputs['orthogonality_loss']
            losses['orthogonality'] = orthogonality_loss
            # print(f"[DEBUG] Orthogonality loss: {orthogonality_loss.item():.6f}")
        else:
            # print(f"[DEBUG] No orthogonality loss found in outputs")
            pass
    
        # Weight losses with uncertainties
        # print(f"[DEBUG] ===== UNCERTAINTY WEIGHTING =====")
        # print(f"[DEBUG] Number of losses: {len(losses)}")
        # print(f"[DEBUG] Loss names: {list(losses.keys())}")
        
        # Define deterministic order for loss-to-log_var mapping
        # This ensures consistent assignment across epochs and checkpoints
        loss_order = ['binary', 'multi', 'rec_video', 'rec_audio', 'triplet', 'orthogonality']
        
        total_loss = 0.0
        
        # Process losses in deterministic order
        for i, name in enumerate(loss_order):
            if name in losses and torch.is_tensor(losses[name]):
                loss = losses[name]
                # Clamp log_var to prevent numerical instability (exp(-log_var) with large log_var)
                log_var_clamped = self.log_vars[i].clamp(-10, 10)
                precision = torch.exp(-log_var_clamped)
                weighted_loss = precision * loss + log_var_clamped
                total_loss += weighted_loss
                
                # print(f"[DEBUG] {name}: loss={loss.item():.6f}, precision={precision.item():.6f}, weighted={weighted_loss.item():.6f}")
            else:
                if name in losses:
                    # print(f"[DEBUG] {name}: non-tensor loss, skipping")
                    pass
                # else:
                #     print(f"[DEBUG] {name}: not in losses dict")

        
        # print(f"[DEBUG] ===== INDIVIDUAL LOSS SUMMARY =====")
        # for name, loss in losses.items():
        #     if isinstance(loss, torch.Tensor):
        #         print(f"[DEBUG] {name}: value = {loss.item():.6f}")
        #     else:
        #         print(f"[DEBUG] {name}: non-tensor value = {loss}")

        # Convert uncertainties to list for easier logging
        uncertainties = torch.exp(self.log_vars).detach().cpu().tolist()
        


        # Add sklearn imports at top of file

        
        # print(f"[DEBUG] ===== LOSS COMPUTATION END =====")
        # print(f"[DEBUG] Final total loss: {total_loss.item():.6f}")
        # print(f"[DEBUG] Uncertainties: {uncertainties}")
        
        return {
            'total_loss': total_loss,
            **{k: v for k, v in losses.items() if torch.is_tensor(v)},  # Only include tensor losses
            'uncertainties': uncertainties  # Return as list instead of tensor
        }

def get_loss_fn():
    """Helper function to create loss function"""
    return UncertaintyWeightedLoss()
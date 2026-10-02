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
    
    def forward(self, features, labels):
        """
        Enhanced Triplet Loss for Deepfake Detection
        
        Strategy:
        - Anchor: Fake videos (B, C, D)
        - Positive: Other fake videos (pull fake cluster together)
        - Negative: Real videos (A) (push real away from fake cluster)
        
        Args:
            features: Dictionary containing static features
            labels: Multi-class labels [B, V] with 0=A(Real), 1=B(Fake), 2=C(Fake), 3=D(Fake)
        """
        #print(f"[DEBUG] TripletLoss forward:")
        #print(f"[DEBUG] Features keys: {features.keys()}")
        #print(f"[DEBUG] Labels shape: {labels.shape}")
        #print(f"[DEBUG] Labels unique values: {torch.unique(labels)}")
        
        losses = []
        B, V = labels.shape[:2]
        
        # Find indices for each class (per batch and video)
        class_indices = {}
        for class_id, class_name in enumerate(['A', 'B', 'C', 'D']):
            indices = (labels == class_id).nonzero(as_tuple=False)  # [num_matches, 2]
            class_indices[class_name] = indices
            #print(f"[DEBUG] Class {class_name} (id={class_id}) indices: {indices.shape[0]} samples")
        
        # Process static features only (as requested)
        if 'static' in features and 'avfs' in features['static']:
            avfs = features['static']['avfs']  # [B, V, T, D]
            #print(f"[DEBUG] Processing static AVFS: {avfs.shape}")
            
            # Triplet 1: B(anchor) - C(positive) - A(negative)
            # Goal: B and C (both fake) should be closer than B and A
            if (len(class_indices['B']) > 0 and 
                len(class_indices['C']) > 0 and 
                len(class_indices['A']) > 0):
                
                #print(f"[DEBUG] Computing B-C-A triplets")
                
                # Extract features for each class
                b_features = self._extract_class_features(avfs, class_indices['B'])  # [num_B, T, D]
                c_features = self._extract_class_features(avfs, class_indices['C'])  # [num_C, T, D]
                a_features = self._extract_class_features(avfs, class_indices['A'])  # [num_A, T, D]
                
                if b_features.size(0) > 0 and c_features.size(0) > 0 and a_features.size(0) > 0:
                    # Use first available sample from each class
                    anchor = b_features[0:1]    # [1, T, D] - Fake B
                    positive = c_features[0:1]  # [1, T, D] - Fake C (similar to B)
                    negative = a_features[0:1]  # [1, T, D] - Real A (dissimilar to B)
                    
                    loss_bca = self._compute_loss(anchor, positive, negative)
                    losses.append(loss_bca)
                    #print(f"[DEBUG] B-C-A triplet loss: {loss_bca.item():.6f}")
            
            # Triplet 2: D(anchor) - B(positive) - A(negative)
            if (len(class_indices['D']) > 0 and 
                len(class_indices['B']) > 0 and 
                len(class_indices['A']) > 0):
                
                #print(f"[DEBUG] Computing D-B-A triplets")
                
                d_features = self._extract_class_features(avfs, class_indices['D'])
                b_features = self._extract_class_features(avfs, class_indices['B'])
                a_features = self._extract_class_features(avfs, class_indices['A'])
                
                if d_features.size(0) > 0 and b_features.size(0) > 0 and a_features.size(0) > 0:
                    anchor = d_features[0:1]    # [1, T, D] - Fake D
                    positive = b_features[0:1]  # [1, T, D] - Fake B (similar to D)
                    negative = a_features[0:1]  # [1, T, D] - Real A (dissimilar to D)
                    
                    loss_dba = self._compute_loss(anchor, positive, negative)
                    losses.append(loss_dba)
                    #print(f"[DEBUG] D-B-A triplet loss: {loss_dba.item():.6f}")
            
            # Triplet 3: C(anchor) - D(positive) - A(negative)
            if (len(class_indices['C']) > 0 and 
                len(class_indices['D']) > 0 and 
                len(class_indices['A']) > 0):
                
                #print(f"[DEBUG] Computing C-D-A triplets")
                
                c_features = self._extract_class_features(avfs, class_indices['C'])
                d_features = self._extract_class_features(avfs, class_indices['D'])
                a_features = self._extract_class_features(avfs, class_indices['A'])
                
                if c_features.size(0) > 0 and d_features.size(0) > 0 and a_features.size(0) > 0:
                    anchor = c_features[0:1]    # [1, T, D] - Fake C
                    positive = d_features[0:1]  # [1, T, D] - Fake D (similar to C)
                    negative = a_features[0:1]  # [1, T, D] - Real A (dissimilar to C)
                    
                    loss_cda = self._compute_loss(anchor, positive, negative)
                    losses.append(loss_cda)
                    #print(f"[DEBUG] C-D-A triplet loss: {loss_cda.item():.6f}")
        
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

class WeightedLoss(nn.Module):
    def __init__(self, loss_weights):
        """
        Initialize loss function with fixed weights.
        
        Args:
            loss_weights: Dict with keys: 'reconstruction', 'orthogonality', 'binary', 'multi', 'triplet'
                          Values should sum to 1.0
        """
        super().__init__()
        
        # Validate weights sum to 1.0
        weight_sum = sum(loss_weights.values())
        assert abs(weight_sum - 1.0) < 1e-6, f"Loss weights must sum to 1.0, got {weight_sum}"
        
        self.loss_weights = loss_weights
        
        # Initialize individual losses
        self.bce_loss = BinaryCrossEntropyLoss()
        self.mce_loss = MultiClassCrossEntropyLoss()
        self.rec_video_loss = ReconstructionLoss('video')
        self.rec_audio_loss = ReconstructionLoss('audio')
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
        #print(f"\n[DEBUG] ===== LOSS COMPUTATION START =====")
        #print(f"[DEBUG] Output keys: {outputs.keys()}")
        #print(f"[DEBUG] LOSS FUNCTION CALLED - THIS SHOULD APPEAR!")
        
        # Access internal outputs from MMD_NET for loss calculation
        internal_outputs = outputs.get('internal', {})
        
        # Per-video losses (enhanced classifier with separated features)
        external_outputs = outputs.get('external', {})
        cls_outputs = external_outputs.get('classifications', {})
        
        #print(f"[DEBUG] External output keys: {external_outputs.keys()}")
        #print(f"[DEBUG] Internal output keys: {internal_outputs.keys()}")
        #print(f"[DEBUG] Target keys: {targets.keys()}")
        
        losses = {}
        
        #print(f"[DEBUG] Loss function - cls_outputs keys: {cls_outputs.keys()}")
        
        # Per-video binary loss (Real/Fake) - trained on common features
        if 'binary' in cls_outputs:
            # targets['labels_binary']: [B, 4] - binary label for each video
            binary_pred = cls_outputs['binary']  # [B, 4, 1]
            binary_targets = targets['labels_binary']  # [B, 4]
            
            #print(f"[DEBUG] Binary loss - pred: {binary_pred.shape}, targets: {binary_targets.shape}")
            
            # Compute loss for each video separately with class weights to handle imbalance
            # Each set has 1 Real (A) and 3 Fake (B,C,D) videos
            # pos_weight = num_negative / num_positive = 3/1 = 3
            pos_weight = torch.tensor(3.0, device=binary_pred.device)
            
            binary_loss = F.binary_cross_entropy_with_logits(
                binary_pred.view(-1),           # [B*4]
                binary_targets.view(-1).float(), # [B*4]
                pos_weight=pos_weight  # Give 3x weight to positive class (Real videos)
            )
            losses['binary'] = binary_loss
            #print(f"[DEBUG] Binary loss computed (with pos_weight=3.0): {binary_loss.item():.6f}")
        
        # Per-video multi-class loss (A,B,C,D) - trained on specific features  
        if 'multi' in cls_outputs:
            # targets['labels_multi']: [B, 4] - class label for each video
            multi_pred = cls_outputs['multi']    # [B, 4, 4]
            multi_targets = targets['labels_multi']  # [B, 4]
            
            #print(f"[DEBUG] Multi-class loss - pred: {multi_pred.shape}, targets: {multi_targets.shape}")
            
            # Compute loss for each video separately
            multi_loss = F.cross_entropy(
                multi_pred.view(-1, 4),    # [B*4, 4]
                multi_targets.view(-1)     # [B*4]
            )
            losses['multi'] = multi_loss
            #print(f"[DEBUG] Multi-class loss computed: {multi_loss.item():.6f}")
        
        # Benefits of per-video losses:
        # 1. Each video contributes individual gradient signal
        # 2. 4x more training samples per batch (B*4 instead of B)
        # 3. Better feature learning - no information loss from pooling
        # 4. Model learns video-level patterns, not set-level averages
        

    
        # Reconstruction losses with swap indices
        if 'reconstructed' in internal_outputs:
            rec = internal_outputs['reconstructed']
            swap_info = rec.get('swap_info', {})
            
            #print(f"[DEBUG] Reconstruction keys: {rec.keys()}")
            if swap_info:
                #print(f"[DEBUG] Swap info keys: {swap_info.keys()}")
                pass
            
            # Video reconstruction
            if 'visual' in rec:
                target_frames = targets['frames']  # [B, 4, 25, 3, 250, 250]
                visual_swap_indices = swap_info.get('visual_swap', None)
                
                #print(f"[DEBUG] Video reconstruction:")
                #print(f"[DEBUG] Target frames: {target_frames.shape}")
                #print(f"[DEBUG] Reconstructed visual: {rec['visual'].shape}")
                #print(f"[DEBUG] Visual swap indices: {visual_swap_indices}")
                
                losses['rec_video'] = self.rec_video_loss(
                    rec['visual'],
                    target_frames,
                    visual_swap_indices
                )
            
            # Audio reconstruction
            if 'audio' in rec:
                target_mels = targets['mels']  # Keep original shape [B, 4, 1, 128, 32]
                audio_swap_indices = swap_info.get('audio_swap', None)
                
                #print(f"[DEBUG] Audio reconstruction:")
                #print(f"[DEBUG] Target mels: {target_mels.shape}")
                #print(f"[DEBUG] Reconstructed audio: {rec['audio'].shape}")
                #print(f"[DEBUG] Audio swap indices: {audio_swap_indices}")
                
                losses['rec_audio'] = self.rec_audio_loss(
                    rec['audio'],
                    target_mels,
                    audio_swap_indices
                )
        
        # REMOVED: Contrastive Loss - Redundant with Orthogonality Loss
        # Orthogonality loss already enforces vfs ⊥ vir separation
        # No need for additional contrastive loss on encoder features
        #print(f"[DEBUG] Contrastive loss skipped - redundant with orthogonality loss")
    
        # REMOVED: Temporal Contrastive Loss - Also redundant
        #print(f"[DEBUG] Temporal contrastive loss skipped - redundant with orthogonality loss")
    
        # Enhanced Triplet Loss on Static Combined Forgery Features
        av_features = internal_outputs.get('av_features', {})
        
        if 'static' in av_features and 'combined' in av_features['static']:
            if 'modality_common' in av_features['static']['combined']:
                # Use modality_common features (these contain forgery information after fusion)
                static_forgery_features = av_features['static']['combined']['modality_common']  # [B, V, 512]
                
                #print(f"[DEBUG] Triplet Loss on Static Combined Forgery Features:")
                #print(f"[DEBUG] Static forgery features: {static_forgery_features.shape}")
                #print(f"[DEBUG] Multi-class labels: {targets['labels_multi'].shape}")
                
                # Expand temporal dimension for triplet loss (it expects [B, V, T, D])
                # Since these are pooled features, we treat them as single time step
                static_forgery_expanded = static_forgery_features.unsqueeze(2)  # [B, V, 1, 512]
                
                triplet_features = {
                    'static': {'avfs': static_forgery_expanded}
                }
                
                triplet_loss = self.triplet_loss(triplet_features, targets['labels_multi'])
                losses['triplet'] = triplet_loss
                #print(f"[DEBUG] Triplet loss (forgery clustering): {triplet_loss.item():.6f}")
            else:
                print(f"[DEBUG] No modality_common features available for triplet loss")
        else:
            print(f"[DEBUG] No static combined features available for triplet loss")
        
        # Orthogonality Loss from AudioEncoderTCN and Visual Encoder
        if 'orthogonality_loss' in internal_outputs:
            orthogonality_loss = internal_outputs['orthogonality_loss']
            losses['orthogonality'] = orthogonality_loss
            #print(f"[DEBUG] Orthogonality loss: {orthogonality_loss.item():.6f}")
        else:
            #print(f"[DEBUG] No orthogonality loss found in outputs")
            pass
    
        # Combine video and audio reconstruction losses into one reconstruction loss
        reconstruction_loss = None
        if 'rec_video' in losses and torch.is_tensor(losses['rec_video']):
            if 'rec_audio' in losses and torch.is_tensor(losses['rec_audio']):
                # Average if both exist
                reconstruction_loss = (losses['rec_video'] + losses['rec_audio']) / 2.0
            else:
                reconstruction_loss = losses['rec_video']
        elif 'rec_audio' in losses and torch.is_tensor(losses['rec_audio']):
            reconstruction_loss = losses['rec_audio']
        
        # Get device from any available tensor
        device = None
        if reconstruction_loss is not None:
            device = reconstruction_loss.device
        elif losses:
            for v in losses.values():
                if torch.is_tensor(v):
                    device = v.device
                    break
        
        if device is None:
            device = torch.device('cpu')
        
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=device)
        
        # Add reconstruction loss to losses dict for logging
        if reconstruction_loss is not None:
            losses['reconstruction'] = reconstruction_loss
        
        # Apply weights to each loss component
        if reconstruction_loss is not None and torch.is_tensor(reconstruction_loss):
            total_loss = total_loss + self.loss_weights['reconstruction'] * reconstruction_loss
        
        if 'orthogonality' in losses and torch.is_tensor(losses['orthogonality']):
            total_loss = total_loss + self.loss_weights['orthogonality'] * losses['orthogonality']
        
        if 'binary' in losses and torch.is_tensor(losses['binary']):
            total_loss = total_loss + self.loss_weights['binary'] * losses['binary']
        
        if 'multi' in losses and torch.is_tensor(losses['multi']):
            total_loss = total_loss + self.loss_weights['multi'] * losses['multi']
        
        if 'triplet' in losses and torch.is_tensor(losses['triplet']):
            total_loss = total_loss + self.loss_weights['triplet'] * losses['triplet']
        
        # For backward compatibility, create dummy uncertainties list (zeros)
        uncertainties = [0.0] * 6
        
        # Compute metrics for static predictions
        if 'static_binary' in cls_outputs:
            metrics = self._compute_classification_metrics(
                cls_outputs['static_binary'],
                cls_outputs['static_multi'],
                targets['labels_binary'].unsqueeze(-1).unsqueeze(-1),
                targets['labels_multi'].unsqueeze(-1)
            )
            losses.update({
                'static_binary_auc': torch.tensor(metrics['binary_auc'], device=targets['labels_binary'].device),
                'static_binary_f1': torch.tensor(metrics['binary_f1'], device=targets['labels_binary'].device),
                'static_multi_f1': torch.tensor(metrics['multi_f1'], device=targets['labels_binary'].device)
            })

        # Compute metrics for temporal predictions if available
        if 'temporal_binary' in cls_outputs:
            metrics = self._compute_classification_metrics(
                cls_outputs['temporal_binary'],
                cls_outputs['temporal_multi'],
                targets['labels_binary'].unsqueeze(-1).unsqueeze(-1),
                targets['labels_multi'].unsqueeze(-1)
            )
            losses.update({
                'temporal_binary_auc': torch.tensor(metrics['binary_auc'], device=targets['labels_binary'].device),
                'temporal_binary_f1': torch.tensor(metrics['binary_f1'], device=targets['labels_binary'].device),
                'temporal_multi_f1': torch.tensor(metrics['multi_f1'], device=targets['labels_binary'].device)
            })

        # Add sklearn imports at top of file
        # from sklearn.metrics import roc_auc_score, f1_score
        
        #print(f"[DEBUG] ===== LOSS COMPUTATION END =====")
        #print(f"[DEBUG] Final total loss: {total_loss.item():.6f}")
        #print(f"[DEBUG] Uncertainties: {uncertainties}")
        
        return {
            'total_loss': total_loss,
            **{k: v for k, v in losses.items() if torch.is_tensor(v)},  # Only include tensor losses
            'uncertainties': uncertainties  # Return as list instead of tensor
        }

def get_loss_fn(loss_weights=None):
    """
    Helper function to create loss function with custom weights.
    
    Args:
        loss_weights: Dict with keys: 'reconstruction', 'orthogonality', 'binary', 'multi', 'triplet'
                     Values should sum to 1.0. If None, uses equal weights (0.2 each).
    
    Returns:
        WeightedLoss instance
    """
    if loss_weights is None:
        # Default: equal weights
        loss_weights = {
            'reconstruction': 0.1,
            'orthogonality': 0.3,
            'binary': 0.25,
            'multi': 0.25,
            'triplet': 0.1
        }
    return WeightedLoss(loss_weights)

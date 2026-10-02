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
        self.pool = nn.AdaptiveAvgPool2d((100, 100))
    
    def forward(self, recon, orig):
        """
        Args:
            recon: Reconstructed video/audio [B*N, C, H, W]
            orig: Original video/audio [B, V, T, C, H, W] for video or [B, V, T, F] for audio
        """
        # Reshape orig to match recon batch dimension
        B, V, T = orig.shape[:3]
        orig = orig.reshape(-1, *orig.shape[3:])  # [B*V*T, C, H, W] or [B*V*T, F]
        
        # Get first half of reconstruction (second half is swapped data)
        recon = recon[:orig.shape[0]]  # Take only first half
        
        # Ensure both tensors are same size
        if self.mode == 'video':
            recon = self.pool(recon)  # [B*V*T, 3, 100, 100]
        else:  # audio
            # Handle audio tensors correctly
            if len(orig.shape) == 2:  # [B*V*T, F]
                orig = orig.unsqueeze(1)  # Add channel dim [B*V*T, 1, F]
            if len(recon.shape) == 4:  # [B*V*T, 1, F, T]
                recon = recon.squeeze(1)  # Remove extra channel dim
            
            # Get target sizes
            freq_dim = orig.size(-1)
            time_dim = orig.size(-2) if len(orig.shape) > 2 else 1
            
            # Reshape and interpolate if needed using torch.nn.functional
            recon = recon.unsqueeze(1)  # Add channel dimension
            recon = torch.nn.functional.interpolate(
                recon, 
                size=(time_dim, freq_dim),
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)  # Remove channel dimension after interpolation
        
        return self.mse(recon, orig)

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
        Args:
            features: Dictionary containing static/temporal features
            labels: Multi-class labels (A,B,C,D)
        """
        losses = []
        B, V = labels.shape[:2]
        
        # Find indices for each class
        class_indices = {
            'A': (labels == 0).nonzero(as_tuple=True),
            'B': (labels == 1).nonzero(as_tuple=True),
            'C': (labels == 2).nonzero(as_tuple=True),
            'D': (labels == 3).nonzero(as_tuple=True)
        }
        
        # Process both static and temporal features
        for feat_type in ['static', 'temporal']:
            if feat_type in features:
                avfs = features[feat_type]['avfs']  # [B, V, T, D]
                
                # B-C-A triplets
                if all(len(class_indices[c][0]) > 0 for c in ['B', 'C', 'A']):
                    anchor = avfs[class_indices['B']]
                    positive = avfs[class_indices['C']]
                    negative = avfs[class_indices['A']]
                    losses.append(self._compute_loss(anchor, positive, negative))
                
                # D-C-A triplets
                if all(len(class_indices[c][0]) > 0 for c in ['D', 'C', 'A']):
                    anchor = avfs[class_indices['D']]
                    positive = avfs[class_indices['C']]
                    negative = avfs[class_indices['A']]
                    losses.append(self._compute_loss(anchor, positive, negative))
                
                # B-D-A triplets
                if all(len(class_indices[c][0]) > 0 for c in ['B', 'D', 'A']):
                    anchor = avfs[class_indices['B']]
                    positive = avfs[class_indices['D']]
                    negative = avfs[class_indices['A']]
                    losses.append(self._compute_loss(anchor, positive, negative))
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=labels.device)

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Reduce number of log vars since we removed combined
        self.log_vars = nn.Parameter(torch.zeros(6))
        
        # Initialize individual losses
        self.bce_loss = BinaryCrossEntropyLoss()
        self.mce_loss = MultiClassCrossEntropyLoss()
        self.rec_video_loss = ReconstructionLoss('video')
        self.rec_audio_loss = ReconstructionLoss('audio')
        self.contrastive_loss = ContrastiveLoss()
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
        #print("\nLoss Computation Shapes:")
        #print_shape_info("Loss Inputs - Outputs", outputs)
        #print_shape_info("Loss Inputs - Targets", targets)
        
        losses = {}
        
        # Static losses (for 4 videos per set)
        cls_outputs = outputs.get('classifications', {})
        
        if 'static_binary' in cls_outputs:
            # BCE for each video [B, 4, 25, 1]
            losses['static_bce'] = self._compute_per_video_loss(
                cls_outputs['static_binary'],  # [B, 4, 25, 1]
                targets['labels_binary'],      # [B, 4]
                self.bce_loss
            )
            
            # MCE for each video [B, 4, 25, 4]
            losses['static_mce'] = self._compute_per_video_loss(
                cls_outputs['static_multi'],  # [B, 4, 25, 4]
                targets['labels_multi'],      # [B, 4]
                self.mce_loss
            )
        
        # Temporal losses (for 4 videos per set)
        if 'temporal_binary' in cls_outputs:
            # BCE for each video [B, 4, 5, 1]
            losses['temporal_bce'] = self._compute_per_video_loss(
                cls_outputs['temporal_binary'],  # [B, 4, 5, 1]
                targets['labels_binary'],        # [B, 4]
                self.bce_loss
            )
            
            # MCE for each video [B, 4, 5, 4]
            losses['temporal_mce'] = self._compute_per_video_loss(
                cls_outputs['temporal_multi'],  # [B, 4, 5, 4]
                targets['labels_multi'],        # [B, 4]
                self.mce_loss
            )
    
        # Reconstruction losses
        if 'reconstructed' in outputs:
            rec = outputs['reconstructed']
            # Video reconstruction
            if 'visual' in rec:
                # Proper reshape of target frames
                target_frames = targets['frames']  # Keep original shape [B, V, T, C, H, W]
                losses['rec_video'] = self.rec_video_loss(
                    rec['visual'],
                    target_frames
                )
            
            # Audio reconstruction
            if 'audio' in rec:
                # Proper reshape of target mels
                target_mels = targets['mels'].squeeze(2)  # Remove extra dim
                losses['rec_audio'] = self.rec_audio_loss(
                    rec['audio'],
                    target_mels
                )
        
        # Enhanced Contrastive Loss using av_features
        av_features = outputs.get('av_features', {})
        
        # Static features contrastive loss
        if 'static' in av_features:
            static_avfs = av_features['static']['combined']['avfs']  # [B, V, T, D]
            static_avfs_rolled = static_avfs.roll(1, dims=1)        # Roll along video dimension
            labels = torch.ones_like(targets['labels_binary'])      # [B, V]
            
            losses['static_contrastive_av'] = self.contrastive_loss(
                static_avfs,
                static_avfs_rolled,
                labels
            )
    
        # Temporal features contrastive loss
        if 'temporal' in av_features:
            temporal_avfs = av_features['temporal']['combined']['avfs']  # [B, V, T, D]
            temporal_avfs_rolled = temporal_avfs.roll(1, dims=1)        # Roll along video dimension
            labels = torch.ones_like(targets['labels_binary'])          # [B, V]
            
            losses['temporal_contrastive_av'] = self.contrastive_loss(
                temporal_avfs,
                temporal_avfs_rolled,
                labels
            )
    
        # Triplet Loss using av_features
        if 'final_combined' in av_features:
            triplet_features = {
                'static': {'avfs': av_features['static']['combined']['avfs']},
                'temporal': {'avfs': av_features['temporal']['combined']['avfs']} if 'temporal' in av_features else None
            }
            losses['triplet'] = self.triplet_loss(triplet_features, targets['labels_multi'])
    
        # Weight losses with uncertainties
        total_loss = 0
        num_losses = len(losses)
        
        for i, (name, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.log_vars[i % len(self.log_vars)])
            total_loss += precision * loss + self.log_vars[i % len(self.log_vars)]
        
        # Normalize by number of losses
        total_loss = total_loss / num_losses
        
        #print("\nIndividual Loss Shapes:")
        for name, loss in losses.items():
            if isinstance(loss, torch.Tensor):
                print(f"{name}: value = {loss.item():.4f}")

        # Convert uncertainties to list for easier logging
        uncertainties = torch.exp(self.log_vars).detach().cpu().tolist()
        
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
        from sklearn.metrics import roc_auc_score, f1_score
        
        return {
            'total_loss': total_loss,
            **{k: v for k, v in losses.items() if torch.is_tensor(v)},  # Only include tensor losses
            'uncertainties': uncertainties  # Return as list instead of tensor
        }

def get_loss_fn():
    """Helper function to create loss function"""
    return UncertaintyWeightedLoss()

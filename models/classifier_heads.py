import torch
import torch.nn as nn
from typing import Dict

class DeepFakeClassifier(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # 4 classes: FAFV, FARV, RAFV, RARV
        )
        
        self.binary = nn.Linear(128, 1)  # Binary: Real vs Fake
    
    def forward(self, x):
        feat = self.classifier[:-1](x)  # Get features before final layer
        multi_class = self.classifier[-1](feat)
        binary = self.binary(feat)
        return multi_class, binary

class MultiModalClassifier(nn.Module):
    """Dual-head classifier for per-video modality-common and modality-specific features"""
    def __init__(self, common_dim=512, specific_dim=512):
        super().__init__()
        
        
        # Binary classifier (Real/Fake) - uses modality-common features
        
        self.binary_classifier = nn.Sequential(
            nn.Linear(common_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Binary: Real vs Fake
        )
        
        # Multi-class classifier (A,B,C,D) - uses modality-specific features
        
        self.multiclass_classifier = nn.Sequential(
            nn.Linear(specific_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # 4 classes: A, B, C, D
        )
        
        # Initialize with Kaiming (He) initialization for ReLU
        self._init_weights()
        
    
    def _init_weights(self):
        """Initialize layers with Kaiming (He) initialization for ReLU activations"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization works well with ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: {
                'modality_common': [B, 4, 512],    # For binary classification
                'modality_specific': [B, 4, 512],  # For multi-class classification
            }
        Returns:
            Per-video classifications (no pooling)
        """
      
        outputs = {}
        
        # Extract features - NO POOLING
        common_features = features['modality_common']    # [B, 4, 512]
        specific_features = features['modality_specific'] # [B, 4, 512]
        

        
        # Reshape for per-video classification
        B, V, D = common_features.shape  # B=batch, V=4 videos, D=512 features
        
        common_flat = common_features.view(B*V, D)      # [B*4, 512]
        specific_flat = specific_features.view(B*V, D)  # [B*4, 512]
        
      
        # Per-video classification (each video gets its own prediction)
        binary_logits = self.binary_classifier(common_flat)      # [B*4, 1]
        multiclass_logits = self.multiclass_classifier(specific_flat)  # [B*4, 4]
        
        
        # Reshape back to batch format
        binary_logits = binary_logits.view(B, V, 1)    # [B, 4, 1]
        multiclass_logits = multiclass_logits.view(B, V, 4)  # [B, 4, 4]
        
        
        outputs.update({
            'binary': binary_logits,      # [B, 4, 1] - Real vs Fake per video
            'multi': multiclass_logits,   # [B, 4, 4] - A vs B vs C vs D per video
        })
        

        return outputs
        


    def _process_features(self, feat, classifier, prefix):
        """Process features and generate classifications
        Args:
            feat: Input features [B, V, T, D] (forgery features only)
            classifier: Classifier module to use
            prefix: Prefix for output dictionary keys
        """
        if torch.is_tensor(feat):
            B, V, T, D = feat.shape
            # Use only forgery features for classification
            forgery = feat.reshape(B*V*T, D)
            multi, binary = classifier(forgery)
            
            return {
                f'{prefix}_multi': multi.reshape(B, V, T, -1),
                f'{prefix}_binary': binary.reshape(B, V, T, -1)
            }
        else:
            raise ValueError(f"Unsupported feature type for {prefix}")
            raise ValueError(f"Unsupported feature type for {prefix}")

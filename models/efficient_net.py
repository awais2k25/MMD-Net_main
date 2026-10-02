import torch
import torch.nn as nn
import numpy as np
import torchvision.models as tv_models
from torch.nn import functional as F
from typing import Dict, List, Tuple, Optional
from .consts import CHUNK_SIZE, MODEL_CONFIGS

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class NonLocalBlock(nn.Module):
    """Non-local Block for capturing long-range dependencies"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels  # 1408 channels from EfficientNet final layer
        self.inter_channels = in_channels // 2  # 704 channels for internal processing
        
        # Query, Key, Value projections
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, 1)  # Value projection
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, 1)  # Query projection
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, 1)  # Key projection
        
        # Output projection and normalization
        self.W = nn.Conv2d(self.inter_channels, self.in_channels, 1)
        self.bn = nn.BatchNorm2d(self.in_channels)
        
        # Add debug flag
        self.debug = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #if self.debug:
            #print("\nNonLocalBlock Debug:")
            #print(f"Input shape: {x.shape}")  # [B*4*25, 1408, 8, 8]
        
        batch_size = x.size(0)  # This is B*4*25 (total frames)
        
        # Value path
        g_x = self.g(x)  # [B*4*25, 704, 8, 8]
        g_x = g_x.view(batch_size, self.inter_channels, -1)  # [B*4*25, 704, 64]
        g_x = g_x.permute(0, 2, 1)  # [B*4*25, 64, 704]
        
        #if self.debug:
            #print(f"Value features shape: {g_x.shape}")
        
        # Query path
        theta_x = self.theta(x)  # [B*4*25, 704, 8, 8]
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)  # [B*4*25, 704, 64]
        theta_x = theta_x.permute(0, 2, 1)  # [B*4*25, 64, 704]
        
        #if self.debug:
            #print(f"Query features shape: {theta_x.shape}")
        
        # Key path
        phi_x = self.phi(x)  # [B*4*25, 704, 8, 8]
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)  # [B*4*25, 704, 64]
        
        #if self.debug:
            #print(f"Key features shape: {phi_x.shape}")
        
        # Compute attention map
        f = torch.matmul(theta_x, phi_x)  # [B*4*25, 64, 64]
        f_div_C = F.softmax(f, dim=-1)  # Normalize attention weights
        
        #if self.debug:
            #print(f"Attention map shape: {f_div_C.shape}")
        
        # Apply attention
        y = torch.matmul(f_div_C, g_x)  # [B*4*25, 64, 704]
        y = y.permute(0, 2, 1).contiguous()  # [B*4*25, 704, 64]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # [B*4*25, 704, 8, 8]
        
        #if self.debug:
            #print(f"After attention shape: {y.shape}")
        
        # Final projection
        W_y = self.W(y)  # [B*4*25, 1408, 8, 8]
        z = self.bn(W_y) + x  # Add residual connection
        
        #if self.debug:
            #print(f"Output shape: {z.shape}\n")
        
        return z

class SophisticatedVisualForgeryHead(nn.Module):
    """Sophisticated forgery detection head with multi-scale spatial analysis"""
    def __init__(self, in_channels: int = 1408, out_dim: int = 512):
        super().__init__()
        
        # Initial bottleneck to reduce 1408 → 512 channels (better information compression)
        self.initial_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Multi-scale spatial convolutions (fine/medium/coarse)
        # Input is now 512 channels after bottleneck
        self.multi_scale_conv = nn.ModuleDict({
            'fine': nn.Conv2d(512, 256, kernel_size=3, padding=1),      # Fine details (192→256)
            'medium': nn.Conv2d(512, 256, kernel_size=5, padding=2),    # Medium patterns (192→256)
            'coarse': nn.Conv2d(512, 256, kernel_size=7, padding=3)     # Coarse structure (192→256)
        })
        
        # Spatial self-attention for artifact localization
        # Input: 768 channels (256*3 from multi-scale)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=768, 
            num_heads=12,  # 768 ÷ 12 = 64 per head
            dropout=0.1, 
            batch_first=True
        )
        
        # SE block for channel-wise attention
        self.se = SEBlock(768)
        
        # High-frequency artifact detector
        self.hf_detector = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),  # 576→768
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Final projection
        self.flatten = nn.Flatten()
        flat_size = 256 * 8 * 8  # 16384 (EfficientNet-B2 produces 8x8 feature maps for 140x140 input)
        self.forgery_projection = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, out_dim)
        )
        
        # Initialize new layers with Kaiming (He) initialization for ReLU
        self._init_weights()
    
    def _init_weights(self):
        """Initialize new layers with Kaiming (He) initialization for ReLU activations"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming initialization for Conv2d with ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Kaiming initialization for Linear layers (works well with ReLU)
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights (PyTorch default is already good, but we can enhance)
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
        
    def forward(self, x):  # [B*4*25, 1408, 8, 8]
        #print(f"[DEBUG] SophisticatedVisualForgeryHead input: {x.shape}")
        
        # Initial bottleneck: reduce 1408 → 512 channels
        x = self.initial_bottleneck(x)  # [B*4*25, 512, 8, 8]
        #print(f"[DEBUG] After bottleneck: {x.shape}")
        
        # Multi-scale spatial analysis
        fine = self.multi_scale_conv['fine'](x)      # [B*4*25, 256, 8, 8]
        medium = self.multi_scale_conv['medium'](x)  # [B*4*25, 256, 8, 8]
        coarse = self.multi_scale_conv['coarse'](x)  # [B*4*25, 256, 8, 8]
        
        #print(f"[DEBUG] Multi-scale outputs - fine: {fine.shape}, medium: {medium.shape}, coarse: {coarse.shape}")
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([fine, medium, coarse], dim=1)  # [B*4*25, 768, 8, 8]
        #print(f"[DEBUG] Multi-scale concatenated: {multi_scale.shape}")
        
        # Spatial self-attention for artifact localization
        B, C, H, W = multi_scale.shape
        spatial_tokens = multi_scale.flatten(2).permute(0, 2, 1)  # [B*4*25, 64, 768]
        #print(f"[DEBUG] Spatial tokens for attention: {spatial_tokens.shape}")
        
        attended, _ = self.spatial_attention(spatial_tokens, spatial_tokens, spatial_tokens)
        attended = attended.permute(0, 2, 1).reshape(B, C, H, W)  # [B*4*25, 768, 8, 8]
        #print(f"[DEBUG] After spatial attention: {attended.shape}")
        
        # Channel attention
        attended = self.se(attended)  # [B*4*25, 768, 8, 8]
        #print(f"[DEBUG] After SE block: {attended.shape}")
        
        # High-frequency artifact detection
        artifacts = self.hf_detector(attended)  # [B*4*25, 256, 8, 8]
        #print(f"[DEBUG] After HF detector: {artifacts.shape}")
        
        # Flatten and project
        flat = self.flatten(artifacts)  # [B*4*25, 16384] (256*8*8)
        #print(f"[DEBUG] After flatten: {flat.shape}")
        
        forgery_features = self.forgery_projection(flat)  # [B*4*25, 512]
        #print(f"[DEBUG] SophisticatedVisualForgeryHead final output: {forgery_features.shape}")
        
        return forgery_features

class SimpleVisualIrrelevantHead(nn.Module):
    """Simple content extraction head for identity/style features"""
    def __init__(self, in_channels: int = 1408, out_dim: int = 512):
        super().__init__()
        
        # Progressive channel reduction: 1408 → 512 → 256
        # Better capacity to distill information from 1408 channels
        self.content_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),  # Initial bottleneck
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Simple spatial pooling (preserve some spatial structure)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Lightweight projection
        self.flatten = nn.Flatten()
        flat_size = 256 * 4 * 4  # 4096
        self.content_projection = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, out_dim)
        )
        
    def forward(self, x):  # [B*4*25, 1408, 8, 8]
        #print(f"[DEBUG] SimpleVisualIrrelevantHead input: {x.shape}")
        
        # Extract content features (progressive: 1408 → 512 → 256)
        content = self.content_extractor(x)  # [B*4*25, 256, 8, 8]
        #print(f"[DEBUG] After content extractor: {content.shape}")
        
        # Spatial pooling
        pooled = self.spatial_pool(content)  # [B*4*25, 256, 4, 4]
        #print(f"[DEBUG] After spatial pooling: {pooled.shape}")
        
        # Flatten and project
        flat = self.flatten(pooled)  # [B*4*25, 4096]
        #print(f"[DEBUG] After flatten: {flat.shape}")
        
        irrelevant_features = self.content_projection(flat)  # [B*4*25, 512]
        #print(f"[DEBUG] SimpleVisualIrrelevantHead final output: {irrelevant_features.shape}")
        
        return irrelevant_features

class EfficientNetExtractor(nn.Module):
    """EfficientNet feature extractor with disentanglement heads using torchvision"""
    def __init__(
        self, 
        version: str = 'b2',
        pretrained: bool = True
    ):
        super().__init__()
        # Load pretrained EfficientNet from torchvision
        # Map version to torchvision model name
        model_name_map = {
            'b0': 'efficientnet_b0',
            'b1': 'efficientnet_b1',
            'b2': 'efficientnet_b2',
            'b3': 'efficientnet_b3',
            'b4': 'efficientnet_b4',
            'b5': 'efficientnet_b5',
            'b6': 'efficientnet_b6',
            'b7': 'efficientnet_b7',
        }
        
        model_name = model_name_map.get(version, 'efficientnet_b2')
        model_func = getattr(tv_models, model_name)
        
        # Load model with pretrained weights
        if pretrained:
            # Use default weights for pretrained models
            self.backbone = model_func(weights='DEFAULT')
        else:
            self.backbone = model_func(weights=None)
        
        # Use dummy input to get actual feature map size (dynamic from MODEL_CONFIGS)
        input_size = MODEL_CONFIGS[version]['input_size'][0]
        dummy_input = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            # Extract features (before classifier)
            features = self.backbone.features(dummy_input)
        feature_size = features.shape[-1]  # Get actual feature map size
        last_channels = features.shape[1]
        # print(f"[DEBUG] Using torchvision {model_name}")
        # print(f"[DEBUG] Last channels: {last_channels}")
        # print(f"[DEBUG] Feature size: {feature_size}")
        
        # Initialize heads with correct channels and feature map size
        self.forgery_head = SophisticatedVisualForgeryHead(last_channels)
        self.irrelevant_head = SimpleVisualIrrelevantHead(last_channels)
        
        # Set chunk size as instance variable
        self.chunk_size = CHUNK_SIZE
        
        # Orthogonality loss weight
        self.orthogonality_weight = 0.1

        self.debug = False  # Add debug flag

    def forward(
        self, 
        batch_data: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        output = {}
        
        # Process static frames (dynamic input size)
        if 'frames' in batch_data:
            #if self.debug:
                #print("\nEfficientNet Forward Pass:")
                #print(f"Input batch frames shape: {batch_data['frames'].shape}")
            
            # Extract dimensions dynamically from input tensor
            B, V, T, C, H, W = batch_data['frames'].shape
            frames = batch_data['frames'].view(-1, C, H, W)
            
            #if self.debug:
                #print(f"Reshaped frames shape: {frames.shape}")  # [B*4*25, 3, H, W]
            
            # Extract features using torchvision (returns tensor, not list)
            last_features = self.backbone.features(frames)  # [B*4*25, 1408, feature_H, feature_W]
            
            #if self.debug:
                #print(f"Backbone features shape: {last_features.shape}")  # [B*4*25, 1408, feature_H, feature_W]
            
            # last_features is already the final feature map (1408 channels for B2)
            
            # Process through heads
            vfs = self.forgery_head(last_features)
            vir = self.irrelevant_head(last_features)
            
            #if self.debug:
                #print(f"Forgery head output shape: {vfs.shape}")    # Should be [B*V*T, 512]
                #print(f"Irrelevant head output shape: {vir.shape}") # Should be [B*V*T, 512]
                #print(f"Reshaped outputs:")
            
            # Reshape features
            vfs_reshaped = vfs.view(B, V, T, -1)  # [B, V, T, 512]
            vir_reshaped = vir.view(B, V, T, -1)  # [B, V, T, 512]
            
            # Compute orthogonality loss
            orthogonality_loss = self.compute_orthogonality_loss(vfs_reshaped, vir_reshaped)
            
            output.update({
                'vfs': vfs_reshaped,
                'vir': vir_reshaped,
                # 'features': last_features.view(B, 4, 25, *last_features.shape[1:]), # raw efficientnet features features: torch.Size([1, 4, 25, 1408, 7, 7])
                'orthogonality_loss': orthogonality_loss
            })
        
        return output
    
    def compute_orthogonality_loss(self, forgery, irrelevant, alpha=0.5):
        """
        Compute orthogonality loss to encourage decorrelation between forgery and irrelevant features
        FIXED: Now uses per-video computation like other losses
        
        Args:
            forgery: Forgery features [B, 4, 25, 512]
            irrelevant: Irrelevant features [B, 4, 25, 512]
            
        Returns:
            torch.Tensor: Orthogonality loss scalar
        """
        #print(f"[DEBUG] Computing orthogonality loss...")
        #print(f"[DEBUG] Forgery features shape: {forgery.shape}")
        #print(f"[DEBUG] Irrelevant features shape: {irrelevant.shape}")
        
        B, V, T, D = forgery.shape  # [B, 4, 25, 512]
        
        # FIXED: Per-video flattening (like other losses)
        # Flatten each video separately: [B, V, T*D] then [B*V, T*D]
        # forgery_per_video = forgery.reshape(B, V, T*D)  # [B, 4, 25*512]
        # irrelevant_per_video = irrelevant.reshape(B, V, T*D)  # [B, 4, 25*512]
        
        # Flatten to per-video samples (consistent with binary/multi losses)
        forgery_flat = forgery.reshape(B*V, T*D)  # [B*4, 25*512] = [4, 12800]
        irrelevant_flat = irrelevant.reshape(B*V, T*D)  # [B*4, 25*512] = [4, 12800]
        
        #print(f"[DEBUG] Per-video flattened forgery: {forgery_flat.shape}")
        #print(f"[DEBUG] Per-video flattened irrelevant: {irrelevant_flat.shape}")
        
        # Normalize features before computing orthogonality to make it more comparable to other losses
        f_norm = F.normalize(forgery_flat, dim=1)  # L2 normalize along feature dimension
        r_norm = F.normalize(irrelevant_flat, dim=1)  # L2 normalize along feature dimension
        
        #print(f"[DEBUG] Normalized forgery features (L2): {f_norm.shape}")
        #print(f"[DEBUG] Normalized irrelevant features (L2): {r_norm.shape}")
        
        # Compute per-video orthogonality (element-wise dot product of normalized features)
        # per_video_correlation = torch.sum(f_norm * r_norm, dim=1)  # [B*4]
        raw_corr = torch.mean(torch.abs(torch.sum(forgery_flat * irrelevant_flat, dim=1))) / (T*D)
        #print(f"[DEBUG] Per-video correlations: {raw_corr.shape}")
        #print(f"[DEBUG] Correlation range: [{raw_corr.min().item():.6f}, {raw_corr.max().item():.6f}]")
        
        # Cosine correlation (scale-insensitive)
        cos_corr = torch.mean(torch.abs(torch.sum(f_norm * r_norm, dim=1)))
        #print(f"[DEBUG] Per-video correlations: {cos_corr.shape}")
        #print(f"[DEBUG] Correlation range: [{cos_corr.min().item():.6f}, {cos_corr.max().item():.6f}]")
        
        # Hybrid
        orthogonality_loss = alpha * raw_corr + (1 - alpha) * cos_corr

        # FIXED: Per-video orthogonality loss (no double normalization)
        # .mean() already normalizes by B*V, so no need to divide by V again
        #print(f"[DEBUG] hybrid orthogonality loss: {orthogonality_loss.item():.6f}")
        
        # Apply weight
        # weighted_loss = self.orthogonality_weight * orthogonality_loss
        # #print(f"[DEBUG] Weighted orthogonality loss: {weighted_loss.item():.6f}")
        
        return orthogonality_loss
    
def get_efficient_net(version: str = 'b2', **kwargs) -> EfficientNetExtractor:
    """Helper function to create EfficientNet with specific version"""
    assert version in ['b0', 'b1', 'b2', 'b3', 'b4'], f"Version {version} not supported"
    return EfficientNetExtractor(version=version, **kwargs)

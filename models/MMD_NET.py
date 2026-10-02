import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

from .efficient_net import get_efficient_net
from .AudioEncoderTCN import AudioEncoderTCN
from .Reconstruction import Reconstruction
from .MMT import get_av_fusion
from .classifier_heads import MultiModalClassifier
from utils.debug_utils import print_shape_info


class MMDNet(nn.Module):
    def __init__(
        self,
        visual_encoder: str = 'b2',
        audio_encoder: str = '18'
    ):
        super().__init__()
        
        # Initialize encoders
        self.visual_encoder = get_efficient_net(
            version=visual_encoder
            # temporal_modeling=False
        ).float()
        self.audio_encoder = AudioEncoderTCN(
            input_dim=1,
            output_dim=512,
            freq_channels=256,
            tcn_channels=256
        ).float()
        
        # ── ABLATION: Reconstruction decoders DISABLED ──────────────────────
        # self.reconstruction = Reconstruction(
        #     feature_dim=512,
        #     efficientnet_version=visual_encoder
        # ).float()
        # ─────────────────────────────────────────────────────────────────────
        
        # Initialize multimodal transformer
        self.av_fusion = get_av_fusion(dim=512).float()
        
        # Initialize classifier
        self.classifier = MultiModalClassifier().float()

        self.debug = True  # Add debug flag

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        #if self.debug:
            #print("\nMMDNet Forward Pass:")
            #print(f"Input frames shape: {batch_data['frames'].shape}")
            #print(f"Input mels shape: {batch_data['mels'].shape}")
        
        # Internal outputs for loss calculation and internal processing
        outputs_internal = {}
        
        visual_input = {'frames': batch_data['frames']}  # [B, 4, 25, 3, H, W]
        audio_input = {'mels': batch_data['mels']}      # [B, 4, 1, 128, 32]
        
        # Encoder Path
        visual_features = self.visual_encoder(visual_input)
        audio_features = self.audio_encoder(audio_input)
        
        # Extract orthogonality losses from both encoders
        audio_orthogonality_loss = audio_features.get('orthogonality_loss', torch.tensor(0.0))
        visual_orthogonality_loss = visual_features.get('orthogonality_loss', torch.tensor(0.0))
        
        # Combine orthogonality losses
        total_orthogonality_loss = audio_orthogonality_loss + visual_orthogonality_loss
        
        #if self.debug:
            #print("\nEncoder Outputs:")
            #print(f"Visual features shapes:")
            #print(f"vfs: {visual_features['vfs'].shape}")
            #print(f"vir: {visual_features['vir'].shape}")
            #print(f"Audio features shapes:")
            #print(f"afs: {audio_features['afs'].shape}")
            #print(f"air: {audio_features['air'].shape}")
            #print(f"Audio Orthogonality loss: {audio_orthogonality_loss.item():.6f}")
            #print(f"Visual Orthogonality loss: {visual_orthogonality_loss.item():.6f}")
            #print(f"Total Orthogonality loss: {total_orthogonality_loss.item():.6f}")
        
        # Store internal outputs for loss calculation
        outputs_internal['encoder_features'] = {
            'visual': visual_features,
            'audio': audio_features
        }
        outputs_internal['orthogonality_loss'] = total_orthogonality_loss
        
        # # Reconstruction Path
        # reconstructed = self.reconstruction(
        #     visual_features['vfs'],
        #     visual_features['vir'],
        #     audio_features['afs'],
        #     audio_features['air'],
        #     enable_swap=self.training
        # )
        # outputs_internal['reconstructed'] = reconstructed
        # ── ABLATION: Reconstruction Path DISABLED ──────────────────────────
        # The reconstruction decoders are completely skipped.
        # This means 'reconstructed' will NOT appear in outputs_internal,
        # and the loss function will naturally skip rec_video/rec_audio losses.
        # ─────────────────────────────────────────────────────────────────────
        
        # Fusion Path
        fusion_input = {
            'static': {
                'original': {
                    'vfs': visual_features['vfs'],
                    'vir': visual_features['vir'],
                    'afs': audio_features['afs'],
                    'air': audio_features['air']
                }
            }
        }

        av_features = self.av_fusion(fusion_input)
        outputs_internal['av_features'] = av_features
        
        # Classification Path
        #if self.debug:
            #print(f"[DEBUG] MMD_NET Classification Path")
            #print(f"[DEBUG] av_features keys: {av_features.keys()}")
            #print(f"[DEBUG] av_features['static'] keys: {av_features['static'].keys()}")
            #print(f"[DEBUG] av_features['static']['combined'] keys: {av_features['static']['combined'].keys()}")
        
        classifier_input = {
            'modality_common': av_features['static']['combined']['modality_common'],      # [B, 4, 512]
            'modality_specific': av_features['static']['combined']['modality_specific'],  # [B, 4, 512]
        }
        
        #if self.debug:
            #print(f"[DEBUG] Classifier input shapes:")
            #print(f"[DEBUG] modality_common: {classifier_input['modality_common'].shape}")
            #print(f"[DEBUG] modality_specific: {classifier_input['modality_specific'].shape}")
        
        classifications = self.classifier(classifier_input)
        
        #if self.debug:
            #print(f"[DEBUG] Classification output keys: {classifications.keys()}")
            #print(f"[DEBUG] Binary predictions: {classifications['binary'].shape}")
            #print(f"[DEBUG] Multi-class predictions: {classifications['multi'].shape}")
        
        # External outputs - only what Main_v2.py actually uses
        outputs_external = {
            'classifications': {
                'binary': classifications['binary'],  # [B, 4, 1] - Real/Fake per video
                'multi': classifications['multi']     # [B, 4, 4] - A/B/C/D per video
            }
        }
        
        # Return both internal and external outputs separately
        return {
            'external': outputs_external,    # For Main_v2.py
            'internal': outputs_internal     # For loss function
        }

def get_mmdnet(**kwargs) -> MMDNet:
    """Helper function to create MMDNet model"""
    return MMDNet(**kwargs)

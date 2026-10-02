"""
Trace tensor shapes through the entire MMD-Net architecture.
Run: python trace_shapes.py
"""
import sys
import torch
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.consts import MODEL_CONFIGS, FRAMES_PER_VIDEO

def trace_shapes():
    """Trace shapes through all modules with a dummy forward pass."""
    print("=" * 70)
    print("MMD-Net Tensor Shape Trace")
    print("=" * 70)
    
    # Import models
    from models.MMD_NET import MMDNet
    
    # Create model
    model = MMDNet()
    model.eval()
    
    B = 1  # batch size
    V = 4  # videos per sample (RVRA, RVFA, FVRA, FVFA)
    T = FRAMES_PER_VIDEO  # frames per video
    input_size = MODEL_CONFIGS['b2']['input_size'][0]  # 140
    
    print(f"\nConfiguration:")
    print(f"  Batch size (B): {B}")
    print(f"  Videos per sample (V): {V}")
    print(f"  Frames per video (T): {T}")
    print(f"  Image input size: {input_size}x{input_size}")
    print(f"  EfficientNet version: b2")
    print(f"  Feature dim (d): 512")
    
    # Create dummy inputs
    dummy_frames = torch.randn(B, V, T, 3, input_size, input_size)
    dummy_mels = torch.randn(B, V, 1, 64, 94)
    
    print(f"\n{'='*70}")
    print("INPUT SHAPES")
    print(f"{'='*70}")
    print(f"  frames: {dummy_frames.shape}  # (B, V, T, C, H, W)")
    print(f"  mels:   {dummy_mels.shape}  # (B, V, C_mel, F, L)")
    
    # ===== VISUAL ENCODER =====
    print(f"\n{'='*70}")
    print("VISUAL ENCODER (EfficientNet-b2)")
    print(f"{'='*70}")
    
    # Reshape
    frames_flat = dummy_frames.view(-1, 3, input_size, input_size)
    print(f"  Reshape for backbone: {frames_flat.shape}  # (B*V*T, 3, H, W)")
    
    # Backbone
    with torch.no_grad():
        backbone_out = model.visual_encoder.backbone.features(frames_flat)
    print(f"  EfficientNet-b2 output: {backbone_out.shape}  # (B*V*T, C_eff, H', W')")
    
    # Forgery Head
    print(f"\n  --- Visual Forgery Head ---")
    fh = model.visual_encoder.forgery_head
    with torch.no_grad():
        x = fh.initial_bottleneck(backbone_out)
        print(f"  After bottleneck (1x1): {x.shape}")
        fine = fh.multi_scale_conv['fine'](x)
        med = fh.multi_scale_conv['medium'](x)
        coarse = fh.multi_scale_conv['coarse'](x)
        print(f"  MS-Conv fine (3x3):   {fine.shape}")
        print(f"  MS-Conv medium (5x5): {med.shape}")
        print(f"  MS-Conv coarse (7x7): {coarse.shape}")
        ms = torch.cat([fine, med, coarse], dim=1)
        print(f"  Concatenated MS-Conv: {ms.shape}")
        B2, C2, H2, W2 = ms.shape
        tokens = ms.flatten(2).permute(0, 2, 1)
        print(f"  Spatial tokens:       {tokens.shape}")
        attended, _ = fh.spatial_attention(tokens, tokens, tokens)
        attended = attended.permute(0, 2, 1).reshape(B2, C2, H2, W2)
        print(f"  After Spatial Attn:   {attended.shape}")
        attended = fh.se(attended)
        print(f"  After SE-Block:       {attended.shape}")
        artifacts = fh.hf_detector(attended)
        print(f"  After HF Detector:    {artifacts.shape}")
        flat = fh.flatten(artifacts)
        print(f"  After Flatten:        {flat.shape}")
        vf = fh.forgery_projection(flat)
        print(f"  Final v_f output:     {vf.shape}")
    
    # Irrelevant Head
    print(f"\n  --- Visual Irrelevant Head ---")
    ih = model.visual_encoder.irrelevant_head
    with torch.no_grad():
        content = ih.content_extractor(backbone_out)
        print(f"  After Content Extract: {content.shape}")
        pooled = ih.spatial_pool(content)
        print(f"  After Spatial Pool:    {pooled.shape}")
        flat = ih.flatten(pooled)
        print(f"  After Flatten:         {flat.shape}")
        vir = ih.content_projection(flat)
        print(f"  Final v_ir output:     {vir.shape}")
    
    # Reshape outputs
    v_f_reshaped = vf.view(B, V, T, -1)
    v_ir_reshaped = vir.view(B, V, T, -1)
    print(f"\n  Reshaped v_f:  {v_f_reshaped.shape}  # (B, V, T, d)")
    print(f"  Reshaped v_ir: {v_ir_reshaped.shape}  # (B, V, T, d)")
    
    # ===== AUDIO ENCODER =====
    print(f"\n{'='*70}")
    print("AUDIO ENCODER (FreqEncoder + TCN)")
    print(f"{'='*70}")
    
    mels_flat = dummy_mels.view(-1, 1, 64, 94)
    print(f"  Reshape for encoder: {mels_flat.shape}  # (B*V, C_mel, F, L)")
    
    ae = model.audio_encoder
    with torch.no_grad():
        # FreqEncoder
        fe = ae.freq_encoder
        x = fe.conv1(mels_flat)
        print(f"\n  --- Frequency Encoder ---")
        print(f"  After Conv1 (3x3, 1→64): {x.shape}")
        x = fe.bn1(x)
        x = fe.relu1(x)
        x = fe.conv2(x)
        print(f"  After Conv2 (3x3, 64→128): {x.shape}")
        x = fe.bn2(x)
        x = fe.relu2(x)
        x = fe.conv3(x)
        print(f"  After Conv3 (3x3, 128→256): {x.shape}")
        x = fe.bn3(x)
        x = fe.relu3(x)
        x = fe.freq_pool(x)
        print(f"  After FreqPool (AdaptAvgPool2d): {x.shape}")
        
        # TCN
        print(f"\n  --- Temporal Convolutional Network ---")
        x_tcn = x.reshape(x.size(0), -1, x.size(-1))
        print(f"  Reshape for TCN: {x_tcn.shape}  # (B*V, C*F', L)")
        tcn_out = ae.tcn(x_tcn)
        print(f"  After TCN (6 dilated layers): {tcn_out.shape}")
        tcn_down = ae.temporal_downsample(tcn_out)
        print(f"  After Temporal Downsample: {tcn_down.shape}  # (B*V, 256, T)")
        
        # Forgery Branch
        print(f"\n  --- Audio Forgery Branch ---")
        fb = ae.forgery_branch
        fine_a = fb.multi_scale_conv['fine'](tcn_down)
        med_a = fb.multi_scale_conv['medium'](tcn_down)
        coarse_a = fb.multi_scale_conv['coarse'](tcn_down)
        print(f"  MS-TempConv fine (k=3):   {fine_a.shape}")
        print(f"  MS-TempConv medium (k=5): {med_a.shape}")
        print(f"  MS-TempConv coarse (k=11): {coarse_a.shape}")
        combined = torch.cat([fine_a, med_a, coarse_a], dim=1)
        print(f"  Concatenated MS-TempConv: {combined.shape}")
        combined_t = combined.permute(0, 2, 1)
        attn_out, _ = fb.artifact_attention(combined_t, combined_t, combined_t)
        print(f"  After Temporal Attention: {attn_out.permute(0, 2, 1).shape}")
        hf_out = fb.hf_detector(attn_out.permute(0, 2, 1))
        print(f"  After HF Detector:        {hf_out.shape}")
        forgery_out = fb.forgery_projection(hf_out)
        print(f"  After Forgery Projection: {forgery_out.shape}")
        af = fb.dropout(forgery_out)
        print(f"  Final a_f output:         {af.shape}")
        
        # Irrelevant Branch
        print(f"\n  --- Audio Irrelevant Branch ---")
        ib = ae.irrelevant_branch
        content_a = ib.content_extractor(tcn_down)
        print(f"  After Content Extractor: {content_a.shape}")
        temp_a = ib.temporal_modeling(content_a)
        print(f"  After Temporal Modeling: {temp_a.shape}")
        air_out = ib.output_projection(temp_a)
        print(f"  Final a_ir output:       {air_out.shape}")
    
    a_f_reshaped = af.view(B, V, T, -1)
    a_ir_reshaped = air_out.view(B, V, T, -1)
    print(f"\n  Reshaped a_f:  {a_f_reshaped.shape}  # (B, V, T, d)")
    print(f"  Reshaped a_ir: {a_ir_reshaped.shape}  # (B, V, T, d)")

    # ===== CROSS-MODAL ALIGNMENT & FUSION (CMAF / MMT) =====
    print(f"\n{'='*70}")
    print("CROSS-MODAL ALIGNMENT & FUSION (CMAF / MMT)")
    print(f"{'='*70}")
    
    cmaf = model.av_fusion
    mmt = cmaf.static_mmt
    with torch.no_grad():
        vfs_v = mmt.proj_v(v_f_reshaped[:, 0])  # Video 0 slice: [B, T, d]
        afs_v = mmt.proj_a(a_f_reshaped[:, 0])  # Video 0 slice: [B, T, d]
        print(f"  Single Video Feature Projections (vfs_v, afs_v): {vfs_v.shape}")
        
        # Multi-Level Cross Attention
        mlca = mmt.multi_level_attention
        print(f"\n  --- Multi-Level Cross Attention ---")
        l_fine_v = mlca.fine_pool(vfs_v.transpose(1, 2)).transpose(1, 2)
        l_med_v = mlca.medium_pool(vfs_v.transpose(1, 2)).transpose(1, 2)
        l_coarse_v = mlca.coarse_pool(vfs_v.transpose(1, 2)).transpose(1, 2)
        print(f"  Pyramid Visual Levels -> Fine (T=25): {l_fine_v.shape}, Medium (T=15): {l_med_v.shape}, Coarse (T=8): {l_coarse_v.shape}")
        
        coarse_att, _ = mlca.attention_levels['coarse'](l_coarse_v, l_coarse_v, l_coarse_v)
        print(f"  Coarse Attended: {coarse_att.shape}")
        med_att, _ = mlca.attention_levels['medium'](l_med_v, l_med_v, l_med_v)
        print(f"  Medium Attended: {med_att.shape}")
        fine_att, _ = mlca.attention_levels['fine'](l_fine_v, l_fine_v, l_fine_v)
        print(f"  Fine Attended:   {fine_att.shape}")
        
        mlca_out = mlca(vfs_v, afs_v)
        print(f"  Fused Multi-Level Feature Output: {mlca_out.shape}  # (B, T, d)")
        
        # Modality Separation
        print(f"\n  --- Modality Separation ---")
        common_f, spec_f = mmt.modality_separation(mlca_out)
        print(f"  Modality Common Features:   {common_f.shape}  # (B, T, d)")
        print(f"  Modality Specific Features: {spec_f.shape}  # (B, T, d)")
        
        # Efficient Temporal Pooling
        print(f"\n  --- Efficient Temporal Pooling ---")
        common_pooled = mmt.temporal_pooling_common(common_f)
        spec_pooled = mmt.temporal_pooling_specific(spec_f)
        print(f"  Pooled Common (per video):   {common_pooled.shape}  # (B, d)")
        print(f"  Pooled Specific (per video): {spec_pooled.shape}  # (B, d)")
        
        # Full MMT forward over all V=4 videos
        mmt_input = {'vfs': v_f_reshaped, 'afs': a_f_reshaped}
        mmt_out = mmt(mmt_input)
        print(f"\n  Full CMAF Output Stack (modality_common):   {mmt_out['modality_common'].shape}  # (B, V, d)")
        print(f"  Full CMAF Output Stack (modality_specific): {mmt_out['modality_specific'].shape}  # (B, V, d)")
        
        # Classifier Head
        print(f"\n  --- Multimodal Classifier Head ---")
        clf_out = model.classifier(mmt_out)
        print(f"  Binary Classification Output: {clf_out['binary'].shape}  # (B, V, 1)")
        print(f"  Multi-Class Classification Output: {clf_out['multi'].shape}  # (B, V, 4)")

    # ===== RECONSTRUCTION MODULE =====
    print(f"\n{'='*70}")
    print("RECONSTRUCTION MODULE (Visual & Audio Decoders)")
    print(f"{'='*70}")
    
    from models.Reconstruction import Reconstruction
    rec = Reconstruction(feature_dim=512, efficientnet_version='b2').eval()
    
    with torch.no_grad():
        # Visual Decoder
        print(f"  --- Visual Decoder ---")
        v_dec = rec.visual_decoder
        v_fused, _, _ = v_dec.fusion(v_f_reshaped, v_ir_reshaped, enable_swap=False)
        print(f"  After Feature Fusion (v_f + v_ir): {v_fused.shape}  # (B, V, T, d)")
        v_mlp_out = v_dec.mlp(v_fused)
        print(f"  After MLP:                        {v_mlp_out.shape}")
        init_size = v_dec.feature_map * 2
        v_reshape = v_mlp_out.reshape(-1, 64, init_size, init_size)
        print(f"  Reshape for ConvTranspose2d:      {v_reshape.shape}  # (B*V*T, 64, 16, 16)")
        v_conv_t = v_dec.conv_transpose(v_reshape)
        print(f"  After ConvTranspose2d:            {v_conv_t.shape}  # (B*V*T, 3, 128, 128)")
        v_recon, _ = v_dec(v_f_reshaped, v_ir_reshaped, enable_swap=False)
        print(f"  Final Visual Reconstruction:      {v_recon.shape}  # (B, V, T, 3, H, W)")
        
        # Audio Decoder
        print(f"\n  --- Audio Decoder ---")
        a_dec = rec.audio_decoder
        a_fused = a_dec.fusion._fuse_without_temporal(a_f_reshaped, a_ir_reshaped)
        print(f"  After Feature Fusion (a_f + a_ir): {a_fused.shape}  # (B, V, T, d)")
        a_temp_in = a_fused.permute(0, 1, 3, 2).reshape(B*V, 512, T)
        a_temp_up = a_dec.temporal_upsample(a_temp_in)
        print(f"  After Temporal Upsample (25→32):  {a_temp_up.shape}  # (B*V, d, 32)")
        a_temp_pooled = a_temp_up.reshape(B, V, 512, 32).permute(0, 1, 3, 2).mean(dim=2)
        print(f"  After Temporal Pooling:           {a_temp_pooled.shape}  # (B, V, d)")
        a_mlp_out = a_dec.mlp(a_temp_pooled)
        print(f"  After MLP:                        {a_mlp_out.shape}")
        a_reshape = a_mlp_out.reshape(-1, 64, 4, 8)
        print(f"  Reshape for ConvTranspose2d:      {a_reshape.shape}  # (B*V, 64, 4, 8)")
        a_conv_t = a_dec.conv_transpose(a_reshape)
        print(f"  After ConvTranspose2d:            {a_conv_t.shape}  # (B*V, 1, 64, 32)")
        a_recon, _ = a_dec(a_f_reshaped, a_ir_reshaped, enable_swap=False)
        print(f"  Final Audio Reconstruction:       {a_recon.shape}  # (B, V, 1, 64, 32)")

    print(f"\n{'='*70}")
    print("SHAPE TRACE COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    trace_shapes()


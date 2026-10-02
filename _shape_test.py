import torch
from models.MMD_NET import get_mmdnet

print("Loading MMDNet...")
model = get_mmdnet(visual_encoder='b2', audio_encoder='18').cuda().eval()

# New shapes: (B=1, V=4, T=25, C=3, H=140, W=140) video, (B=1, V=4, C=1, mel=64, t=32) audio
dummy = {
    'frames': torch.randn(1, 4, 25, 3, 140, 140).cuda(),
    'mels':   torch.randn(1, 4, 1, 64, 32).cuda(),
}
print("Running forward pass with new shapes: frames=(1,4,25,3,140,140)  mels=(1,4,1,64,32)")
with torch.no_grad():
    out = model(dummy)

print(f"Binary output: {out['external']['classifications']['binary'].shape}")
print(f"Multi  output: {out['external']['classifications']['multi'].shape}")
print("✅ Shape test passed!")

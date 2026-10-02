import torch
from models.AudioEncoderTCN import AudioEncoderTCN

model = AudioEncoderTCN(input_dim=1, output_dim=512, freq_channels=256, tcn_channels=256)

# Test 1-video batch
x1 = torch.randn(2, 1, 1, 64, 32)
out1 = model(x1)
print(f"[1-video] Forgery features: {out1['afs'].shape}")

# Test 4-video batch
x4 = torch.randn(2, 4, 1, 64, 32)
out4 = model(x4)
print(f"[4-video] Forgery features: {out4['afs'].shape}")


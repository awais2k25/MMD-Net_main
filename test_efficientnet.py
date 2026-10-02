import torch
from models.efficient_net import get_efficient_net

model = get_efficient_net(version='b2')

# Test 4-video batch
x4 = {'frames': torch.randn(2, 4, 25, 3, 250, 250)}
out4 = model(x4)
print(f"[4-video] vfs features: {out4['vfs'].shape}")

# Test 1-video batch
x1 = {'frames': torch.randn(2, 1, 25, 3, 250, 250)}
try:
    out1 = model(x1)
    print(f"[1-video] vfs features: {out1['vfs'].shape}")
except Exception as e:
    import traceback
    traceback.print_exc()


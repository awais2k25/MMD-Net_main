import torch
from models.Reconstruction import Reconstruction

recon = Reconstruction(feature_dim=512, efficientnet_version='b2')
vf_static = torch.randn(2, 1, 25, 512)
virr = torch.randn(2, 1, 25, 512)
af_static = torch.randn(2, 1, 25, 512)
airr = torch.randn(2, 1, 25, 512)

try:
    outputs = recon(vf_static, virr, af_static, airr, enable_swap=False)
    print("Reconstruction successful with V=1.")
except Exception as e:
    import traceback
    traceback.print_exc()


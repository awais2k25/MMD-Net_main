import torch
from models.MMD_NET import get_mmdnet

try:
    model = get_mmdnet(visual_encoder='b2', audio_encoder='18')
    model.eval()
    
    # 4-video batch
    batch_4 = {
        'frames': torch.randn(2, 4, 25, 3, 140, 140),
        'mels': torch.randn(2, 4, 1, 64, 32)
    }
    
    print("Testing 4-video batch...")
    out4 = model(batch_4)
    print("4-video batch passed")
    
    
    # 1-video batch
    batch_1 = {
        'frames': torch.randn(2, 1, 25, 3, 140, 140),
        'mels': torch.randn(2, 1, 1, 64, 32)
    }
    print("Testing 1-video batch...")
    out1 = model(batch_1)
    print("1-video batch passed")
    
except Exception as e:
    import traceback
    traceback.print_exc()


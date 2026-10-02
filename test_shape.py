import torch
import torch.nn.functional as F
q = torch.randn(1, 3200, 128)
k = torch.randn(1, 3200, 128)
v = torch.randn(1, 3200, 128)
try:
    attn = torch.nn.MultiheadAttention(512, 8)
    attn(q, k, v)
except Exception as e:
    print(e)

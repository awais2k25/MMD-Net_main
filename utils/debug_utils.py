import torch
from typing import Dict, Any

def print_shape_info(name: str, tensor_or_dict: Any, indent: str = ''):
    """Print shape information for tensors or nested dictionaries"""
    if isinstance(tensor_or_dict, torch.Tensor):
        print(f"{indent}{name}: shape = {tensor_or_dict.shape}")
    elif isinstance(tensor_or_dict, dict):
        print(f"{indent}{name}:")
        for k, v in tensor_or_dict.items():
            print_shape_info(k, v, indent + '  ')
    elif isinstance(tensor_or_dict, (list, tuple)):
        print(f"{indent}{name}: (list/tuple) length = {len(tensor_or_dict)}")
        for i, item in enumerate(tensor_or_dict):
            print_shape_info(f"item_{i}", item, indent + '  ')

import os
import sys
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.MMD_NET import get_mmdnet
import torch
from prettytable import PrettyTable

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_size():
    """Analyze model size and parameters per component"""
    model = get_mmdnet()
    
    # Create table
    table = PrettyTable(['Module', 'Parameters'])
    
    # Add main components
    components = {
        'Visual Encoder': model.visual_encoder,
        'Audio Encoder': model.audio_encoder,
        'Reconstruction': model.reconstruction,
        'AV Fusion': model.av_fusion,
        'Classifier': model.classifier
    }
    
    total_params = 0
    for name, component in components.items():
        params = sum(p.numel() for p in component.parameters() if p.requires_grad)
        total_params += params
        table.add_row([name, f'{params:,}'])
    
    table.add_row(['Total', f'{total_params:,}'])
    print(table)
    
    # Memory analysis
    print("\nMemory Analysis:")
    print(f"Model Size (MB): {total_params * 4 / (1024*1024):.2f}")
    
    # CUDA memory if available
    if torch.cuda.is_available():
        print("\nCUDA Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

def analyze_feature_dimensions():
    """Analyze feature dimensions throughout the model"""
    model = get_mmdnet()
    model.eval()  # Set to eval mode
    
    # Create dummy input matching dataset format
    B, V = 2, 4  # Batch size, Videos per set
    
    # Create dummy input matching expected format
    batch = {
        'frames': torch.randn(B, V, 25, 3, 250, 250),   # [B, 4, 25, 3, 250, 250]
        'mels': torch.randn(B, V, 1, 128, 32),           # [B, 4, 1, 128, 32] — channel dim added
    }
    
    # Debug info before forward pass
    print("\nInput Batch Structure:")
    def print_tensor_shape(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                print(f"{prefix}{k}: {tuple(v.shape)}")
            elif isinstance(v, dict):
                print(f"{prefix}{k}:")
                print_tensor_shape(v, prefix + '  ')
    
    print_tensor_shape(batch)
    
    # Add hooks with better error handling
    hooks = []
    def hook_fn(name):
        def hook(module, input, output):
            print(f"\n{name} output shapes:")
            try:
                if isinstance(output, torch.Tensor):
                    print(f"  tensor: {output.shape}")
                elif isinstance(output, dict):
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: {v.shape}")
                        elif isinstance(v, dict):
                            print(f"  {k}:")
                            for sk, sv in v.items():
                                if isinstance(sv, torch.Tensor):
                                    print(f"    {sk}: {sv.shape}")
                                elif isinstance(sv, dict):
                                    print(f"    {sk}: <nested dict>")
            except Exception as e:
                print(f"  Error in hook {name}: {str(e)}")
        return hook
    
    # Register hooks
    for name, module in [
        ('Visual Encoder', model.visual_encoder),
        ('Audio Encoder', model.audio_encoder),
        ('Reconstruction', model.reconstruction),
        ('AV Fusion', model.av_fusion),
        ('Classifier', model.classifier)
    ]:
        hooks.append(module.register_forward_hook(hook_fn(name)))
    
    try:
        # Forward pass with error handling
        with torch.no_grad():
            print("\nStarting forward pass...")
            outputs = model(batch)
            
            print("\nFinal Output Structure:")
            print_tensor_shape(outputs)
            
    except Exception as e:
        print(f"\nError during forward pass: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

if __name__ == '__main__':
    print("Model Parameter Analysis:")
    analyze_model_size()
    print("\nFeature Dimension Analysis:")
    analyze_feature_dimensions()

#!/usr/bin/env python3
"""
Script to check EfficientNet model input sizes and compare timm vs torchvision implementations.

Official Torchvision Input Sizes:
    'b0': (224, 256), 'b1': (240, 256), 'b2': (288, 288), 'b3': (300, 320),
    'b4': (380, 384), 'b5': (456, 489), 'b6': (528, 561), 'b7': (600, 633)
    
Note: Torchvision uses (height, width) format, but typically uses square inputs
in practice, so the larger dimension is used.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights
import timm
from typing import Dict, List, Tuple
import sys

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Official torchvision EfficientNet input sizes (width, height) as provided by user
# These are the exact official sizes from PyTorch documentation
# Format: (width, height) - converted to (height, width) for PyTorch tensor usage
TORCHVISION_OFFICIAL_SIZES = {
    'b0': (256, 224),  # (width, height) -> (224, 256) for PyTorch
    'b1': (256, 240),  # (width, height) -> (240, 256) for PyTorch
    'b2': (288, 288),  # (width, height) -> (288, 288) for PyTorch
    'b3': (320, 300),  # (width, height) -> (300, 320) for PyTorch
    'b4': (384, 380),  # (width, height) -> (380, 384) for PyTorch
    'b5': (489, 456),  # (width, height) -> (456, 489) for PyTorch
    'b6': (561, 528),  # (width, height) -> (528, 561) for PyTorch
    'b7': (633, 600),  # (width, height) -> (600, 633) for PyTorch
}

# Convert to (height, width) format for PyTorch tensor usage
# PyTorch uses (batch, channels, height, width) format
TORCHVISION_PYTORCH_SIZES = {
    'b0': (224, 256),  # (height, width) for torch.randn(1, 3, 224, 256)
    'b1': (240, 256),  # (height, width) for torch.randn(1, 3, 240, 256)
    'b2': (288, 288),  # (height, width) for torch.randn(1, 3, 288, 288)
    'b3': (300, 320),  # (height, width) for torch.randn(1, 3, 300, 320)
    'b4': (380, 384),  # (height, width) for torch.randn(1, 3, 380, 384)
    'b5': (456, 489),  # (height, width) for torch.randn(1, 3, 456, 489)
    'b6': (528, 561),  # (height, width) for torch.randn(1, 3, 528, 561)
    'b7': (600, 633),  # (height, width) for torch.randn(1, 3, 600, 633)
}

def check_timm_efficientnet_models():
    """Check timm EfficientNet models and their expected input sizes"""
    print("=" * 80)
    print("TIMM EFFICIENTNET MODELS")
    print("=" * 80)
    print("Using timm.data.resolve_model_data_config() to get official input sizes")
    print("=" * 80)
    
    timm_models = [
        # EfficientNet V1
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
        'efficientnet_b3', 
        # 'efficientnet_b4', 'efficientnet_b5',
        # 'efficientnet_b6', 'efficientnet_b7',
        # EfficientNet V2 (using tf_ prefix as they correspond to tf_ models)
        'tf_efficientnetv2_b0', 'tf_efficientnetv2_b1', 'tf_efficientnetv2_b2',
        'tf_efficientnetv2_b3', 'tf_efficientnetv2_s', 'tf_efficientnetv2_m',
        'tf_efficientnetv2_l'
    ]
    
    timm_results = {}
    
    for model_name in timm_models:
        try:
            # Try to create model, if it fails with V2 names, try alternative names
            try:
                model = timm.create_model(model_name, pretrained=False)
            except RuntimeError as e:
                # If V2 model fails, try without tf_ prefix or with different naming
                if 'v2' in model_name or 'efficientnetv2' in model_name:
                    # Try alternative names
                    alt_names = [
                        model_name.replace('tf_', ''),
                        model_name.replace('efficientnetv2', 'efficientnet_v2'),
                        model_name.replace('tf_efficientnetv2', 'efficientnet_v2'),
                    ]
                    model = None
                    for alt_name in alt_names:
                        try:
                            model = timm.create_model(alt_name, pretrained=False)
                            model_name = alt_name  # Update to working name
                            break
                        except:
                            continue
                    if model is None:
                        raise e
                else:
                    raise e
            
            # Count parameters
            num_params = count_parameters(model)
            
            # Get default input size from timm config (official way)
            config = timm.data.resolve_model_data_config(model)
            input_size_raw = config['input_size']
            
            # Handle different formats: could be (height, width) or (channels, height, width)
            if len(input_size_raw) == 3:
                # Format: (channels, height, width) - extract just height and width
                _, default_h, default_w = input_size_raw
                default_input_size = (default_h, default_w)
            elif len(input_size_raw) == 2:
                # Format: (height, width)
                default_input_size = input_size_raw
            else:
                # Fallback
                default_input_size = (224, 224)
            
            # Get version (e.g., 'b2' from 'efficientnet_b2', 's' from 'tf_efficientnetv2_s')
            version = model_name.split('_')[-1]
            
            # Check if this is a V2 model
            is_v2 = 'v2' in model_name or 'efficientnetv2' in model_name.lower()
            
            # Get torchvision official size for comparison (only for V1 models)
            tv_official = None
            if not is_v2:
                tv_official = TORCHVISION_PYTORCH_SIZES.get(version, None)
            
            # Test with different input sizes
            # Include your current size (100x100), timm default, torchvision official, and a few others
            test_sizes = [
                (100, 100),  # Your current size
                default_input_size,  # Timm official (height, width)
            ]
            
            # Add torchvision official size if different (only for V1 models)
            if tv_official and tv_official != default_input_size:
                test_sizes.append(tv_official)
            
            # Add a couple more for comparison
            if version == 'b2':
                test_sizes.extend([(260, 260), (288, 288)])  # Common sizes for B2
            elif is_v2:
                # For V2 models, add some common sizes
                if default_input_size[0] > 300:
                    test_sizes.extend([(256, 256), (384, 384)])
                else:
                    test_sizes.extend([(224, 224), (256, 256)])
            
            output_info = {}
            
            for h, w in test_sizes:
                try:
                    # Test with features_only=True (like your code)
                    test_model = timm.create_model(
                        model_name, 
                        pretrained=False,
                        features_only=True,
                        out_indices=(4,)
                    )
                    test_model.eval()
                    
                    dummy_input = torch.randn(1, 3, h, w)
                    with torch.no_grad():
                        features = test_model(dummy_input)
                        last_features = features[-1]
                        output_shape = last_features.shape
                        output_info[(h, w)] = {
                            'shape': output_shape,
                            'channels': output_shape[1],
                            'spatial': (output_shape[2], output_shape[3])
                        }
                except Exception as e:
                    output_info[(h, w)] = {'error': str(e)}
            
            timm_results[model_name] = {
                'default_input_size': default_input_size,
                'torchvision_official': tv_official,
                'num_parameters': num_params,
                'outputs': output_info
            }
            
            print(f"\n{model_name}:")
            print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
            print(f"  TIMM Official Input Size: {default_input_size} (height, width)")
            if tv_official:
                print(f"  Torchvision Official Size: {tv_official} (height, width) = {TORCHVISION_OFFICIAL_SIZES.get(version)} (width, height)")
            print(f"  Output Info:")
            for size, info in output_info.items():
                if 'error' in info:
                    print(f"    Input {size}: ERROR - {info['error']}")
                else:
                    print(f"    Input {size}:")
                    print(f"      Output shape: {info['shape']}")
                    print(f"      Channels: {info['channels']}")
                    print(f"      Spatial: {info['spatial']}")
                    
        except Exception as e:
            print(f"\n{model_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            timm_results[model_name] = {'error': str(e)}
    
    return timm_results


def check_torchvision_efficientnet_models():
    """Check torchvision EfficientNet models and their expected input sizes"""
    print("\n" + "=" * 80)
    print("TORCHVISION EFFICIENTNET MODELS")
    print("=" * 80)
    print("Official Torchvision Input Sizes (from PyTorch documentation):")
    for version, size in TORCHVISION_OFFICIAL_SIZES.items():
        print(f"  B{version[1:]}: {size[0]}x{size[1]} (width x height) = {TORCHVISION_PYTORCH_SIZES[version]} (height x width for PyTorch)")
    print("=" * 80)
    
    tv_models_list = [
        # EfficientNet V1
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        'efficientnet_b3', 
        # 'efficientnet_b4', 'efficientnet_b5',
        # 'efficientnet_b6', 'efficientnet_b7',
        # EfficientNet V2
        'efficientnet_v2_s', 'efficientnet_v2_m'
    ]
    
    tv_results = {}
    
    for model_name in tv_models_list:
        try:
            # Create model
            model_func = getattr(tv_models, model_name)
            
            # Handle V2 models with weights
            if model_name == 'efficientnet_v2_s':
                model = model_func(weights=None)
            elif model_name == 'efficientnet_v2_m':
                model = model_func(weights=None)
            else:
                model = model_func(pretrained=False)
            
            model.eval()
            
            # Count parameters
            num_params = count_parameters(model)
            
            # Get version and official size
            version = model_name.split('_')[-1]
            
            # V2 models have different sizes - we'll detect them
            if model_name.startswith('efficientnet_v2'):
                # V2 models: use default or detect from model
                official_size = (384, 384)  # Common default for V2
                if model_name == 'efficientnet_v2_s':
                    official_size = (384, 384)
                elif model_name == 'efficientnet_v2_m':
                    official_size = (480, 480)
            else:
                official_size = TORCHVISION_PYTORCH_SIZES.get(version, (224, 224))  # (height, width)
            
            # Test with official size, your current size (100x100), and a few others
            test_sizes = [
                (140, 140),  # Your current size
                official_size,  # Official torchvision size (height, width)
            ]
            
            # Add a couple more for B2 specifically (using square for comparison)
            if version == 'b2':
                test_sizes.extend([(260, 260), (288, 288)])
            
            output_info = {}
            
            for h, w in test_sizes:
                try:
                    dummy_input = torch.randn(1, 3, h, w)
                    with torch.no_grad():
                        # Get features before classifier
                        # For torchvision, we need to get features manually
                        features = model.features(dummy_input)
                        # Get avgpool output
                        avgpool_out = model.avgpool(features)
                        output_shape = avgpool_out.shape
                        
                        output_info[(h, w)] = {
                            'features_shape': features.shape,
                            'avgpool_shape': output_shape,
                            'channels': features.shape[1],
                            'spatial': (features.shape[2], features.shape[3])
                        }
                except Exception as e:
                    output_info[(h, w)] = {'error': str(e)}
            
            tv_results[model_name] = {
                'official_size': official_size,
                'num_parameters': num_params,
                'outputs': output_info
            }
            
            if model_name.startswith('efficientnet_v2'):
                print(f"\n{model_name}:")
                print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
                print(f"  Official Input Size: {official_size} (height, width)")
            else:
                original_size = TORCHVISION_OFFICIAL_SIZES.get(version, (224, 224))
                print(f"\n{model_name}:")
                print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
                print(f"  Official Input Size: {official_size} (height, width) = {original_size} (width, height)")
            
            print(f"  Output Info:")
            for size, info in output_info.items():
                if 'error' in info:
                    print(f"    Input {size}: ERROR - {info['error']}")
                else:
                    print(f"    Input {size}:")
                    print(f"      Features shape: {info['features_shape']}")
                    print(f"      Channels: {info['channels']}")
                    print(f"      Spatial: {info['spatial']}")
                    print(f"      AvgPool shape: {info['avgpool_shape']}")
                    
        except Exception as e:
            print(f"\n{model_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            tv_results[model_name] = {'error': str(e)}
    
    return tv_results


def compare_features_only_capability():
    """Compare features_only capability between timm and torchvision"""
    print("\n" + "=" * 80)
    print("FEATURES_ONLY CAPABILITY COMPARISON")
    print("=" * 80)
    
    print("\nTIMM:")
    print("  ✓ Supports features_only=True parameter")
    print("  ✓ Can specify out_indices to get specific layers")
    print("  ✓ Returns list of feature maps at different scales")
    print("  ✓ Easy to get intermediate features")
    print("  Example: timm.create_model('efficientnet_b2', features_only=True, out_indices=(4,))")
    
    print("\nTORCHVISION:")
    print("  ✗ Does NOT have features_only parameter (but you CAN get features)")
    print("  ✓ Can manually access model.features(input) to get feature maps")
    print("  ✓ Returns final feature map (before classifier)")
    print("  ✗ Need to manually extract intermediate features (no out_indices)")
    print("  Example: model.features(input) - returns final feature map only")
    print("  Note: Both can extract features, but timm has more convenient API")


def test_your_current_config():
    """Test your current configuration (EfficientNet-B2 with 100x100 input)"""
    print("\n" + "=" * 80)
    print("YOUR CURRENT CONFIGURATION TEST")
    print("=" * 80)
    
    print("\nTesting: EfficientNet-B2 with 100x100 input (TIMM)")
    try:
        model = timm.create_model(
            'efficientnet_b2',
            pretrained=False,
            features_only=True,
            out_indices=(4,)
        )
        model.eval()
        
        # Get official sizes
        config = timm.data.resolve_model_data_config(model)
        input_size_raw = config['input_size']
        
        # Handle different formats: could be (height, width) or (channels, height, width)
        if len(input_size_raw) == 3:
            # Format: (channels, height, width) - extract just height and width
            _, timm_h, timm_w = input_size_raw
            timm_official = (timm_h, timm_w)
        elif len(input_size_raw) == 2:
            # Format: (height, width)
            timm_official = input_size_raw
        else:
            timm_official = (260, 260)  # Fallback
        
        tv_official = TORCHVISION_PYTORCH_SIZES['b2']  # (height, width)
        tv_official_original = TORCHVISION_OFFICIAL_SIZES['b2']  # (width, height)
        
        # Test with your current size
        print(f"\n1. Your Current Size (100x100):")
        dummy_input = torch.randn(1, 3, 100, 100)
        with torch.no_grad():
            features = model(dummy_input)
            last_features = features[-1]
            
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output features shape: {last_features.shape}")
        print(f"  Channels: {last_features.shape[1]}")
        print(f"  Spatial size: {last_features.shape[2]}x{last_features.shape[3]}")
        
        # Test with TIMM official size
        print(f"\n2. TIMM Official Size {timm_official} (height, width):")
        dummy_input_timm = torch.randn(1, 3, timm_official[0], timm_official[1])
        with torch.no_grad():
            features_timm = model(dummy_input_timm)
            last_features_timm = features_timm[-1]
            
        print(f"  Input shape: {dummy_input_timm.shape}")
        print(f"  Output features shape: {last_features_timm.shape}")
        print(f"  Channels: {last_features_timm.shape[1]}")
        print(f"  Spatial size: {last_features_timm.shape[2]}x{last_features_timm.shape[3]}")
        
        # Test with Torchvision official size
        print(f"\n3. Torchvision Official Size {tv_official} (height, width) = {tv_official_original} (width, height):")
        dummy_input_tv = torch.randn(1, 3, tv_official[0], tv_official[1])
        with torch.no_grad():
            features_tv = model(dummy_input_tv)
            last_features_tv = features_tv[-1]
            
        print(f"  Input shape: {dummy_input_tv.shape}")
        print(f"  Output features shape: {last_features_tv.shape}")
        print(f"  Channels: {last_features_tv.shape[1]}")
        print(f"  Spatial size: {last_features_tv.shape[2]}x{last_features_tv.shape[3]}")
        
        # Compare differences
        print("\n" + "-" * 80)
        print("COMPARISON:")
        print("-" * 80)
        
        # Compare with TIMM official
        spatial_ratio_timm = (last_features.shape[2] * last_features.shape[3]) / (last_features_timm.shape[2] * last_features_timm.shape[3])
        print(f"\n100x100 vs TIMM Official ({timm_official}):")
        print(f"  Spatial reduction: {last_features_timm.shape[2]}x{last_features_timm.shape[3]} → {last_features.shape[2]}x{last_features.shape[3]}")
        print(f"  Spatial information: {spatial_ratio_timm:.2%} of TIMM official")
        
        # Compare with Torchvision official
        spatial_ratio_tv = (last_features.shape[2] * last_features.shape[3]) / (last_features_tv.shape[2] * last_features_tv.shape[3])
        print(f"\n100x100 vs Torchvision Official ({tv_official} = {tv_official_original}):")
        print(f"  Spatial reduction: {last_features_tv.shape[2]}x{last_features_tv.shape[3]} → {last_features.shape[2]}x{last_features.shape[3]}")
        print(f"  Spatial information: {spatial_ratio_tv:.2%} of Torchvision official")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_specific_input_sizes():
    """Test EfficientNet-B2 and EfficientNet-V2 with specific input sizes
    Testing with shape (T, C, H, W) where T is number of frames (75 or 25)
    Includes both timm and torchvision models
    """
    print("\n" + "=" * 80)
    print("SPECIFIC INPUT SIZE TESTING")
    print("=" * 80)
    print("Testing with shape (T, C, H, W) where T=frames, C=3, H=W=100 or official sizes")
    print("=" * 80)
    
    # Models to test: (display_name, model_name, model_type, model_func)
    models_to_test = [
        ('EfficientNet-B2 (timm)', 'efficientnet_b2', 'timm', None),
        ('EfficientNet-V2-S (timm)', 'tf_efficientnetv2_s', 'timm', None),
        ('EfficientNet-B2 (torchvision)', 'efficientnet_b2', 'torchvision', tv_models.efficientnet_b2),
        ('EfficientNet-V2-S (torchvision)', 'efficientnet_v2_s', 'torchvision', tv_models.efficientnet_v2_s),
        ('EfficientNet-V2-M (torchvision)', 'efficientnet_v2_m', 'torchvision', tv_models.efficientnet_v2_m),
    ]
    
    # Test configurations: (num_frames, height, width)
    test_configs = [
        (25, 140, 140)  # 75 frames, 100x100 spatial
        # (25, 140, 140),  # 25 frames, 100x100 spatial
    ]
    
    # Torchvision V2 official sizes
    torchvision_v2_sizes = {
        'efficientnet_v2_s': (250, 250),
        'efficientnet_v2_m': (480, 480),
    }
    
    for model_display_name, model_name, model_type, model_func in models_to_test:
        print(f"\n{model_display_name}:")
        
        # Try to create model
        model = None
        original_model_name = model_name
        official_size = None
        
        if model_type == 'timm':
            try:
                model = timm.create_model(model_name, pretrained=False, features_only=True, out_indices=(4,))
            except:
                # Try alternatives
                alt_names = [
                    model_name.replace('tf_', ''),
                    model_name.replace('efficientnetv2', 'efficientnet_v2'),
                ]
                for alt_name in alt_names:
                    try:
                        model = timm.create_model(alt_name, pretrained=False, features_only=True, out_indices=(4,))
                        model_name = alt_name
                        break
                    except:
                        continue
            
            if model is None:
                print(f"  ERROR: Could not create timm model {original_model_name}")
                continue
            
            # Get official input size from timm
            try:
                config = timm.data.resolve_model_data_config(model)
                input_size_raw = config['input_size']
                if len(input_size_raw) == 3:
                    _, official_h, official_w = input_size_raw
                    official_size = (official_h, official_w)
                elif len(input_size_raw) == 2:
                    official_size = input_size_raw
                else:
                    official_size = (256, 256)
            except:
                official_size = (256, 256)
                
        elif model_type == 'torchvision':
            try:
                if model_func is None:
                    model_func = getattr(tv_models, model_name)
                model = model_func(weights=None)
            except Exception as e:
                print(f"  ERROR: Could not create torchvision model {model_name}: {e}")
                continue
            
            # Get official input size for torchvision
            if model_name in torchvision_v2_sizes:
            #     official_size = torchvision_v2_sizes[model_name]
            # elif model_name == 'efficientnet_b2':
            #     official_size = TORCHVISION_PYTORCH_SIZES.get('b2', (288, 288))
            # else:
            #     # Try to detect from model or use default
                official_size = (288, 288)  # Default for V2 models
        
        if model is None:
            print(f"  ERROR: Could not create model {original_model_name}")
            continue
        
        if official_size is None:
            print(f"  ERROR: Could not determine official size for {original_model_name}")
            official_size = (256, 256)  # Default fallback
        
        model.eval()
        
        num_params = count_parameters(model)
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  Official Input Size: {official_size} (height, width)")
        
        # Add official size configs
        test_configs_with_official = test_configs + [
            (75, official_size[0], official_size[1]),  # 75 frames, official H, W
            (25, official_size[0], official_size[1]),  # 25 frames, official H, W
        ]
        
        print(f"  Output Info for different configurations (T, H, W):")
        for T, H, W in test_configs_with_official:
            try:
                # Create input with shape (T, C, H, W) where T=frames, C=3
                dummy_input = torch.randn(T, 3, H, W)
                
                with torch.no_grad():
                    # Process all frames as a batch (T frames processed together)
                    # if model_type == 'timm':
                    #     features = model(dummy_input)
                    #     last_features = features[-1]  # Shape: (T, C_out, H_out, W_out)
                    if model_type == 'torchvision':
                        # Torchvision returns tensor directly from features()
                        last_features = model.features(dummy_input)  # Shape: (T, C_out, H_out, W_out)
                    
                    output_shape = last_features.shape
                    
                print(f"    Input shape (T={T}, C=3, H={H}, W={W}): {dummy_input.shape}")
                print(f"      Output shape: {output_shape} (T={T}, C={output_shape[1]}, H={output_shape[2]}, W={output_shape[3]})")
                print(f"      Channels: {output_shape[1]}")
                print(f"      Spatial per frame: ({output_shape[2]}, {output_shape[3]})")
                print(f"      Total spatial locations: {output_shape[2] * output_shape[3]} per frame")
                print(f"      Total features across all frames: {output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]:,}")
                
                # Calculate spatial ratio compared to official size
                official_dummy = torch.randn(1, 3, official_size[0], official_size[1])
                with torch.no_grad():
                    if model_type == 'timm':
                        official_features = model(official_dummy)
                        official_output = official_features[-1]
                    elif model_type == 'torchvision':
                        official_output = model.features(official_dummy)
                    
                    official_spatial = official_output.shape[2] * official_output.shape[3]
                    test_spatial = output_shape[2] * output_shape[3]
                    spatial_ratio = test_spatial / official_spatial if official_spatial > 0 else 0
                    print(f"      Spatial info per frame: {spatial_ratio:.2%} of official size")
                    
            except Exception as e:
                print(f"    Input (T={T}, H={H}, W={W}): ERROR - {e}")
                import traceback
                traceback.print_exc()


def recommendations():
    """Provide recommendations"""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Get official sizes for B2
    try:
        model = timm.create_model('efficientnet_b2', pretrained=False)
        config = timm.data.resolve_model_data_config(model)
        input_size_raw = config['input_size']
        
        # Handle different formats: could be (height, width) or (channels, height, width)
        if len(input_size_raw) == 3:
            # Format: (channels, height, width) - extract just height and width
            _, timm_h, timm_w = input_size_raw
            timm_official = (timm_h, timm_w)
        elif len(input_size_raw) == 2:
            # Format: (height, width)
            timm_official = input_size_raw
        else:
            timm_official = (260, 260)  # Fallback
    except:
        timm_official = (260, 260)  # Fallback
    
    tv_official = TORCHVISION_PYTORCH_SIZES['b2']  # (height, width)
    tv_official_original = TORCHVISION_OFFICIAL_SIZES['b2']  # (width, height)
    
    print(f"""
1. TIMM vs TORCHVISION:
   - TIMM is BETTER for your use case because:
     * Supports features_only=True (crucial for your feature extraction)
     * Can easily get intermediate layers with out_indices
     * More flexible and feature-rich
     * Better for feature extraction tasks
     * Official B2 input size: {timm_official} (height, width)
   
   - TORCHVISION is better for:
     * Classification tasks
     * Simpler integration
     * Standard use cases
     * Official B2 input size: {tv_official} (height, width) = {tv_official_original} (width, height)

2. INPUT SIZE IMPACT:
   - Your current 100x100 input reduces spatial resolution significantly
   - TIMM EfficientNet-B2 expects: {timm_official} (height, width)
   - Torchvision EfficientNet-B2 expects: {tv_official} (height, width) = {tv_official_original} (width, height)
   - With 100x100: you get ~4x4 feature maps (vs ~9x9 with standard)
   - This reduces spatial information by ~80%

3. RECOMMENDED INPUT SIZES:
   - Minimum: 160x160 (gives ~5x5 feature maps)
   - Good balance: 224x224 (gives ~7x7 feature maps)
   - TIMM official: {timm_official} (height, width) - optimal for timm
   - Torchvision official: {tv_official} (height, width) = {tv_official_original} (width, height) - optimal for torchvision

4. YOUR CURRENT SETUP:
   - Your code using timm is correct ✓
   - Using features_only=True is the right approach ✓
   - The 352 channels you're seeing might be from a different layer
   - Check if out_indices=(4,) is giving you the final layer
   - Consider using out_indices=(-1,) to get the last layer explicitly
   - Or use out_indices=(5,) if available for the true final layer

5. HOW TO GET INPUT/OUTPUT SIZES (from timm docs):
   - Use timm.data.resolve_model_data_config(model) to get official input size
   - Use features_only=True with out_indices to control which layers you get
   - Test with dummy input to verify actual output dimensions
    """)


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("EFFICIENTNET MODEL COMPARISON SCRIPT")
    print("=" * 80)
    
    # Check timm models
    # timm_results = check_timm_efficientnet_models()
    
    # Check torchvision models
    # tv_results = check_torchvision_efficientnet_models()
    
    # Compare features_only
    # compare_features_only_capability()
    
    # Test specific input sizes
    test_specific_input_sizes()
    
    # Test current config
    # test_your_current_config()
    
    # Recommendations
    # recommendations()
    
    print("\n" + "=" * 80)
    print("SCRIPT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


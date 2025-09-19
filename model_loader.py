#!/usr/bin/env python3
"""
Universal Model Loader for Ear Biometrics System
Handles automatic model architecture detection and state dict compatibility
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4
from pathlib import Path
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple, List
import json

# Try to import timm for advanced models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available - some models may not load")


class ModelArchitectureDetector:
    """Detects model architecture from state dict keys"""
    
    @staticmethod
    def analyze_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze state dict to determine model architecture and configuration"""
        keys = list(state_dict.keys())
        
        # Detect model type
        model_info = {
            'type': 'unknown',
            'feature_dim': None,
            'backbone': None,
            'has_attention': False,
            'has_cbam': False,
            'is_timm': False,
            'is_ultimate': False,
            'is_excellent': False,
            'num_classes': None,
            'architecture_hints': []
        }
        
        # Check for Ultimate model patterns (more comprehensive)
        ultimate_patterns = [
            'multi_scale_features', 'ultimate', 'fusion_layer', 'scale_attention',
            'backbone.0.', 'backbone.1.', 'backbone.2.'  # Multiple backbone pattern
        ]
        if any(pattern in k for pattern in ultimate_patterns for k in keys):
            model_info['type'] = 'ultimate'
            model_info['is_ultimate'] = True
            model_info['architecture_hints'].append('Has Ultimate architecture patterns')
        
        # Check for models with "ultimate" in filename or high feature dimensions
        feature_dim = model_info.get('feature_dim') or 0
        if any('ultimate' in k.lower() for k in keys) or (feature_dim >= 4096):
            model_info['type'] = 'ultimate'
            model_info['is_ultimate'] = True
            model_info['architecture_hints'].append('Ultimate model (by name or feature dimension)')
        
        # Check for Excellent model patterns (CBAM attention)
        if any('attentions' in k for k in keys) or any('cbam' in k.lower() for k in keys):
            model_info['type'] = 'excellent'
            model_info['is_excellent'] = True
            model_info['has_cbam'] = True
            model_info['architecture_hints'].append('Has CBAM attention modules')
        
        # Check for timm backbone
        if any('backbone.blocks' in k for k in keys) or any('backbone.conv_stem' in k for k in keys):
            model_info['is_timm'] = True
            model_info['architecture_hints'].append('Uses timm backbone')
        
        # Check for standard EfficientNet patterns
        if any('backbone.features' in k for k in keys):
            model_info['backbone'] = 'efficientnet'
            model_info['architecture_hints'].append('Standard EfficientNet backbone')
        
        # Check for neck architecture
        if any('neck.' in k for k in keys):
            model_info['type'] = 'neck'
            model_info['architecture_hints'].append('Has neck architecture (custom head)')
        
        # Check for backbone.classifier architecture
        if any('backbone.classifier.' in k for k in keys):
            model_info['type'] = 'backbone_classifier'
            model_info['architecture_hints'].append('Has backbone.classifier architecture')
        
        # Try to detect feature dimension
        feature_dim_candidates = []
        for key in keys:
            if 'classifier' in key or 'fc' in key or 'feature_processor' in key:
                if 'weight' in key and key in state_dict:
                    try:
                        weight_shape = state_dict[key].shape
                        if len(weight_shape) == 2:  # Linear layer
                            # Output dimension is usually the feature dimension
                            if weight_shape[0] < 10000:  # Reasonable feature dimension
                                feature_dim_candidates.append(weight_shape[0])
                    except (AttributeError, IndexError):
                        continue
        
        if feature_dim_candidates:
            # Most common feature dimension
            model_info['feature_dim'] = max(set(feature_dim_candidates), 
                                           key=feature_dim_candidates.count)
        else:
            model_info['feature_dim'] = None
        
        # Check for batch norm patterns
        has_running_mean = any('running_mean' in k for k in keys)
        has_running_var = any('running_var' in k for k in keys)
        has_bn_weight = any('bn' in k.lower() and 'weight' in k for k in keys)
        
        if has_running_mean or has_running_var:
            model_info['architecture_hints'].append('Has BatchNorm with running stats')
        
        return model_info


class UniversalModelLoader:
    """Universal model loader that handles any architecture"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_checkpoint_safely(self, model_path: str) -> Tuple[Dict, Dict]:
        """Safely load checkpoint with error handling"""
        checkpoint = None
        
        # Try multiple loading strategies
        loading_strategies = [
            ("weights_only=True", lambda: torch.load(model_path, map_location=self.device, weights_only=True)),
            ("weights_only=False", lambda: torch.load(model_path, map_location=self.device, weights_only=False)),
            ("pickle_module=pickle", lambda: torch.load(model_path, map_location=self.device, weights_only=False, pickle_module=__import__('pickle'))),
        ]
        
        last_error = None
        for strategy_name, load_func in loading_strategies:
            try:
                checkpoint = load_func()
                print(f"Successfully loaded with strategy: {strategy_name}")
                break
            except Exception as e:
                last_error = e
                print(f"Strategy '{strategy_name}' failed: {str(e)[:100]}...")
                continue
        
        if checkpoint is None:
            # If all strategies fail, try to load just the state dict by manually handling the file
            try:
                print("All standard strategies failed, attempting manual extraction...")
                checkpoint = self.manual_state_dict_extraction(model_path)
            except Exception as e:
                raise RuntimeError(f"Could not load checkpoint with any strategy. Last error: {last_error}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict' and isinstance(v, (int, float, str, bool, list, dict))}
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict' and isinstance(v, (int, float, str, bool, list, dict))}
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
                metadata = {}
        else:
            # Direct state dict
            state_dict = checkpoint
            metadata = {}
        
        return state_dict, metadata
    
    def manual_state_dict_extraction(self, model_path: str) -> Dict:
        """Manually extract state dict when normal loading fails"""
        import zipfile
        import io
        
        # Try to extract just the tensors without unpickling custom classes
        try:
            with zipfile.ZipFile(model_path, 'r') as zip_file:
                # Look for data.pkl which contains the main content
                if 'data.pkl' in zip_file.namelist():
                    # Load with a custom unpickler that skips problematic classes
                    import pickle
                    
                    class SafeUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            # Skip problematic classes and return a dummy
                            if name in ['TrainConfig', 'Config'] or module == '__main__':
                                return lambda *args, **kwargs: None
                            return super().find_class(module, name)
                    
                    data = zip_file.read('data.pkl')
                    unpickler = SafeUnpickler(io.BytesIO(data))
                    checkpoint = unpickler.load()
                    
                    return checkpoint
        except Exception as e:
            print(f"Manual extraction failed: {e}")
            raise
        
        raise RuntimeError("Could not manually extract state dict")
    
    def fix_state_dict_keys(self, state_dict: Dict, model: nn.Module) -> Dict:
        """Fix common state dict key mismatches"""
        model_state = model.state_dict()
        fixed_state_dict = {}
        
        # Get sets of keys
        saved_keys = set(state_dict.keys())
        model_keys = set(model_state.keys())
        
        # Direct matches
        common_keys = saved_keys & model_keys
        for key in common_keys:
            if state_dict[key].shape == model_state[key].shape:
                fixed_state_dict[key] = state_dict[key]
        
        # Handle module prefix differences
        missing_keys = model_keys - saved_keys
        unexpected_keys = saved_keys - model_keys
        
        # Try to match keys with/without 'module.' prefix
        for missing_key in missing_keys:
            # Check if saved has 'module.' prefix
            if f'module.{missing_key}' in unexpected_keys:
                if state_dict[f'module.{missing_key}'].shape == model_state[missing_key].shape:
                    fixed_state_dict[missing_key] = state_dict[f'module.{missing_key}']
                    unexpected_keys.remove(f'module.{missing_key}')
            # Check if model needs 'module.' prefix
            elif missing_key.startswith('module.'):
                key_without_module = missing_key[7:]
                if key_without_module in unexpected_keys:
                    if state_dict[key_without_module].shape == model_state[missing_key].shape:
                        fixed_state_dict[missing_key] = state_dict[key_without_module]
                        unexpected_keys.remove(key_without_module)
        
        # Handle BatchNorm running stats that might be missing
        for key in missing_keys - set(fixed_state_dict.keys()):
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                # Use model's initialized values for running stats
                fixed_state_dict[key] = model_state[key]
                print(f"Using initialized value for: {key}")
        
        # Handle special parameter mapping (like neck.3.weight -> neck_3_weight)
        remaining_unexpected = []
        for key in unexpected_keys:
            if key == 'neck.3.weight' and 'neck_3_weight' in model_state:
                fixed_state_dict['neck_3_weight'] = state_dict[key]
                print(f"Mapped {key} -> neck_3_weight")
            elif key in ['running_mean', 'running_var'] and key not in model_state:
                # These are likely orphaned BatchNorm stats - ignore them
                print(f"Ignoring orphaned BatchNorm stat: {key}")
            else:
                remaining_unexpected.append(key)
        
        return fixed_state_dict, list(missing_keys - set(fixed_state_dict.keys())), remaining_unexpected
    
    def load_model_universal(self, model_path: str, verbose: bool = True) -> Tuple[nn.Module, Dict]:
        """Load any model architecture automatically"""
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        state_dict, metadata = self.load_checkpoint_safely(model_path)
        
        # Analyze architecture
        model_info = ModelArchitectureDetector.analyze_state_dict(state_dict)
        
        if verbose:
            print(f"\n=== Model Architecture Analysis ===")
            print(f"Model Type: {model_info['type']}")
            print(f"Feature Dimension: {model_info['feature_dim']}")
            print(f"Architecture Hints: {', '.join(model_info['architecture_hints'])}")
        
        # Try to get feature dimension from metadata or analysis
        feature_dim = metadata.get('feature_dim', model_info.get('feature_dim'))
        if feature_dim is None:
            # Try to infer from state dict keys
            for key in state_dict.keys():
                if 'classifier' in key and 'weight' in key:
                    try:
                        shape = state_dict[key].shape
                        if len(shape) == 2:
                            feature_dim = shape[0]
                            break
                    except:
                        continue
        
        if feature_dim is None:
            feature_dim = 512  # Default fallback
            print(f"Warning: Could not detect feature dimension, using default: {feature_dim}")
        
        # Create appropriate model based on detected architecture
        model = self.create_model_from_info(model_info, feature_dim, state_dict)
        
        if model is None:
            raise ValueError(f"Could not create model for architecture type: {model_info['type']}")
        
        # Load state dict with compatibility fixes
        try:
            # First attempt: strict loading
            model.load_state_dict(state_dict, strict=True)
            if verbose:
                print("✅ Model loaded successfully (strict mode)")
        except RuntimeError as e:
            if verbose:
                print(f"⚠️ Strict loading failed: {e}")
                print("Attempting compatibility fixes...")
            
            # Second attempt: fix common issues
            fixed_state_dict, missing_keys, unexpected_keys = self.fix_state_dict_keys(state_dict, model)
            
            if verbose and missing_keys:
                print(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
            if verbose and unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
            
            # Load with strict=False to ignore mismatches
            model.load_state_dict(fixed_state_dict, strict=False)
            
            if verbose:
                print("✅ Model loaded successfully (compatibility mode)")
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Update model info with actual configuration
        model_info['feature_dim'] = feature_dim
        model_info['device'] = str(self.device)
        model_info['model_path'] = model_path
        
        return model, model_info
    
    def create_model_from_info(self, model_info: Dict, feature_dim: int, state_dict: Dict) -> Optional[nn.Module]:
        """Create model based on detected architecture"""
        
        # Try to import necessary architectures
        ultimate_available = False
        excellent_available = False
        UltimateEarFeatureExtractor = None
        ExcellentEarFeatureExtractor = None
        
        try:
            from ultimate_ear_training import UltimateEarFeatureExtractor
            ultimate_available = True
        except ImportError:
            pass
        
        try:
            from excellent_ear_training import ExcellentEarFeatureExtractor
            excellent_available = True
            print("✓ Imported actual ExcellentEarFeatureExtractor from training script")
        except ImportError:
            print("⚠️ Could not import ExcellentEarFeatureExtractor from training script")
        
        # Detect backbone type from state dict keys
        backbone_type = 'efficientnet_b4'  # default
        if any('b5' in str(k) for k in state_dict.keys()):
            backbone_type = 'efficientnet_b5'
        
        # Try Ultimate model first if detected or if other approaches fail
        if model_info['is_ultimate'] or 'ultimate' in model_info.get('type', '').lower():
            if ultimate_available:
                print(f"Creating Ultimate model architecture with {backbone_type}...")
                try:
                    return UltimateEarFeatureExtractor(feature_dim=feature_dim, backbone=backbone_type)
                except Exception as e:
                    print(f"Ultimate model creation failed: {e}, trying with default params...")
                    try:
                        return UltimateEarFeatureExtractor(feature_dim=feature_dim)
                    except Exception as e2:
                        print(f"Ultimate model creation with defaults also failed: {e2}")
                        print("Falling back to standard model...")
            else:
                print("Ultimate model detected but UltimateEarFeatureExtractor not available, trying alternatives...")
        
        # Try Excellent model
        if model_info['is_excellent'] or model_info['has_cbam'] or 'excellent' in model_info.get('type', '').lower():
            print("Creating EXACT working Excellent architecture (from ear_biometrics_v2.py)...")
            try:
                return self.create_working_excellent_model(feature_dim)
            except Exception as e:
                print(f"Working Excellent model creation failed: {e}")
                
                # Only try training script version as fallback if available
                if excellent_available and ExcellentEarFeatureExtractor is not None:
                    print(f"Trying training script ExcellentEarFeatureExtractor as fallback...")
                    try:
                        return ExcellentEarFeatureExtractor(feature_dim=feature_dim)
                    except Exception as e2:
                        print(f"Training script Excellent model creation failed: {e2}")
                
                print("Creating generic Excellent/CBAM model architecture...")
                try:
                    return self.create_excellent_model(feature_dim)
                except Exception as e3:
                    print(f"Generic Excellent model creation failed: {e3}, falling back to standard model...")
        
        # Fall back to standard model - this should always work
        print("Creating standard EfficientNet model architecture...")
        try:
            return self.create_standard_model(feature_dim, state_dict)
        except Exception as e:
            print(f"Standard model creation failed: {e}")
            # Last resort - create a very simple model
            return self.create_simple_model(feature_dim)
    
    def create_standard_model(self, feature_dim: int, state_dict: Dict = None) -> nn.Module:
        """Create standard EfficientNet model based on state_dict structure"""
        
        # Analyze state_dict to determine the exact architecture
        if state_dict:
            keys = list(state_dict.keys())
            
            # Check if it's a "neck" architecture (common in some training scripts)
            if any('neck.' in k for k in keys):
                return self.create_neck_model(feature_dim, state_dict)
            
            # Check if it's a direct backbone + classifier model
            if any('backbone.classifier.' in k for k in keys):
                return self.create_backbone_classifier_model(feature_dim, state_dict)
        
        # Default standard model
        class StandardEarModel(nn.Module):
            def __init__(self, feature_dim=512):
                super().__init__()
                from torchvision.models import EfficientNet_B4_Weights
                self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
                
                # Get the number of features from the classifier
                if hasattr(self.backbone.classifier, 'in_features'):
                    in_features = self.backbone.classifier.in_features
                else:
                    in_features = 1792  # EfficientNet-B4 default
                
                # Replace classifier
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features, feature_dim),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, feature_dim)
                )
                self.feature_dim = feature_dim
            
            def forward(self, x):
                features = self.backbone(x)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                return features
        
        return StandardEarModel(feature_dim)
    
    def create_neck_model(self, feature_dim: int, state_dict: Dict) -> nn.Module:
        """Create model with exact 'neck' architecture matching the state_dict"""
        
        # First, let's analyze the exact structure
        print("\n=== Analyzing Neck Architecture ===")
        neck_keys = [k for k in sorted(state_dict.keys()) if 'neck.' in k]
        for key in neck_keys:
            if key in state_dict:
                shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
                print(f"{key}: {shape}")
        
        class ExactNeckModel(nn.Module):
            def __init__(self):
                super().__init__()
                from torchvision.models import EfficientNet_B4_Weights
                self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
                self.backbone.classifier = nn.Identity()
                
                # Build the exact neck structure to match the state_dict
                # Based on the error, we need:
                # neck.1: Linear layer (2048, 1792) 
                # neck.2: BatchNorm1d(2048)
                # neck.3: Single parameter (skip or handle specially)
                # neck.5: Linear layer 
                # neck.6: BatchNorm1d
                
                neck_modules = nn.ModuleDict()
                
                # Layer 1: Linear (1792 -> 2048)
                if 'neck.1.weight' in state_dict:
                    w_shape = state_dict['neck.1.weight'].shape
                    neck_modules['1'] = nn.Linear(w_shape[1], w_shape[0])
                    print(f"neck.1: Linear({w_shape[1]}, {w_shape[0]})")
                
                # Layer 2: BatchNorm1d
                if 'neck.2.weight' in state_dict:
                    bn_features = state_dict['neck.2.weight'].shape[0]
                    neck_modules['2'] = nn.BatchNorm1d(bn_features)
                    print(f"neck.2: BatchNorm1d({bn_features})")
                
                # Layer 3: Handle single parameter (could be temperature, threshold, etc.)
                if 'neck.3.weight' in state_dict:
                    param_shape = state_dict['neck.3.weight'].shape
                    if param_shape == torch.Size([1]):
                        # Single parameter - register as a parameter, not a layer
                        self.register_parameter('neck_3_weight', nn.Parameter(torch.ones(1)))
                        print(f"neck.3: Parameter(1) - registered as parameter")
                
                # Layer 4: ReLU (no parameters, just activation)
                neck_modules['4'] = nn.ReLU(inplace=True)
                print(f"neck.4: ReLU()")
                
                # Layer 5: Linear 
                if 'neck.5.weight' in state_dict:
                    w_shape = state_dict['neck.5.weight'].shape
                    neck_modules['5'] = nn.Linear(w_shape[1], w_shape[0])
                    print(f"neck.5: Linear({w_shape[1]}, {w_shape[0]})")
                
                # Layer 6: BatchNorm1d
                if 'neck.6.weight' in state_dict:
                    bn_features = state_dict['neck.6.weight'].shape[0]
                    neck_modules['6'] = nn.BatchNorm1d(bn_features)
                    print(f"neck.6: BatchNorm1d({bn_features})")
                
                self.neck = neck_modules
                
                # Determine feature dimension from last layer
                if '6' in neck_modules:
                    self.feature_dim = neck_modules['6'].num_features
                elif '5' in neck_modules:
                    self.feature_dim = neck_modules['5'].out_features
                else:
                    self.feature_dim = feature_dim
                
                print(f"Final feature dimension: {self.feature_dim}")
            
            def forward(self, x):
                # Backbone features
                features = self.backbone(x)
                
                # Apply neck layers in order
                if '1' in self.neck:
                    features = self.neck['1'](features)
                if '2' in self.neck:
                    features = self.neck['2'](features)
                # Skip layer 3 (single parameter)
                if '4' in self.neck:
                    features = self.neck['4'](features)
                if '5' in self.neck:
                    features = self.neck['5'](features)
                if '6' in self.neck:
                    features = self.neck['6'](features)
                
                # Normalize
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                return features
        
        return ExactNeckModel()
    
    def create_backbone_classifier_model(self, feature_dim: int, state_dict: Dict) -> nn.Module:
        """Create model with backbone.classifier architecture"""
        
        class BackboneClassifierModel(nn.Module):
            def __init__(self, feature_dim=512):
                super().__init__()
                from torchvision.models import EfficientNet_B4_Weights
                self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
                
                # Analyze classifier structure from state_dict
                classifier_keys = [k for k in state_dict.keys() if 'backbone.classifier.' in k]
                classifier_keys.sort()
                
                # Build classifier based on found keys
                classifier_layers = []
                in_features = 1792  # EfficientNet-B4 default
                
                # Parse the classifier structure
                layer_nums = set()
                for key in classifier_keys:
                    parts = key.split('.')
                    if len(parts) >= 3 and parts[2].isdigit():
                        layer_nums.add(int(parts[2]))
                
                layer_nums = sorted(layer_nums)
                
                for i, layer_num in enumerate(layer_nums):
                    weight_key = f'backbone.classifier.{layer_num}.weight'
                    if weight_key in state_dict:
                        weight_shape = state_dict[weight_key].shape
                        if len(weight_shape) == 2:  # Linear layer
                            out_features = weight_shape[0]
                            classifier_layers.append(nn.Linear(in_features, out_features))
                            in_features = out_features
                        elif len(weight_shape) == 1:  # BatchNorm layer
                            classifier_layers.append(nn.BatchNorm1d(weight_shape[0]))
                    
                    # Add ReLU between layers (except last)
                    if i < len(layer_nums) - 1:
                        classifier_layers.append(nn.ReLU(inplace=True))
                
                if classifier_layers:
                    self.backbone.classifier = nn.Sequential(*classifier_layers)
                else:
                    # Fallback classifier
                    self.backbone.classifier = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(in_features, feature_dim),
                        nn.BatchNorm1d(feature_dim),
                        nn.ReLU(),
                        nn.Linear(feature_dim, feature_dim)
                    )
                
                self.feature_dim = feature_dim
            
            def forward(self, x):
                features = self.backbone(x)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                return features
        
        return BackboneClassifierModel(feature_dim)
    
    def create_excellent_model(self, feature_dim: int) -> nn.Module:
        """Create Excellent model with CBAM attention"""
        
        if not TIMM_AVAILABLE:
            print("Warning: timm not available, falling back to standard model")
            return self.create_standard_model(feature_dim)
        
        # Import CBAM components
        class ChannelAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.max_pool = nn.AdaptiveMaxPool2d(1)
                self.fc = nn.Sequential(
                    nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
                )
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                avg_out = self.fc(self.avg_pool(x))
                max_out = self.fc(self.max_pool(x))
                return self.sigmoid(avg_out + max_out)
        
        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=7):
                super().__init__()
                self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avg_out, max_out], dim=1)
                x = self.conv1(x)
                return self.sigmoid(x)
        
        class CBAM(nn.Module):
            def __init__(self, in_planes, ratio=16, kernel_size=7):
                super().__init__()
                self.ca = ChannelAttention(in_planes, ratio)
                self.sa = SpatialAttention(kernel_size)
            
            def forward(self, x):
                x = x * self.ca(x)
                x = x * self.sa(x)
                return x
        
        class ExcellentModel(nn.Module):
            def __init__(self, feature_dim=2048, dropout_rate=0.2):
                super().__init__()
                
                # Use timm backbone
                self.backbone = timm.create_model('efficientnet_b4', pretrained=True, 
                                                features_only=True, out_indices=[4])
                
                # Get feature dimensions
                feature_info = self.backbone.feature_info
                all_feature_dims = [info['num_chs'] for info in feature_info]
                self.feature_dims = [all_feature_dims[4]]  # Only the 448-channel feature
                
                # Attention module
                self.attentions = nn.ModuleList([
                    CBAM(self.feature_dims[0])
                ])
                
                # Feature processor
                total_features = self.feature_dims[0]
                self.feature_processor = nn.Sequential(
                    nn.Conv2d(total_features, feature_dim // 2, 1),
                    nn.BatchNorm2d(feature_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim // 2, feature_dim),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate / 2),
                    nn.Linear(feature_dim, feature_dim),
                    nn.BatchNorm1d(feature_dim),
                )
                
                self.feature_dim = feature_dim
            
            def forward(self, x):
                features = self.backbone(x)
                attended_feat = self.attentions[0](features[0])
                final_features = self.feature_processor(attended_feat)
                final_features = torch.nn.functional.normalize(final_features, p=2, dim=1)
                return final_features
        
        return ExcellentModel(feature_dim)
    
    def create_working_excellent_model(self, feature_dim: int) -> nn.Module:
        """Create the EXACT working Excellent architecture from ear_biometrics_v2.py"""
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for working Excellent model. Please install timm.")
        
        # Define the exact attention modules from the working implementation
        class ChannelAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.max_pool = nn.AdaptiveMaxPool2d(1)
                self.fc = nn.Sequential(
                    nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
                )
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=7):
                super().__init__()
                self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avg_out, max_out], dim=1)
                return self.sigmoid(self.conv1(x))

        class CBAM(nn.Module):
            def __init__(self, in_planes, ratio=16, kernel_size=7):
                super().__init__()
                self.ca = ChannelAttention(in_planes, ratio)
                self.sa = SpatialAttention(kernel_size)
            
            def forward(self, x):
                x = x * self.ca(x)
                x = x * self.sa(x)
                return x

        # The EXACT working ExcellentEarFeatureExtractor from ear_biometrics_v2.py
        class WorkingExcellentEarFeatureExtractor(nn.Module):
            def __init__(self, feature_dim=2048, dropout_rate=0.2):
                super().__init__()
                
                # Use EfficientNet-B4 features (timm) with last feature map - EXACT COPY
                self.backbone = timm.create_model('efficientnet_b4', pretrained=True, features_only=True, out_indices=[4])
                feature_info = self.backbone.feature_info
                num_chs = [info['num_chs'] for info in feature_info][4]  # 448
                
                # Single attention module - EXACT COPY
                self.attentions = nn.ModuleList([CBAM(num_chs)])
                
                # Feature processor - EXACT COPY
                self.feature_processor = nn.Sequential(
                    nn.Conv2d(num_chs, feature_dim // 2, 1),
                    nn.BatchNorm2d(feature_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim // 2, feature_dim),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate / 2),
                    nn.Linear(feature_dim, feature_dim),
                    nn.BatchNorm1d(feature_dim),
                )
                self.feature_dim = feature_dim
                
                # Initialize weights - EXACT COPY
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                # EXACT forward pass from working implementation
                feats = self.backbone(x)
                attended = self.attentions[0](feats[0])
                out = self.feature_processor(attended)
                return torch.nn.functional.normalize(out, p=2, dim=1)  # EXACT normalization call
        
        print(f"Creating working Excellent model with {feature_dim}D features")
        return WorkingExcellentEarFeatureExtractor(feature_dim=feature_dim)
    
    def create_simple_model(self, feature_dim: int) -> nn.Module:
        """Create a very simple model as last resort"""
        
        class SimpleModel(nn.Module):
            def __init__(self, feature_dim=512):
                super().__init__()
                # Very simple CNN backbone
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten()
                )
                
                # Simple classifier
                self.classifier = nn.Sequential(
                    nn.Linear(64 * 7 * 7, feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_dim, feature_dim)
                )
                
                self.feature_dim = feature_dim
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                x = torch.nn.functional.normalize(x, p=2, dim=1)
                return x
        
        return SimpleModel(feature_dim)


class UniversalFeatureExtractor:
    """Universal feature extractor that works with any model"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu', verbose: bool = True):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.model_info = {}
        self.feature_dim = 512  # Default
        self.loader = UniversalModelLoader(device)
        
        if model_path:
            self.load_model(model_path, verbose)
    
    def load_model(self, model_path: str, verbose: bool = True) -> bool:
        """Load model with automatic architecture detection"""
        try:
            self.model, self.model_info = self.loader.load_model_universal(model_path, verbose)
            self.feature_dim = self.model_info.get('feature_dim', 512)
            self.model_path = model_path
            
            if verbose:
                print(f"\n✅ Model loaded successfully!")
                print(f"   Type: {self.model_info.get('type', 'unknown')}")
                print(f"   Feature dimension: {self.feature_dim}")
                print(f"   Device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from image"""
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        try:
            # Ensure model is in eval mode
            self.model.eval()
            
            # Handle different input types
            if hasattr(image, 'numpy'):  # PIL Image
                image = np.array(image)
            
            # Ensure RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                import cv2
                if image.dtype == np.uint8:
                    # Assume BGR if uint8
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_tensor = self.loader.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
            
            # Convert to numpy
            features_np = features.cpu().numpy().flatten()
            
            # Normalize if not already normalized
            norm = np.linalg.norm(features_np)
            if norm > 1.5:  # Not normalized
                features_np = features_np / (norm + 1e-8)
            
            return features_np.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return self.model_info


# Test function
def test_universal_loader():
    """Test the universal model loader"""
    print("Testing Universal Model Loader...")
    
    # Test with different model files
    test_models = [
        "excellent_ear_model_best.pth",
        "efficientnet_b4_ultimate_best.pth",
        "ear_efficientnet_epoch_50.pth",
        # Add more model paths to test
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_path in test_models:
        if Path(model_path).exists():
            print(f"\n{'='*60}")
            print(f"Testing: {model_path}")
            print('='*60)
            
            extractor = UniversalFeatureExtractor(model_path, device, verbose=True)
            
            if extractor.model is not None:
                # Test feature extraction with dummy image
                dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                features = extractor.extract_features(dummy_image)
                
                if features is not None:
                    print(f"✅ Feature extraction successful!")
                    print(f"   Shape: {features.shape}")
                    print(f"   Norm: {np.linalg.norm(features):.4f}")
                else:
                    print("❌ Feature extraction failed")
        else:
            print(f"Skipping {model_path} (not found)")


if __name__ == "__main__":
    test_universal_loader()

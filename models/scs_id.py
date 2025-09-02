# models/scs_id.py - FIXED VERSION
"""
SCS-ID: Squeezed ConvSeek for Intrusion Detection
Lightweight CNN architecture combining SqueezeNet efficiency with ConvSeek pattern extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FireModule(nn.Module):
    """
    Fire module from SqueezeNet - key component for parameter reduction
    """
    def __init__(self, input_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        
        # Squeeze layer
        self.squeeze = nn.Conv1d(input_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Expand layers
        self.expand1x1 = nn.Conv1d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv1d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for computational efficiency
    """
    def __init__(self, input_channels, output_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            input_channels, input_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=input_channels
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SCSIDModel(nn.Module):
    """
    SCS-ID: Squeezed ConvSeek Model
    Combines SqueezeNet efficiency with optimized convolutional operations
    """
    def __init__(self, input_features=42, num_classes=15):
        super(SCSIDModel, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        
        # Input transformation - convert 1D features to appropriate format
        # Input: [batch, 1, features, 1] -> [batch, 8, features]
        self.input_conv = nn.Conv2d(1, 8, kernel_size=(1, 1))
        self.input_bn = nn.BatchNorm2d(8)
        
        # Reshape to 1D for efficient processing
        # After reshape: [batch, 8, features]
        
        # Fire modules for parameter efficiency (SqueezeNet inspired)
        self.fire1 = FireModule(8, 4, 8, 8)      # 16 output channels
        self.fire2 = FireModule(16, 8, 16, 16)   # 32 output channels
        self.fire3 = FireModule(32, 8, 16, 16)   # 32 output channels
        
        # Depthwise separable convolutions for spatial-temporal patterns
        self.depthwise_conv1 = DepthwiseSeparableConv1d(32, 64, kernel_size=3, padding=1)
        self.depthwise_conv2 = DepthwiseSeparableConv1d(64, 32, kernel_size=3, padding=1)
        
        # Batch normalization and activation
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        
        # Global pooling for dimensionality reduction
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final classification layers
        self.fc1 = nn.Linear(64, 128)  # 32 (avg) + 32 (max) = 64
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Final dropout and batch norm
        self.fc_dropout = nn.Dropout(0.5)
        self.fc_bn = nn.BatchNorm1d(128)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: [batch, 1, features, 1] (from main training loop)
        batch_size = x.size(0)
        
        # Initial convolution and normalization
        x = self.input_conv(x)          # [batch, 8, features, 1]
        x = self.input_bn(x)
        x = F.relu(x)
        
        # Reshape for 1D convolutions
        x = x.squeeze(-1)               # [batch, 8, features]
        
        # Fire modules for parameter efficiency
        x = self.fire1(x)               # [batch, 16, features]
        x = self.fire2(x)               # [batch, 32, features]
        x = self.fire3(x)               # [batch, 32, features]
        
        # Depthwise separable convolutions
        x = self.depthwise_conv1(x)     # [batch, 64, features]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.depthwise_conv2(x)     # [batch, 32, features]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # [batch, 32]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [batch, 32]
        
        # Combine pooling results
        x = torch.cat([avg_pool, max_pool], dim=1)      # [batch, 64]
        
        # Final classification layers
        x = self.fc1(x)                 # [batch, 128]
        x = self.fc_bn(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        
        x = self.fc2(x)                 # [batch, 64]
        x = F.relu(x)
        
        x = self.fc3(x)                 # [batch, num_classes]
        
        return x
    
    def count_parameters(self):
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_feature_maps(self, x):
        """Extract feature maps for analysis"""
        feature_maps = {}
        
        # Input processing
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = x.squeeze(-1)
        feature_maps['input_processed'] = x
        
        # Fire modules
        x = self.fire1(x)
        feature_maps['fire1'] = x
        
        x = self.fire2(x)
        feature_maps['fire2'] = x
        
        x = self.fire3(x)
        feature_maps['fire3'] = x
        
        # Depthwise convolutions
        x = self.depthwise_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        feature_maps['depthwise1'] = x
        
        x = self.depthwise_conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        feature_maps['depthwise2'] = x
        
        return feature_maps

def create_scs_id_model(input_features=42, num_classes=15, apply_pruning=False, pruning_ratio=0.3):
    """
    Factory function to create SCS-ID model with optional pruning
    """
    model = SCSIDModel(input_features=input_features, num_classes=num_classes)
    
    if apply_pruning:
        model = apply_structured_pruning(model, pruning_ratio)
    
    return model

def apply_structured_pruning(model, pruning_ratio=0.3):
    """
    Apply structured pruning to reduce model parameters by specified ratio
    """
    print(f"🔧 Applying structured pruning (ratio: {pruning_ratio})")
    
    try:
        import torch.nn.utils.prune as prune
        
        # Get all conv and linear layers
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                modules_to_prune.append((module, 'weight'))
        
        # Apply pruning
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Remove pruning re-parametrization to make it permanent
        for module, param in modules_to_prune:
            prune.remove(module, param)
        
        # Count remaining parameters
        total_params, trainable_params = model.count_parameters()
        print(f"   ✅ Pruning complete. Remaining parameters: {trainable_params:,}")
        
    except ImportError:
        print("   ⚠️ Pruning not available. Continuing without pruning.")
    
    return model

def apply_quantization(model):
    """
    Apply INT8 quantization for inference efficiency
    """
    print("🔧 Applying INT8 quantization...")
    
    try:
        # Prepare model for quantization
        model.eval()
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d, nn.Conv2d},
            dtype=torch.qint8
        )
        
        print("   ✅ Quantization complete")
        return model_quantized
        
    except Exception as e:
        print(f"   ⚠️ Quantization failed: {e}. Continuing without quantization.")
        return model

# Model testing and validation
def test_scs_id_model():
    """Test SCS-ID model with sample data"""
    print("🧪 Testing SCS-ID Model")
    print("=" * 40)
    
    # Create model
    model = create_scs_id_model(input_features=42, num_classes=15)
    
    # Count parameters
    total_params, trainable_params = model.count_parameters()
    print(f"📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 32
    input_features = 42
    
    # Input format expected by the model: [batch, 1, features, 1]
    sample_input = torch.randn(batch_size, 1, input_features, 1)
    
    print(f"\n🔄 Forward Pass Test:")
    print(f"   Input shape: {sample_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test with pruning
    print(f"\n🔧 Testing with pruning:")
    model_pruned = create_scs_id_model(input_features=42, num_classes=15, 
                                      apply_pruning=True, pruning_ratio=0.3)
    
    total_params_pruned, trainable_params_pruned = model_pruned.count_parameters()
    reduction = (1 - trainable_params_pruned / trainable_params) * 100
    
    print(f"   Pruned parameters: {trainable_params_pruned:,}")
    print(f"   Parameter reduction: {reduction:.1f}%")
    
    # Test quantization
    print(f"\n🔧 Testing quantization:")
    model_quantized = apply_quantization(model)
    
    with torch.no_grad():
        output_quantized = model_quantized(sample_input)
        print(f"   Quantized output shape: {output_quantized.shape}")
    
    print("\n✅ SCS-ID model test completed successfully!")
    
    return model

if __name__ == "__main__":
    test_scs_id_model()
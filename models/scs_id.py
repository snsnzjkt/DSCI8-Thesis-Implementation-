# models/scs_id.py - SCS-ID: Squeezed ConvSeek for Intrusion Detection
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

try:
    from config import config
except ImportError:
    # Fallback config
    class Config:
        NUM_CLASSES = 15
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        SELECTED_FEATURES = 42
        PRUNING_RATIO = 0.3
    config = Config()

class FireModule(nn.Module):
    """
    SqueezeNet Fire Module optimized for network traffic analysis
    Based on Iandola et al. (2016) with adaptations for 1D network features
    """
    def __init__(self, input_channels: int, squeeze_channels: int, 
                 expand1x1_channels: int, expand3x3_channels: int):
        super(FireModule, self).__init__()
        
        # Squeeze layer - reduce dimensionality
        self.squeeze = nn.Conv1d(input_channels, squeeze_channels, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm1d(squeeze_channels)
        
        # Expand layers - capture different scales of patterns
        self.expand_1x1 = nn.Conv1d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand_1x1_bn = nn.BatchNorm1d(expand1x1_channels)
        
        self.expand_3x3 = nn.Conv1d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand_3x3_bn = nn.BatchNorm1d(expand3x3_channels)
        
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Squeeze phase
        squeeze_out = self.activation(self.squeeze_bn(self.squeeze(x)))
        squeeze_out = self.dropout(squeeze_out)
        
        # Expand phase
        expand_1x1_out = self.activation(self.expand_1x1_bn(self.expand_1x1(squeeze_out)))
        expand_3x3_out = self.activation(self.expand_3x3_bn(self.expand_3x3(squeeze_out)))
        
        # Concatenate expand outputs
        return torch.cat([expand_1x1_out, expand_3x3_out], dim=1)

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution for 1D network traffic data
    Based on Howard et al. (2017) MobileNet concept
    Reduces parameters while maintaining representational power
    """
    def __init__(self, input_channels: int, output_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(DepthwiseSeparableConv1d, self).__init__()
        
        # Depthwise convolution - each input channel convolved separately
        self.depthwise = nn.Conv1d(
            input_channels, input_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            groups=input_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm1d(input_channels)
        
        # Pointwise convolution - 1x1 conv to combine features
        self.pointwise = nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm1d(output_channels)
        
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.activation(self.depthwise_bn(self.depthwise(x)))
        x = self.activation(self.pointwise_bn(self.pointwise(x)))
        return x

class ConvSeekBlock(nn.Module):
    """
    ConvSeek mechanism for efficient pattern extraction in network traffic
    Combines attention mechanisms with efficient convolutions
    """
    def __init__(self, input_channels: int, output_channels: int):
        super(ConvSeekBlock, self).__init__()
        
        self.depthwise_sep = DepthwiseSeparableConv1d(input_channels, output_channels)
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(output_channels, output_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels // 4, output_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(output_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply depthwise separable convolution
        conv_out = self.depthwise_sep(x)
        
        # Apply channel attention
        channel_att = self.channel_attention(conv_out)
        conv_out = conv_out * channel_att
        
        # Apply spatial attention
        spatial_att = self.spatial_attention(conv_out)
        conv_out = conv_out * spatial_att
        
        return conv_out

class SCS_ID(nn.Module):
    """
    SCS-ID: Squeezed ConvSeek for Intrusion Detection
    
    Combines SqueezeNet efficiency with ConvSeek pattern extraction
    Optimized for campus network intrusion detection
    
    Key Features:
    - Fire modules for parameter efficiency
    - Depthwise separable convolutions for computational efficiency
    - ConvSeek blocks for enhanced pattern recognition
    - Structured pruning support
    - INT8 quantization ready
    """
    
    def __init__(self, input_features: int = 42, num_classes: int = 15, 
                 dropout_rate: float = 0.2):
        super(SCS_ID, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        
        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Fire modules for efficient feature extraction
        self.fire1 = FireModule(64, 16, 64, 64)    # Output: 128 channels
        self.fire2 = FireModule(128, 16, 64, 64)   # Output: 128 channels
        
        # ConvSeek blocks for pattern recognition
        self.convseek1 = ConvSeekBlock(128, 64)
        self.convseek2 = ConvSeekBlock(64, 32)
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head with structured pruning support
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using He initialization for ReLU networks"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through SCS-ID model
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reshape for 1D convolution: (batch_size, 1, features)
        x = x.unsqueeze(1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Fire modules for efficient feature extraction
        x = self.fire1(x)
        x = self.fire2(x)
        
        # ConvSeek blocks for pattern recognition
        x = self.convseek1(x)
        x = self.convseek2(x)
        
        # Global feature aggregation
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x):
        """Extract intermediate feature maps for analysis"""
        features = {}
        
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        features['input_proj'] = x
        
        x = self.fire1(x)
        features['fire1'] = x
        
        x = self.fire2(x)
        features['fire2'] = x
        
        x = self.convseek1(x)
        features['convseek1'] = x
        
        x = self.convseek2(x)
        features['convseek2'] = x
        
        return features
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def apply_structured_pruning(self, pruning_ratio: float = 0.3):
        """
        Apply structured pruning to reduce model size
        Based on L1-norm importance of filters
        """
        print(f"Applying structured pruning with ratio: {pruning_ratio}")
        
        # Get modules that can be pruned (Conv1d layers)
        prunable_modules = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d) and 'classifier' not in name:
                prunable_modules.append((name, module))
        
        # Calculate importance scores and prune
        for name, module in prunable_modules:
            if module.weight.size(0) > 8:  # Don't prune if too few filters
                num_filters = module.weight.size(0)
                num_to_prune = int(num_filters * pruning_ratio)
                
                if num_to_prune > 0:
                    # Calculate L1 norm of each filter
                    filter_norms = torch.norm(module.weight.data, p=1, dim=(1, 2))
                    
                    # Get indices of filters to keep (highest norms)
                    _, indices = torch.topk(filter_norms, num_filters - num_to_prune)
                    indices = indices.sort()[0]
                    
                    # Prune the filters
                    module.weight.data = module.weight.data[indices]
                    if module.bias is not None:
                        module.bias.data = module.bias.data[indices]
                    
                    print(f"Pruned {name}: {num_filters} -> {len(indices)} filters")
        
        print("Structured pruning completed")
    
    def prepare_for_quantization(self):
        """Prepare model for INT8 quantization"""
        # Fuse conv-bn-relu patterns for better quantization
        torch.quantization.fuse_modules_qat(self, [
            ['input_proj.0', 'input_proj.1', 'input_proj.2']
        ])
        
        # Set quantization config
        self.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self, inplace=True)
        
        print("Model prepared for quantization")

def create_scs_id_model(input_features: int = 42, num_classes: int = 15, 
                       apply_pruning: bool = True, pruning_ratio: float = 0.3):
    """
    Factory function to create optimized SCS-ID model
    
    Args:
        input_features: Number of input features (default 42 from DeepSeek RL)
        num_classes: Number of output classes
        apply_pruning: Whether to apply structured pruning
        pruning_ratio: Ratio of filters to prune
    
    Returns:
        Optimized SCS-ID model
    """
    model = SCS_ID(input_features, num_classes)
    
    if apply_pruning:
        model.apply_structured_pruning(pruning_ratio)
    
    total_params, trainable_params = model.count_parameters()
    print(f"SCS-ID Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Input features: {input_features}")
    print(f"  Output classes: {num_classes}")
    
    return model

# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    model = create_scs_id_model()
    
    # Test forward pass
    batch_size = 32
    input_features = 42
    x = torch.randn(batch_size, input_features)
    
    print(f"\nTesting forward pass:")
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
        # Test feature extraction
        features = model.get_feature_maps(x)
        print(f"\nFeature map shapes:")
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}")
    
    print("\nSCS-ID model test completed successfully!")
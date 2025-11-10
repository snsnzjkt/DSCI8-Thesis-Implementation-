"""
SCS-ID: Optimized Squeezed ConvSeek for Intrusion Detection
Enhanced for high accuracy and low FPR while maintaining efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChannelAttention(nn.Module):
    """Simplified channel attention module"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduced_channels = max(channels // reduction, 4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        # Global average pooling
        y = self.pool(x).view(b, c)
        # Channel-wise weights
        y = self.fc(y).view(b, c, 1)
        # Apply attention
        return x * y

class OptimizedFireModule(nn.Module):
    """Enhanced Fire Module with residual connections and attention"""
    def __init__(self, input_channels, squeeze_channels, expand_channels):
        super().__init__()
        
        # Fixed channel sizes
        self.input_channels = input_channels
        self.squeeze_channels = max(8, squeeze_channels)  # Minimum 8 channels
        self.expand_channels = max(16, expand_channels)  # Minimum 16 channels
        
        # Each branch gets half the expand channels
        self.branch_channels = self.expand_channels // 2
        
        # Squeeze path
        self.squeeze = nn.Sequential(
            nn.Conv1d(input_channels, self.squeeze_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.squeeze_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 expansion path
        self.expand1x1 = nn.Sequential(
            nn.Conv1d(self.squeeze_channels, self.branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 expansion path (depthwise separable)
        self.expand3x3 = nn.Sequential(
            nn.Conv1d(self.squeeze_channels, self.squeeze_channels, 
                     kernel_size=3, padding=1, groups=self.squeeze_channels, bias=False),
            nn.BatchNorm1d(self.squeeze_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.squeeze_channels, self.branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = ChannelAttention(expand_channels)
        
        # Residual connection if dimensions match
        self.residual = input_channels == expand_channels
        if not self.residual:
            self.residual_conv = nn.Conv1d(input_channels, expand_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        
        # Squeeze
        out = self.squeeze(x)
        
        # Expand
        out1x1 = self.expand1x1(out)
        out3x3 = self.expand3x3(out)
        
        # Combine expand paths
        out = torch.cat([out1x1, out3x3], dim=1)
        
        # Apply attention
        out = self.attention(out)
        
        # Optional residual connection
        if self.residual and self.input_channels == self.expand_channels:
            out = out + identity
        elif hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(identity)
            
        return out

class EnhancedConvSeekBlock(nn.Module):
    """Ultra-efficient ConvSeek block with extreme parameter reduction"""
    def __init__(self, input_channels, output_channels, kernel_size=3, dropout_rate=0.3):
        super().__init__()
        
        # Reduce intermediate channels aggressively
        intermediate_channels = max(min(input_channels // 8, output_channels // 8), 4)
        
        # Efficient depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size, 
                     padding=kernel_size//2, groups=input_channels, bias=False),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(inplace=True)
        )
        
        # Two-step pointwise with extreme reduction
        self.pointwise1 = nn.Conv1d(input_channels, intermediate_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.pointwise2 = nn.Conv1d(intermediate_channels, output_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(output_channels)
        
        self.attention = ChannelAttention(output_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Efficient residual connection
        self.residual = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else None

    def forward(self, x):
        identity = x
        
        # Multi-stage efficient convolution
        out = self.depthwise(x)
        out = self.pointwise1(out)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.pointwise2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        # Apply attention and dropout
        out = self.attention(out)
        out = self.dropout(out)
        
        # Optional residual connection
        if self.residual is not None:
            identity = self.residual(identity)
        
        return out + identity

class OptimizedSCSID(nn.Module):
    """
    Ultra-efficient SCS-ID architecture with strict parameter control:
    - Fixed channel progression
    - Efficient feature extraction
    - Stable gradients
    - Parameter efficiency
    """
    def __init__(self, input_features, num_classes, base_channels=32):
        super().__init__()
        
        # Fixed channel progression
        self.base_channels = 32  # Fixed base channels
        channels = [32, 48, 64]  # Progressive channel growth
        
        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, self.base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.base_channels),
            nn.ReLU(inplace=True)
        )
        self.input_attention = ChannelAttention(self.base_channels)
        
        # Feature extraction path
        self.fire1 = OptimizedFireModule(
            channels[0],  # 32
            channels[0] // 2,  # 16
            channels[1]  # 48
        )
        self.fire2 = OptimizedFireModule(
            channels[1],  # 48
            channels[1] // 2,  # 24
            channels[2]  # 64
        )
        
        channels = [32, 48, 64]  # Redefine for clarity
        
        # Feature refinement
        self.convseek1 = EnhancedConvSeekBlock(
            channels[2],  # 64
            channels[1]   # 48
        )
        self.convseek2 = EnhancedConvSeekBlock(
            channels[1],  # 48
            channels[0]   # 32
        )
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[1], bias=False),  # 32 -> 48
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Light dropout
            nn.Linear(channels[1], num_classes)  # 48 -> num_classes
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Reshape input: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)
        
        # Enhanced feature extraction
        x = self.input_conv(x)
        x = self.input_attention(x)
        
        # Fire modules with residual connections
        x = self.fire1(x)
        x = self.fire2(x)
        
        # ConvSeek blocks with attention
        x = self.convseek1(x)
        x = self.convseek2(x)
        
        # Global pooling and classification
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x

    def get_compression_stats(self):
        """Calculate model compression statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 32 / 8 / 1024  # Size in KB
        return {
            'total_parameters': total_params,
            'model_size_kb': total_size,
            'architecture': [
                (name, sum(p.numel() for p in module.parameters()))
                for name, module in self.named_children()
            ]
        }
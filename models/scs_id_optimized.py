"""
SCS-ID: Optimized Squeezed ConvSeek for Intrusion Detection
Enhanced for high accuracy and low FPR while maintaining efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChannelAttention(nn.Module):
    """Channel attention module for focusing on important feature channels"""
    def __init__(self, channels, reduction=8):  # Changed reduction from 16 to 8
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Ensure minimum intermediate channels
        reduced_channels = max(channels // reduction, 4)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class OptimizedFireModule(nn.Module):
    """
    Enhanced Fire Module with balanced efficiency-performance design
    
    Features:
    1. Optimized squeeze ratios for parameter reduction
    2. Channel attention for performance preservation
    3. Residual connections for stable training
    4. Memory-efficient operations
    5. Balanced channel reduction strategy
    """
    def __init__(self, input_channels, squeeze_channels, expand_channels):
        super().__init__()
        
        # Aggressive squeeze ratio for maximum parameter efficiency
        squeeze_ratio = 4  # Much more aggressive ratio
        squeeze_channels = max(input_channels // squeeze_ratio, 4)  # Reduced minimum channels
        
        # Memory-efficient squeeze operation
        self.squeeze = nn.Sequential(
            nn.Conv1d(input_channels, squeeze_channels, kernel_size=1),
            nn.BatchNorm1d(squeeze_channels),
            nn.ReLU(inplace=True)
        )
        
        # Balanced expand paths with optimized channel distribution
        # Calculate balanced expand channels
        total_expand = expand_channels  # Keep total channels as specified
        expand1x1_channels = total_expand // 2
        expand3x3_channels = total_expand - expand1x1_channels  # Ensure we use exactly total_expand channels
        
        self.expand1x1 = nn.Sequential(
            nn.Conv1d(squeeze_channels, expand1x1_channels, kernel_size=1),
            nn.BatchNorm1d(expand1x1_channels),
            nn.ReLU(inplace=True)
        )
        
        self.expand3x3 = nn.Sequential(
            nn.Conv1d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(expand3x3_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = ChannelAttention(expand_channels)
        
        # Residual connection if dimensions match
        self.residual = input_channels == expand_channels
        if not self.residual:
            self.residual_conv = nn.Conv1d(input_channels, expand_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        
        out = self.squeeze(x)
        out = torch.cat([self.expand1x1(out), self.expand3x3(out)], dim=1)
        out = self.attention(out)
        
        if self.residual:
            out += identity
        else:
            out += self.residual_conv(identity)
        
        return out

class EnhancedConvSeekBlock(nn.Module):
    """Improved ConvSeek block with attention and dynamic regularization"""
    def __init__(self, input_channels, output_channels, kernel_size=3, dropout_rate=0.3):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size, padding=kernel_size//2, groups=input_channels),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = ChannelAttention(output_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Efficient residual connection
        self.residual = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else None

    def forward(self, x):
        identity = x
        
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.attention(out)
        out = self.dropout(out)
        
        if self.residual is not None:
            identity = self.residual(identity)
        
        return out + identity

class OptimizedSCSID(nn.Module):
    """
    Optimized SCS-ID architecture with:
    - Enhanced feature extraction
    - Efficient parameter usage
    - Improved regularization
    - Built-in compression
    """
    def __init__(self, input_features, num_classes, base_channels=32):  # Reduced from 64 to 32
        super().__init__()
        
        # Using smaller base channels for efficiency
        self.base_channels = base_channels
        
        # Track total parameters for model comparison
        self.total_parameters = 0
        self.total_parameters_after_pruning = 0
        
        # Efficient input processing with minimal channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, self.base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.base_channels),
            nn.ReLU(inplace=True)
        )
        self.input_attention = ChannelAttention(self.base_channels)
        
        # Aggressive feature extraction with minimal channels
        self.fire1 = OptimizedFireModule(base_channels, base_channels//4, base_channels//2)
        self.fire2 = OptimizedFireModule(base_channels//2, base_channels//8, base_channels//2)
        
        # Minimal pattern recognition with efficient channels
        self.convseek1 = EnhancedConvSeekBlock(base_channels//2, base_channels//4)
        self.convseek2 = EnhancedConvSeekBlock(base_channels//4, base_channels//8)
        
        # Ultra-efficient classifier with minimal dimensions
        self.pool = nn.AdaptiveAvgPool1d(1)
        final_channels = base_channels//8  # Match the last convseek output (after aggressive reduction)
        
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, max(16, final_channels)),  # Ensure minimum width
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Reduced dropout for smaller network
            nn.Linear(max(16, final_channels), num_classes)
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
                
    def count_parameters(self):
        """Count total parameters and parameters after pruning"""
        total_params = sum(p.numel() for p in self.parameters())
        self.total_parameters = total_params
        
        # Count non-zero parameters (after potential pruning)
        nonzero_params = sum(p.nonzero().size(0) for p in self.parameters())
        self.total_parameters_after_pruning = nonzero_params
        
        return total_params, nonzero_params

    def forward(self, x):
        # Input shape: [batch, features]
        x = x.unsqueeze(1)  # Shape: [batch, 1, features]
        
        # Initial convolution
        x = self.input_conv(x)  # Shape: [batch, base_channels, features]
        x = self.input_attention(x)
        
        # Fire modules
        x = self.fire1(x)  # Shape: [batch, base_channels, features]
        x = self.fire2(x)  # Shape: [batch, base_channels*2, features]
        
        # ConvSeek blocks
        x = self.convseek1(x)  # Shape: [batch, base_channels, features]
        x = self.convseek2(x)  # Shape: [batch, base_channels//2, features]
        
        # Global pooling
        x = self.pool(x)  # Shape: [batch, base_channels//2, 1]
        x = x.squeeze(-1)  # Shape: [batch, base_channels//2]
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
# models/model_metrics.py - Placeholder for model metrics functionality
# This file provides basic model metrics functionality

import torch
import torch.nn as nn

class ModelMetricsTracker:
    """Basic model metrics tracker"""
    
    def __init__(self):
        self.metrics = {}
    
    def reset(self):
        """Reset metrics"""
        self.metrics = {}
    
    def update(self, **kwargs):
        """Update metrics with new values"""
        self.metrics.update(kwargs)
    
    def get(self, key, default=None):
        """Get metric value"""
        return self.metrics.get(key, default)

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_conv1d(module, input, output):
    """Count FLOPS for Conv1d layer"""
    input = input[0]
    batch_size = input.shape[0]
    output_dims = output.shape[1] * output.shape[2]
    kernel_dims = module.kernel_size[0]
    in_channels = module.in_channels
    out_channels = module.out_channels
    
    filters_per_channel = out_channels // module.groups
    conv_per_position_flops = int(torch.prod(torch.tensor(kernel_dims))) * in_channels // module.groups
    
    active_elements_count = batch_size * output_dims
    overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
    
    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_channels * active_elements_count
        
    overall_flops = overall_conv_flops + bias_flops
    return overall_flops

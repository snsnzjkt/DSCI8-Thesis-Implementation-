"""
Model Optimizer: Implements pruning and quantization for SCS-ID
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, Any, List

class ModelOptimizer:
    """
    Optimizer for SCS-ID model to improve computational efficiency
    while maintaining detection performance.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.prune_amount = config.get('prune_amount', 0.5)
        self.min_channels = config.get('min_channels', 4)
    
    def structured_pruning(self):
        """Apply structured pruning to reduce model size"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                # Calculate importance scores
                importance = self._calculate_channel_importance(module)
                
                # Keep only top channels
                n_channels = module.out_channels
                n_keep = max(int(n_channels * (1 - self.prune_amount)), self.min_channels)
                
                # Get indices of top channels
                top_indices = torch.topk(importance, n_keep).indices
                
                # Create pruning mask
                mask = torch.zeros_like(importance)
                mask[top_indices] = 1
                
                # Apply structured pruning
                prune.custom_from_mask(module, name='weight', mask=mask)
    
    def _calculate_channel_importance(self, module: nn.Conv1d) -> torch.Tensor:
        """Calculate importance of each channel based on L1-norm"""
        weight = module.weight.data
        return torch.norm(weight.view(weight.size(0), -1), p=1, dim=1)
    
    def quantize_model(self):
        """Apply quantization to reduce model size"""
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        # Calibration would happen during training
        torch.quantization.convert(self.model, inplace=True)
    
    def apply_optimizations(self):
        """Apply all optimizations to the model"""
        print("🔧 Applying model optimizations...")
        
        # 1. Structured Pruning
        print("   📊 Applying structured pruning...")
        self.structured_pruning()
        
        # 2. Quantization
        print("   📊 Applying quantization...")
        self.quantize_model()
        
        # Calculate and print metrics
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   ✅ Optimizations complete")
        print(f"   📉 Final parameter count: {param_count:,}")
        
        return self.model
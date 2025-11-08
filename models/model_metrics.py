"""
Model Metrics Tracker for accurate parameter counting and performance measurement
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
import time

class ModelMetricsTracker:
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    
    @staticmethod
    def measure_inference_time(model: nn.Module, input_size: Tuple[int, ...], 
                             device: str, num_runs: int = 100) -> Dict[str, float]:
        """Measure average inference time"""
        model.eval()
        model = model.to(device)
        dummy_input = torch.randn(input_size).to(device)
        
        # Warmup runs
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Actual timing
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize() if device == 'cuda' else None
                times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        return {
            'avg_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times)
        }
    
    @staticmethod
    def calculate_flops(model: nn.Module, input_size: Tuple[int, ...]) -> int:
        """Calculate FLOPs for the model"""
        def count_conv1d(m: nn.Conv1d, x: torch.Tensor, y: torch.Tensor) -> int:
            x = x[0]  # Remove batch dimension
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size[0]
            flops = out_channels * x.shape[1] * (2 * in_channels * kernel_size - 1)
            return flops
        
        flops = 0
        hooks = []
        
        def add_hooks(m: nn.Module) -> None:
            if isinstance(m, nn.Conv1d):
                hooks.append(m.register_forward_hook(
                    lambda m, x, y: flops += count_conv1d(m, x, y)))
        
        model.apply(add_hooks)
        dummy_input = torch.randn(input_size)
        with torch.no_grad():
            _ = model(dummy_input)
        
        for hook in hooks:
            hook.remove()
        
        return flops
    
    @staticmethod
    def measure_memory_usage(model: nn.Module, input_size: Tuple[int, ...], 
                           device: str) -> Dict[str, float]:
        """Measure peak memory usage during inference"""
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        model = model.to(device)
        dummy_input = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        if device == 'cuda':
            max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        else:
            import psutil
            max_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        return {
            'peak_memory_mb': max_memory
        }
    
    @staticmethod
    def collect_all_metrics(model: nn.Module, input_size: Tuple[int, ...], 
                          device: str) -> Dict[str, Any]:
        """Collect all model metrics in one call"""
        metrics = {}
        
        # 1. Parameter counts
        metrics.update(ModelMetricsTracker.count_parameters(model))
        
        # 2. Inference time
        metrics.update(ModelMetricsTracker.measure_inference_time(model, input_size, device))
        
        # 3. FLOPs
        metrics['flops'] = ModelMetricsTracker.calculate_flops(model, input_size)
        
        # 4. Memory usage
        metrics.update(ModelMetricsTracker.measure_memory_usage(model, input_size, device))
        
        return metrics
#!/usr/bin/env python3
"""
Quick GPU Test for SCS-ID Model
Tests GPU functionality with a small training example
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.scs_id import SCSIDModel
from config import config

def test_gpu_training():
    """Quick test to verify GPU training works"""
    
    print("🧪 GPU Training Test")
    print("=" * 30)
    print(f"Device: {config.DEVICE}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model and move to GPU
    print("\n🏗️  Creating SCS-ID model...")
    model = SCSIDModel(
        input_features=config.SELECTED_FEATURES,  # Use 42 selected features
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model on device: {next(model.parameters()).device}")
    
    # Create dummy data
    print("\n📊 Creating dummy data...")
    batch_size = 8
    # SCS-ID expects input shape: [batch, 1, features, 1] with 42 selected features
    dummy_data = torch.randn(batch_size, 1, config.SELECTED_FEATURES, 1).to(config.DEVICE)
    dummy_labels = torch.randint(0, config.NUM_CLASSES, (batch_size,)).to(config.DEVICE)
    
    print(f"Data shape: {dummy_data.shape}")
    print(f"Data device: {dummy_data.device}")
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_data)
        print(f"Output shape: {outputs.shape}")
        print(f"Output device: {outputs.device}")
    
    # Test training step
    print("\n🚀 Testing training step...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    outputs = model(dummy_data)
    loss = criterion(outputs, dummy_labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print("✅ GPU training test successful!")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

if __name__ == "__main__":
    test_gpu_training()
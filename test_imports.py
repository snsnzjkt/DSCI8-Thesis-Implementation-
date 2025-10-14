#!/usr/bin/env python3
"""
Test script to isolate the import issue
"""

print("Testing imports...")

try:
    print("1. Testing torch import...")
    import torch
    print("✅ torch imported successfully")
    
    print("2. Testing config import...")
    from config import config
    print("✅ config imported successfully")
    
    print("3. Testing scs_id model import...")
    from models.scs_id import create_scs_id_model
    print("✅ scs_id model imported successfully")
    
    print("4. Testing model creation...")
    model = create_scs_id_model(input_features=42, num_classes=3)
    print("✅ Model created successfully")
    
    print("5. Testing count_parameters...")
    total_params, trainable_params = model.count_parameters()
    print(f"✅ Parameters counted: {total_params:,} total, {trainable_params:,} trainable")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
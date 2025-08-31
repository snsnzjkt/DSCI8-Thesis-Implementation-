#!/usr/bin/env python3
"""
GPU Setup Verification Script for SCS-ID
Run this to verify your GPU setup before training
"""

import torch
import sys

def check_cuda_availability():
    """Check CUDA availability and setup"""
    print("🔍 GPU Setup Verification for SCS-ID")
    print("=" * 50)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        # GPU Information
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**3)
        
        print(f"GPU Count: {gpu_count}")
        print(f"Current GPU: {current_gpu}")
        print(f"GPU Name: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Test GPU functionality
        print("\n🧪 Testing GPU functionality...")
        try:
            # Create a test tensor and move to GPU
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor.t())
            print("✅ GPU tensor operations working correctly")
            
            # Memory usage
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"GPU Memory Allocated: {allocated:.1f} MB")
            print(f"GPU Memory Reserved: {reserved:.1f} MB")
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            return False
            
    else:
        print("\n🚨 CUDA not available. Reasons might be:")
        print("   1. No NVIDIA GPU installed")
        print("   2. CUDA toolkit not installed")
        print("   3. PyTorch installed without CUDA support")
        print("   4. GPU drivers not properly installed")
        
        # Check PyTorch installation
        print(f"\nPyTorch Version: {torch.__version__}")
        print("To install PyTorch with CUDA support:")
        print("   pip uninstall torch")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        return False
    
    return True

def check_training_requirements():
    """Check if system meets SCS-ID training requirements"""
    print("\n📋 SCS-ID Training Requirements Check")
    print("=" * 50)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Memory requirements for CIC-IDS2017 dataset
        min_memory = 4.0  # GB minimum for your thesis requirements
        recommended_memory = 8.0  # GB recommended
        
        if gpu_memory >= recommended_memory:
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB (Excellent - above recommended)")
        elif gpu_memory >= min_memory:
            print(f"⚠️  GPU Memory: {gpu_memory:.1f} GB (Acceptable - meets minimum)")
        else:
            print(f"❌ GPU Memory: {gpu_memory:.1f} GB (Insufficient - need at least {min_memory} GB)")
    
    # Check if we can handle the expected batch size
    try:
        batch_size = 32  # From your config
        num_features = 78  # From your thesis
        
        if torch.cuda.is_available():
            # Test memory allocation for expected batch size
            test_batch = torch.randn(batch_size, 1, num_features).cuda()
            print(f"✅ Batch size {batch_size} memory test passed")
            
            # Cleanup
            del test_batch
            torch.cuda.empty_cache()
            
    except RuntimeError as e:
        print(f"❌ Batch size test failed: {e}")
        print("   Consider reducing batch size in config.py")

def main():
    """Main function"""
    print("SCS-ID: A Squeezed ConvSeek for Efficient Intrusion Detection")
    print("GPU Setup Verification\n")
    
    # Run checks
    gpu_ok = check_cuda_availability()
    
    if gpu_ok:
        check_training_requirements()
        print("\n🎉 GPU setup looks good! You can proceed with training.")
        print("   Run: python experiments/train_baseline.py")
        print("   Then: python experiments/train_scs_id.py")
    else:
        print("\n🔧 Please fix GPU setup issues before training.")
        
if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 863a7fffe5e3487c38210d00ab6dade768918527

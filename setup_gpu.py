#!/usr/bin/env python3
"""
GPU Setup Script for SCS-ID Project
This script helps set up GPU acceleration for the project.
"""

import subprocess
import sys
import os

def check_nvidia_driver():
    """Check if NVIDIA driver is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA Driver detected")
            print(result.stdout.split('\n')[0])  # First line with driver info
            return True
        else:
            print("❌ NVIDIA driver not found")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi command not found. Please install NVIDIA drivers.")
        return False

def check_cuda_toolkit():
    """Check if CUDA toolkit is installed"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA Toolkit detected")
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ CUDA Toolkit not found")
            return False
    except FileNotFoundError:
        print("❌ nvcc command not found. CUDA Toolkit may not be installed.")
        return False

def install_pytorch_gpu():
    """Install PyTorch with GPU support"""
    print("\n🔧 Installing PyTorch with CUDA support...")
    
    # Uninstall CPU version first
    print("Removing CPU-only PyTorch...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'])
    
    # Install GPU version
    print("Installing PyTorch with CUDA 11.8 support...")
    cmd = [
        sys.executable, '-m', 'pip', 'install', 
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/cu118'
    ]
    
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("✅ PyTorch with CUDA installed successfully")
        return True
    else:
        print("❌ Failed to install PyTorch with CUDA")
        return False

def verify_pytorch_gpu():
    """Verify PyTorch GPU setup"""
    try:
        import torch
        print(f"\n📋 PyTorch Configuration:")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        return torch.cuda.is_available()
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def main():
    print("🚀 SCS-ID GPU Setup Verification")
    print("=" * 50)
    
    # Check NVIDIA driver
    driver_ok = check_nvidia_driver()
    
    if not driver_ok:
        print("\n❌ Setup Required:")
        print("1. Install NVIDIA GPU drivers from: https://www.nvidia.com/drivers/")
        return
    
    # Check CUDA toolkit
    cuda_ok = check_cuda_toolkit()
    
    if not cuda_ok:
        print("\n⚠️  CUDA Toolkit Options:")
        print("Option 1: Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
        print("Option 2: Use PyTorch's bundled CUDA (proceed with installation)")
        
        response = input("\nDo you want to install PyTorch with bundled CUDA? (y/n): ")
        if response.lower() == 'y':
            if install_pytorch_gpu():
                verify_pytorch_gpu()
        return
    
    # If CUDA is available, install PyTorch
    print("\n🎯 CUDA Toolkit detected. Installing PyTorch...")
    if install_pytorch_gpu():
        verify_pytorch_gpu()

if __name__ == "__main__":
    main()
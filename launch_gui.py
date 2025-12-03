#!/usr/bin/env python3
"""
Network IDS GUI Launcher
Launch the Network Intrusion Detection System GUI for testing SCS-ID vs Baseline CNN models
"""

import sys
import os
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_models():
    """Check if trained models are available"""
    baseline_model = Path("results/baseline_model.pth")
    scs_id_model = Path("results/scs_id_best_model.pth")
    
    if not baseline_model.exists():
        print("⚠️  Warning: Baseline model not found at results/baseline_model.pth")
        return False
    
    if not scs_id_model.exists():
        print("⚠️  Warning: SCS-ID model not found at results/scs_id_best_model.pth")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 Network IDS GUI Launcher")
    print("=" * 50)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        return 1
    
    # Check models
    print("🔍 Checking trained models...")
    models_available = check_models()
    if not models_available:
        print("⚠️  Some models are missing. The GUI will still run but some features may not work.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    print("✅ All checks passed!")
    print("\n🎯 Starting Network IDS GUI...")
    print("📊 Features available:")
    print("   • Single Sample Testing")
    print("   • Batch Testing")
    print("   • Live Monitoring Simulation")
    print("   • Model Performance Comparison")
    print("\n" + "=" * 50)
    
    try:
        # Launch the GUI
        from network_ids_gui import NetworkIDSGui
        app = NetworkIDSGui()
        app.run()
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
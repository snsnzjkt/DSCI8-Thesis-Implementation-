# clean_checkpoints.py - Clean up incompatible model checkpoints
"""
This script removes potentially incompatible model checkpoints
Use this if you encounter size mismatch errors when switching between different dataset configurations
"""

import os
import glob
from pathlib import Path

def clean_checkpoints():
    """Remove all model checkpoints to start fresh"""
    print("🧹 Cleaning up model checkpoints...")
    
    # Find all checkpoint files
    checkpoint_patterns = [
        "results/best_baseline_model.pth",
        "results/baseline_model.pth", 
        "results/scs_id_best_model.pth",
        "results/scs_id_model.pth",
        "results/**/*.pth"
    ]
    
    removed_count = 0
    
    for pattern in checkpoint_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"   🗑️  Removed: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to remove {file_path}: {e}")
    
    if removed_count == 0:
        print("   ✅ No checkpoints found to clean")
    else:
        print(f"   ✅ Cleaned {removed_count} checkpoint file(s)")
    
    print("\n💡 You can now run training scripts without checkpoint conflicts!")

if __name__ == "__main__":
    print("🧹 Model Checkpoint Cleaner")
    print("=" * 40)
    clean_checkpoints()
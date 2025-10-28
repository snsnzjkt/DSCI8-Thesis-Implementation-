# experiments/deepseek_feature_selection_only.py
"""
DeepSeek RL Feature Selection - Standalone Script
Runs ONLY DeepSeek RL feature selection and saves results

This separates the time-intensive DeepSeek RL (30-60 min) from SCS-ID training,
allowing you to:
1. Run DeepSeek RL once and save selected features
2. Reuse features for multiple SCS-ID training runs
3. Faster iteration when tuning SCS-ID parameters

Usage: python experiments/deepseek_feature_selection_only.py
"""
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pickle
import time
import os
from sklearn.model_selection import train_test_split
from config import config
from models.deepseek_rl import DeepSeekRL


def load_processed_data():
    """Load preprocessed CIC-IDS2017 data"""
    print("\n" + "="*70)
    print("LOADING PREPROCESSED DATA FOR DEEPSEEK RL")
    print("="*70)
    
    data_path = Path(config.DATA_DIR) / "processed" / "processed_data.pkl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    num_classes = data['num_classes']
    class_names = data.get('class_names', [f"Class_{i}" for i in range(num_classes)])
    feature_names = data.get('feature_names', [f"Feature_{i}" for i in range(X_train.shape[1])])
    
    print(f"✓ Data loaded successfully!")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   Classes: {num_classes}")
    
    return X_train, X_test, y_train, y_test, num_classes, class_names, feature_names


def run_deepseek_feature_selection():
    """Run DeepSeek RL feature selection and save results"""
    print("\n" + "="*70)
    print("DEEPSEEK RL FEATURE SELECTION - STANDALONE")
    print("="*70)
    print("⏰ Estimated time: 30-60 minutes")
    print("🎯 Goal: Select optimal 42 features from 78 using reinforcement learning")
    print("💾 Results will be saved for fast SCS-ID training reuse")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test, num_classes, class_names, feature_names = load_processed_data()
    
    # Create train/validation split for DeepSeek RL
    print(f"\n🔄 Creating train/validation split...")
    split_ratio = 0.8
    split_idx = int(split_ratio * len(X_train))
    
    X_train_rl = X_train[:split_idx]
    X_val_rl = X_train[split_idx:]
    y_train_rl = y_train[:split_idx]
    y_val_rl = y_train[split_idx:]
    
    print(f"   Train: {len(X_train_rl):,} samples")
    print(f"   Validation: {len(X_val_rl):,} samples")
    
    # Initialize DeepSeek RL
    print(f"\n🧠 Initializing DeepSeek RL...")
    episodes = 50  # Reduced from the default value
    target_features = config.SELECTED_FEATURES
    
    deepseek_rl = DeepSeekRL(max_features=target_features)
    print(f"   Target features: {target_features}")
    print(f"   Training episodes: {episodes}")
    
    # Train DeepSeek RL
    print(f"\n🚀 Starting DeepSeek RL training with {episodes} episodes...")
    print("   This will take less time due to reduced episodes")
    
    start_time = time.time()
    deepseek_rl.fit(
        X_train_rl, y_train_rl, 
        X_val_rl, y_val_rl, 
        episodes=episodes, 
        verbose=True
    )
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    print("   Monitoring GPU utilization during training...")
    
    # Check GPU utilization
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   GPU in use: {gpu_name}")
            print(f"   Current GPU utilization: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
        else:
            print("   No GPU detected. Training is running on CPU.")
    except ImportError:
        print("   PyTorch is not installed. Unable to check GPU utilization.")
    
    # Get selected features
    selected_features = deepseek_rl.get_selected_features()
    
    print(f"\n✅ DeepSeek RL training complete!")
    print(f"   Time: {training_time:.1f}s ({training_time/60:.2f} minutes)")
    print(f"   Selected {len(selected_features)} features from {len(feature_names)}")
    print(f"   Feature indices: {sorted(selected_features)}")
    
    # Transform all datasets with selected features
    print(f"\n🔄 Transforming datasets with selected features...")
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    print(f"   Original: {X_train.shape} -> Selected: {X_train_selected.shape}")
    print(f"   Test: {X_test.shape} -> Selected: {X_test_selected.shape}")
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    deepseek_dir = f"{config.RESULTS_DIR}/deepseek_rl"
    os.makedirs(deepseek_dir, exist_ok=True)
    
    # Save training plots
    try:
        plot_path = f"{deepseek_dir}/training_history.png"
        deepseek_rl.plot_training_history(plot_path)
        print(f"   📊 Training plots saved: {plot_path}")
    except Exception as e:
        print(f"   ⚠️ Could not save training plots: {e}")
    
    # Prepare results
    results = {
        # Feature selection results
        'selected_features': selected_features.tolist(),
        'selected_feature_names': [feature_names[i] for i in selected_features],
        'original_feature_count': len(feature_names),
        'selected_feature_count': len(selected_features),
        
        # Training metadata
        'method': 'DeepSeek_RL',
        'episodes': episodes,
        'target_features': target_features,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        
        # DeepSeek RL specific data
        'deepseek_rl_object': deepseek_rl,
        'training_history': getattr(deepseek_rl, 'training_history', []),
        'convergence_history': getattr(deepseek_rl, 'convergence_history', []),
        
        # Configuration
        'config': {
            'episodes': episodes,
            'target_features': target_features,
            'original_features': len(feature_names),
            'split_ratio': split_ratio
        },
        
        # Transformed datasets ready for SCS-ID training
        'X_train_selected': X_train_selected,
        'X_test_selected': X_test_selected,
        'y_train': y_train,
        'y_test': y_test,
        'num_classes': num_classes,
        'class_names': class_names,
        'feature_names_selected': [feature_names[i] for i in selected_features]
    }
    
    # Save complete results
    results_file = f"{config.RESULTS_DIR}/deepseek_feature_selection_complete.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save lightweight version (features only, no datasets)
    lightweight_results = {
        'selected_features': selected_features.tolist(),
        'selected_feature_names': [feature_names[i] for i in selected_features],
        'original_feature_count': len(feature_names),
        'selected_feature_count': len(selected_features),
        'method': 'DeepSeek_RL',
        'episodes': episodes,
        'training_time_minutes': training_time / 60,
        'config': results['config']
    }
    
    lightweight_file = f"{config.RESULTS_DIR}/deepseek_features_only.pkl"
    with open(lightweight_file, 'wb') as f:
        pickle.dump(lightweight_results, f)
    
    # Print summary
    print(f"\n" + "="*70)
    print("🎉 DEEPSEEK RL FEATURE SELECTION COMPLETE")
    print("="*70)
    print(f"⏱️  Total time: {training_time/60:.2f} minutes")
    print(f"🎯 Selected features: {len(selected_features)} from {len(feature_names)}")
    print(f"📋 Feature indices: {sorted(selected_features)}")
    print(f"\n📁 Files saved:")
    print(f"   📦 Complete results: {results_file}")
    print(f"   🏷️  Features only: {lightweight_file}")
    print(f"   📊 Training plots: {deepseek_dir}/training_history.png")
    print(f"\n🚀 Next step: Run SCS-ID training with pre-selected features:")
    print(f"   python experiments/train_scs_id_fast.py")
    print("="*70)
    
    return selected_features, results


if __name__ == "__main__":
    try:
        selected_features, results = run_deepseek_feature_selection()
        print("✅ DeepSeek RL feature selection completed successfully!")
    except Exception as e:
        print(f"❌ Error during DeepSeek RL feature selection: {e}")
        import traceback
        traceback.print_exc()
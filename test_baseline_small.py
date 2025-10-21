# test_baseline_small.py - Test baseline training with synthetic small data
"""
Quick test script for baseline CNN training with synthetic data
No need for full dataset download - generates small test data
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from experiments.train_baseline import BaselineTrainer, BaselineCNN
from config import config

def generate_synthetic_data(n_samples=1000, n_features=78, n_classes=16):
    """Generate synthetic intrusion detection data"""
    print("🎲 Generating synthetic data...")
    
    # Generate classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),  # 70% informative features
        n_redundant=int(n_features * 0.1),    # 10% redundant features
        n_clusters_per_class=2,
        n_classes=n_classes,
        class_sep=1.5,  # Good class separation
        random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   📊 Generated {n_samples:,} samples with {n_features} features")
    print(f"   📊 Training: {len(X_train):,} samples")
    print(f"   📊 Test: {len(X_test):,} samples")
    print(f"   🏷️  Classes: {n_classes}")
    
    return X_train, X_test, y_train, y_test

def create_synthetic_dataset():
    """Create and save synthetic dataset in the expected format"""
    print("🏗️ Creating synthetic dataset...")
    
    # Generate data
    X_train, X_test, y_train, y_test = generate_synthetic_data(
        n_samples=1000,    # Small dataset for quick testing
        n_features=78,     # Match CIC-IDS2017 features
        n_classes=16       # Match attack types + benign
    )
    
    # Create class names (simulate CIC-IDS2017 classes)
    class_names = [
        'BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration',
        'Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - Sql Injection',
        'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest',
        'DoS Hulk', 'DoS GoldenEye', 'Heartbleed', 'DoS slowloris'
    ]
    
    # Create feature names
    feature_names = [f'feature_{i:02d}' for i in range(78)]
    
    # Prepare data dictionary
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'num_classes': 16,
        'class_names': class_names,
        'feature_names': feature_names,
        'preprocessing_info': {
            'method': 'synthetic_generation',
            'scaler': 'StandardScaler',
            'dataset_size': len(X_train) + len(X_test)
        }
    }
    
    # Create directories
    processed_dir = Path(config.DATA_DIR) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the synthetic dataset
    processed_file = processed_dir / "processed_data.pkl"
    with open(processed_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"   ✅ Synthetic dataset saved to: {processed_file}")
    print(f"   📁 Ready for baseline training!")
    
    return data

class QuickTestConfig:
    """Quick test configuration with reduced parameters"""
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    BATCH_SIZE = 16        # Smaller batch size
    LEARNING_RATE = 1e-3   # Higher learning rate for faster convergence
    EPOCHS = 5             # Much fewer epochs for quick testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

def run_quick_baseline_test():
    """Run a quick baseline test with synthetic data"""
    print("🚀 Starting Quick Baseline Test")
    print("=" * 50)
    
    # Override config for quick testing
    global config
    config = QuickTestConfig()
    
    # Generate synthetic data
    create_synthetic_dataset()
    
    # Run baseline training
    print("\n🏋️ Starting baseline training...")
    trainer = BaselineTrainer()
    
    try:
        model, accuracy, f1 = trainer.train_model()
        
        print("\n" + "=" * 50)
        print("✅ QUICK TEST COMPLETED!")
        print("=" * 50)
        print(f"🎯 Final Results:")
        print(f"   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📊 F1-Score: {f1:.4f}")
        print(f"   ⚡ Quick test successful!")
        print("\n💡 Ready to run full training with real dataset!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_threshold_optimizer():
    """Test just the threshold optimizer with synthetic data"""
    print("\n🎯 Testing Threshold Optimizer...")
    
    from models.threshold_optimizer import ThresholdOptimizer
    
    # Generate small synthetic data
    np.random.seed(42)
    n_samples = 500
    
    # Simulate multi-class predictions (3 classes: benign, attack1, attack2)
    y_true = np.random.randint(0, 3, n_samples)
    
    # Simulate prediction probabilities
    y_pred_proba = np.random.rand(n_samples, 3)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    print(f"   📊 Testing with {n_samples} synthetic predictions")
    
    try:
        # Test threshold optimization
        optimizer = ThresholdOptimizer(target_fpr=0.05)  # 5% FPR for testing
        
        results = optimizer.optimize_threshold(y_true, y_pred_proba, verbose=True)
        metrics = optimizer.calculate_metrics_with_threshold(y_true, y_pred_proba, verbose=True)
        
        print("   ✅ Threshold optimizer test successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Threshold optimizer test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Baseline Training - Quick Test Mode")
    print("=" * 60)
    print("This script tests the baseline training with synthetic data")
    print("No need to download the full CIC-IDS2017 dataset!")
    print("=" * 60)
    
    # Test threshold optimizer first
    threshold_ok = test_threshold_optimizer()
    
    if threshold_ok:
        # Run full quick test
        success = run_quick_baseline_test()
        
        if success:
            print("\n🎉 All tests passed! Your baseline implementation is working correctly.")
            print("💡 Next steps:")
            print("   1. Download full dataset: python data/download_dataset.py")
            print("   2. Preprocess data: python data/preprocess.py")
            print("   3. Run full training: python experiments/train_baseline.py")
        else:
            print("\n⚠️  Some tests failed. Check the errors above.")
    else:
        print("\n⚠️  Threshold optimizer test failed. Please check the implementation.")
import numpy as np
import torch
from models.baseline_cnn import BaselineCNN
from models.scs_id_optimized import OptimizedSCSID
from network_ids_gui import NetworkDataSimulator, ModelTester
from pathlib import Path

def test_improved_performance():
    """Test that SCS-ID now shows better attack detection than baseline"""
    print("🧪 Testing Improved SCS-ID vs Baseline Performance")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize components
    simulator = NetworkDataSimulator()
    tester = ModelTester(device=device, input_size=120)
    
    # Load models
    tester.load_models()
    
    if not tester.baseline_model or not tester.scs_id_model:
        print("❌ Models not loaded properly")
        return
    
    print("✅ Both models loaded successfully")
    
    # Generate test data with diverse attack types
    print(f"\n📊 Generating diverse test data...")
    test_data = simulator.generate_batch(batch_size=200, benign_ratio=0.4, num_features=120)
    
    # Check what we generated
    attack_types = []
    for label in test_data['labels']:
        attack_types.append(simulator.attack_types[label])
    
    print(f"Generated {len(test_data['features'])} samples:")
    from collections import Counter
    label_counts = Counter(test_data['labels'])
    for label, count in sorted(label_counts.items()):
        attack_name = simulator.attack_types[label]
        percentage = (count / len(test_data['labels'])) * 100
        print(f"   {attack_name}: {count} samples ({percentage:.1f}%)")
    
    # Test both models
    print(f"\n🔍 Testing Model Predictions...")
    results = tester.predict(test_data['features'], model_type='both')
    
    if 'baseline' not in results or 'scs_id' not in results:
        print("❌ Failed to get predictions from both models")
        return
    
    baseline_preds = results['baseline']['predictions']
    scs_id_preds = results['scs_id']['predictions']
    true_labels = test_data['labels']
    
    print(f"\n📊 Prediction Analysis:")
    
    # Baseline performance
    baseline_correct = np.sum(baseline_preds == true_labels)
    baseline_accuracy = baseline_correct / len(true_labels)
    baseline_unique = len(set(baseline_preds))
    
    print(f"\nBaseline CNN:")
    print(f"   Overall Accuracy: {baseline_accuracy:.3f} ({baseline_correct}/{len(true_labels)})")
    print(f"   Unique predictions: {baseline_unique}/15 classes")
    
    baseline_pred_counts = Counter(baseline_preds)
    print(f"   Prediction distribution:")
    for pred_class in sorted(baseline_pred_counts.keys()):
        count = baseline_pred_counts[pred_class]
        percentage = (count / len(baseline_preds)) * 100
        attack_name = simulator.attack_types[pred_class]
        print(f"      {attack_name}: {count} ({percentage:.1f}%)")
    
    # SCS-ID performance  
    scs_id_correct = np.sum(scs_id_preds == true_labels)
    scs_id_accuracy = scs_id_correct / len(true_labels)
    scs_id_unique = len(set(scs_id_preds))
    
    print(f"\nSCS-ID Model:")
    print(f"   Overall Accuracy: {scs_id_accuracy:.3f} ({scs_id_correct}/{len(true_labels)})")
    print(f"   Unique predictions: {scs_id_unique}/15 classes")
    
    scs_id_pred_counts = Counter(scs_id_preds)
    print(f"   Prediction distribution:")
    for pred_class in sorted(scs_id_pred_counts.keys()):
        count = scs_id_pred_counts[pred_class]
        percentage = (count / len(scs_id_preds)) * 100
        attack_name = simulator.attack_types[pred_class]
        print(f"      {attack_name}: {count} ({percentage:.1f}%)")
    
    # Attack-specific performance (non-BENIGN classes)
    attack_mask = np.array(true_labels) != 0  # Non-benign samples
    if np.sum(attack_mask) > 0:
        baseline_attack_accuracy = np.mean(baseline_preds[attack_mask] == np.array(true_labels)[attack_mask])
        scs_id_attack_accuracy = np.mean(scs_id_preds[attack_mask] == np.array(true_labels)[attack_mask])
        
        print(f"\n🎯 Attack Detection Performance:")
        print(f"   Baseline Attack Detection: {baseline_attack_accuracy:.3f}")
        print(f"   SCS-ID Attack Detection: {scs_id_attack_accuracy:.3f}")
        
        if scs_id_attack_accuracy > baseline_attack_accuracy:
            improvement = scs_id_attack_accuracy - baseline_attack_accuracy
            print(f"   ✅ SCS-ID is {improvement:.3f} better at detecting attacks!")
        else:
            print(f"   ⚠️ SCS-ID needs more improvement")
    
    # Model comparison summary
    print(f"\n🏆 Summary:")
    print(f"="*40)
    
    if scs_id_accuracy > baseline_accuracy:
        improvement = scs_id_accuracy - baseline_accuracy
        print(f"✅ SCS-ID Overall Performance: +{improvement:.3f} better")
    
    if scs_id_unique > baseline_unique:
        print(f"✅ SCS-ID Diversity: {scs_id_unique} vs {baseline_unique} classes")
    
    if scs_id_unique > 2 and baseline_unique <= 2:
        print(f"✅ SCS-ID shows much better class diversity!")
        
    # Calculate detection rates for specific attack types
    print(f"\n🔍 Per-Attack-Type Detection Rates:")
    for attack_idx in range(1, 15):  # Skip BENIGN (index 0)
        if attack_idx < len(simulator.attack_types):
            attack_name = simulator.attack_types[attack_idx]
            attack_samples = np.array(true_labels) == attack_idx
            
            if np.sum(attack_samples) > 0:
                baseline_detection = np.mean(baseline_preds[attack_samples] == attack_idx)
                scs_id_detection = np.mean(scs_id_preds[attack_samples] == attack_idx)
                
                print(f"   {attack_name}:")
                print(f"      Baseline: {baseline_detection:.3f}")
                print(f"      SCS-ID: {scs_id_detection:.3f}")
                
                if scs_id_detection > baseline_detection:
                    print(f"      ✅ SCS-ID better by {scs_id_detection - baseline_detection:.3f}")

if __name__ == "__main__":
    test_improved_performance()
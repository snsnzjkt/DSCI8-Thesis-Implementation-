import torch
import numpy as np
import pandas as pd
from pathlib import Path
from models.baseline_cnn import BaselineCNN
from models.scs_id_optimized import OptimizedSCSID
from data.preprocess import CICIDSPreprocessor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_model_predictions():
    """Analyze both models' predictions to understand the bias issue"""
    print("🔍 Analyzing Model Predictions and Performance")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load processed test data
    processed_dir = Path('data/processed')
    if (processed_dir / 'test_data.pkl').exists():
        print("📊 Loading test data...")
        with open(processed_dir / 'test_data.pkl', 'rb') as f:
            test_data = pd.read_pickle(f)
        
        print(f"Test data shape: {test_data['features'].shape}")
        print(f"Test labels shape: {test_data['labels'].shape}")
        
        # Load label encoder
        with open(processed_dir / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pd.read_pickle(f)
        
        class_names = label_encoder.classes_
        print(f"Classes: {class_names}")
        
        # Check class distribution in test data
        unique_labels, counts = np.unique(test_data['labels'], return_counts=True)
        print(f"\n📊 Test Data Class Distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
            percentage = (count / len(test_data['labels'])) * 100
            print(f"   {class_name}: {count} samples ({percentage:.2f}%)")
    else:
        print("❌ No test data found. Generating sample data...")
        # Create sample test data
        preprocessor = CICIDSPreprocessor()
        
        # Use some samples for testing
        n_samples = 1000
        features = np.random.randn(n_samples, 120)  # 120 features to match models
        
        # Create balanced labels for testing
        n_classes = 15
        labels = np.random.randint(0, n_classes, n_samples)
        
        test_data = {'features': features, 'labels': labels}
        class_names = [f"Class_{i}" for i in range(n_classes)]
        class_names[0] = "BENIGN"  # First class is benign
        
        print(f"Generated test data shape: {features.shape}")
    
    # Load models
    print(f"\n🔧 Loading Models...")
    
    # Load Baseline CNN
    baseline_model = None
    try:
        baseline_model = BaselineCNN(input_features=120, num_classes=15)
        if Path('results/baseline/best_baseline_model.pth').exists():
            checkpoint = torch.load('results/baseline/best_baseline_model.pth', map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                baseline_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                baseline_model.load_state_dict(checkpoint)
            baseline_model.to(device)
            baseline_model.eval()
            print("✅ Baseline CNN loaded successfully")
        else:
            baseline_model = None
            print("❌ Baseline model file not found")
    except Exception as e:
        print(f"❌ Failed to load Baseline CNN: {e}")
        baseline_model = None
    
    # Load SCS-ID Model
    scs_id_model = None
    try:
        scs_id_model = OptimizedSCSID(input_features=120, num_classes=15)
        if Path('results/scs_id/scs_id_best_model.pth').exists():
            checkpoint = torch.load('results/scs_id/scs_id_best_model.pth', map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                scs_id_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                scs_id_model.load_state_dict(checkpoint)
            scs_id_model.to(device)
            scs_id_model.eval()
            print("✅ SCS-ID loaded successfully")
        else:
            scs_id_model = None
            print("❌ SCS-ID model file not found")
    except Exception as e:
        print(f"❌ Failed to load SCS-ID: {e}")
        scs_id_model = None
    
    if not baseline_model and not scs_id_model:
        print("❌ No models could be loaded!")
        return
    
    # Prepare test data
    features = test_data['features']
    labels = test_data['labels']
    
    # Take a subset for analysis (first 500 samples)
    n_test = min(500, len(features))
    test_features = features[:n_test]
    test_labels = labels[:n_test]
    
    # Convert to tensors
    test_tensor = torch.FloatTensor(test_features).to(device)
    
    print(f"\n🧪 Testing with {n_test} samples...")
    
    # Make predictions
    results = {}
    
    if baseline_model:
        print("Testing Baseline CNN...")
        with torch.no_grad():
            baseline_input = test_tensor.unsqueeze(1)  # Add channel dimension for CNN
            baseline_output = baseline_model(baseline_input)
            baseline_probs = torch.softmax(baseline_output, dim=1)
            baseline_preds = torch.argmax(baseline_probs, dim=1).cpu().numpy()
            baseline_confidence = torch.max(baseline_probs, dim=1)[0].cpu().numpy()
            
        results['baseline'] = {
            'predictions': baseline_preds,
            'confidence': baseline_confidence,
            'probabilities': baseline_probs.cpu().numpy()
        }
    
    if scs_id_model:
        print("Testing SCS-ID...")
        with torch.no_grad():
            scs_id_output = scs_id_model(test_tensor)
            scs_id_probs = torch.softmax(scs_id_output, dim=1)
            scs_id_preds = torch.argmax(scs_id_probs, dim=1).cpu().numpy()
            scs_id_confidence = torch.max(scs_id_probs, dim=1)[0].cpu().numpy()
            
        results['scs_id'] = {
            'predictions': scs_id_preds,
            'confidence': scs_id_confidence,
            'probabilities': scs_id_probs.cpu().numpy()
        }
    
    # Analyze results
    print(f"\n📊 Prediction Analysis:")
    print("="*60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        preds = model_results['predictions']
        conf = model_results['confidence']
        
        # Check prediction distribution
        pred_counts = Counter(preds)
        print(f"   Prediction distribution:")
        for pred_class in sorted(pred_counts.keys()):
            count = pred_counts[pred_class]
            percentage = (count / len(preds)) * 100
            class_name = class_names[pred_class] if pred_class < len(class_names) else f"Class_{pred_class}"
            print(f"      {class_name}: {count} ({percentage:.1f}%)")
        
        # Check if predicting only one class (bias issue)
        unique_preds = len(set(preds))
        print(f"   Unique predictions: {unique_preds}/15 classes")
        
        if unique_preds == 1:
            print(f"   ⚠️  MODEL IS BIASED - Only predicting class {preds[0]} ({class_names[preds[0]]})")
        elif unique_preds < 5:
            print(f"   ⚠️  MODEL HAS LIMITED DIVERSITY - Only {unique_preds} classes predicted")
        else:
            print(f"   ✅ Model shows good diversity")
        
        # Average confidence
        print(f"   Average confidence: {np.mean(conf):.4f}")
        print(f"   Confidence std: {np.std(conf):.4f}")
    
    # Compare models if both loaded
    if 'baseline' in results and 'scs_id' in results:
        print(f"\n🆚 Model Comparison:")
        baseline_preds = results['baseline']['predictions']
        scs_id_preds = results['scs_id']['predictions']
        
        agreement = np.mean(baseline_preds == scs_id_preds) * 100
        print(f"   Agreement between models: {agreement:.1f}%")
        
        # Check which model is more diverse
        baseline_unique = len(set(baseline_preds))
        scs_id_unique = len(set(scs_id_preds))
        
        if scs_id_unique > baseline_unique:
            print(f"   ✅ SCS-ID is more diverse ({scs_id_unique} vs {baseline_unique} classes)")
        elif baseline_unique > scs_id_unique:
            print(f"   ❌ Baseline is more diverse ({baseline_unique} vs {scs_id_unique} classes)")
        else:
            print(f"   ➡️  Both models equally diverse ({baseline_unique} classes)")
    
    # Suggestions for improvement
    print(f"\n💡 Suggestions for Improvement:")
    print("="*60)
    
    if any(len(set(results[model]['predictions'])) <= 2 for model in results):
        print("1. 🎯 Class Imbalance Issue Detected:")
        print("   - Models are heavily biased toward one or few classes")
        print("   - Consider retraining with balanced class weights")
        print("   - Use focal loss or class-balanced loss functions")
        print("   - Apply SMOTE or other oversampling techniques")
        
        print("\n2. 🔧 Model Architecture Improvements:")
        print("   - Add class weights during training")
        print("   - Use different loss functions (focal loss, weighted cross-entropy)")
        print("   - Adjust learning rate and training epochs")
        print("   - Apply dropout and regularization")
        
        print("\n3. 🎲 Data Improvements:")
        print("   - Ensure balanced training data")
        print("   - Apply data augmentation for minority classes")
        print("   - Feature engineering for better class separation")
    
    return results

if __name__ == "__main__":
    analyze_model_predictions()
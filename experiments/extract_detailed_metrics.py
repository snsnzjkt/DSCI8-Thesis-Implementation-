# -*- coding: utf-8 -*-
# experiments/extract_detailed_metrics.py - Extract all available metrics from actual results
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_and_inspect_results():
    """Load and inspect all available results"""
    results_dir = Path("results")
    
    print("🔍 Detailed Inspection of Actual Results")
    print("="*50)
    
    # Load baseline results
    baseline_path = results_dir / "baseline" / "baseline_results.pkl"
    with open(baseline_path, 'rb') as f:
        baseline_data = pickle.load(f)
    
    # Load SCS-ID results
    scs_id_path = results_dir / "scs_id" / "scs_id_optimized_results.pkl"
    with open(scs_id_path, 'rb') as f:
        scs_id_data = pickle.load(f)
    
    print("\n📊 BASELINE CNN DETAILED METRICS:")
    print("="*40)
    
    # Extract detailed baseline metrics
    if 'test_accuracy' in baseline_data:
        print(f"Test Accuracy: {baseline_data['test_accuracy']:.6f} ({baseline_data['test_accuracy']*100:.2f}%)")
    
    if 'f1_score' in baseline_data:
        print(f"F1-Score: {baseline_data['f1_score']:.6f}")
    
    if 'model_parameters' in baseline_data:
        print(f"Model Parameters: {baseline_data['model_parameters']:,}")
    
    if 'training_time' in baseline_data:
        print(f"Training Time: {baseline_data['training_time']:.2f} seconds")
    
    # Extract classification report if available
    if 'classification_report' in baseline_data:
        print("\nClassification Report Keys:")
        if isinstance(baseline_data['classification_report'], dict):
            for key in baseline_data['classification_report'].keys():
                print(f"  - {key}")
    
    # Extract threshold optimization results
    if 'threshold_optimization' in baseline_data:
        thresh_opt = baseline_data['threshold_optimization']
        print(f"\nThreshold Optimization:")
        if isinstance(thresh_opt, dict):
            for key, value in thresh_opt.items():
                print(f"  {key}: {value}")
    
    print("\n📊 SCS-ID DETAILED METRICS:")
    print("="*35)
    
    # Extract detailed SCS-ID metrics
    if 'test_accuracy' in scs_id_data:
        print(f"Test Accuracy: {scs_id_data['test_accuracy']:.6f} ({scs_id_data['test_accuracy']*100:.2f}%)")
    
    if 'f1_score' in scs_id_data:
        print(f"F1-Score: {scs_id_data['f1_score']:.6f}")
    
    if 'training_time' in scs_id_data:
        print(f"Training Time: {scs_id_data['training_time']:.2f} seconds")
    
    # Extract model stats if available
    if 'model_stats' in scs_id_data:
        model_stats = scs_id_data['model_stats']
        print(f"\nModel Statistics:")
        if isinstance(model_stats, dict):
            for key, value in model_stats.items():
                print(f"  {key}: {value}")
    
    # Extract threshold results
    if 'threshold_results' in scs_id_data:
        thresh_res = scs_id_data['threshold_results']
        print(f"\nThreshold Results:")
        if isinstance(thresh_res, dict):
            for key, value in thresh_res.items():
                print(f"  {key}: {value}")
    
    # Calculate actual improvements
    print("\n📈 ACTUAL IMPROVEMENTS:")
    print("="*25)
    
    baseline_acc = baseline_data.get('test_accuracy', 0)
    scs_id_acc = scs_id_data.get('test_accuracy', 0)
    acc_improvement = scs_id_acc - baseline_acc
    acc_percent = (acc_improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
    
    print(f"Accuracy Improvement: {baseline_acc:.6f} → {scs_id_acc:.6f}")
    print(f"Absolute Improvement: +{acc_improvement:.6f}")
    print(f"Percentage Improvement: +{acc_percent:.3f}%")
    
    baseline_f1 = baseline_data.get('f1_score', 0)
    scs_id_f1 = scs_id_data.get('f1_score', 0)
    f1_improvement = scs_id_f1 - baseline_f1
    f1_percent = (f1_improvement / baseline_f1) * 100 if baseline_f1 > 0 else 0
    
    print(f"\nF1-Score Improvement: {baseline_f1:.6f} → {scs_id_f1:.6f}")
    print(f"Absolute Improvement: +{f1_improvement:.6f}")
    print(f"Percentage Improvement: +{f1_percent:.3f}%")
    
    # Parameter comparison
    baseline_params = baseline_data.get('model_parameters', 0)
    scs_id_params = None
    if 'model_stats' in scs_id_data and isinstance(scs_id_data['model_stats'], dict):
        scs_id_params = scs_id_data['model_stats'].get('total_params', None)
    
    if baseline_params and scs_id_params:
        param_reduction = (1 - scs_id_params / baseline_params) * 100
        print(f"\nParameter Comparison:")
        print(f"Baseline Parameters: {baseline_params:,}")
        print(f"SCS-ID Parameters: {scs_id_params:,}")
        print(f"Parameter Reduction: {param_reduction:.1f}%")
    
    # Training time comparison
    baseline_time = baseline_data.get('training_time', 0)
    scs_id_time = scs_id_data.get('training_time', 0)
    
    if baseline_time and scs_id_time:
        time_ratio = scs_id_time / baseline_time
        print(f"\nTraining Time Comparison:")
        print(f"Baseline Training Time: {baseline_time:.2f} seconds")
        print(f"SCS-ID Training Time: {scs_id_time:.2f} seconds")
        print(f"Time Ratio (SCS-ID/Baseline): {time_ratio:.2f}x")
    
    # Extract confusion matrix data if available
    if 'predictions' in baseline_data and 'labels' in baseline_data:
        print(f"\n📊 CONFUSION MATRIX DATA AVAILABLE:")
        print(f"Baseline - Predictions shape: {np.array(baseline_data['predictions']).shape}")
        print(f"Baseline - Labels shape: {np.array(baseline_data['labels']).shape}")
    
    if 'predictions' in scs_id_data and 'labels' in scs_id_data:
        print(f"SCS-ID - Predictions shape: {np.array(scs_id_data['predictions']).shape}")
        print(f"SCS-ID - Labels shape: {np.array(scs_id_data['labels']).shape}")
    
    # Save detailed comparison
    output_path = results_dir / "detailed_metrics_comparison.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("DETAILED METRICS COMPARISON\n")
        f.write("="*40 + "\n\n")
        
        f.write("ACCURACY METRICS:\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.6f} ({baseline_acc*100:.2f}%)\n")
        f.write(f"SCS-ID Accuracy: {scs_id_acc:.6f} ({scs_id_acc*100:.2f}%)\n")
        f.write(f"Improvement: +{acc_improvement:.6f} (+{acc_percent:.3f}%)\n\n")
        
        f.write("F1-SCORE METRICS:\n")
        f.write(f"Baseline F1: {baseline_f1:.6f}\n")
        f.write(f"SCS-ID F1: {scs_id_f1:.6f}\n")
        f.write(f"Improvement: +{f1_improvement:.6f} (+{f1_percent:.3f}%)\n\n")
        
        if baseline_params and scs_id_params:
            f.write("PARAMETER METRICS:\n")
            f.write(f"Baseline Parameters: {baseline_params:,}\n")
            f.write(f"SCS-ID Parameters: {scs_id_params:,}\n")
            f.write(f"Reduction: {param_reduction:.1f}%\n\n")
        
        if baseline_time and scs_id_time:
            f.write("TRAINING TIME METRICS:\n")
            f.write(f"Baseline Time: {baseline_time:.2f}s\n")
            f.write(f"SCS-ID Time: {scs_id_time:.2f}s\n")
            f.write(f"Ratio: {time_ratio:.2f}x\n\n")
        
        f.write(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n✅ Detailed comparison saved to: {output_path}")
    
    return {
        'baseline_data': baseline_data,
        'scs_id_data': scs_id_data,
        'comparison': {
            'accuracy_improvement': acc_improvement,
            'accuracy_percent': acc_percent,
            'f1_improvement': f1_improvement,
            'f1_percent': f1_percent,
            'parameter_reduction': param_reduction if baseline_params and scs_id_params else None,
            'time_ratio': time_ratio if baseline_time and scs_id_time else None
        }
    }

if __name__ == "__main__":
    results = load_and_inspect_results()
    print("\n[SUCCESS] Detailed metrics extraction completed!")
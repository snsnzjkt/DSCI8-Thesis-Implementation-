#!/usr/bin/env python3
"""
Extract per-class test results for thesis documentation
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def extract_per_class_results():
    """Extract detailed per-class results from saved model data"""
    
    print("📊 Extracting Per-Class Test Results")
    print("=" * 50)
    
    # Load baseline results
    baseline_path = Path("results/baseline/baseline_results.pkl")
    scs_id_path = Path("results/scs_id/scs_id_optimized_results.pkl")
    
    if not baseline_path.exists() or not scs_id_path.exists():
        print("❌ Model result files not found")
        return
    
    # Load the data
    with open(baseline_path, 'rb') as f:
        baseline_data = pickle.load(f)
    
    with open(scs_id_path, 'rb') as f:
        scs_id_data = pickle.load(f)
    
    print("\n📊 BASELINE CNN PER-CLASS RESULTS:")
    print("-" * 40)
    
    # Extract classification report
    if 'classification_report' in baseline_data:
        baseline_report = baseline_data['classification_report']
        print("Classification Report Keys:", list(baseline_report.keys()))
        
        # Create per-class results table
        classes = [k for k in baseline_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        print(f"\nClasses found: {len(classes)}")
        for cls in classes[:10]:  # Show first 10
            if isinstance(baseline_report[cls], dict):
                precision = baseline_report[cls].get('precision', 0)
                recall = baseline_report[cls].get('recall', 0)
                f1 = baseline_report[cls].get('f1-score', 0)
                support = baseline_report[cls].get('support', 0)
                print(f"  {cls}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, N={support}")
    
    print("\n📊 SCS-ID PER-CLASS RESULTS:")
    print("-" * 40)
    
    if 'classification_report' in scs_id_data:
        scs_id_report = scs_id_data['classification_report']
        classes = [k for k in scs_id_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        print(f"\nClasses found: {len(classes)}")
        for cls in classes[:10]:  # Show first 10
            if isinstance(scs_id_report[cls], dict):
                precision = scs_id_report[cls].get('precision', 0)
                recall = scs_id_report[cls].get('recall', 0)
                f1 = scs_id_report[cls].get('f1-score', 0)
                support = scs_id_report[cls].get('support', 0)
                print(f"  {cls}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, N={support}")
    
    # Create comprehensive comparison table
    print("\n📋 Creating Comprehensive Comparison Table...")
    
    if 'classification_report' in baseline_data and 'classification_report' in scs_id_data:
        baseline_report = baseline_data['classification_report']
        scs_id_report = scs_id_data['classification_report']
        
        # Get all classes
        all_classes = set()
        all_classes.update([k for k in baseline_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
        all_classes.update([k for k in scs_id_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
        all_classes = sorted(list(all_classes))
        
        # Create comparison data
        comparison_data = []
        
        for cls in all_classes:
            baseline_metrics = baseline_report.get(cls, {})
            scs_id_metrics = scs_id_report.get(cls, {})
            
            if isinstance(baseline_metrics, dict) and isinstance(scs_id_metrics, dict):
                row = {
                    'Class': cls,
                    'Baseline_Precision': baseline_metrics.get('precision', 0),
                    'Baseline_Recall': baseline_metrics.get('recall', 0),
                    'Baseline_F1': baseline_metrics.get('f1-score', 0),
                    'Baseline_Support': baseline_metrics.get('support', 0),
                    'SCS_ID_Precision': scs_id_metrics.get('precision', 0),
                    'SCS_ID_Recall': scs_id_metrics.get('recall', 0),
                    'SCS_ID_F1': scs_id_metrics.get('f1-score', 0),
                    'SCS_ID_Support': scs_id_metrics.get('support', 0),
                }
                
                # Calculate improvements
                row['Precision_Improvement'] = row['SCS_ID_Precision'] - row['Baseline_Precision']
                row['Recall_Improvement'] = row['SCS_ID_Recall'] - row['Baseline_Recall']
                row['F1_Improvement'] = row['SCS_ID_F1'] - row['Baseline_F1']
                
                comparison_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(comparison_data)
        output_path = Path("results/per_class_comparison.csv")
        df.to_csv(output_path, index=False)
        
        print(f"✅ Per-class comparison saved to: {output_path}")
        
        # Print summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Total Classes: {len(comparison_data)}")
        print(f"Average Baseline F1: {df['Baseline_F1'].mean():.4f}")
        print(f"Average SCS-ID F1: {df['SCS_ID_F1'].mean():.4f}")
        print(f"Average F1 Improvement: {df['F1_Improvement'].mean():.4f}")
        print(f"Classes with F1 Improvement: {(df['F1_Improvement'] > 0).sum()}")
        
        return df
    
    return None

if __name__ == "__main__":
    extract_per_class_results()
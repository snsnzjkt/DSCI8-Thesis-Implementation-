import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def create_comprehensive_test_results():
    """Create comprehensive test results CSV with predictions, labels, and analysis"""
    
    # Load both result files
    with open("results/baseline/baseline_results.pkl", 'rb') as f:
        baseline_results = pickle.load(f)
    
    with open("results/scs_id/scs_id_optimized_results.pkl", 'rb') as f:
        scs_id_results = pickle.load(f)
    
    # Extract data
    true_labels = baseline_results['labels']  # Both should have same test set
    baseline_preds = baseline_results['predictions']
    scs_id_preds = scs_id_results['predictions']
    class_names = baseline_results['class_names']
    
    # Create comprehensive DataFrame
    test_results_df = pd.DataFrame({
        'true_labels': true_labels,
        'baseline_predictions': baseline_preds,
        'scs_id_predictions': scs_id_preds,
        'true_class_names': [class_names[int(label)] for label in true_labels],
        'baseline_pred_names': [class_names[int(pred)] for pred in baseline_preds],
        'scs_id_pred_names': [class_names[int(pred)] for pred in scs_id_preds],
        'baseline_correct': [int(true == pred) for true, pred in zip(true_labels, baseline_preds)],
        'scs_id_correct': [int(true == pred) for true, pred in zip(true_labels, scs_id_preds)]
    })
    
    # Save to CSV
    test_results_df.to_csv('comprehensive_test_results.csv', index=False)
    print(f"Saved comprehensive test results to: comprehensive_test_results.csv")
    print(f"Shape: {test_results_df.shape}")
    print(f"Columns: {list(test_results_df.columns)}")
    
    # Create summary statistics
    summary_stats = {
        'Total_Test_Samples': len(true_labels),
        'Baseline_Accuracy': sum(test_results_df['baseline_correct']) / len(true_labels),
        'SCS_ID_Accuracy': sum(test_results_df['scs_id_correct']) / len(true_labels),
        'Baseline_Correct': sum(test_results_df['baseline_correct']),
        'SCS_ID_Correct': sum(test_results_df['scs_id_correct']),
        'Unique_Classes': len(class_names),
        'Class_Names': class_names
    }
    
    # Save summary
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('test_results_summary.csv', index=False)
    print(f"\nSaved summary to: test_results_summary.csv")
    
    # Create per-class breakdown
    per_class_results = []
    for i, class_name in enumerate(class_names):
        class_mask = test_results_df['true_labels'] == i
        if class_mask.sum() > 0:
            baseline_class_acc = test_results_df.loc[class_mask, 'baseline_correct'].mean()
            scs_id_class_acc = test_results_df.loc[class_mask, 'scs_id_correct'].mean()
            
            per_class_results.append({
                'class_id': i,
                'class_name': class_name,
                'sample_count': class_mask.sum(),
                'baseline_accuracy': baseline_class_acc,
                'scs_id_accuracy': scs_id_class_acc,
                'accuracy_improvement': scs_id_class_acc - baseline_class_acc
            })
    
    per_class_df = pd.DataFrame(per_class_results)
    per_class_df.to_csv('per_class_test_results.csv', index=False)
    print(f"Saved per-class results to: per_class_test_results.csv")
    
    return test_results_df, summary_stats, per_class_df

# Create all result files
test_df, summary, per_class = create_comprehensive_test_results()

print("\n" + "="*60)
print("TEST RESULTS SUMMARY")
print("="*60)
print(f"Total test samples: {summary['Total_Test_Samples']:,}")
print(f"Baseline accuracy: {summary['Baseline_Accuracy']:.4f}")
print(f"SCS-ID accuracy: {summary['SCS_ID_Accuracy']:.4f}")
print(f"Accuracy improvement: {summary['SCS_ID_Accuracy'] - summary['Baseline_Accuracy']:.4f}")
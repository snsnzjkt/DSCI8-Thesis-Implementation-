# experiments/run_deepseek_feature_selection.py
"""
Complete DeepSeek RL Feature Selection Pipeline
Integrates with SCS-ID framework for optimal feature selection (78 → 42)

This script:
1. Loads preprocessed CIC-IDS2017 data
2. Trains DeepSeek RL agent
3. Evaluates selected features
4. Saves results for SCS-ID training
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.deepseek_rl import DeepSeekRL, evaluate_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data(data_path='data/processed'):
    """
    Load preprocessed CIC-IDS2017 dataset
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    print("="*70)
    print("📂 Loading Preprocessed Data")
    print("="*70)
    
    try:
        # Load preprocessed data
        X_train = np.load(f'{data_path}/X_train.npy')
        X_val = np.load(f'{data_path}/X_val.npy')
        X_test = np.load(f'{data_path}/X_test.npy')
        y_train = np.load(f'{data_path}/y_train.npy')
        y_val = np.load(f'{data_path}/y_val.npy')
        y_test = np.load(f'{data_path}/y_test.npy')
        
        # Load feature names if available
        try:
            with open(f'{data_path}/feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
        except:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {len(np.unique(y_train))}")
        print("="*70 + "\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
        
    except FileNotFoundError:
        print("⚠️  Preprocessed data not found!")
        print("   Please run data preprocessing first:")
        print("   $ python data/preprocess.py")
        sys.exit(1)


def train_deepseek_rl(X_train, y_train, X_val, y_val, 
                       target_features=42, episodes=200, output_dir='results/deepseek_rl'):
    """
    Train DeepSeek RL feature selector
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        target_features: Number of features to select (default: 42)
        episodes: Training episodes
        output_dir: Directory to save results
        
    Returns:
        Trained DeepSeekRL instance and selected features
    """
    print("\n" + "="*70)
    print("🧠 Training DeepSeek RL Feature Selector")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DeepSeek RL
    deepseek = DeepSeekRL(max_features=target_features)
    
    # Train the model
    history = deepseek.fit(
        X_train, y_train, 
        X_val, y_val,
        episodes=episodes,
        target_network_update=10,
        verbose=True
    )
    
    # Get selected features
    selected_features = deepseek.get_selected_features()
    
    # Save model
    model_path = f'{output_dir}/deepseek_rl_model.pth'
    deepseek.save_model(model_path)
    
    # Save selected feature indices
    np.save(f'{output_dir}/selected_features.npy', selected_features)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{output_dir}/training_history.csv', index=False)
    
    # Plot training history
    deepseek.plot_training_history(save_path=f'{output_dir}/training_history.png')
    
    print(f"\n✅ Model and results saved to: {output_dir}")
    
    return deepseek, selected_features


def evaluate_selected_features(X_train, y_train, X_test, y_test, 
                                selected_features, feature_names, output_dir='results/deepseek_rl'):
    """
    Evaluate the selected features using Random Forest
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        selected_features: Indices of selected features
        feature_names: List of feature names
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("📊 Evaluating Selected Features")
    print("="*70)
    
    # Transform data using selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Selected features: {X_train_selected.shape[1]}")
    print(f"Reduction: {(1 - X_train_selected.shape[1]/X_train.shape[1])*100:.1f}%\n")
    
    # Train Random Forest on selected features
    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train_selected, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test_selected)
    
    # Classification report
    print("\n📈 Classification Report (Selected Features):")
    print("-"*70)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'{output_dir}/classification_report.csv')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - DeepSeek RL Selected Features', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance evaluation
    print("\n🔍 Evaluating Feature Importance...")
    importance_results = evaluate_feature_importance(
        X_train, y_train,
        selected_features=selected_features,
        top_k=20,
        feature_names=feature_names
    )
    
    # Save feature importance
    if importance_results:
        importance_df = pd.DataFrame({
            'feature_idx': selected_features,
            'feature_name': [feature_names[i] for i in selected_features],
            'importance_score': importance_results['combined_scores'],
            'rf_importance': importance_results['rf_importances'],
            'perm_importance': importance_results['perm_importances']
        })
        importance_df = importance_df.sort_values('importance_score', ascending=False)
        importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_20 = importance_df.head(20)
        plt.barh(range(len(top_20)), top_20['importance_score'].to_numpy())
        plt.yticks(range(len(top_20)), top_20['feature_name'].tolist())
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 20 Feature Importance (DeepSeek RL)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate false positive rate
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fp_rate = fp.sum() / (fp.sum() + tn.sum() + 1e-10)
    
    # Save metrics summary
    metrics_summary = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positive_rate': fp_rate,
        'original_features': X_train.shape[1],
        'selected_features': len(selected_features),
        'reduction_percentage': (1 - len(selected_features)/X_train.shape[1])*100
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(f'{output_dir}/metrics_summary.csv', index=False)
    
    print("\n" + "="*70)
    print("📊 Performance Metrics Summary")
    print("="*70)
    print(f"Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"FP Rate:         {fp_rate:.4f} ({fp_rate*100:.2f}%)")
    print(f"Feature Reduction: {metrics_summary['reduction_percentage']:.1f}%")
    print("="*70)
    
    # Thesis requirements check
    print("\n" + "="*70)
    print("📋 Thesis Requirements Check")
    print("="*70)
    accuracy_target = 0.99
    fp_reduction_target = 0.20  # 20% reduction from baseline
    feature_target = 42
    
    print(f"✓ Feature Selection: {len(selected_features)}/{feature_target} features")
    print(f"  Status: {'✅ PASS' if len(selected_features) == feature_target else '⚠️  NEEDS ADJUSTMENT'}")
    
    print(f"✓ Detection Accuracy: {accuracy:.4f}")
    print(f"  Target: >{accuracy_target:.2f}")
    print(f"  Status: {'✅ PASS' if accuracy > accuracy_target else '⚠️  NEEDS IMPROVEMENT'}")
    
    print(f"✓ False Positive Rate: {fp_rate:.4f}")
    print(f"  Status: {'✅ LOW' if fp_rate < 0.02 else '⚠️  MODERATE' if fp_rate < 0.05 else '❌ HIGH'}")
    
    print("="*70 + "\n")
    
    return metrics_summary


def compare_with_baseline(X_train, y_train, X_test, y_test, 
                          selected_features, output_dir='results/deepseek_rl'):
    """
    Compare selected features with using all features
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        selected_features: Indices of selected features
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("⚖️  Comparing with Baseline (All Features)")
    print("="*70)
    
    from sklearn.metrics import accuracy_score, f1_score
    import time
    
    # Train on ALL features
    print("\n1️⃣  Training on ALL features...")
    rf_all = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    rf_all.fit(X_train, y_train)
    train_time_all = time.time() - start_time
    
    start_time = time.time()
    y_pred_all = rf_all.predict(X_test)
    inference_time_all = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    acc_all = accuracy_score(y_test, y_pred_all)
    f1_all = f1_score(y_test, y_pred_all, average='weighted')
    
    # Train on SELECTED features
    print("2️⃣  Training on SELECTED features...")
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    rf_selected = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    rf_selected.fit(X_train_selected, y_train)
    train_time_selected = time.time() - start_time
    
    start_time = time.time()
    y_pred_selected = rf_selected.predict(X_test_selected)
    inference_time_selected = (time.time() - start_time) / len(X_test_selected) * 1000
    
    acc_selected = accuracy_score(y_test, y_pred_selected)
    f1_selected = f1_score(y_test, y_pred_selected, average='weighted')
    
    # Calculate improvements
    print("\n" + "="*70)
    print("📊 Comparison Results")
    print("="*70)
    print(f"\n{'Metric':<30} {'All Features':<20} {'Selected Features':<20} {'Improvement':<15}")
    print("-"*85)
    print(f"{'Number of Features':<30} {X_train.shape[1]:<20} {len(selected_features):<20} "
          f"{-(1-len(selected_features)/X_train.shape[1])*100:>13.1f}%")
    print(f"{'Accuracy':<30} {acc_all:<20.4f} {acc_selected:<20.4f} "
          f"{(acc_selected-acc_all)*100:>13.2f}%")
    print(f"{'F1-Score':<30} {f1_all:<20.4f} {f1_selected:<20.4f} "
          f"{(f1_selected-f1_all)*100:>13.2f}%")
    print(f"{'Training Time (s)':<30} {train_time_all:<20.2f} {train_time_selected:<20.2f} "
          f"{-(1-train_time_selected/train_time_all)*100:>13.1f}%")
    print(f"{'Inference Time (ms/sample)':<30} {inference_time_all:<20.4f} {inference_time_selected:<20.4f} "
          f"{-(1-inference_time_selected/inference_time_all)*100:>13.1f}%")
    print("="*70 + "\n")
    
    # Save comparison
    comparison_df = pd.DataFrame({
        'metric': ['features', 'accuracy', 'f1_score', 'train_time_s', 'inference_time_ms'],
        'all_features': [X_train.shape[1], acc_all, f1_all, train_time_all, inference_time_all],
        'selected_features': [len(selected_features), acc_selected, f1_selected, 
                              train_time_selected, inference_time_selected],
        'improvement_pct': [
            -(1-len(selected_features)/X_train.shape[1])*100,
            (acc_selected-acc_all)*100,
            (f1_selected-f1_all)*100,
            -(1-train_time_selected/train_time_all)*100,
            -(1-inference_time_selected/inference_time_all)*100
        ]
    })
    comparison_df.to_csv(f'{output_dir}/baseline_comparison.csv', index=False)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    metrics = ['Accuracy', 'F1-Score']
    all_vals = [acc_all, f1_all]
    selected_vals = [acc_selected, f1_selected]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, all_vals, width, label='All Features', alpha=0.8)
    axes[0].bar(x + width/2, selected_vals, width, label='Selected Features', alpha=0.8)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Efficiency comparison
    efficiency_metrics = ['Features', 'Train Time', 'Inference Time']
    all_eff = [X_train.shape[1], train_time_all, inference_time_all]
    selected_eff = [len(selected_features), train_time_selected, inference_time_selected]
    
    # Normalize for visualization
    all_eff_norm = [v/max(all_eff[i], selected_eff[i]) for i, v in enumerate(all_eff)]
    selected_eff_norm = [v/max(all_eff[i], selected_eff[i]) for i, v in enumerate(selected_eff)]
    
    x2 = np.arange(len(efficiency_metrics))
    axes[1].bar(x2 - width/2, all_eff_norm, width, label='All Features', alpha=0.8)
    axes[1].bar(x2 + width/2, selected_eff_norm, width, label='Selected Features', alpha=0.8)
    axes[1].set_ylabel('Normalized Value (lower is better)')
    axes[1].set_title('Efficiency Comparison', fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(efficiency_metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison results saved to: {output_dir}")


def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("🚀 DeepSeek RL Feature Selection - Complete Pipeline")
    print("="*70 + "\n")
    
    # Configuration
    TARGET_FEATURES = 42  # As per thesis
    EPISODES = 200  # Training episodes
    OUTPUT_DIR = 'results/deepseek_rl'
    
    # Step 1: Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_preprocessed_data()
    
    # Step 2: Train DeepSeek RL
    deepseek, selected_features = train_deepseek_rl(
        X_train, y_train, X_val, y_val,
        target_features=TARGET_FEATURES,
        episodes=EPISODES,
        output_dir=OUTPUT_DIR
    )
    
    # Step 3: Evaluate selected features
    metrics = evaluate_selected_features(
        X_train, y_train, X_test, y_test,
        selected_features, feature_names,
        output_dir=OUTPUT_DIR
    )
    
    # Step 4: Compare with baseline
    compare_with_baseline(
        X_train, y_train, X_test, y_test,
        selected_features,
        output_dir=OUTPUT_DIR
    )
    
    # Final summary
    print("\n" + "="*70)
    print("✅ Pipeline Complete!")
    print("="*70)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • deepseek_rl_model.pth - Trained model")
    print("  • selected_features.npy - Selected feature indices")
    print("  • training_history.csv - Training progress")
    print("  • training_history.png - Training visualization")
    print("  • classification_report.csv - Performance metrics")
    print("  • confusion_matrix.png - Confusion matrix")
    print("  • feature_importance.csv - Feature importance scores")
    print("  • feature_importance.png - Feature importance plot")
    print("  • metrics_summary.csv - Overall metrics")
    print("  • baseline_comparison.csv - Comparison with all features")
    print("  • comparison_visualization.png - Visual comparison")
    print("\n" + "="*70)
    print("\n🎯 Next Steps:")
    print("   1. Use selected features for SCS-ID training")
    print("   2. Run: python experiments/train_scs_id.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
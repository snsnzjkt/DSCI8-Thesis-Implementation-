# -*- coding: utf-8 -*-
# experiments/per_class_analysis.py - Comprehensive Per-Class Performance Analysis
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    accuracy_score, roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import config
except ImportError:
    class Config:
        RESULTS_DIR = "results"
        DATA_DIR = "data"
        DEVICE = "cpu"  # Default to CPU
    config = Config()

class PerClassAnalyzer:
    """
    Comprehensive per-class performance analysis for CIC-IDS2017 dataset
    Analyzes precision, recall, F1-score, and misclassification patterns for 15 attack types
    """
    
    def __init__(self):
        self.baseline_results = None
        self.scs_id_results = None
        self.class_names = None
        self.analysis_results = {}
        
    def load_results(self):
        """Load baseline and SCS-ID model results"""
        print("📊 Loading model results for per-class analysis...")
        
        baseline_path = Path(config.RESULTS_DIR) / "baseline" / "baseline_results.pkl"
        scs_id_path = Path(config.RESULTS_DIR) / "scs_id" / "scs_id_optimized_results.pkl"
        
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline results not found: {baseline_path}")
        if not scs_id_path.exists():
            raise FileNotFoundError(f"SCS-ID results not found: {scs_id_path}")
            
        with open(baseline_path, 'rb') as f:
            self.baseline_results = pickle.load(f)
        with open(scs_id_path, 'rb') as f:
            self.scs_id_results = pickle.load(f)
            
        self.class_names = self.baseline_results.get('class_names', [])
        print(f"   ✅ Loaded results for {len(self.class_names)} classes")
        print(f"   📋 Classes: {', '.join(self.class_names[:5])}..." if len(self.class_names) > 5 else f"   📋 Classes: {', '.join(self.class_names)}")
        
    def analyze_per_class_metrics(self):
        """Analyze precision, recall, and F1-score for each class"""
        print("\n🎯 Analyzing per-class metrics...")
        
        # Get predictions and labels
        baseline_preds = np.array(self.baseline_results['predictions'])
        scs_id_preds = np.array(self.scs_id_results['predictions'])
        true_labels = np.array(self.baseline_results['labels'])
        
        # Calculate detailed metrics for both models
        baseline_report = classification_report(true_labels, baseline_preds, 
                                              target_names=self.class_names, 
                                              output_dict=True, zero_division=0)
        scs_id_report = classification_report(true_labels, scs_id_preds, 
                                            target_names=self.class_names, 
                                            output_dict=True, zero_division=0)
        
        # Create comprehensive comparison DataFrame
        metrics_data = []
        for class_name in self.class_names:
            if class_name in baseline_report and class_name in scs_id_report:
                baseline_metrics = baseline_report[class_name]
                scs_id_metrics = scs_id_report[class_name]
                
                metrics_data.append({
                    'Class': class_name,
                    'Attack_Type': 'Normal' if class_name == 'BENIGN' else 'Attack',
                    'Baseline_Precision': baseline_metrics['precision'],
                    'SCS-ID_Precision': scs_id_metrics['precision'],
                    'Precision_Improvement': scs_id_metrics['precision'] - baseline_metrics['precision'],
                    'Baseline_Recall': baseline_metrics['recall'],
                    'SCS-ID_Recall': scs_id_metrics['recall'],
                    'Recall_Improvement': scs_id_metrics['recall'] - baseline_metrics['recall'],
                    'Baseline_F1': baseline_metrics['f1-score'],
                    'SCS-ID_F1': scs_id_metrics['f1-score'],
                    'F1_Improvement': scs_id_metrics['f1-score'] - baseline_metrics['f1-score'],
                    'Support': baseline_metrics['support']
                })
        
        self.per_class_df = pd.DataFrame(metrics_data)
        self.analysis_results['per_class_metrics'] = self.per_class_df
        
        # Print summary
        print(f"   📈 Average F1 Improvement: {self.per_class_df['F1_Improvement'].mean():.4f}")
        print(f"   🎯 Classes with F1 improvement: {(self.per_class_df['F1_Improvement'] > 0).sum()}/{len(self.per_class_df)}")
        
        return self.per_class_df
    
    def analyze_confusion_matrices(self):
        """Generate and analyze confusion matrices"""
        print("\n🔍 Analyzing confusion matrices...")
        
        baseline_preds = np.array(self.baseline_results['predictions'])
        scs_id_preds = np.array(self.scs_id_results['predictions'])
        true_labels = np.array(self.baseline_results['labels'])
        
        # Calculate confusion matrices
        self.baseline_cm = confusion_matrix(true_labels, baseline_preds)
        self.scs_id_cm = confusion_matrix(true_labels, scs_id_preds)
        
        # Calculate per-class error rates
        baseline_errors = self._calculate_class_error_rates(self.baseline_cm)
        scs_id_errors = self._calculate_class_error_rates(self.scs_id_cm)
        
        self.analysis_results['confusion_matrices'] = {
            'baseline': self.baseline_cm,
            'scs_id': self.scs_id_cm,
            'baseline_errors': baseline_errors,
            'scs_id_errors': scs_id_errors
        }
        
        print(f"   ✅ Confusion matrices calculated ({len(self.class_names)}x{len(self.class_names)})")
        
    def _calculate_class_error_rates(self, cm):
        """Calculate per-class false positive and false negative rates"""
        n_classes = len(self.class_names)
        error_rates = {}
        
        for i, class_name in enumerate(self.class_names):
            # True Positives, False Positives, False Negatives, True Negatives
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp  # Sum of column i minus diagonal
            fn = np.sum(cm[i, :]) - tp  # Sum of row i minus diagonal
            tn = np.sum(cm) - tp - fp - fn
            
            # Calculate rates
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            error_rates[class_name] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'fpr': fpr, 'fnr': fnr, 'precision': precision, 'recall': recall
            }
            
        return error_rates
    
    def rank_attack_difficulty(self):
        """Rank attack types by detection difficulty"""
        print("\n🏆 Ranking attack types by detection difficulty...")
        
        # Create difficulty ranking based on multiple metrics
        attack_classes = self.per_class_df[self.per_class_df['Attack_Type'] == 'Attack'].copy()
        
        # Calculate difficulty score (lower F1, higher difficulty)
        attack_classes['Baseline_Difficulty'] = 1 - attack_classes['Baseline_F1']
        attack_classes['SCS-ID_Difficulty'] = 1 - attack_classes['SCS-ID_F1']
        attack_classes['Avg_Difficulty'] = (attack_classes['Baseline_Difficulty'] + attack_classes['SCS-ID_Difficulty']) / 2
        
        # Sort by difficulty (hardest first)
        difficulty_ranking = attack_classes.sort_values('Avg_Difficulty', ascending=False)
        
        self.analysis_results['difficulty_ranking'] = difficulty_ranking
        
        print("   📊 Attack Difficulty Ranking (Hardest to Easiest):")
        for i, (_, row) in enumerate(difficulty_ranking.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['Class']:<25} (Avg F1: {1-row['Avg_Difficulty']:.3f})")
            
        return difficulty_ranking
    
    def analyze_misclassification_patterns(self):
        """Analyze common misclassification patterns"""
        print("\n🔍 Analyzing misclassification patterns...")
        
        # Find most common misclassifications for both models
        baseline_errors = self._find_common_misclassifications(self.baseline_cm, "Baseline")
        scs_id_errors = self._find_common_misclassifications(self.scs_id_cm, "SCS-ID")
        
        self.analysis_results['misclassification_patterns'] = {
            'baseline': baseline_errors,
            'scs_id': scs_id_errors
        }
        
        print("   📊 Top 5 Misclassification Patterns:")
        print("   " + "="*60)
        print("   Baseline CNN:")
        for i, (true_class, pred_class, count, pct) in enumerate(baseline_errors[:5], 1):
            print(f"   {i}. {true_class} → {pred_class}: {count} ({pct:.2f}%)")
        
        print("\n   SCS-ID:")
        for i, (true_class, pred_class, count, pct) in enumerate(scs_id_errors[:5], 1):
            print(f"   {i}. {true_class} → {pred_class}: {count} ({pct:.2f}%)")
    
    def _find_common_misclassifications(self, cm, model_name):
        """Find the most common misclassification patterns"""
        misclassifications = []
        total_predictions = np.sum(cm)
        
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:  # Misclassification
                    count = cm[i, j]
                    percentage = (count / total_predictions) * 100
                    misclassifications.append((
                        self.class_names[i],  # True class
                        self.class_names[j],  # Predicted class
                        count,
                        percentage
                    ))
        
        # Sort by count (most common first)
        return sorted(misclassifications, key=lambda x: x[2], reverse=True)
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating per-class analysis visualizations...")
        
        # Create output directory
        output_dir = Path(config.RESULTS_DIR) / "per_class_analysis"
        output_dir.mkdir(exist_ok=True)
        
        # 1. Per-class metrics comparison
        self._create_metrics_comparison_plot(output_dir)
        
        # 2. Confusion matrix heatmaps
        self._create_confusion_matrix_heatmaps(output_dir)
        
        # 3. Class difficulty ranking
        self._create_difficulty_ranking_plot(output_dir)
        
        # 4. Error rate comparison
        self._create_error_rate_comparison(output_dir)
        
        # 5. Misclassification pattern analysis
        self._create_misclassification_analysis(output_dir)
        
        print(f"   ✅ All visualizations saved to: {output_dir}")
    
    def _create_metrics_comparison_plot(self, output_dir):
        """Create per-class metrics comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sort by F1 improvement for better visualization
        df_sorted = self.per_class_df.sort_values('F1_Improvement', ascending=True)
        
        # 1. Precision comparison - Plot SCS-ID first, then Baseline on top
        axes[0,0].barh(range(len(df_sorted)), df_sorted['SCS-ID_Precision'], 
                      alpha=0.6, label='SCS-ID', color='#4ECDC4')
        axes[0,0].barh(range(len(df_sorted)), df_sorted['Baseline_Precision'], 
                      alpha=0.8, label='Baseline', color='#FF6B6B')
        axes[0,0].set_yticks(range(len(df_sorted)))
        axes[0,0].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                  for name in df_sorted['Class']], fontsize=8)
        axes[0,0].set_xlabel('Precision')
        axes[0,0].set_title('Per-Class Precision Comparison')
        axes[0,0].legend(['SCS-ID', 'Baseline'])  # Update legend order to match plot order
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Recall comparison - Plot SCS-ID first, then Baseline on top
        axes[0,1].barh(range(len(df_sorted)), df_sorted['SCS-ID_Recall'], 
                      alpha=0.6, label='SCS-ID', color='#4ECDC4')
        axes[0,1].barh(range(len(df_sorted)), df_sorted['Baseline_Recall'], 
                      alpha=0.8, label='Baseline', color='#FF6B6B')
        axes[0,1].set_yticks(range(len(df_sorted)))
        axes[0,1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                  for name in df_sorted['Class']], fontsize=8)
        axes[0,1].set_xlabel('Recall')
        axes[0,1].set_title('Per-Class Recall Comparison')
        axes[0,1].legend(['SCS-ID', 'Baseline'])  # Update legend order to match plot order
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. F1-Score comparison - Plot SCS-ID first, then Baseline on top
        axes[1,0].barh(range(len(df_sorted)), df_sorted['SCS-ID_F1'], 
                      alpha=0.6, label='SCS-ID', color='#4ECDC4')
        axes[1,0].barh(range(len(df_sorted)), df_sorted['Baseline_F1'], 
                      alpha=0.8, label='Baseline', color='#FF6B6B')
        axes[1,0].set_yticks(range(len(df_sorted)))
        axes[1,0].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                  for name in df_sorted['Class']], fontsize=8)
        axes[1,0].set_xlabel('F1-Score')
        axes[1,0].set_title('Per-Class F1-Score Comparison')
        axes[1,0].legend(['SCS-ID', 'Baseline'])  # Update legend order to match plot order
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Improvement heatmap
        improvements = df_sorted[['Precision_Improvement', 'Recall_Improvement', 'F1_Improvement']].values
        im = axes[1,1].imshow(improvements.T, cmap='RdYlGn', aspect='auto')
        axes[1,1].set_xticks(range(len(df_sorted)))
        axes[1,1].set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                  for name in df_sorted['Class']], rotation=45, ha='right', fontsize=8)
        axes[1,1].set_yticks(range(3))
        axes[1,1].set_yticklabels(['Precision', 'Recall', 'F1-Score'])
        axes[1,1].set_title('Performance Improvements (SCS-ID vs Baseline)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1,1], orientation='horizontal', pad=0.1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Per-class metrics comparison saved")
    
    def _create_confusion_matrix_heatmaps(self, output_dir):
        """Create confusion matrix heatmaps"""
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Normalize confusion matrices
        baseline_cm_norm = self.baseline_cm.astype('float') / self.baseline_cm.sum(axis=1)[:, np.newaxis]
        scs_id_cm_norm = self.scs_id_cm.astype('float') / self.scs_id_cm.sum(axis=1)[:, np.newaxis]
        
        # Handle NaN values (classes with no samples)
        baseline_cm_norm = np.nan_to_num(baseline_cm_norm)
        scs_id_cm_norm = np.nan_to_num(scs_id_cm_norm)
        
        # 1. Baseline confusion matrix
        sns.heatmap(baseline_cm_norm, annot=False, cmap='Blues', ax=axes[0],
                   xticklabels=[name[:10] for name in self.class_names],
                   yticklabels=[name[:10] for name in self.class_names])
        axes[0].set_title('Baseline CNN - Confusion Matrix (Normalized)')
        axes[0].set_xlabel('Predicted Class')
        axes[0].set_ylabel('True Class')
        
        # 2. SCS-ID confusion matrix
        sns.heatmap(scs_id_cm_norm, annot=False, cmap='Greens', ax=axes[1],
                   xticklabels=[name[:10] for name in self.class_names],
                   yticklabels=[name[:10] for name in self.class_names])
        axes[1].set_title('SCS-ID - Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted Class')
        axes[1].set_ylabel('True Class')
        
        # 3. Difference matrix (SCS-ID - Baseline)
        diff_matrix = scs_id_cm_norm - baseline_cm_norm
        sns.heatmap(diff_matrix, annot=False, cmap='RdBu_r', center=0, ax=axes[2],
                   xticklabels=[name[:10] for name in self.class_names],
                   yticklabels=[name[:10] for name in self.class_names])
        axes[2].set_title('Improvement Matrix (SCS-ID - Baseline)')
        axes[2].set_xlabel('Predicted Class')
        axes[2].set_ylabel('True Class')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Confusion matrix heatmaps saved")
    
    def _create_difficulty_ranking_plot(self, output_dir):
        """Create attack difficulty ranking plot"""
        if 'difficulty_ranking' not in self.analysis_results:
            return
            
        difficulty_df = self.analysis_results['difficulty_ranking'].head(10)
        
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar plot - Plot SCS-ID first, then Baseline on top
        y_pos = range(len(difficulty_df))
        
        plt.barh(y_pos, 1 - difficulty_df['SCS-ID_Difficulty'], 
                alpha=0.6, label='SCS-ID F1', color='#4ECDC4')
        plt.barh(y_pos, 1 - difficulty_df['Baseline_Difficulty'], 
                alpha=0.8, label='Baseline F1', color='#FF6B6B')
        
        plt.yticks(y_pos, [name[:25] + '...' if len(name) > 25 else name 
                          for name in difficulty_df['Class']])
        plt.xlabel('F1-Score (Higher = Easier to Detect)')
        plt.title('Attack Type Detection Difficulty Ranking\n(Top 10 Most Challenging)')
        plt.legend(['SCS-ID F1', 'Baseline F1'])  # Update legend order to match plot order
        plt.grid(True, alpha=0.3)
        
        # Add improvement annotations
        for i, (_, row) in enumerate(difficulty_df.iterrows()):
            improvement = row['F1_Improvement']
            if improvement > 0:
                plt.annotate(f'+{improvement:.3f}', 
                           xy=(1 - row['SCS-ID_Difficulty'], i),
                           xytext=(5, 0), textcoords='offset points',
                           fontsize=8, color='green', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'attack_difficulty_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Attack difficulty ranking saved")
    
    def _create_error_rate_comparison(self, output_dir):
        """Create error rate comparison plot"""
        # Extract error rates for comparison
        baseline_errors = self.analysis_results['confusion_matrices']['baseline_errors']
        scs_id_errors = self.analysis_results['confusion_matrices']['scs_id_errors']
        
        # Create DataFrame for error rates
        error_data = []
        for class_name in self.class_names:
            if class_name in baseline_errors and class_name in scs_id_errors:
                error_data.append({
                    'Class': class_name,
                    'Baseline_FPR': baseline_errors[class_name]['fpr'],
                    'SCS-ID_FPR': scs_id_errors[class_name]['fpr'],
                    'Baseline_FNR': baseline_errors[class_name]['fnr'],
                    'SCS-ID_FNR': scs_id_errors[class_name]['fnr'],
                    'FPR_Improvement': baseline_errors[class_name]['fpr'] - scs_id_errors[class_name]['fpr'],
                    'FNR_Improvement': baseline_errors[class_name]['fnr'] - scs_id_errors[class_name]['fnr']
                })
        
        error_df = pd.DataFrame(error_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. False Positive Rate comparison
        axes[0,0].scatter(error_df['Baseline_FPR'], error_df['SCS-ID_FPR'], 
                         alpha=0.7, s=60, color='#FF6B6B')
        axes[0,0].plot([0, max(error_df['Baseline_FPR'])], [0, max(error_df['Baseline_FPR'])], 
                      'k--', alpha=0.5, label='Equal Performance')
        axes[0,0].set_xlabel('Baseline FPR')
        axes[0,0].set_ylabel('SCS-ID FPR')
        axes[0,0].set_title('False Positive Rate Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. False Negative Rate comparison
        axes[0,1].scatter(error_df['Baseline_FNR'], error_df['SCS-ID_FNR'], 
                         alpha=0.7, s=60, color='#4ECDC4')
        axes[0,1].plot([0, max(error_df['Baseline_FNR'])], [0, max(error_df['Baseline_FNR'])], 
                      'k--', alpha=0.5, label='Equal Performance')
        axes[0,1].set_xlabel('Baseline FNR')
        axes[0,1].set_ylabel('SCS-ID FNR')
        axes[0,1].set_title('False Negative Rate Comparison')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. FPR improvements
        error_df_sorted = error_df.sort_values('FPR_Improvement', ascending=True)
        bars1 = axes[1,0].barh(range(len(error_df_sorted)), error_df_sorted['FPR_Improvement'], 
                              color=['green' if x > 0 else 'red' for x in error_df_sorted['FPR_Improvement']])
        axes[1,0].set_yticks(range(len(error_df_sorted)))
        axes[1,0].set_yticklabels([name[:15] for name in error_df_sorted['Class']], fontsize=8)
        axes[1,0].set_xlabel('FPR Improvement (Baseline - SCS-ID)')
        axes[1,0].set_title('False Positive Rate Improvements')
        axes[1,0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. FNR improvements
        error_df_sorted = error_df.sort_values('FNR_Improvement', ascending=True)
        bars2 = axes[1,1].barh(range(len(error_df_sorted)), error_df_sorted['FNR_Improvement'], 
                              color=['green' if x > 0 else 'red' for x in error_df_sorted['FNR_Improvement']])
        axes[1,1].set_yticks(range(len(error_df_sorted)))
        axes[1,1].set_yticklabels([name[:15] for name in error_df_sorted['Class']], fontsize=8)
        axes[1,1].set_xlabel('FNR Improvement (Baseline - SCS-ID)')
        axes[1,1].set_title('False Negative Rate Improvements')
        axes[1,1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Error rate comparison saved")
    
    def _create_misclassification_analysis(self, output_dir):
        """Create misclassification pattern analysis"""
        baseline_errors = self.analysis_results['misclassification_patterns']['baseline'][:10]
        scs_id_errors = self.analysis_results['misclassification_patterns']['scs_id'][:10]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. Baseline top misclassifications
        if baseline_errors:
            true_classes = [f"{err[0][:15]}..." if len(err[0]) > 15 else err[0] for err in baseline_errors]
            pred_classes = [f"{err[1][:15]}..." if len(err[1]) > 15 else err[1] for err in baseline_errors]
            counts = [err[2] for err in baseline_errors]
            
            bars1 = axes[0].barh(range(len(baseline_errors)), counts, color='#FF6B6B', alpha=0.7)
            axes[0].set_yticks(range(len(baseline_errors)))
            axes[0].set_yticklabels([f"{tc} → {pc}" for tc, pc in zip(true_classes, pred_classes)], fontsize=9)
            axes[0].set_xlabel('Number of Misclassifications')
            axes[0].set_title('Top 10 Misclassification Patterns - Baseline CNN')
            axes[0].grid(True, alpha=0.3)
            
            # Add percentage labels
            for i, (bar, (_, _, count, pct)) in enumerate(zip(bars1, baseline_errors)):
                axes[0].text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                           f'{pct:.2f}%', va='center', fontsize=8)
        
        # 2. SCS-ID top misclassifications
        if scs_id_errors:
            true_classes = [f"{err[0][:15]}..." if len(err[0]) > 15 else err[0] for err in scs_id_errors]
            pred_classes = [f"{err[1][:15]}..." if len(err[1]) > 15 else err[1] for err in scs_id_errors]
            counts = [err[2] for err in scs_id_errors]
            
            bars2 = axes[1].barh(range(len(scs_id_errors)), counts, color='#4ECDC4', alpha=0.7)
            axes[1].set_yticks(range(len(scs_id_errors)))
            axes[1].set_yticklabels([f"{tc} → {pc}" for tc, pc in zip(true_classes, pred_classes)], fontsize=9)
            axes[1].set_xlabel('Number of Misclassifications')
            axes[1].set_title('Top 10 Misclassification Patterns - SCS-ID')
            axes[1].grid(True, alpha=0.3)
            
            # Add percentage labels
            for i, (bar, (_, _, count, pct)) in enumerate(zip(bars2, scs_id_errors)):
                axes[1].text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                           f'{pct:.2f}%', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'misclassification_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Misclassification pattern analysis saved")
    
    def generate_detailed_report(self):
        """Generate comprehensive per-class analysis report"""
        print("\n📋 Generating detailed per-class analysis report...")
        
        output_dir = Path(config.RESULTS_DIR) / "per_class_analysis"
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / "per_class_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PER-CLASS PERFORMANCE ANALYSIS REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"CIC-IDS2017 Dataset Analysis - {len(self.class_names)} Classes\n\n")
            
            # 1. Executive Summary
            f.write("1. EXECUTIVE SUMMARY\n")
            f.write("-"*20 + "\n")
            avg_f1_improvement = self.per_class_df['F1_Improvement'].mean()
            classes_improved = (self.per_class_df['F1_Improvement'] > 0).sum()
            total_classes = len(self.per_class_df)
            
            f.write(f"Average F1-Score Improvement: {avg_f1_improvement:.4f}\n")
            f.write(f"Classes with Improved Performance: {classes_improved}/{total_classes} ({classes_improved/total_classes*100:.1f}%)\n")
            f.write(f"Best F1 Improvement: {self.per_class_df['F1_Improvement'].max():.4f} ({self.per_class_df.loc[self.per_class_df['F1_Improvement'].idxmax(), 'Class']})\n")
            f.write(f"Worst F1 Change: {self.per_class_df['F1_Improvement'].min():.4f} ({self.per_class_df.loc[self.per_class_df['F1_Improvement'].idxmin(), 'Class']})\n\n")
            
            # 2. Per-Class Detailed Results
            f.write("2. DETAILED PER-CLASS RESULTS\n")
            f.write("-"*30 + "\n")
            f.write(f"{'Class':<25} {'Baseline F1':<12} {'SCS-ID F1':<12} {'Improvement':<12} {'Support':<10}\n")
            f.write("-"*80 + "\n")
            
            for _, row in self.per_class_df.sort_values('F1_Improvement', ascending=False).iterrows():
                class_name = row['Class'][:24]  # Truncate if too long
                f.write(f"{class_name:<25} {row['Baseline_F1']:<12.4f} {row['SCS-ID_F1']:<12.4f} "
                       f"{row['F1_Improvement']:<12.4f} {row['Support']:<10.0f}\n")
            
            # 3. Attack Difficulty Ranking
            if 'difficulty_ranking' in self.analysis_results:
                f.write("\n3. ATTACK DIFFICULTY RANKING\n")
                f.write("-"*28 + "\n")
                f.write("Ranking based on average F1-score (lower = more difficult)\n\n")
                
                difficulty_df = self.analysis_results['difficulty_ranking']
                for i, (_, row) in enumerate(difficulty_df.head(10).iterrows(), 1):
                    avg_f1 = (row['Baseline_F1'] + row['SCS-ID_F1']) / 2
                    f.write(f"{i:2d}. {row['Class']:<30} (Avg F1: {avg_f1:.4f})\n")
            
            # 4. Top Misclassification Patterns
            f.write("\n4. TOP MISCLASSIFICATION PATTERNS\n")
            f.write("-"*34 + "\n")
            
            f.write("Baseline CNN:\n")
            baseline_errors = self.analysis_results['misclassification_patterns']['baseline'][:5]
            for i, (true_class, pred_class, count, pct) in enumerate(baseline_errors, 1):
                f.write(f"{i}. {true_class} → {pred_class}: {count} ({pct:.2f}%)\n")
            
            f.write("\nSCS-ID:\n")
            scs_id_errors = self.analysis_results['misclassification_patterns']['scs_id'][:5]
            for i, (true_class, pred_class, count, pct) in enumerate(scs_id_errors, 1):
                f.write(f"{i}. {true_class} → {pred_class}: {count} ({pct:.2f}%)\n")
            
            # 5. Class-Specific Insights
            f.write("\n5. CLASS-SPECIFIC INSIGHTS\n")
            f.write("-"*26 + "\n")
            
            # Most improved classes
            top_improved = self.per_class_df.nlargest(3, 'F1_Improvement')
            f.write("Most Improved Classes:\n")
            for _, row in top_improved.iterrows():
                f.write(f"• {row['Class']}: +{row['F1_Improvement']:.4f} F1 improvement\n")
            
            # Classes needing attention
            needs_attention = self.per_class_df[self.per_class_df['SCS-ID_F1'] < 0.9]
            if not needs_attention.empty:
                f.write(f"\nClasses Needing Attention (F1 < 0.9):\n")
                for _, row in needs_attention.iterrows():
                    f.write(f"• {row['Class']}: F1 = {row['SCS-ID_F1']:.4f}\n")
            
            f.write(f"\nReport generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"   ✅ Detailed report saved to: {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run the complete per-class analysis pipeline"""
        print("🚀 Starting Comprehensive Per-Class Analysis")
        print("="*60)
        
        try:
            # Load data
            self.load_results()
            
            # Run analyses
            self.analyze_per_class_metrics()
            self.analyze_confusion_matrices()
            self.rank_attack_difficulty()
            self.analyze_misclassification_patterns()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate report
            report_path = self.generate_detailed_report()
            
            print("\n" + "="*60)
            print("✅ PER-CLASS ANALYSIS COMPLETE!")
            print("="*60)
            print("📊 Key Findings:")
            avg_improvement = self.per_class_df['F1_Improvement'].mean()
            improved_classes = (self.per_class_df['F1_Improvement'] > 0).sum()
            total_classes = len(self.per_class_df)
            
            print(f"   📈 Average F1 Improvement: {avg_improvement:.4f}")
            print(f"   🎯 Classes Improved: {improved_classes}/{total_classes} ({improved_classes/total_classes*100:.1f}%)")
            print(f"   📋 Detailed Report: {report_path}")
            print(f"   📊 Visualizations: {Path(config.RESULTS_DIR) / 'per_class_analysis'}")
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Run per-class analysis"""
    analyzer = PerClassAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n[SUCCESS] Per-class analysis completed successfully!")
    else:
        print("\n[ERROR] Per-class analysis failed. Check the error messages above.")

if __name__ == "__main__":
    main()
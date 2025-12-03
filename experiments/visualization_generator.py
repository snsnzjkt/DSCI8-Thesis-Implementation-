"""
Comprehensive Visualization Generator
Creates all required charts and plots for model comparison analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List
from datetime import datetime

class ComprehensiveVisualizer:
    """Generate comprehensive visualizations for model comparison"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = {
            'baseline': '#FF6B6B',
            'scs_id': '#4ECDC4',
            'improvement': '#45B7D1',
            'significant': '#96CEB4',
            'non_significant': '#FFEAA7'
        }
        
        # Attack types
        self.attack_types = [
            'BENIGN', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
            'DoS slowloris', 'FTP-Patator', 'SSH-Patator', 'Web Attack - Brute Force',
            'Web Attack - Sql Injection', 'Web Attack - XSS', 'Infiltration',
            'Bot', 'PortScan', 'Heartbleed'
        ]
        
    def load_results(self, results_file: str = "complete_results.pkl"):
        """Load comparison results"""
        with open(self.results_dir / results_file, 'rb') as f:
            self.results = pickle.load(f)
        
    def create_computational_efficiency_charts(self):
        """Create charts for Hypothesis 1: Computational Efficiency"""
        print("📊 Creating computational efficiency visualizations...")
        
        baseline_comp = self.results['baseline']['computational']
        scs_id_comp = self.results['scs_id']['computational']
        efficiency = self.results['comparison']['efficiency_reductions']
        
        # 1. Parameter Count Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Parameter count comparison
        params = [baseline_comp['total_parameters'], scs_id_comp['total_parameters']]
        models = ['Baseline CNN\\n(Ayeni et al.)', 'SCS-ID\\n(Proposed)']
        bars1 = ax1.bar(models, params, color=[self.colors['baseline'], self.colors['scs_id']])
        ax1.set_ylabel('Number of Parameters')
        ax1.set_title('Parameter Count Comparison', fontweight='bold', fontsize=14)
        
        # Add value labels
        for bar, param in zip(bars1, params):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{param:,}', ha='center', va='bottom', fontweight='bold')
        
        # Memory usage comparison
        memory = [baseline_comp['peak_memory_mb'], scs_id_comp['peak_memory_mb']]
        bars2 = ax2.bar(models, memory, color=[self.colors['baseline'], self.colors['scs_id']])
        ax2.set_ylabel('Peak Memory Usage (MB)')
        ax2.set_title('Memory Utilization Comparison', fontweight='bold', fontsize=14)
        
        for bar, mem in zip(bars2, memory):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem:.2f} MB', ha='center', va='bottom', fontweight='bold')
        
        # Inference latency comparison
        latency = [baseline_comp['inference_latency_ms'], scs_id_comp['inference_latency_ms']]
        bars3 = ax3.bar(models, latency, color=[self.colors['baseline'], self.colors['scs_id']])
        ax3.set_ylabel('Inference Latency (ms per connection)')
        ax3.set_title('Inference Latency Comparison', fontweight='bold', fontsize=14)
        
        for bar, lat in zip(bars3, latency):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{lat:.3f} ms', ha='center', va='bottom', fontweight='bold')
        
        # Efficiency improvements
        improvements = [
            efficiency['parameter_count_reduction'],
            efficiency['memory_utilization_reduction'],
            efficiency['inference_latency_improvement']
        ]
        metrics = ['Parameter\\nReduction', 'Memory\\nReduction', 'Latency\\nImprovement']
        bars4 = ax4.bar(metrics, improvements, color=self.colors['improvement'])
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('Efficiency Improvements (SCS-ID vs Baseline)', fontweight='bold', fontsize=14)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for bar, imp in zip(bars4, improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:.1f}%', ha='center', va='bottom' if imp > 0 else 'top', 
                    fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'computational_efficiency_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def create_performance_comparison_charts(self):
        """Create charts for overall performance comparison"""
        print("📊 Creating performance comparison visualizations...")
        
        baseline_perf = self.results['baseline']['performance']
        scs_id_perf = self.results['scs_id']['performance']
        
        # Overall performance metrics comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main performance metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        baseline_vals = [
            baseline_perf['accuracy'],
            baseline_perf['precision_macro'],
            baseline_perf['recall_macro'],
            baseline_perf['f1_macro']
        ]
        scs_id_vals = [
            scs_id_perf['accuracy'],
            scs_id_perf['precision_macro'],
            scs_id_perf['recall_macro'],
            scs_id_perf['f1_macro']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline CNN', 
                       color=self.colors['baseline'])
        bars2 = ax1.bar(x + width/2, scs_id_vals, width, label='SCS-ID', 
                       color=self.colors['scs_id'])
        
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Comparison', fontweight='bold', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.set_ylim([0, 1.1])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # False Positive Rate and False Alarm Rate
        fpr_far_metrics = ['False Positive Rate', 'False Alarm Rate']
        baseline_fpr_far = [baseline_perf['fpr_overall'], baseline_perf['far_overall']]
        scs_id_fpr_far = [scs_id_perf['fpr_overall'], scs_id_perf['far_overall']]
        
        x2 = np.arange(len(fpr_far_metrics))
        bars3 = ax2.bar(x2 - width/2, baseline_fpr_far, width, label='Baseline CNN', 
                       color=self.colors['baseline'])
        bars4 = ax2.bar(x2 + width/2, scs_id_fpr_far, width, label='SCS-ID', 
                       color=self.colors['scs_id'])
        
        ax2.set_ylabel('Rate')
        ax2.set_title('False Positive Rate & False Alarm Rate', fontweight='bold', fontsize=14)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(fpr_far_metrics)
        ax2.legend()
        
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # AUC-ROC and MCC
        advanced_metrics = ['AUC-ROC', 'MCC']
        baseline_advanced = [baseline_perf['auc_roc'], baseline_perf['mcc_overall']]
        scs_id_advanced = [scs_id_perf['auc_roc'], scs_id_perf['mcc_overall']]
        
        x3 = np.arange(len(advanced_metrics))
        bars5 = ax3.bar(x3 - width/2, baseline_advanced, width, label='Baseline CNN', 
                       color=self.colors['baseline'])
        bars6 = ax3.bar(x3 + width/2, scs_id_advanced, width, label='SCS-ID', 
                       color=self.colors['scs_id'])
        
        ax3.set_ylabel('Score')
        ax3.set_title('AUC-ROC and Matthews Correlation Coefficient', fontweight='bold', fontsize=14)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(advanced_metrics)
        ax3.legend()
        
        for bars in [bars5, bars6]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Statistical significance overview
        h1_stats = self.results['statistical_tests']['hypothesis_1']
        comp_metrics = list(h1_stats.keys())
        significance = [h1_stats[metric]['significant'] for metric in comp_metrics]
        p_values = [h1_stats[metric]['p_value'] for metric in comp_metrics]
        
        colors = [self.colors['significant'] if sig else self.colors['non_significant'] 
                 for sig in significance]
        bars7 = ax4.bar(comp_metrics, p_values, color=colors)
        ax4.set_ylabel('p-value')
        ax4.set_title('Statistical Significance (Hypothesis 1)', fontweight='bold', fontsize=14)
        ax4.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        ax4.set_yscale('log')
        ax4.legend()
        
        # Rotate x-axis labels
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, p_val in zip(bars7, p_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{p_val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'performance_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def create_per_attack_analysis(self):
        """Create per-attack type analysis charts for Hypothesis 2"""
        print("📊 Creating per-attack analysis visualizations...")
        
        baseline_perf = self.results['baseline']['performance']
        scs_id_perf = self.results['scs_id']['performance']
        
        # Metrics to analyze
        metrics_info = {
            'f1_per_class': {'title': 'F1-Score by Attack Type', 'ylabel': 'F1-Score'},
            'precision_per_class': {'title': 'Precision by Attack Type', 'ylabel': 'Precision'},
            'recall_per_class': {'title': 'Recall by Attack Type', 'ylabel': 'Recall'},
            'fpr_per_class': {'title': 'False Positive Rate by Attack Type', 'ylabel': 'FPR'}
        }
        
        for metric_key, info in metrics_info.items():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            
            # Get data
            baseline_data = baseline_perf[metric_key]
            scs_id_data = scs_id_perf[metric_key]
            
            # Ensure we have the right number of attack types
            n_attacks = min(len(baseline_data), len(scs_id_data), len(self.attack_types))
            attack_names = self.attack_types[:n_attacks]
            baseline_values = baseline_data[:n_attacks]
            scs_id_values = scs_id_data[:n_attacks]
            
            # Comparison chart
            x = np.arange(len(attack_names))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline CNN', 
                           color=self.colors['baseline'], alpha=0.8)
            bars2 = ax1.bar(x + width/2, scs_id_values, width, label='SCS-ID', 
                           color=self.colors['scs_id'], alpha=0.8)
            
            ax1.set_ylabel(info['ylabel'])
            ax1.set_title(info['title'], fontweight='bold', fontsize=14)
            ax1.set_xticks(x)
            ax1.set_xticklabels(attack_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Statistical significance for each attack type
            if 'hypothesis_2' in self.results['statistical_tests']:
                h2_stats = self.results['statistical_tests']['hypothesis_2']
                if metric_key in h2_stats:
                    significance_data = []
                    p_values = []
                    
                    for attack in attack_names:
                        if attack in h2_stats[metric_key]:
                            sig_data = h2_stats[metric_key][attack]
                            significance_data.append(sig_data['significant'])
                            p_values.append(sig_data['p_value'])
                        else:
                            significance_data.append(False)
                            p_values.append(1.0)
                    
                    # Create significance chart
                    colors = [self.colors['significant'] if sig else self.colors['non_significant'] 
                             for sig in significance_data]
                    bars3 = ax2.bar(attack_names, p_values, color=colors, alpha=0.8)
                    ax2.set_ylabel('p-value')
                    ax2.set_title(f'Statistical Significance - {info["title"]}', 
                                 fontweight='bold', fontsize=14)
                    ax2.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
                    ax2.set_yscale('log')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    # Add significance labels
                    for bar, sig, p_val in zip(bars3, significance_data, p_values):
                        height = bar.get_height()
                        label = '✓' if sig else '✗'
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                label, ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / f'per_attack_{metric_key}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
    def create_confusion_matrices(self):
        """Create confusion matrix comparisons"""
        print("📊 Creating confusion matrix visualizations...")
        
        baseline_cm = self.results['baseline']['performance']['confusion_matrix']
        scs_id_cm = self.results['scs_id']['performance']['confusion_matrix']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Baseline confusion matrix
        sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Baseline CNN - Confusion Matrix', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # SCS-ID confusion matrix
        sns.heatmap(scs_id_cm, annot=True, fmt='d', cmap='Greens', ax=ax2)
        ax2.set_title('SCS-ID - Confusion Matrix', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'confusion_matrices_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        print("📊 Creating summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        baseline_perf = self.results['baseline']['performance']
        scs_id_perf = self.results['scs_id']['performance']
        baseline_comp = self.results['baseline']['computational']
        scs_id_comp = self.results['scs_id']['computational']
        efficiency = self.results['comparison']['efficiency_reductions']
        
        # Key metrics comparison (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        key_metrics = ['F1-Score', 'Accuracy', 'FPR']
        baseline_key = [baseline_perf['f1_macro'], baseline_perf['accuracy'], baseline_perf['fpr_overall']]
        scs_id_key = [scs_id_perf['f1_macro'], scs_id_perf['accuracy'], scs_id_perf['fpr_overall']]
        
        x = np.arange(len(key_metrics))
        width = 0.35
        bars1 = ax1.bar(x - width/2, baseline_key, width, label='Baseline CNN', 
                       color=self.colors['baseline'])
        bars2 = ax1.bar(x + width/2, scs_id_key, width, label='SCS-ID', 
                       color=self.colors['scs_id'])
        ax1.set_title('Key Performance Metrics', fontweight='bold', fontsize=16)
        ax1.set_xticks(x)
        ax1.set_xticklabels(key_metrics)
        ax1.legend()
        
        # Efficiency improvements (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        improvements = [
            efficiency['parameter_count_reduction'],
            efficiency['memory_utilization_reduction'], 
            efficiency['inference_latency_improvement']
        ]
        imp_labels = ['Parameters', 'Memory', 'Latency']
        bars3 = ax2.bar(imp_labels, improvements, color=self.colors['improvement'])
        ax2.set_title('Efficiency Improvements (%)', fontweight='bold', fontsize=16)
        ax2.set_ylabel('Improvement (%)')
        
        # Per-attack F1 scores (middle)
        ax3 = fig.add_subplot(gs[1, :])
        n_attacks = min(len(baseline_perf['f1_per_class']), len(self.attack_types))
        attack_subset = self.attack_types[:n_attacks]
        baseline_f1 = baseline_perf['f1_per_class'][:n_attacks]
        scs_id_f1 = scs_id_perf['f1_per_class'][:n_attacks]
        
        x = np.arange(len(attack_subset))
        bars4 = ax3.bar(x - width/2, baseline_f1, width, label='Baseline CNN', 
                       color=self.colors['baseline'])
        bars5 = ax3.bar(x + width/2, scs_id_f1, width, label='SCS-ID', 
                       color=self.colors['scs_id'])
        ax3.set_title('F1-Score by Attack Type', fontweight='bold', fontsize=16)
        ax3.set_xticks(x)
        ax3.set_xticklabels(attack_subset, rotation=45, ha='right')
        ax3.legend()
        
        # Statistical significance summary (bottom left)
        ax4 = fig.add_subplot(gs[2, :2])
        h1_stats = self.results['statistical_tests']['hypothesis_1']
        comp_metrics = list(h1_stats.keys())
        significance = [h1_stats[metric]['significant'] for metric in comp_metrics]
        p_values = [h1_stats[metric]['p_value'] for metric in comp_metrics]
        
        colors = [self.colors['significant'] if sig else self.colors['non_significant'] 
                 for sig in significance]
        bars6 = ax4.bar(comp_metrics, p_values, color=colors)
        ax4.set_title('Statistical Significance (Hypothesis 1)', fontweight='bold', fontsize=16)
        ax4.set_ylabel('p-value')
        ax4.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        ax4.set_yscale('log')
        ax4.legend()
        
        # Model size comparison (bottom right)
        ax5 = fig.add_subplot(gs[2, 2:])
        model_sizes = [baseline_comp['total_parameters'], scs_id_comp['total_parameters']]
        model_names = ['Baseline\\nCNN', 'SCS-ID']
        bars7 = ax5.bar(model_names, model_sizes, color=[self.colors['baseline'], self.colors['scs_id']])
        ax5.set_title('Model Size Comparison', fontweight='bold', fontsize=16)
        ax5.set_ylabel('Parameters')
        
        # Summary text (bottom)
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        summary_text = f"""
COMPREHENSIVE MODEL COMPARISON SUMMARY
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

HYPOTHESIS 1 - COMPUTATIONAL EFFICIENCY:
• Parameter Reduction: {efficiency['parameter_count_reduction']:.1f}%
• Memory Reduction: {efficiency['memory_utilization_reduction']:.1f}%
• Latency Improvement: {efficiency['inference_latency_improvement']:.1f}%

HYPOTHESIS 2 - DETECTION PERFORMANCE:
• SCS-ID F1-Score: {scs_id_perf['f1_macro']:.4f} vs Baseline: {baseline_perf['f1_macro']:.4f}
• SCS-ID FPR: {scs_id_perf['fpr_overall']:.4f} vs Baseline: {baseline_perf['fpr_overall']:.4f}
• SCS-ID Accuracy: {scs_id_perf['accuracy']:.4f} vs Baseline: {baseline_perf['accuracy']:.4f}

STATISTICAL SIGNIFICANCE:
• Parameter Count: {'✓ Significant' if h1_stats['parameter_count']['significant'] else '✗ Not Significant'}
• Memory Usage: {'✓ Significant' if h1_stats['memory_usage']['significant'] else '✗ Not Significant'}  
• Inference Latency: {'✓ Significant' if h1_stats['inference_latency']['significant'] else '✗ Not Significant'}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('SCS-ID vs Baseline CNN: Comprehensive Analysis Dashboard', 
                     fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'comprehensive_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def generate_all_visualizations(self):
        """Generate all required visualizations"""
        print("🎨 Generating comprehensive visualizations...")
        print("=" * 50)
        
        # Create all visualization types
        self.create_computational_efficiency_charts()
        self.create_performance_comparison_charts()
        self.create_per_attack_analysis()
        self.create_confusion_matrices()
        self.create_summary_dashboard()
        
        print("✅ All visualizations generated successfully!")
        print(f"📁 Visualizations saved to: {self.viz_dir}")
        
        return self.viz_dir

def main():
    """Generate visualizations from comparison results"""
    results_dir = Path("results/comprehensive_comparison")
    
    if not (results_dir / "complete_results.pkl").exists():
        print("❌ Results file not found. Please run comprehensive_model_comparison.py first.")
        return
    
    visualizer = ComprehensiveVisualizer(results_dir)
    visualizer.load_results()
    viz_dir = visualizer.generate_all_visualizations()
    
    return viz_dir

if __name__ == "__main__":
    main()
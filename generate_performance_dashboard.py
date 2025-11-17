import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformanceDashboard:
    def __init__(self):
        self.load_data()
        self.setup_directories()
    
    def load_data(self):
        """Load all necessary data"""
        # Load test results
        self.test_results = pd.read_csv('comprehensive_test_results.csv')
        self.per_class_results = pd.read_csv('per_class_test_results.csv')
        self.summary = pd.read_csv('test_results_summary.csv')
        
        # Load model results
        with open('results/baseline/baseline_results.pkl', 'rb') as f:
            self.baseline_results = pickle.load(f)
        
        with open('results/scs_id/scs_id_optimized_results.pkl', 'rb') as f:
            self.scs_id_results = pickle.load(f)
    
    def setup_directories(self):
        """Create output directory"""
        import os
        os.makedirs('results/performance_dashboard', exist_ok=True)
    
    def create_static_dashboard(self):
        """Create comprehensive static dashboard"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Comprehensive Performance Dashboard: SCS-ID vs Baseline CNN', fontsize=20, fontweight='bold')
        
        # 1. Overall Accuracy Comparison
        models = ['Baseline CNN', 'SCS-ID']
        accuracies = [self.baseline_results['test_accuracy'], self.scs_id_results['test_accuracy']]
        bars = axes[0,0].bar(models, accuracies, color=['#FF6B6B', '#4ECDC4'])
        axes[0,0].set_title('Overall Test Accuracy', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0.99, 1.0)
        for i, (model, acc) in enumerate(zip(models, accuracies)):
            axes[0,0].text(i, acc + 0.0001, f'{acc:.4f}', ha='center', fontweight='bold')
        
        # 2. F1-Score Comparison
        f1_scores = [self.baseline_results['f1_score'], self.scs_id_results['f1_score']]
        bars = axes[0,1].bar(models, f1_scores, color=['#FF6B6B', '#4ECDC4'])
        axes[0,1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].set_ylim(0.99, 1.0)
        for i, (model, f1) in enumerate(zip(models, f1_scores)):
            axes[0,1].text(i, f1 + 0.0001, f'{f1:.4f}', ha='center', fontweight='bold')
        
        # 3. Training Time Comparison
        training_times = [self.baseline_results['training_time']/3600, self.scs_id_results['training_time']/3600]
        bars = axes[0,2].bar(models, training_times, color=['#FF6B6B', '#4ECDC4'])
        axes[0,2].set_title('Training Time (Hours)', fontsize=14, fontweight='bold')
        axes[0,2].set_ylabel('Hours')
        for i, (model, time) in enumerate(zip(models, training_times)):
            axes[0,2].text(i, time + 0.1, f'{time:.1f}h', ha='center', fontweight='bold')
        
        # 4. Per-Class Accuracy Heatmap
        per_class_data = self.per_class_results[['class_name', 'baseline_accuracy', 'scs_id_accuracy']].set_index('class_name')
        im = axes[1,0].imshow(per_class_data.T, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.0)
        axes[1,0].set_title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
        axes[1,0].set_xticks(range(len(per_class_data.index)))
        axes[1,0].set_xticklabels(per_class_data.index, rotation=45, ha='right')
        axes[1,0].set_yticks([0, 1])
        axes[1,0].set_yticklabels(['Baseline', 'SCS-ID'])
        plt.colorbar(im, ax=axes[1,0])
        
        # 5. Model Parameters Comparison
        params_baseline = self.baseline_results['model_parameters']
        params_scs_id = self.scs_id_results['model_stats']['total_parameters']
        reduction = (params_baseline - params_scs_id) / params_baseline * 100
        
        bars = axes[1,1].bar(['Baseline CNN', 'SCS-ID'], [params_baseline/1000, params_scs_id/1000], 
                            color=['#FF6B6B', '#4ECDC4'])
        axes[1,1].set_title(f'Model Parameters (Reduction: {reduction:.1f}%)', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('Parameters (K)')
        for i, params in enumerate([params_baseline/1000, params_scs_id/1000]):
            axes[1,1].text(i, params + 1, f'{params:.1f}K', ha='center', fontweight='bold')
        
        # 6. Training Convergence
        epochs_baseline = range(1, len(self.baseline_results['train_accuracies']) + 1)
        epochs_scs_id = range(1, len(self.scs_id_results['train_accuracies']) + 1)
        
        axes[1,2].plot(epochs_baseline, self.baseline_results['train_accuracies'], 'r-', label='Baseline Train', alpha=0.7)
        axes[1,2].plot(epochs_baseline, self.baseline_results['val_accuracies'], 'r--', label='Baseline Val', alpha=0.7)
        axes[1,2].plot(epochs_scs_id, self.scs_id_results['train_accuracies'], 'b-', label='SCS-ID Train', alpha=0.7)
        axes[1,2].plot(epochs_scs_id, self.scs_id_results['val_accuracies'], 'b--', label='SCS-ID Val', alpha=0.7)
        axes[1,2].set_title('Training Convergence', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Epoch')
        axes[1,2].set_ylabel('Accuracy')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        # 7. Confusion Matrix Comparison (Top Classes)
        top_classes = self.per_class_results.nlargest(5, 'sample_count')['class_name'].tolist()
        class_names = self.baseline_results['class_names']
        
        # Create confusion matrices for top classes only
        true_labels = np.array(self.test_results['true_labels'])
        baseline_preds = np.array(self.test_results['baseline_predictions'])
        scs_id_preds = np.array(self.test_results['scs_id_predictions'])
        
        # Filter for top classes
        top_class_indices = [class_names.index(cls) for cls in top_classes if cls in class_names]
        mask = np.isin(true_labels, top_class_indices)
        
        cm_baseline = confusion_matrix(true_labels[mask], baseline_preds[mask], labels=top_class_indices)
        cm_scs_id = confusion_matrix(true_labels[mask], scs_id_preds[mask], labels=top_class_indices)
        
        # Plot baseline confusion matrix
        im1 = axes[2,0].imshow(cm_baseline, interpolation='nearest', cmap='Blues')
        axes[2,0].set_title('Baseline CNN - Top 5 Classes', fontsize=12, fontweight='bold')
        tick_marks = np.arange(len(top_classes))
        axes[2,0].set_xticks(tick_marks)
        axes[2,0].set_yticks(tick_marks)
        axes[2,0].set_xticklabels([cls[:8] for cls in top_classes], rotation=45)
        axes[2,0].set_yticklabels([cls[:8] for cls in top_classes])
        
        # Plot SCS-ID confusion matrix  
        im2 = axes[2,1].imshow(cm_scs_id, interpolation='nearest', cmap='Greens')
        axes[2,1].set_title('SCS-ID - Top 5 Classes', fontsize=12, fontweight='bold')
        axes[2,1].set_xticks(tick_marks)
        axes[2,1].set_yticks(tick_marks)
        axes[2,1].set_xticklabels([cls[:8] for cls in top_classes], rotation=45)
        axes[2,1].set_yticklabels([cls[:8] for cls in top_classes])
        
        # 8. Accuracy Improvement by Class
        improvement = self.per_class_results['accuracy_improvement'].values
        class_names_short = [name[:10] for name in self.per_class_results['class_name']]
        
        colors = ['green' if x > 0 else 'red' for x in improvement]
        bars = axes[2,2].barh(range(len(improvement)), improvement, color=colors, alpha=0.7)
        axes[2,2].set_title('Accuracy Improvement by Class', fontsize=12, fontweight='bold')
        axes[2,2].set_xlabel('Accuracy Improvement')
        axes[2,2].set_yticks(range(len(improvement)))
        axes[2,2].set_yticklabels(class_names_short)
        axes[2,2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/performance_dashboard/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Overall Accuracy Comparison', 'F1-Score Comparison', 'Model Parameters',
                'Per-Class Accuracy', 'Training Convergence', 'Sample Distribution',
                'Accuracy Improvement', 'Confusion Matrix Heatmap', 'Key Metrics Summary'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "heatmap"}, {"type": "table"}]
            ]
        )
        
        # 1. Overall Accuracy
        fig.add_trace(go.Bar(
            x=['Baseline CNN', 'SCS-ID'],
            y=[self.baseline_results['test_accuracy'], self.scs_id_results['test_accuracy']],
            name='Accuracy',
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f"{acc:.4f}" for acc in [self.baseline_results['test_accuracy'], self.scs_id_results['test_accuracy']]],
            textposition='auto'
        ), row=1, col=1)
        
        # 2. F1-Score
        fig.add_trace(go.Bar(
            x=['Baseline CNN', 'SCS-ID'],
            y=[self.baseline_results['f1_score'], self.scs_id_results['f1_score']],
            name='F1-Score',
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f"{f1:.4f}" for f1 in [self.baseline_results['f1_score'], self.scs_id_results['f1_score']]],
            textposition='auto',
            showlegend=False
        ), row=1, col=2)
        
        # 3. Model Parameters
        params_baseline = self.baseline_results['model_parameters']
        params_scs_id = self.scs_id_results['model_stats']['total_parameters']
        
        fig.add_trace(go.Bar(
            x=['Baseline CNN', 'SCS-ID'],
            y=[params_baseline/1000, params_scs_id/1000],
            name='Parameters (K)',
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f"{params/1000:.1f}K" for params in [params_baseline, params_scs_id]],
            textposition='auto',
            showlegend=False
        ), row=1, col=3)
        
        # 4. Per-Class Accuracy
        fig.add_trace(go.Bar(
            x=self.per_class_results['class_name'],
            y=self.per_class_results['baseline_accuracy'],
            name='Baseline',
            marker_color='#FF6B6B',
            opacity=0.7
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            x=self.per_class_results['class_name'],
            y=self.per_class_results['scs_id_accuracy'],
            name='SCS-ID',
            marker_color='#4ECDC4',
            opacity=0.7
        ), row=2, col=1)
        
        # 5. Training Convergence
        epochs_baseline = list(range(1, len(self.baseline_results['train_accuracies']) + 1))
        epochs_scs_id = list(range(1, len(self.scs_id_results['train_accuracies']) + 1))
        
        fig.add_trace(go.Scatter(
            x=epochs_baseline,
            y=self.baseline_results['train_accuracies'],
            mode='lines',
            name='Baseline Train',
            line=dict(color='red', dash='solid')
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=epochs_scs_id,
            y=self.scs_id_results['train_accuracies'],
            mode='lines',
            name='SCS-ID Train',
            line=dict(color='blue', dash='solid')
        ), row=2, col=2)
        
        # 6. Sample Distribution
        fig.add_trace(go.Bar(
            x=self.per_class_results['class_name'],
            y=self.per_class_results['sample_count'],
            name='Sample Count',
            marker_color='skyblue',
            showlegend=False
        ), row=2, col=3)
        
        # 7. Accuracy Improvement
        improvement = self.per_class_results['accuracy_improvement']
        colors = ['green' if x > 0 else 'red' for x in improvement]
        
        fig.add_trace(go.Bar(
            x=improvement,
            y=self.per_class_results['class_name'],
            orientation='h',
            name='Improvement',
            marker_color=colors,
            showlegend=False
        ), row=3, col=1)
        
        # 8. Summary Table
        summary_data = [
            ['Total Test Samples', f"{len(self.test_results):,}"],
            ['Baseline Accuracy', f"{self.baseline_results['test_accuracy']:.4f}"],
            ['SCS-ID Accuracy', f"{self.scs_id_results['test_accuracy']:.4f}"],
            ['Accuracy Improvement', f"{self.scs_id_results['test_accuracy'] - self.baseline_results['test_accuracy']:.4f}"],
            ['Parameter Reduction', f"{(params_baseline - params_scs_id)/params_baseline*100:.1f}%"],
            ['Training Time (hrs)', f"B: {self.baseline_results['training_time']/3600:.1f}, S: {self.scs_id_results['training_time']/3600:.1f}"]
        ]
        
        fig.add_trace(go.Table(
            header=dict(values=['Metric', 'Value'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[[row[0] for row in summary_data], [row[1] for row in summary_data]],
                      fill_color='lavender',
                      align='left')
        ), row=3, col=3)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Interactive Performance Dashboard: SCS-ID vs Baseline CNN",
            title_x=0.5,
            showlegend=True
        )
        
        # Save interactive dashboard
        pyo.plot(fig, filename='results/performance_dashboard/interactive_dashboard.html', auto_open=False)
        
        return fig
    
    def generate_summary_report(self):
        """Generate summary report"""
        params_baseline = self.baseline_results['model_parameters']
        params_scs_id = self.scs_id_results['model_stats']['total_parameters']
        param_reduction = (params_baseline - params_scs_id) / params_baseline * 100
        
        report = f"""
PERFORMANCE DASHBOARD SUMMARY REPORT
=====================================

Model Comparison: SCS-ID vs Baseline CNN
Dataset: CIC-IDS2017 (Test Set: {len(self.test_results):,} samples)

OVERALL PERFORMANCE:
-------------------
• Baseline CNN Accuracy: {self.baseline_results['test_accuracy']:.4f} ({self.baseline_results['test_accuracy']*100:.2f}%)
• SCS-ID Accuracy: {self.scs_id_results['test_accuracy']:.4f} ({self.scs_id_results['test_accuracy']*100:.2f}%)
• Accuracy Improvement: +{(self.scs_id_results['test_accuracy'] - self.baseline_results['test_accuracy'])*100:.2f}%

F1-SCORE:
---------
• Baseline CNN: {self.baseline_results['f1_score']:.4f}
• SCS-ID: {self.scs_id_results['f1_score']:.4f}
• F1-Score Improvement: +{(self.scs_id_results['f1_score'] - self.baseline_results['f1_score'])*100:.2f}%

MODEL EFFICIENCY:
----------------
• Parameter Reduction: {param_reduction:.1f}% ({params_baseline:,} -> {params_scs_id:,})
• Training Time: Baseline {self.baseline_results['training_time']/3600:.1f}h vs SCS-ID {self.scs_id_results['training_time']/3600:.1f}h

BEST PERFORMING CLASSES (Top 3):
--------------------------------"""
        
        top_3_improved = self.per_class_results.nlargest(3, 'accuracy_improvement')
        for idx, row in top_3_improved.iterrows():
            report += f"\n• {row['class_name']}: +{row['accuracy_improvement']*100:.2f}% improvement"
        
        report += f"""

DASHBOARD FILES GENERATED:
-------------------------
• Static Dashboard: results/performance_dashboard/comprehensive_dashboard.png
• Interactive Dashboard: results/performance_dashboard/interactive_dashboard.html
• Summary Report: results/performance_dashboard/dashboard_summary.txt

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open('results/performance_dashboard/dashboard_summary.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        return report

# Create the dashboard
dashboard = PerformanceDashboard()

print("Creating Static Dashboard...")
static_fig = dashboard.create_static_dashboard()

print("Creating Interactive Dashboard...")
interactive_fig = dashboard.create_interactive_dashboard()

print("Generating Summary Report...")
summary = dashboard.generate_summary_report()

print("\n" + "="*60)
print("PERFORMANCE DASHBOARD COMPLETED!")
print("="*60)
print("Files created:")
print("• Static: results/performance_dashboard/comprehensive_dashboard.png")
print("• Interactive: results/performance_dashboard/interactive_dashboard.html") 
print("• Summary: results/performance_dashboard/dashboard_summary.txt")
print("="*60)
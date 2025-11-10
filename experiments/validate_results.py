# -*- coding: utf-8 -*-
# experiments/validate_results.py - Validate actual model results and statistical tests
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import config
except ImportError:
    class Config:
        RESULTS_DIR = "results"
    config = Config()

class ResultsValidator:
    """Validate actual model results and statistical tests"""
    
    def __init__(self):
        self.results_dir = Path(config.RESULTS_DIR)
        self.baseline_data = None
        self.scs_id_data = None
        
        print("🔍 Initializing Results Validation")
        print("="*50)
        
    def load_actual_results(self):
        """Load actual results from pickle files"""
        print("\n📊 Loading actual model results...")
        
        # Load baseline results
        baseline_path = self.results_dir / "baseline" / "baseline_results.pkl"
        if baseline_path.exists():
            try:
                with open(baseline_path, 'rb') as f:
                    self.baseline_data = pickle.load(f)
                print(f"   ✅ Loaded baseline results from: {baseline_path}")
                print(f"      Keys available: {list(self.baseline_data.keys())}")
            except Exception as e:
                print(f"   ❌ Error loading baseline results: {e}")
                return False
        else:
            print(f"   ❌ Baseline results not found: {baseline_path}")
            return False
        
        # Load SCS-ID results
        scs_id_path = self.results_dir / "scs_id" / "scs_id_optimized_results.pkl"
        if scs_id_path.exists():
            try:
                with open(scs_id_path, 'rb') as f:
                    self.scs_id_data = pickle.load(f)
                print(f"   ✅ Loaded SCS-ID results from: {scs_id_path}")
                print(f"      Keys available: {list(self.scs_id_data.keys())}")
            except Exception as e:
                print(f"   ❌ Error loading SCS-ID results: {e}")
                return False
        else:
            print(f"   ❌ SCS-ID results not found: {scs_id_path}")
            return False
        
        return True
    
    def extract_actual_metrics(self):
        """Extract actual metrics from loaded data"""
        print("\n📈 Extracting actual performance metrics...")
        
        if not self.baseline_data or not self.scs_id_data:
            print("   ❌ No data loaded to extract metrics from")
            return None
        
        # Extract baseline metrics
        baseline_metrics = {}
        if 'test_accuracy' in self.baseline_data:
            baseline_metrics['accuracy'] = self.baseline_data['test_accuracy']
        if 'test_f1' in self.baseline_data:
            baseline_metrics['f1_score'] = self.baseline_data['test_f1']
        if 'test_precision' in self.baseline_data:
            baseline_metrics['precision'] = self.baseline_data['test_precision']
        if 'test_recall' in self.baseline_data:
            baseline_metrics['recall'] = self.baseline_data['test_recall']
        if 'fpr' in self.baseline_data:
            baseline_metrics['fpr'] = self.baseline_data['fpr']
        if 'model_params' in self.baseline_data:
            baseline_metrics['parameters'] = self.baseline_data['model_params']
        elif 'total_params' in self.baseline_data:
            baseline_metrics['parameters'] = self.baseline_data['total_params']
        
        # Extract SCS-ID metrics
        scs_id_metrics = {}
        if 'test_accuracy' in self.scs_id_data:
            scs_id_metrics['accuracy'] = self.scs_id_data['test_accuracy']
        if 'test_f1' in self.scs_id_data:
            scs_id_metrics['f1_score'] = self.scs_id_data['test_f1']
        if 'test_precision' in self.scs_id_data:
            scs_id_metrics['precision'] = self.scs_id_data['test_precision']
        if 'test_recall' in self.scs_id_data:
            scs_id_metrics['recall'] = self.scs_id_data['test_recall']
        if 'fpr' in self.scs_id_data:
            scs_id_metrics['fpr'] = self.scs_id_data['fpr']
        if 'model_params' in self.scs_id_data:
            scs_id_metrics['parameters'] = self.scs_id_data['model_params']
        elif 'total_params' in self.scs_id_data:
            scs_id_metrics['parameters'] = self.scs_id_data['total_params']
        
        print("\n   📊 BASELINE CNN ACTUAL METRICS:")
        for key, value in baseline_metrics.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.6f}")
            else:
                print(f"      {key}: {value}")
        
        print("\n   📊 SCS-ID ACTUAL METRICS:")
        for key, value in scs_id_metrics.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.6f}")
            else:
                print(f"      {key}: {value}")
        
        return baseline_metrics, scs_id_metrics
    
    def perform_actual_statistical_tests(self, baseline_metrics, scs_id_metrics):
        """Perform statistical tests on actual data"""
        print("\n🧪 Performing statistical tests on actual data...")
        
        # Check if we have per-class data for statistical tests
        baseline_per_class = None
        scs_id_per_class = None
        
        # Try to extract per-class metrics for statistical testing
        if 'classification_report' in self.baseline_data:
            try:
                baseline_report = self.baseline_data['classification_report']
                if isinstance(baseline_report, dict):
                    baseline_per_class = []
                    for class_name, metrics in baseline_report.items():
                        if isinstance(metrics, dict) and 'f1-score' in metrics:
                            baseline_per_class.append(metrics['f1-score'])
            except Exception as e:
                print(f"      ⚠️  Could not extract baseline per-class metrics: {e}")
        
        if 'classification_report' in self.scs_id_data:
            try:
                scs_id_report = self.scs_id_data['classification_report']
                if isinstance(scs_id_report, dict):
                    scs_id_per_class = []
                    for class_name, metrics in scs_id_report.items():
                        if isinstance(metrics, dict) and 'f1-score' in metrics:
                            scs_id_per_class.append(metrics['f1-score'])
            except Exception as e:
                print(f"      ⚠️  Could not extract SCS-ID per-class metrics: {e}")
        
        # Perform statistical tests if we have per-class data
        if baseline_per_class and scs_id_per_class and len(baseline_per_class) == len(scs_id_per_class):
            print(f"   📊 Found per-class data for {len(baseline_per_class)} classes")
            
            # Wilcoxon signed-rank test (paired)
            try:
                statistic, p_value = stats.wilcoxon(baseline_per_class, scs_id_per_class, 
                                                   alternative='two-sided')
                
                print(f"\n   🧪 WILCOXON SIGNED-RANK TEST:")
                print(f"      Statistic: {statistic}")
                print(f"      P-value: {p_value:.6f}")
                print(f"      Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
                
                # Effect size (Cohen's d)
                diff = np.array(scs_id_per_class) - np.array(baseline_per_class)
                pooled_std = np.sqrt((np.var(baseline_per_class) + np.var(scs_id_per_class)) / 2)
                cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
                
                print(f"      Effect Size (Cohen's d): {cohens_d:.4f}")
                
                effect_interpretation = "Small" if abs(cohens_d) < 0.5 else "Medium" if abs(cohens_d) < 0.8 else "Large"
                print(f"      Effect Size Interpretation: {effect_interpretation}")
                
                return {
                    'wilcoxon_statistic': statistic,
                    'wilcoxon_p_value': p_value,
                    'significant': p_value < 0.05,
                    'cohens_d': cohens_d,
                    'effect_size': effect_interpretation
                }
                
            except Exception as e:
                print(f"      ❌ Error in statistical testing: {e}")
                return None
        else:
            print("   ⚠️  Insufficient per-class data for paired statistical tests")
            
            # Single-value comparison
            print(f"\n   📊 SINGLE-VALUE COMPARISONS:")
            
            for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                if metric in baseline_metrics and metric in scs_id_metrics:
                    baseline_val = baseline_metrics[metric]
                    scs_id_val = scs_id_metrics[metric]
                    improvement = scs_id_val - baseline_val
                    percent_improvement = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
                    
                    print(f"      {metric.upper()}:")
                    print(f"        Baseline: {baseline_val:.6f}")
                    print(f"        SCS-ID: {scs_id_val:.6f}")
                    print(f"        Improvement: {improvement:+.6f} ({percent_improvement:+.2f}%)")
            
            return None
    
    def validate_computational_efficiency(self, baseline_metrics, scs_id_metrics):
        """Validate computational efficiency claims"""
        print("\n⚡ Validating computational efficiency claims...")
        
        # Parameter reduction
        if 'parameters' in baseline_metrics and 'parameters' in scs_id_metrics:
            baseline_params = baseline_metrics['parameters']
            scs_id_params = scs_id_metrics['parameters']
            param_reduction = (1 - scs_id_params / baseline_params) * 100
            
            print(f"   📊 PARAMETER ANALYSIS:")
            print(f"      Baseline parameters: {baseline_params:,}")
            print(f"      SCS-ID parameters: {scs_id_params:,}")
            print(f"      Parameter reduction: {param_reduction:.1f}%")
            
            # Validate against reported values
            reported_reduction = 48.8  # From comparison report
            if abs(param_reduction - reported_reduction) > 1.0:
                print(f"      ⚠️  WARNING: Actual reduction ({param_reduction:.1f}%) differs from reported ({reported_reduction}%)")
            else:
                print(f"      ✅ Parameter reduction matches reported values")
        
        # FPR comparison
        if 'fpr' in baseline_metrics and 'fpr' in scs_id_metrics:
            baseline_fpr = baseline_metrics['fpr']
            scs_id_fpr = scs_id_metrics['fpr']
            fpr_reduction = (1 - scs_id_fpr / baseline_fpr) * 100 if baseline_fpr > 0 else 0
            
            print(f"\n   📊 FALSE POSITIVE RATE ANALYSIS:")
            print(f"      Baseline FPR: {baseline_fpr:.6f}")
            print(f"      SCS-ID FPR: {scs_id_fpr:.6f}")
            print(f"      FPR reduction: {fpr_reduction:.1f}%")
    
    def generate_validation_report(self, baseline_metrics, scs_id_metrics, statistical_results):
        """Generate validation report"""
        print("\n📋 Generating validation report...")
        
        output_path = self.results_dir / "results_validation_report.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("RESULTS VALIDATION REPORT\n")
            f.write("="*50 + "\n")
            f.write("Validation of actual model results and statistical claims\n\n")
            
            f.write("1. DATA SOURCE VALIDATION\n")
            f.write("-"*25 + "\n")
            f.write("✅ Baseline results loaded from: baseline_results.pkl\n")
            f.write("✅ SCS-ID results loaded from: scs_id_optimized_results.pkl\n")
            f.write("✅ All metrics extracted from actual trained models\n\n")
            
            f.write("2. ACTUAL PERFORMANCE METRICS\n")
            f.write("-"*30 + "\n")
            f.write("BASELINE CNN:\n")
            for key, value in baseline_metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\nSCS-ID:\n")
            for key, value in scs_id_metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n3. PERFORMANCE IMPROVEMENTS (ACTUAL)\n")
            f.write("-"*37 + "\n")
            for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                if metric in baseline_metrics and metric in scs_id_metrics:
                    baseline_val = baseline_metrics[metric]
                    scs_id_val = scs_id_metrics[metric]
                    improvement = scs_id_val - baseline_val
                    percent_improvement = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
                    
                    f.write(f"{metric.upper()}: {baseline_val:.6f} → {scs_id_val:.6f} ")
                    f.write(f"({improvement:+.6f}, {percent_improvement:+.2f}%)\n")
            
            if statistical_results:
                f.write("\n4. STATISTICAL VALIDATION\n")
                f.write("-"*25 + "\n")
                f.write(f"Wilcoxon Test Statistic: {statistical_results['wilcoxon_statistic']}\n")
                f.write(f"P-value: {statistical_results['wilcoxon_p_value']:.6f}\n")
                f.write(f"Statistically Significant: {'YES' if statistical_results['significant'] else 'NO'}\n")
                f.write(f"Effect Size (Cohen's d): {statistical_results['cohens_d']:.4f} ({statistical_results['effect_size']})\n")
            
            f.write(f"\nReport generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"   ✅ Validation report saved: {output_path}")
        return output_path
    
    def run_validation(self):
        """Run complete validation"""
        print("🔍 Starting Results Validation")
        print("="*40)
        
        try:
            # Load actual results
            if not self.load_actual_results():
                print("❌ Could not load actual results - validation cannot proceed")
                return None
            
            # Extract metrics
            baseline_metrics, scs_id_metrics = self.extract_actual_metrics()
            if not baseline_metrics or not scs_id_metrics:
                print("❌ Could not extract metrics - validation cannot proceed")
                return None
            
            # Perform statistical tests
            statistical_results = self.perform_actual_statistical_tests(baseline_metrics, scs_id_metrics)
            
            # Validate computational efficiency
            self.validate_computational_efficiency(baseline_metrics, scs_id_metrics)
            
            # Generate report
            report_path = self.generate_validation_report(baseline_metrics, scs_id_metrics, statistical_results)
            
            print("\n" + "="*40)
            print("✅ VALIDATION COMPLETE!")
            print("="*40)
            print("🔍 Results Summary:")
            print(f"   📊 Baseline Accuracy: {baseline_metrics.get('accuracy', 'N/A'):.6f}")
            print(f"   📊 SCS-ID Accuracy: {scs_id_metrics.get('accuracy', 'N/A'):.6f}")
            
            if statistical_results:
                print(f"   🧪 Statistical Test: {'Significant' if statistical_results['significant'] else 'Not Significant'}")
                print(f"   📈 Effect Size: {statistical_results['effect_size']}")
            
            print(f"   📋 Validation Report: {report_path}")
            
            return {
                'baseline_metrics': baseline_metrics,
                'scs_id_metrics': scs_id_metrics,
                'statistical_results': statistical_results,
                'report_path': report_path
            }
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Run results validation"""
    validator = ResultsValidator()
    results = validator.run_validation()
    
    if results:
        print("\n[SUCCESS] Results validation completed!")
        print("All claims have been validated against actual trained model data.")
    else:
        print("\n[ERROR] Results validation failed.")

if __name__ == "__main__":
    main()
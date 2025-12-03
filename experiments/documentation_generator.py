"""
Comprehensive Results Documentation Generator
Creates detailed .txt files with all statistical results, hypothesis testing outcomes,
and model comparison summaries according to scientific reporting standards.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List
from datetime import datetime
import json

class ResultsDocumentationGenerator:
    """Generate comprehensive text documentation for model comparison results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.docs_dir = self.results_dir / "documentation"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Attack types for detailed reporting
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
            
    def generate_hypothesis_1_report(self):
        """Generate detailed report for Hypothesis 1: Computational Efficiency"""
        report_path = self.docs_dir / "hypothesis_1_computational_efficiency_report.txt"
        
        baseline_comp = self.results['baseline']['computational']
        scs_id_comp = self.results['scs_id']['computational']
        efficiency = self.results['comparison']['efficiency_reductions']
        h1_stats = self.results['statistical_tests']['hypothesis_1']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HYPOTHESIS 1: COMPUTATIONAL EFFICIENCY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Hypothesis statement
            f.write("HYPOTHESIS STATEMENT:\n")
            f.write("-" * 40 + "\n")
            f.write("H0: There was no significant improvement in computational efficiency metrics\n")
            f.write("    (measured by parameter count, inference latency, and memory utilization)\n")
            f.write("    when using the proposed SCS-ID model with the Squeezed ConvSeek\n")
            f.write("    architecture compared to Ayeni's CNN model in campus network environments.\n\n")
            f.write("H1: There was a significant improvement in computational efficiency metrics\n")
            f.write("    (measured by parameter count, inference latency, and memory utilization)\n")
            f.write("    when using the proposed SCS-ID model with the Squeezed ConvSeek\n")
            f.write("    architecture compared to Ayeni's CNN model.\n\n")
            
            # Raw measurements
            f.write("COMPUTATIONAL EFFICIENCY MEASUREMENTS:\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. PARAMETER COUNT ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"   Baseline CNN (Ayeni et al.):  {baseline_comp['total_parameters']:,} parameters\n")
            f.write(f"   SCS-ID (Proposed):           {scs_id_comp['total_parameters']:,} parameters\n")
            f.write(f"   Parameter Count Reduction:   {efficiency['parameter_count_reduction']:.2f}%\n")
            f.write(f"   Formula: PCR = (1 - P_SCS-ID/P_baseline) × 100%\n")
            f.write(f"            PCR = (1 - {scs_id_comp['total_parameters']:,}/{baseline_comp['total_parameters']:,}) × 100%\n")
            f.write(f"            PCR = {efficiency['parameter_count_reduction']:.2f}%\n\n")
            
            f.write("2. INFERENCE LATENCY ANALYSIS:\n")
            f.write("-" * 32 + "\n")
            f.write(f"   Baseline CNN Latency:        {baseline_comp['inference_latency_ms']:.4f} ms per connection\n")
            f.write(f"   SCS-ID Latency:              {scs_id_comp['inference_latency_ms']:.4f} ms per connection\n")
            f.write(f"   Latency Improvement:         {efficiency['inference_latency_improvement']:.2f}%\n")
            f.write(f"   Formula: IL = T_processing/n × 1000 ms\n")
            f.write(f"   Where n = 1000 connections for standardization\n\n")
            
            f.write("3. MEMORY UTILIZATION ANALYSIS:\n")
            f.write("-" * 33 + "\n")
            f.write(f"   Baseline CNN Memory:         {baseline_comp['peak_memory_mb']:.2f} MB\n")
            f.write(f"   SCS-ID Memory:               {scs_id_comp['peak_memory_mb']:.2f} MB\n")
            f.write(f"   Memory Utilization Reduction: {efficiency['memory_utilization_reduction']:.2f}%\n")
            f.write(f"   Formula: MUR = (1 - M_SCS-ID/M_baseline) × 100%\n")
            f.write(f"            MUR = (1 - {scs_id_comp['peak_memory_mb']:.2f}/{baseline_comp['peak_memory_mb']:.2f}) × 100%\n")
            f.write(f"            MUR = {efficiency['memory_utilization_reduction']:.2f}%\n\n")
            
            # Statistical analysis
            f.write("STATISTICAL SIGNIFICANCE TESTING:\n")
            f.write("=" * 50 + "\n\n")
            
            for metric_name, stats in h1_stats.items():
                f.write(f"{metric_name.upper().replace('_', ' ')} STATISTICAL TEST:\n")
                f.write("-" * 40 + "\n")
                f.write(f"   Sample Size:                 n = 30 (bootstrap samples)\n")
                f.write(f"   Significance Level:          alpha = 0.05\n")
                f.write(f"   Normality Test (Shapiro-Wilk): W = {stats['shapiro_stat']:.4f}, p = {stats['shapiro_p']:.4f}\n")
                f.write(f"   Data Distribution:           {'Normal' if stats['is_normal'] else 'Non-normal'}\n")
                f.write(f"   Statistical Test Used:       {stats['test_used']}\n")
                f.write(f"   Test Statistic:              {stats['test_statistic']:.4f}\n")
                f.write(f"   p-value:                     {stats['p_value']:.6f}\n")
                f.write(f"   Effect Size (Cohen's d):     {stats['effect_size']:.4f}\n")
                f.write(f"   Mean Difference:             {stats['mean_difference']:.6f}\n")
                f.write(f"   Baseline Mean:               {stats['baseline_mean']:.6f}\n")
                f.write(f"   SCS-ID Mean:                 {stats['scs_id_mean']:.6f}\n")
                f.write(f"   Improvement Percentage:      {stats['improvement_percent']:.2f}%\n")
                
                # Significance interpretation
                if stats['significant']:
                    f.write(f"   Result:                      ✓ SIGNIFICANT (p < 0.05)\n")
                    f.write(f"   Interpretation:              Reject H0, Accept H1\n")
                    f.write(f"                               Significant improvement observed\n")
                else:
                    f.write(f"   Result:                      ✗ NOT SIGNIFICANT (p ≥ 0.05)\n")
                    f.write(f"   Interpretation:              Fail to reject H0\n")
                    f.write(f"                               No significant improvement observed\n")
                
                f.write("\n")
            
            # Overall conclusion
            f.write("HYPOTHESIS 1 CONCLUSION:\n")
            f.write("=" * 30 + "\n")
            
            significant_metrics = [name for name, stats in h1_stats.items() if stats['significant']]
            total_metrics = len(h1_stats)
            
            f.write(f"Significant Improvements: {len(significant_metrics)}/{total_metrics} metrics\n")
            f.write(f"Significant Metrics: {', '.join(significant_metrics) if significant_metrics else 'None'}\n\n")
            
            if len(significant_metrics) >= 2:
                f.write("CONCLUSION: HYPOTHESIS 1 (H1) IS SUPPORTED\n")
                f.write("The SCS-ID model demonstrates statistically significant improvements\n")
                f.write("in computational efficiency compared to the baseline CNN model.\n")
            else:
                f.write("CONCLUSION: HYPOTHESIS 1 (H1) IS NOT FULLY SUPPORTED\n")
                f.write("The SCS-ID model does not demonstrate sufficient statistically\n")
                f.write("significant improvements in computational efficiency.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
        print(f"✅ Hypothesis 1 report generated: {report_path}")
        
    def generate_hypothesis_2_report(self):
        """Generate detailed report for Hypothesis 2: Detection Performance"""
        report_path = self.docs_dir / "hypothesis_2_detection_performance_report.txt"
        
        baseline_perf = self.results['baseline']['performance']
        scs_id_perf = self.results['scs_id']['performance']
        h2_stats = self.results['statistical_tests']['hypothesis_2']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HYPOTHESIS 2: DETECTION PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Hypothesis statement
            f.write("HYPOTHESIS STATEMENT:\n")
            f.write("-" * 40 + "\n")
            f.write("H0: There was no statistically significant difference in detection\n")
            f.write("    performance (accuracy, F1-score, and false positive rate) when\n")
            f.write("    using the SCS-ID architecture compared to Ayeni's CNN model\n")
            f.write("    while operating under various campus network traffic conditions.\n\n")
            f.write("H1: There was a statistically significant difference in detection\n")
            f.write("    performance (accuracy, F1-score, and false positive rate) when\n")
            f.write("    using the SCS-ID architecture compared to Ayeni's CNN model.\n\n")
            
            # Overall performance metrics
            f.write("OVERALL DETECTION PERFORMANCE METRICS:\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("CLASSIFICATION PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Metric                    | Baseline CNN | SCS-ID    | Difference\n")
            f.write("-" * 65 + "\n")
            f.write(f"Accuracy                  | {baseline_perf['accuracy']:.6f}   | {scs_id_perf['accuracy']:.6f} | {scs_id_perf['accuracy'] - baseline_perf['accuracy']:+.6f}\n")
            f.write(f"Precision (Macro Avg)     | {baseline_perf['precision_macro']:.6f}   | {scs_id_perf['precision_macro']:.6f} | {scs_id_perf['precision_macro'] - baseline_perf['precision_macro']:+.6f}\n")
            f.write(f"Recall (Macro Avg)        | {baseline_perf['recall_macro']:.6f}   | {scs_id_perf['recall_macro']:.6f} | {scs_id_perf['recall_macro'] - baseline_perf['recall_macro']:+.6f}\n")
            f.write(f"F1-Score (Macro Avg)      | {baseline_perf['f1_macro']:.6f}   | {scs_id_perf['f1_macro']:.6f} | {scs_id_perf['f1_macro'] - baseline_perf['f1_macro']:+.6f}\n")
            f.write(f"False Positive Rate       | {baseline_perf['fpr_overall']:.6f}   | {scs_id_perf['fpr_overall']:.6f} | {scs_id_perf['fpr_overall'] - baseline_perf['fpr_overall']:+.6f}\n")
            f.write(f"False Alarm Rate          | {baseline_perf['far_overall']:.6f}   | {scs_id_perf['far_overall']:.6f} | {scs_id_perf['far_overall'] - baseline_perf['far_overall']:+.6f}\n")
            f.write(f"AUC-ROC                   | {baseline_perf['auc_roc']:.6f}   | {scs_id_perf['auc_roc']:.6f} | {scs_id_perf['auc_roc'] - baseline_perf['auc_roc']:+.6f}\n")
            f.write(f"Matthews Correlation Coeff| {baseline_perf['mcc_overall']:.6f}   | {scs_id_perf['mcc_overall']:.6f} | {scs_id_perf['mcc_overall'] - baseline_perf['mcc_overall']:+.6f}\n\n")
            
            # Per-attack type analysis (main focus for Hypothesis 2)
            f.write("PER-ATTACK TYPE PERFORMANCE ANALYSIS:\n")
            f.write("=" * 50 + "\n\n")
            
            # Performance metrics per attack
            metrics_info = {
                'precision_per_class': 'Precision',
                'recall_per_class': 'Recall',
                'f1_per_class': 'F1-Score',
                'fpr_per_class': 'False Positive Rate'
            }
            
            for metric_key, metric_name in metrics_info.items():
                f.write(f"{metric_name.upper()} BY ATTACK TYPE:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Attack Type':<25} | {'Baseline':<10} | {'SCS-ID':<10} | {'Diff':<10} | {'Significant'}\n")
                f.write("-" * 80 + "\n")
                
                baseline_values = baseline_perf[metric_key]
                scs_id_values = scs_id_perf[metric_key]
                
                n_attacks = min(len(baseline_values), len(scs_id_values), len(self.attack_types))
                
                for i in range(n_attacks):
                    attack_name = self.attack_types[i]
                    baseline_val = baseline_values[i]
                    scs_id_val = scs_id_values[i]
                    difference = scs_id_val - baseline_val
                    
                    # Check significance
                    significant = "N/A"
                    if metric_key in h2_stats and attack_name in h2_stats[metric_key]:
                        sig_data = h2_stats[metric_key][attack_name]
                        significant = "✓ Yes" if sig_data['significant'] else "✗ No"
                    
                    f.write(f"{attack_name:<25} | {baseline_val:<10.6f} | {scs_id_val:<10.6f} | {difference:<+10.6f} | {significant}\n")
                
                f.write("\n")
            
            # Statistical significance testing per attack type
            f.write("STATISTICAL SIGNIFICANCE TESTING (PER-ATTACK TYPE):\n")
            f.write("=" * 60 + "\n\n")
            
            for metric_key, metric_name in metrics_info.items():
                if metric_key not in h2_stats:
                    continue
                    
                f.write(f"{metric_name.upper()} - STATISTICAL TEST RESULTS:\n")
                f.write("-" * 50 + "\n")
                
                metric_stats = h2_stats[metric_key]
                significant_attacks = []
                total_attacks = 0
                
                for attack_name in self.attack_types:
                    if attack_name not in metric_stats:
                        continue
                        
                    total_attacks += 1
                    stats = metric_stats[attack_name]
                    
                    f.write(f"\n{attack_name}:\n")
                    f.write(f"  Sample Size:              n = 30 (bootstrap samples)\n")
                    f.write(f"  Significance Level:       alpha = 0.05\n")
                    f.write(f"  Normality Test:           W = {stats['shapiro_stat']:.4f}, p = {stats['shapiro_p']:.4f}\n")
                    f.write(f"  Data Distribution:        {'Normal' if stats['is_normal'] else 'Non-normal'}\n")
                    f.write(f"  Statistical Test:         {stats['test_used']}\n")
                    f.write(f"  Test Statistic:           {stats['test_statistic']:.4f}\n")
                    f.write(f"  p-value:                  {stats['p_value']:.6f}\n")
                    f.write(f"  Effect Size:              {stats['effect_size']:.4f}\n")
                    f.write(f"  Mean Difference:          {stats['mean_difference']:.6f}\n")
                    f.write(f"  Baseline Value:           {stats['baseline_value']:.6f}\n")
                    f.write(f"  SCS-ID Value:             {stats['scs_id_value']:.6f}\n")
                    
                    if stats['significant']:
                        f.write(f"  Result:                   ✓ SIGNIFICANT (p < 0.05)\n")
                        f.write(f"  Interpretation:           Significant difference detected\n")
                        significant_attacks.append(attack_name)
                    else:
                        f.write(f"  Result:                   ✗ NOT SIGNIFICANT (p ≥ 0.05)\n")
                        f.write(f"  Interpretation:           No significant difference\n")
                
                f.write(f"\n{metric_name} Summary:\n")
                f.write(f"  Significant Attacks: {len(significant_attacks)}/{total_attacks}\n")
                f.write(f"  Significant Attack Types: {', '.join(significant_attacks) if significant_attacks else 'None'}\n\n")
            
            # Overall hypothesis conclusion
            f.write("HYPOTHESIS 2 CONCLUSION:\n")
            f.write("=" * 30 + "\n")
            
            # Count total significant differences across all metrics and attacks
            total_tests = 0
            significant_tests = 0
            
            for metric_key in metrics_info.keys():
                if metric_key in h2_stats:
                    for attack_name in self.attack_types:
                        if attack_name in h2_stats[metric_key]:
                            total_tests += 1
                            if h2_stats[metric_key][attack_name]['significant']:
                                significant_tests += 1
            
            f.write(f"Total Statistical Tests Conducted: {total_tests}\n")
            f.write(f"Statistically Significant Results: {significant_tests}\n")
            f.write(f"Percentage of Significant Results: {(significant_tests/total_tests*100):.1f}%\n\n")
            
            # Decision based on proportion of significant results
            if significant_tests / total_tests >= 0.3:  # At least 30% significant
                f.write("CONCLUSION: HYPOTHESIS 2 (H1) IS SUPPORTED\n")
                f.write("There are statistically significant differences in detection\n")
                f.write("performance between SCS-ID and Baseline CNN models across\n")
                f.write("multiple attack types and performance metrics.\n")
            else:
                f.write("CONCLUSION: HYPOTHESIS 2 (H1) IS NOT SUPPORTED\n")
                f.write("There are insufficient statistically significant differences\n")
                f.write("in detection performance between the models to support\n")
                f.write("the alternative hypothesis.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
        print(f"✅ Hypothesis 2 report generated: {report_path}")
        
    def generate_comprehensive_summary(self):
        """Generate comprehensive summary document"""
        summary_path = self.docs_dir / "comprehensive_analysis_summary.txt"
        
        baseline_perf = self.results['baseline']['performance']
        scs_id_perf = self.results['scs_id']['performance']
        baseline_comp = self.results['baseline']['computational']
        scs_id_comp = self.results['scs_id']['computational']
        efficiency = self.results['comparison']['efficiency_reductions']
        h1_stats = self.results['statistical_tests']['hypothesis_1']
        h2_stats = self.results['statistical_tests']['hypothesis_2']
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE MODEL COMPARISON ANALYSIS SUMMARY\n")
            f.write("SCS-ID vs Baseline CNN (Ayeni et al.)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: CIC-IDS2017 Campus Network Traffic\n")
            f.write(f"Test Samples: {len(self.results['baseline']['performance']['predictions']):,}\n")
            f.write(f"Attack Types Analyzed: {len(self.attack_types)}\n\n")
            
            # Executive summary
            f.write("EXECUTIVE SUMMARY:\n")
            f.write("=" * 20 + "\n")
            f.write("This analysis compares the proposed SCS-ID (Squeezed ConvSeek for Intrusion\n")
            f.write("Detection) model against Ayeni et al.'s baseline CNN model across two main\n")
            f.write("hypotheses: computational efficiency improvements and detection performance\n")
            f.write("differences in campus network environments.\n\n")
            
            # Key findings
            f.write("KEY FINDINGS:\n")
            f.write("-" * 15 + "\n")
            
            # Hypothesis 1 summary
            h1_significant = sum(1 for stats in h1_stats.values() if stats['significant'])
            f.write(f"1. COMPUTATIONAL EFFICIENCY (Hypothesis 1):\n")
            f.write(f"   • Parameter Reduction: {efficiency['parameter_count_reduction']:.1f}% fewer parameters\n")
            f.write(f"   • Memory Reduction: {efficiency['memory_utilization_reduction']:.1f}% less memory usage\n")
            f.write(f"   • Latency Improvement: {efficiency['inference_latency_improvement']:.1f}% faster inference\n")
            f.write(f"   • Statistical Significance: {h1_significant}/3 metrics significant\n")
            f.write(f"   • Conclusion: {'SUPPORTED' if h1_significant >= 2 else 'NOT FULLY SUPPORTED'}\n\n")
            
            # Hypothesis 2 summary
            total_h2_tests = sum(len(metric_stats) for metric_stats in h2_stats.values())
            significant_h2_tests = sum(
                sum(1 for stats in metric_stats.values() if stats['significant'])
                for metric_stats in h2_stats.values()
            )
            
            f.write(f"2. DETECTION PERFORMANCE (Hypothesis 2):\n")
            f.write(f"   • Overall F1-Score: Baseline {baseline_perf['f1_macro']:.4f} vs SCS-ID {scs_id_perf['f1_macro']:.4f}\n")
            f.write(f"   • Overall Accuracy: Baseline {baseline_perf['accuracy']:.4f} vs SCS-ID {scs_id_perf['accuracy']:.4f}\n")
            f.write(f"   • False Positive Rate: Baseline {baseline_perf['fpr_overall']:.4f} vs SCS-ID {scs_id_perf['fpr_overall']:.4f}\n")
            f.write(f"   • Per-Attack Significance: {significant_h2_tests}/{total_h2_tests} tests significant\n")
            f.write(f"   • Conclusion: {'SUPPORTED' if significant_h2_tests/total_h2_tests >= 0.3 else 'NOT SUPPORTED'}\n\n")
            
            # Model specifications
            f.write("MODEL SPECIFICATIONS:\n")
            f.write("-" * 22 + "\n")
            f.write(f"Baseline CNN (Ayeni et al.):\n")
            f.write(f"  • Parameters: {baseline_comp['total_parameters']:,}\n")
            f.write(f"  • Memory Usage: {baseline_comp['peak_memory_mb']:.2f} MB\n")
            f.write(f"  • Model Size: {baseline_comp['model_size_mb']:.2f} MB\n")
            f.write(f"  • Inference Latency: {baseline_comp['inference_latency_ms']:.4f} ms/connection\n\n")
            
            f.write(f"SCS-ID (Proposed):\n")
            f.write(f"  • Parameters: {scs_id_comp['total_parameters']:,}\n")
            f.write(f"  • Memory Usage: {scs_id_comp['peak_memory_mb']:.2f} MB\n")
            f.write(f"  • Model Size: {scs_id_comp['model_size_mb']:.2f} MB\n")
            f.write(f"  • Inference Latency: {scs_id_comp['inference_latency_ms']:.4f} ms/connection\n\n")
            
            # Statistical methodology
            f.write("STATISTICAL METHODOLOGY:\n")
            f.write("-" * 25 + "\n")
            f.write("• Significance Level: alpha = 0.05\n")
            f.write("• Bootstrap Sampling: n = 30 samples per comparison\n")
            f.write("• Normality Testing: Shapiro-Wilk test\n")
            f.write("• Parametric Test: Paired t-test (when data is normal)\n")
            f.write("• Non-parametric Test: Wilcoxon signed-rank test (when non-normal)\n")
            f.write("• Effect Size: Cohen's d for paired samples\n\n")
            
            # Detailed results table
            f.write("DETAILED PERFORMANCE COMPARISON:\n")
            f.write("-" * 35 + "\n")
            f.write(f"{'Metric':<30} | {'Baseline':<12} | {'SCS-ID':<12} | {'Improvement':<12}\n")
            f.write("-" * 80 + "\n")
            
            performance_metrics = [
                ('Accuracy', baseline_perf['accuracy'], scs_id_perf['accuracy']),
                ('Precision (Macro)', baseline_perf['precision_macro'], scs_id_perf['precision_macro']),
                ('Recall (Macro)', baseline_perf['recall_macro'], scs_id_perf['recall_macro']),
                ('F1-Score (Macro)', baseline_perf['f1_macro'], scs_id_perf['f1_macro']),
                ('False Positive Rate', baseline_perf['fpr_overall'], scs_id_perf['fpr_overall']),
                ('False Alarm Rate', baseline_perf['far_overall'], scs_id_perf['far_overall']),
                ('AUC-ROC', baseline_perf['auc_roc'], scs_id_perf['auc_roc']),
                ('Matthews Corr. Coeff', baseline_perf['mcc_overall'], scs_id_perf['mcc_overall'])
            ]
            
            for metric_name, baseline_val, scs_id_val in performance_metrics:
                if 'false' in metric_name.lower():
                    improvement = ((baseline_val - scs_id_val) / baseline_val) * 100  # Lower is better
                    improvement_text = f"{improvement:+.2f}%"
                else:
                    improvement = ((scs_id_val - baseline_val) / baseline_val) * 100  # Higher is better
                    improvement_text = f"{improvement:+.2f}%"
                
                f.write(f"{metric_name:<30} | {baseline_val:<12.6f} | {scs_id_val:<12.6f} | {improvement_text:<12}\n")
            
            f.write("\n")
            
            # Computational improvements table
            f.write("COMPUTATIONAL IMPROVEMENTS:\n")
            f.write("-" * 28 + "\n")
            f.write(f"{'Metric':<25} | {'Baseline':<15} | {'SCS-ID':<15} | {'Reduction':<12}\n")
            f.write("-" * 75 + "\n")
            
            comp_metrics = [
                ('Parameters', f"{baseline_comp['total_parameters']:,}", f"{scs_id_comp['total_parameters']:,}", 
                 f"{efficiency['parameter_count_reduction']:.1f}%"),
                ('Memory (MB)', f"{baseline_comp['peak_memory_mb']:.2f}", f"{scs_id_comp['peak_memory_mb']:.2f}", 
                 f"{efficiency['memory_utilization_reduction']:.1f}%"),
                ('Latency (ms)', f"{baseline_comp['inference_latency_ms']:.4f}", f"{scs_id_comp['inference_latency_ms']:.4f}", 
                 f"{efficiency['inference_latency_improvement']:.1f}%")
            ]
            
            for metric_name, baseline_val, scs_id_val, reduction in comp_metrics:
                f.write(f"{metric_name:<25} | {baseline_val:<15} | {scs_id_val:<15} | {reduction:<12}\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 17 + "\n")
            
            if h1_significant >= 2:
                f.write("1. COMPUTATIONAL EFFICIENCY: The SCS-ID model is recommended for\n")
                f.write("   deployment in resource-constrained campus environments due to\n")
                f.write("   significant improvements in computational efficiency.\n\n")
            else:
                f.write("1. COMPUTATIONAL EFFICIENCY: Further optimization may be needed\n")
                f.write("   to achieve statistically significant computational improvements.\n\n")
            
            if significant_h2_tests / total_h2_tests >= 0.3:
                f.write("2. DETECTION PERFORMANCE: The SCS-ID model shows sufficient\n")
                f.write("   performance differences to warrant consideration for deployment\n")
                f.write("   in campus network intrusion detection systems.\n\n")
            else:
                f.write("2. DETECTION PERFORMANCE: The models show similar detection\n")
                f.write("   performance. Choice should be based on computational requirements.\n\n")
            
            # Limitations
            f.write("LIMITATIONS:\n")
            f.write("-" * 13 + "\n")
            f.write("• Analysis based on CIC-IDS2017 dataset specific to campus networks\n")
            f.write("• Bootstrap sampling used to generate statistical comparisons\n")
            f.write("• Results may not generalize to other network environments\n")
            f.write("• Model performance may vary with different hyperparameter settings\n\n")
            
            # Future work
            f.write("FUTURE WORK:\n")
            f.write("-" * 13 + "\n")
            f.write("• Validation on additional campus network datasets\n")
            f.write("• Real-time deployment testing in actual campus environments\n")
            f.write("• Cross-validation with other intrusion detection datasets\n")
            f.write("• Investigation of model interpretability and explainability\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF COMPREHENSIVE ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n")
            
        print(f"✅ Comprehensive summary generated: {summary_path}")
        
    def generate_all_documentation(self):
        """Generate all documentation files"""
        print("📝 Generating comprehensive documentation...")
        print("=" * 50)
        
        self.generate_hypothesis_1_report()
        self.generate_hypothesis_2_report()
        self.generate_comprehensive_summary()
        
        print("\\n✅ All documentation generated successfully!")
        print(f"📁 Documentation saved to: {self.docs_dir}")
        
        # List generated files
        print("\\nGenerated Files:")
        for file_path in self.docs_dir.glob("*.txt"):
            print(f"  📄 {file_path.name}")
            
        return self.docs_dir

def main():
    """Generate documentation from comparison results"""
    results_dir = Path("results/comprehensive_comparison")
    
    if not (results_dir / "complete_results.pkl").exists():
        print("❌ Results file not found. Please run comprehensive_model_comparison.py first.")
        return
    
    doc_generator = ResultsDocumentationGenerator(results_dir)
    doc_generator.load_results()
    docs_dir = doc_generator.generate_all_documentation()
    
    return docs_dir

if __name__ == "__main__":
    main()
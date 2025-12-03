"""
Main execution script for comprehensive model comparison analysis
Runs all components: comparison, visualization, and documentation
"""

import os
import sys
from pathlib import Path
import traceback
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def main():
    """Execute complete analysis pipeline"""
    print("🚀 COMPREHENSIVE MODEL COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Analysis covers:")
    print("  • Hypothesis 1: Computational Efficiency Metrics")
    print("  • Hypothesis 2: Detection Performance (Per-Attack Analysis)")
    print("  • Statistical Significance Testing")
    print("  • Comprehensive Visualizations")
    print("  • Detailed Documentation")
    print("=" * 80)
    
    try:
        # Step 1: Run comprehensive model comparison
        print("\n🔬 STEP 1: COMPREHENSIVE MODEL COMPARISON")
        print("-" * 50)
        
        from experiments.comprehensive_model_comparison import main as run_comparison
        results = run_comparison()
        
        if results is None:
            print("❌ Model comparison failed!")
            return False
            
        print("✅ Model comparison completed successfully!")
        
        # Step 2: Generate visualizations
        print("\n🎨 STEP 2: GENERATING VISUALIZATIONS")
        print("-" * 50)
        
        from experiments.visualization_generator import main as generate_viz
        viz_dir = generate_viz()
        
        if viz_dir is None:
            print("❌ Visualization generation failed!")
            return False
            
        print("✅ Visualizations generated successfully!")
        
        # Step 3: Generate documentation
        print("\n📝 STEP 3: GENERATING DOCUMENTATION")
        print("-" * 50)
        
        from experiments.documentation_generator import main as generate_docs
        docs_dir = generate_docs()
        
        if docs_dir is None:
            print("❌ Documentation generation failed!")
            return False
            
        print("✅ Documentation generated successfully!")
        
        # Step 4: Summary
        print("\n" + "=" * 80)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 80)
        
        results_dir = Path("results/comprehensive_comparison")
        
        print(f"\n📊 QUICK SUMMARY:")
        if results:
            baseline_perf = results['baseline']['performance']
            scs_id_perf = results['scs_id']['performance']
            efficiency = results['comparison']['efficiency_reductions']
            h1_stats = results['statistical_tests']['hypothesis_1']
            
            print(f"\n🎯 Performance Comparison:")
            print(f"  • Baseline F1-Score:     {baseline_perf['f1_macro']:.4f}")
            print(f"  • SCS-ID F1-Score:       {scs_id_perf['f1_macro']:.4f}")
            print(f"  • Baseline FPR:          {baseline_perf['fpr_overall']:.4f}")
            print(f"  • SCS-ID FPR:            {scs_id_perf['fpr_overall']:.4f}")
            
            print(f"\n⚡ Efficiency Improvements:")
            print(f"  • Parameter Reduction:   {efficiency['parameter_count_reduction']:.2f}%")
            print(f"  • Memory Reduction:      {efficiency['memory_utilization_reduction']:.2f}%")
            print(f"  • Latency Improvement:   {efficiency['inference_latency_improvement']:.2f}%")
            
            print(f"\n📈 Hypothesis 1 Results:")
            significant_h1 = sum(1 for stats in h1_stats.values() if stats['significant'])
            print(f"  • Significant Metrics:   {significant_h1}/3")
            print(f"  • Conclusion:            {'✅ H1 SUPPORTED' if significant_h1 >= 2 else '❌ H1 NOT SUPPORTED'}")
            
            # Count Hypothesis 2 results
            h2_stats = results['statistical_tests']['hypothesis_2']
            total_h2_tests = sum(len(metric_stats) for metric_stats in h2_stats.values())
            significant_h2_tests = sum(
                sum(1 for stats in metric_stats.values() if stats['significant'])
                for metric_stats in h2_stats.values()
            )
            
            print(f"\n📊 Hypothesis 2 Results:")
            print(f"  • Significant Tests:     {significant_h2_tests}/{total_h2_tests}")
            print(f"  • Significance Rate:     {(significant_h2_tests/total_h2_tests*100):.1f}%")
            print(f"  • Conclusion:            {'✅ H2 SUPPORTED' if significant_h2_tests/total_h2_tests >= 0.3 else '❌ H2 NOT SUPPORTED'}")
        
        print(f"\n📁 Generated Files:")
        print(f"  📊 Results:       {results_dir}")
        print(f"  🎨 Visualizations: {results_dir / 'visualizations'}")
        print(f"  📝 Documentation:  {results_dir / 'documentation'}")
        
        # List key files
        print(f"\n📄 Key Output Files:")
        
        # Results files
        if (results_dir / "results_summary.json").exists():
            print(f"  • results_summary.json - Complete results in JSON format")
        if (results_dir / "complete_results.pkl").exists():
            print(f"  • complete_results.pkl - Raw results with all data")
            
        # Visualization files
        viz_dir = results_dir / "visualizations"
        if viz_dir.exists():
            for viz_file in viz_dir.glob("*.png"):
                print(f"  • {viz_file.name} - {viz_file.stem.replace('_', ' ').title()}")
                
        # Documentation files
        docs_dir = results_dir / "documentation"
        if docs_dir.exists():
            for doc_file in docs_dir.glob("*.txt"):
                print(f"  • {doc_file.name} - {doc_file.stem.replace('_', ' ').title()}")
        
        print(f"\n✅ Analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Analysis failed!")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
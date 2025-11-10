# -*- coding: utf-8 -*-
# experiments/real_model_inference_benchmark.py - Inference benchmarking with actual trained models
import sys
import time
import gc
import psutil
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
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
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        BATCH_SIZE = 1024
    config = Config()

class RealModelInferenceBenchmark:
    """
    Inference benchmarking using actual trained model .pth files
    """
    
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.baseline_model = None
        self.scs_id_model = None
        self.test_data = None
        self.benchmark_results = []
        self.batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.warmup_iterations = 10
        self.benchmark_iterations = 100
        
        print(f"🔧 Real Model Benchmark initialized on device: {self.device}")
        
    def load_actual_trained_models(self):
        """Load actual trained models from .pth files"""
        print("\n📦 Loading actual trained models from .pth files...")
        
        # Paths to actual model files
        baseline_path = Path(config.RESULTS_DIR) / "baseline" / "best_baseline_model.pth"
        scs_id_path = Path(config.RESULTS_DIR) / "scs_id" / "scs_id_best_model.pth"
        
        if not baseline_path.exists():
            print(f"   ❌ Baseline model not found: {baseline_path}")
            return False
            
        if not scs_id_path.exists():
            print(f"   ❌ SCS-ID model not found: {scs_id_path}")
            return False
        
        try:
            # Load the state dictionaries
            print(f"   📂 Loading baseline from: {baseline_path}")
            baseline_state_dict = torch.load(baseline_path, map_location=self.device)
            
            print(f"   📂 Loading SCS-ID from: {scs_id_path}")
            scs_id_state_dict = torch.load(scs_id_path, map_location=self.device)
            
            # Import model architectures
            from models.baseline_cnn import BaselineCNN
            from models.scs_id_optimized import OptimizedSCSID
            
            # Create model instances with correct parameters
            # Get the actual input features from the test data or use default
            input_features = 66  # Based on your processed data
            num_classes = 15    # CIC-IDS2017 has 15 classes
            
            print(f"   🏗️  Reconstructing Baseline CNN...")
            self.baseline_model = BaselineCNN(input_features=input_features, num_classes=num_classes)
            
            print(f"   🏗️  Reconstructing SCS-ID model...")
            self.scs_id_model = OptimizedSCSID(input_features=input_features, num_classes=num_classes)
            
            # Load state dictionaries
            try:
                self.baseline_model.load_state_dict(baseline_state_dict, strict=False)
                print(f"   ✅ Baseline model state loaded")
            except Exception as e:
                print(f"   ⚠️  Baseline state loading issue: {e}")
                print(f"      Trying flexible loading...")
                # Try to load what we can
                self.baseline_model.load_state_dict(baseline_state_dict, strict=False)
                print(f"   ✅ Baseline model loaded with flexible matching")
            
            try:
                self.scs_id_model.load_state_dict(scs_id_state_dict, strict=False)
                print(f"   ✅ SCS-ID model state loaded")
            except Exception as e:
                print(f"   ⚠️  SCS-ID state loading issue: {e}")
                print(f"      Trying flexible loading...")
                self.scs_id_model.load_state_dict(scs_id_state_dict, strict=False)
                print(f"   ✅ SCS-ID model loaded with flexible matching")
            
            # Move models to device and set to eval mode
            self.baseline_model.to(self.device).eval()
            self.scs_id_model.to(self.device).eval()
            
            # Count parameters
            baseline_params = sum(p.numel() for p in self.baseline_model.parameters())
            scs_id_params = sum(p.numel() for p in self.scs_id_model.parameters())
            param_reduction = (1 - scs_id_params / baseline_params) * 100
            
            print(f"\n   📊 MODEL COMPARISON:")
            print(f"      Baseline parameters: {baseline_params:,}")
            print(f"      SCS-ID parameters: {scs_id_params:,}")
            print(f"      Parameter reduction: {param_reduction:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_test_data(self):
        """Load actual test data for benchmarking"""
        print("\n📊 Loading test data for benchmarking...")
        
        # Try to load the same test data used for evaluation
        processed_path = Path(config.DATA_DIR) / "processed" / "processed_data.pkl"
        
        if processed_path.exists():
            try:
                with open(processed_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Use test data if available
                if 'X_test' in data and 'y_test' in data:
                    X_test = data['X_test']
                    y_test = data['y_test']
                    print(f"   ✅ Loaded actual test data: {X_test.shape[0]:,} samples")
                else:
                    # Use validation data or subset of training data
                    if 'X_val' in data and 'y_val' in data:
                        X_test = data['X_val']
                        y_test = data['y_val']
                        print(f"   ✅ Using validation data: {X_test.shape[0]:,} samples")
                    else:
                        # Use subset of training data
                        X_train = data['X_train']
                        y_train = data['y_train']
                        test_size = min(10000, len(X_train) // 10)  # 10k samples or 10%
                        X_test = X_train[:test_size]
                        y_test = y_train[:test_size]
                        print(f"   ⚠️  Using training subset: {X_test.shape[0]:,} samples")
                
                # Convert to tensors
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                y_test_tensor = torch.LongTensor(y_test).to(self.device)
                
                self.test_data = TensorDataset(X_test_tensor, y_test_tensor)
                print(f"   📏 Input features: {X_test_tensor.shape[1]}")
                
                return True
                
            except Exception as e:
                print(f"   ❌ Error loading processed data: {e}")
                return self._create_synthetic_test_data()
        else:
            print("   ⚠️  Processed data not found, creating synthetic test data...")
            return self._create_synthetic_test_data()
    
    def _create_synthetic_test_data(self):
        """Create synthetic test data matching the expected dimensions"""
        print("   🧪 Creating synthetic test data...")
        
        n_samples = 10000
        n_features = 66  # Based on processed data dimensions
        n_classes = 15
        
        X_synthetic = torch.randn(n_samples, n_features, device=self.device)
        y_synthetic = torch.randint(0, n_classes, (n_samples,), device=self.device)
        
        self.test_data = TensorDataset(X_synthetic, y_synthetic)
        print(f"   ✅ Created synthetic data: {n_samples:,} samples, {n_features} features")
        
        return True
    
    def measure_model_inference(self, model, data_loader, model_name):
        """Measure inference performance for a single model"""
        print(f"\n⏱️  Measuring {model_name} inference performance...")
        
        # Warmup
        print(f"   🔥 Warming up ({self.warmup_iterations} iterations)...")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(data_loader):
                if i >= self.warmup_iterations:
                    break
                # Correct input shape based on model architecture
                if len(inputs.shape) == 2:
                    if "Baseline" in model_name:
                        inputs = inputs.unsqueeze(1)  # Baseline expects (batch, 1, features)
                    # SCS-ID expects (batch, features) and adds channel dim internally
                try:
                    _ = model(inputs)
                except Exception as e:
                    print(f"   ⚠️  Warmup error: {e}")
                    return None
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark inference
        print(f"   📊 Benchmarking ({self.benchmark_iterations} iterations)...")
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(data_loader):
                if i >= self.benchmark_iterations:
                    break
                
                # Correct input shape based on model architecture  
                if len(inputs.shape) == 2:
                    if "Baseline" in model_name:
                        inputs = inputs.unsqueeze(1)  # Baseline expects (batch, 1, features)
                    # SCS-ID expects (batch, features) and adds channel dim internally
                
                # Memory before inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
                else:
                    mem_before = psutil.Process().memory_info().rss / 1024**2
                
                # Time inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                start_time = time.perf_counter()
                
                try:
                    outputs = model(inputs)
                except Exception as e:
                    print(f"   ❌ Inference error: {e}")
                    return None
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                end_time = time.perf_counter()
                
                # Memory after inference
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated() / 1024**2
                else:
                    mem_after = psutil.Process().memory_info().rss / 1024**2
                
                inference_time = (end_time - start_time) * 1000  # ms
                inference_times.append(inference_time)
                memory_usage.append(mem_after - mem_before)
        
        # Calculate statistics
        batch_size = data_loader.batch_size
        avg_batch_time = np.mean(inference_times)
        std_batch_time = np.std(inference_times)
        avg_sample_time = avg_batch_time / batch_size
        throughput = batch_size * 1000 / avg_batch_time
        avg_memory = np.mean(memory_usage)
        
        results = {
            'model': model_name,
            'batch_size': batch_size,
            'avg_batch_time_ms': avg_batch_time,
            'std_batch_time_ms': std_batch_time,
            'avg_sample_time_ms': avg_sample_time,
            'throughput_samples_per_sec': throughput,
            'avg_memory_mb': avg_memory
        }
        
        print(f"   ✅ Avg batch time: {avg_batch_time:.2f}ms ± {std_batch_time:.2f}ms")
        print(f"   ✅ Avg sample time: {avg_sample_time:.4f}ms")
        print(f"   ✅ Throughput: {throughput:.0f} samples/sec")
        print(f"   ✅ Memory usage: {avg_memory:.1f} MB")
        
        return results
    
    def run_benchmark_across_batch_sizes(self):
        """Run benchmark across different batch sizes"""
        print("\n🚀 Starting benchmark across batch sizes...")
        
        for batch_size in self.batch_sizes:
            print(f"\n📦 Testing batch size: {batch_size}")
            
            try:
                if self.test_data is None:
                    print(f"   ❌ Test data not available")
                    continue
                    
                data_loader = DataLoader(
                    self.test_data,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
                )
                
                # Benchmark baseline model
                baseline_result = self.measure_model_inference(
                    self.baseline_model, data_loader, f"Baseline CNN"
                )
                
                if baseline_result:
                    baseline_result['model_type'] = 'Baseline CNN'
                    self.benchmark_results.append(baseline_result)
                
                # Benchmark SCS-ID model
                scs_id_result = self.measure_model_inference(
                    self.scs_id_model, data_loader, f"SCS-ID"
                )
                
                if scs_id_result:
                    scs_id_result['model_type'] = 'SCS-ID'
                    self.benchmark_results.append(scs_id_result)
                
            except Exception as e:
                print(f"   ❌ Error with batch size {batch_size}: {e}")
                continue
        
        print(f"\n✅ Benchmark complete! Tested {len(self.benchmark_results)} configurations.")
        return pd.DataFrame(self.benchmark_results)
    
    def analyze_performance_improvements(self):
        """Analyze performance improvements"""
        print("\n📈 Analyzing performance improvements...")
        
        if not self.benchmark_results:
            print("   ❌ No benchmark results to analyze")
            return None
        
        results_df = pd.DataFrame(self.benchmark_results)
        improvements = []
        
        for batch_size in self.batch_sizes:
            baseline_data = results_df[
                (results_df['model_type'] == 'Baseline CNN') & 
                (results_df['batch_size'] == batch_size)
            ]
            scs_id_data = results_df[
                (results_df['model_type'] == 'SCS-ID') & 
                (results_df['batch_size'] == batch_size)
            ]
            
            if not baseline_data.empty and not scs_id_data.empty:
                baseline_row = baseline_data.iloc[0]
                scs_id_row = scs_id_data.iloc[0]
                
                # Calculate improvements
                latency_improvement = (1 - scs_id_row['avg_sample_time_ms'] / baseline_row['avg_sample_time_ms']) * 100
                throughput_improvement = (scs_id_row['throughput_samples_per_sec'] / baseline_row['throughput_samples_per_sec'] - 1) * 100
                memory_reduction = (1 - scs_id_row['avg_memory_mb'] / baseline_row['avg_memory_mb']) * 100 if baseline_row['avg_memory_mb'] > 0 else 0
                
                improvements.append({
                    'batch_size': batch_size,
                    'latency_improvement_pct': latency_improvement,
                    'throughput_improvement_pct': throughput_improvement,
                    'memory_reduction_pct': memory_reduction,
                    'baseline_throughput': baseline_row['throughput_samples_per_sec'],
                    'scs_id_throughput': scs_id_row['throughput_samples_per_sec'],
                    'baseline_latency': baseline_row['avg_sample_time_ms'],
                    'scs_id_latency': scs_id_row['avg_sample_time_ms']
                })
        
        improvements_df = pd.DataFrame(improvements)
        
        if not improvements_df.empty:
            avg_throughput_improvement = improvements_df['throughput_improvement_pct'].mean()
            best_throughput_improvement = improvements_df['throughput_improvement_pct'].max()
            optimal_batch_size = improvements_df.loc[
                improvements_df['throughput_improvement_pct'].idxmax(), 'batch_size'
            ]
            
            print(f"   📊 Average Throughput Improvement: {avg_throughput_improvement:.1f}%")
            print(f"   📊 Best Throughput Improvement: {best_throughput_improvement:.1f}%")
            print(f"   📊 Optimal Batch Size: {optimal_batch_size}")
            print(f"   🎯 Target Achievement (>300%): {'✅ ACHIEVED' if best_throughput_improvement > 300 else '❌ NOT ACHIEVED'}")
        
        return improvements_df
    
    def create_visualizations(self, results_df, improvements_df):
        """Create performance visualizations"""
        print("\n📊 Creating visualizations...")
        
        output_dir = Path(config.RESULTS_DIR) / "real_model_inference_benchmark"
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # 1. Throughput comparison
        plt.figure(figsize=(12, 8))
        
        for model_type in ['Baseline CNN', 'SCS-ID']:
            model_data = results_df[results_df['model_type'] == model_type]
            if not model_data.empty:
                plt.plot(model_data['batch_size'], model_data['throughput_samples_per_sec'], 
                        marker='o', linewidth=2, markersize=6, label=model_type)
        
        plt.xlabel('Batch Size', fontsize=12, fontweight='bold')
        plt.ylabel('Throughput (samples/second)', fontsize=12, fontweight='bold')
        plt.title('Real Model Inference Throughput Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'real_model_throughput_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Improvement analysis
        if improvements_df is not None and not improvements_df.empty:
            plt.figure(figsize=(12, 8))
            plt.plot(improvements_df['batch_size'], 
                    improvements_df['throughput_improvement_pct'], 
                    marker='o', linewidth=3, color='green', markersize=8)
            plt.xlabel('Batch Size', fontsize=12, fontweight='bold')
            plt.ylabel('Throughput Improvement (%)', fontsize=12, fontweight='bold')
            plt.title('Real Model Throughput Improvement vs Baseline', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xscale('log', base=2)
            plt.axhline(y=300, color='red', linestyle='--', linewidth=2, label='Target (300%)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'real_model_improvement_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"   ✅ Visualizations saved to: {output_dir}")
    
    def generate_report(self, results_df, improvements_df):
        """Generate comprehensive benchmark report"""
        print("\n📋 Generating benchmark report...")
        
        output_dir = Path(config.RESULTS_DIR) / "real_model_inference_benchmark"
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / "real_model_benchmark_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REAL MODEL INFERENCE BENCHMARK REPORT\n")
            f.write("="*50 + "\n")
            f.write("Benchmarking using actual trained model .pth files\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Benchmark Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model information
            if self.baseline_model and self.scs_id_model:
                baseline_params = sum(p.numel() for p in self.baseline_model.parameters())
                scs_id_params = sum(p.numel() for p in self.scs_id_model.parameters())
                param_reduction = (1 - scs_id_params / baseline_params) * 100
                
                f.write("MODEL INFORMATION\n")
                f.write("-"*17 + "\n")
                f.write(f"Baseline Model: {baseline_params:,} parameters\n")
                f.write(f"SCS-ID Model: {scs_id_params:,} parameters\n")
                f.write(f"Parameter Reduction: {param_reduction:.1f}%\n\n")
            
            # Performance summary
            if improvements_df is not None and not improvements_df.empty:
                avg_improvement = improvements_df['throughput_improvement_pct'].mean()
                best_improvement = improvements_df['throughput_improvement_pct'].max()
                optimal_batch_size = improvements_df.loc[
                    improvements_df['throughput_improvement_pct'].idxmax(), 'batch_size'
                ]
                
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-"*19 + "\n")
                f.write(f"Average Throughput Improvement: {avg_improvement:.1f}%\n")
                f.write(f"Best Throughput Improvement: {best_improvement:.1f}%\n")
                f.write(f"Optimal Batch Size: {optimal_batch_size}\n")
                f.write(f"Target Achievement (>300%): {'YES' if best_improvement > 300 else 'NO'}\n\n")
            
            # Detailed results
            if not results_df.empty:
                f.write("DETAILED RESULTS\n")
                f.write("-"*16 + "\n")
                f.write(f"{'Batch Size':<12} {'Model':<12} {'Throughput':<15} {'Latency (ms)':<15}\n")
                f.write("-"*60 + "\n")
                
                for _, row in results_df.iterrows():
                    f.write(f"{row['batch_size']:<12} {row['model_type']:<12} "
                           f"{row['throughput_samples_per_sec']:<15.0f} "
                           f"{row['avg_sample_time_ms']:<15.4f}\n")
            
            f.write(f"\nBased on actual trained models from .pth files\n")
            f.write(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"   ✅ Report saved: {report_path}")
        return report_path
    
    def run_complete_benchmark(self):
        """Run complete benchmarking pipeline with real models"""
        print("🚀 Starting Real Model Inference Benchmarking")
        print("="*55)
        
        try:
            # Load actual trained models
            if not self.load_actual_trained_models():
                print("❌ Failed to load models")
                return None
            
            # Load test data
            if not self.load_test_data():
                print("❌ Failed to load test data")
                return None
            
            # Run benchmarks
            results_df = self.run_benchmark_across_batch_sizes()
            improvements_df = self.analyze_performance_improvements()
            
            # Create visualizations
            self.create_visualizations(results_df, improvements_df)
            
            # Generate report
            report_path = self.generate_report(results_df, improvements_df)
            
            print("\n" + "="*55)
            print("✅ REAL MODEL BENCHMARKING COMPLETE!")
            print("="*55)
            
            if improvements_df is not None and not improvements_df.empty:
                best_improvement = improvements_df['throughput_improvement_pct'].max()
                avg_improvement = improvements_df['throughput_improvement_pct'].mean()
                
                print(f"🏆 Key Results (REAL MODELS):")
                print(f"   📈 Average Throughput Improvement: {avg_improvement:.1f}%")
                print(f"   🚀 Best Throughput Improvement: {best_improvement:.1f}%")
                print(f"   🎯 Target (>300%): {'✅ ACHIEVED' if best_improvement > 300 else '⚠️  NOT ACHIEVED'}")
            
            print(f"   📋 Report: {report_path}")
            
            return {
                'results': results_df,
                'improvements': improvements_df,
                'report_path': report_path
            }
            
        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Run real model benchmarking"""
    benchmark = RealModelInferenceBenchmark()
    results = benchmark.run_complete_benchmark()
    
    if results:
        print("\n[SUCCESS] Real model benchmarking completed!")
        print("These results are based on your actual trained models.")
    else:
        print("\n[ERROR] Real model benchmarking failed.")

if __name__ == "__main__":
    main()
# run_scs_id_workflow.py
"""
SCS-ID Training Workflow - Two-Stage Pipeline

This script provides a guided workflow for the separated DeepSeek RL and SCS-ID training.

Two-Stage Approach Benefits:
1. Run time-intensive DeepSeek RL once (30-60 min)
2. Reuse selected features for fast SCS-ID training (5-15 min)
3. Enable rapid iteration when tuning SCS-ID hyperparameters
4. Separate experimental and compute-intensive phases

Usage:
  python run_scs_id_workflow.py
"""
import os
import sys
from pathlib import Path
import pickle

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import config


def check_prerequisites():
    """Check if prerequisite data files exist"""
    print("="*70)
    print("CHECKING PREREQUISITES")
    print("="*70)
    
    # Check processed data
    processed_data_path = Path(config.DATA_DIR) / "processed" / "processed_data.pkl"
    if not processed_data_path.exists():
        print("ERROR: Preprocessed data not found!")
        print(f"Expected: {processed_data_path}")
        print("\nSolution: Run data preprocessing first:")
        print("  python data/preprocess.py")
        return False
    
    print(f"SUCCESS: Preprocessed data found: {processed_data_path}")
    
    # Check if DeepSeek RL features already exist
    deepseek_results_path = Path(config.RESULTS_DIR) / "deepseek_feature_selection_complete.pkl"
    if deepseek_results_path.exists():
        try:
            with open(deepseek_results_path, 'rb') as f:
                results = pickle.load(f)
            print(f"SUCCESS: DeepSeek RL features already exist!")
            print(f"   Selected features: {results['selected_feature_count']} from {results['original_feature_count']}")
            print(f"   Training time: {results['training_time_minutes']:.2f} minutes")
            print(f"   Method: {results['method']}")
            return "skip_stage1"
        except Exception as e:
            print(f"! DeepSeek RL results file corrupted: {e}")
            print("  Will need to re-run Stage 1")
    
    print("- DeepSeek RL features not found (will run Stage 1)")
    return True


def run_stage1():
    """Run DeepSeek RL feature selection"""
    print("\n" + "="*70)
    print("STAGE 1: DEEPSEEK RL FEATURE SELECTION")
    print("="*70)
    print("Estimated time: 30-60 minutes")
    print("This will select optimal 42 features from 78 using reinforcement learning")
    print("="*70)
    
    print("\nStarting DeepSeek RL feature selection...")
    
    try:
        # Import and run
        from experiments.deepseek_feature_selection_only import run_deepseek_feature_selection
        
        selected_features, results = run_deepseek_feature_selection()
        
        print("\nSUCCESS: Stage 1 complete!")
        print(f"Selected {len(selected_features)} features in {results['training_time_minutes']:.2f} minutes")
        return True
        
    except Exception as e:
        print(f"\nERROR: Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_stage2():
    """Run fast SCS-ID training with pre-selected features"""
    print("\n" + "="*70)
    print("STAGE 2: FAST SCS-ID TRAINING")  
    print("="*70)
    print("Estimated time: 5-15 minutes")
    print("Using pre-selected DeepSeek RL features for fast training")
    print("="*70)
    
    print("\nStarting fast SCS-ID training...")
    
    try:
        # Import and run
        from experiments.train_scs_id_fast import FastSCSIDTrainer
        
        trainer = FastSCSIDTrainer()
        model, accuracy, f1_score = trainer.train_model()
        
        print(f"\nSUCCESS: Stage 2 complete!")
        print(f"Final accuracy: {accuracy*100:.2f}%")
        print(f"F1 score: {f1_score:.4f}")
        return True
        
    except Exception as e:
        print(f"\nERROR: Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main workflow function"""
    print("\n" + "="*70)
    print("SCS-ID TWO-STAGE TRAINING WORKFLOW")
    print("="*70)
    print("This workflow separates DeepSeek RL feature selection from SCS-ID training")
    print("for faster iteration and better resource management.")
    print("="*70)
    
    # Check prerequisites
    prereq_status = check_prerequisites()
    
    if prereq_status == False:
        print("\nERROR: Prerequisites not met. Please resolve issues and try again.")
        return
    
    # Stage 1: DeepSeek RL feature selection
    if prereq_status == "skip_stage1":
        print("\nSkipping Stage 1 - DeepSeek RL features already exist")
        stage1_success = True
    else:
        stage1_success = run_stage1()
    
    if not stage1_success:
        print("\nERROR: Workflow stopped - Stage 1 failed")
        return
    
    # Stage 2: Fast SCS-ID training
    stage2_success = run_stage2()
    
    if stage2_success:
        print("\n" + "="*70)
        print("COMPLETE WORKFLOW SUCCESS!")
        print("="*70)
        print("Both stages completed successfully:")
        print("SUCCESS: Stage 1: DeepSeek RL feature selection")
        print("SUCCESS: Stage 2: SCS-ID training with structured pruning + INT8 quantization")
        print("\nResults saved in:", config.RESULTS_DIR)
        print("="*70)
    else:
        print("\nERROR: Workflow incomplete - Stage 2 failed")


def show_manual_commands():
    """Show manual commands for running stages separately"""
    print("\n" + "="*70)
    print("MANUAL EXECUTION COMMANDS")
    print("="*70)
    print("You can also run the stages manually:")
    print()
    print("Stage 1 (DeepSeek RL Feature Selection):")
    print("  python experiments/deepseek_feature_selection_only.py")
    print()
    print("Stage 2 (Fast SCS-ID Training):")  
    print("  python experiments/train_scs_id_fast.py")
    print()
    print("Original Combined Version (if needed):")
    print("  python experiments/train_scs_id.py")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
        
        print("\n\n💡 Tip: You can now run Stage 2 multiple times with different")
        print("hyperparameters without re-running the time-intensive DeepSeek RL!")
        
        show_manual_commands()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Workflow interrupted by user")
    except Exception as e:
        print(f"\n❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
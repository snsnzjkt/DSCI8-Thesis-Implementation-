import pickle
import pandas as pd
import numpy as np

def examine_results_file(filepath):
    """Examine the contents of a results pickle file"""
    print(f"\n{'='*60}")
    print(f"Examining: {filepath}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Type: {type(results)}")
        
        if isinstance(results, dict):
            print(f"Keys: {list(results.keys())}")
            for key, value in results.items():
                print(f"\n{key}:")
                if isinstance(value, (list, tuple, np.ndarray)):
                    print(f"  Type: {type(value)}, Shape/Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    if hasattr(value, 'shape'):
                        print(f"  Shape: {value.shape}")
                elif isinstance(value, (int, float)):
                    print(f"  Value: {value}")
                elif isinstance(value, str):
                    print(f"  Value: {value}")
                else:
                    print(f"  Type: {type(value)}")
        
        return results
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# Examine both result files
baseline_results = examine_results_file("results/baseline/baseline_results.pkl")
scs_id_results = examine_results_file("results/scs_id/scs_id_optimized_results.pkl")

# If results contain test predictions, save them to CSV
def save_test_results_to_csv(results, model_name):
    if results and isinstance(results, dict):
        # Look for common test result keys
        test_keys = ['y_test', 'y_pred', 'test_predictions', 'predictions', 'test_labels', 'true_labels']
        
        data_to_save = {}
        for key in test_keys:
            if key in results:
                data_to_save[key] = results[key]
        
        if data_to_save:
            df = pd.DataFrame(data_to_save)
            csv_filename = f"{model_name}_test_results.csv"
            df.to_csv(csv_filename, index=False)
            print(f"\nSaved test results to: {csv_filename}")
            print(f"Columns: {list(df.columns)}")
            print(f"Shape: {df.shape}")
            return csv_filename
    return None

# Save test results to CSV files
baseline_csv = save_test_results_to_csv(baseline_results, "baseline")
scs_id_csv = save_test_results_to_csv(scs_id_results, "scs_id")
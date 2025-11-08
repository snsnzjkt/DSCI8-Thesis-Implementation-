"""
Test script for DeepSeek RL Optimized with subset of data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data.preprocess import CICIDSPreprocessor
from models.deepseek_rl_optimized import OptimizedDeepSeekRL
from sklearn.model_selection import train_test_split
from config import config

def load_subset_data():
    """Load a small subset of data for testing"""
    preprocessor = CICIDSPreprocessor()
    
    # Load only Monday data as test subset
    monday_file = Path(config.DATA_DIR) / "raw" / "Monday-WorkingHours.pcap_ISCX.csv"
    
    # Read only first 10000 rows for quick testing
    df = pd.read_csv(monday_file, nrows=10000)
    
    # Basic preprocessing steps
    X = df.drop(['Label'], axis=1) if 'Label' in df.columns else df
    y = df['Label'] if 'Label' in df.columns else pd.Series(['BENIGN'] * len(df))
    
    # Convert any string columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except:
                print(f"Warning: Dropping non-numeric column {col}")
                X = X.drop(columns=[col])
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Apply standard scaling
    X = preprocessor.scaler.fit_transform(X)
    y = preprocessor.label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def main():
    print("Loading subset of data for testing...")
    X_train, X_test, y_train, y_test = load_subset_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize optimized model with reduced parameters
    model = OptimizedDeepSeekRL(
        max_features=20,     # Reduced for testing
        sample_ratio=0.5,    # Use 50% of training data
        use_cache=True       # Enable caching for faster testing
    )
    
    print("\nStarting feature selection...")
    model.fit(X_train, y_train, X_test, y_test)
    
    selected_features = model.selected_features_idx
    if selected_features is not None:
        print("\nSelected features indices:", selected_features)
        print(f"Number of selected features: {len(selected_features)}")
    else:
        print("\nNo features were selected.")

    # Get feature selection environment stats
    env = getattr(model, 'env', None)
    if env is not None:
        print(f"\nBest F1 Score: {env.best_f1:.4f}")
        print(f"Best FP Rate: {env.best_fp_rate:.4f}")
    else:
        print("\nNo performance metrics available yet.")

if __name__ == "__main__":
    main()
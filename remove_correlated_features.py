#!/usr/bin/env python3
"""
Remove highly correlated features from CIC-IDS2017 dataset based on multicollinearity analysis
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def remove_correlated_features():
    """Remove highly correlated features identified in multicollinearity analysis"""
    
    # Features recommended for removal (21 features)
    features_to_remove = [
        "Subflow Fwd Packets", "Subflow Bwd Packets", "Subflow Fwd Bytes", 
        "Subflow Bwd Bytes", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "SYN Flag Count", "Fwd Header Length.1", "Average Packet Size",
        "Fwd IAT Total", "Flow IAT Max", "Bwd Packet Length Std",
        "Packet Length Std", "Fwd IAT Max", "Idle Mean", "Bwd Packet Length Max",
        "Total Backward Packets", "Total Length of Fwd Packets", 
        "Total Length of Bwd Packets", "Fwd Packet Length Mean", "Bwd Packet Length Mean"
    ]
    
    print(f"Removing {len(features_to_remove)} highly correlated features...")
    
    # Load processed data
    processed_dir = Path("data/processed")
    processed_file = processed_dir / "processed_data.pkl"
    
    with open(processed_file, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test'] 
    feature_names = data['feature_names']
    
    print(f"   Original features: {len(feature_names)}")
    
    # Find indices of features to remove
    indices_to_remove = []
    removed_count = 0
    for feature in features_to_remove:
        if feature in feature_names:
            indices_to_remove.append(feature_names.index(feature))
            print(f"   Removing: {feature}")
            removed_count += 1
        else:
            print(f"   Not found: {feature}")
    
    # Remove features
    indices_to_keep = [i for i in range(len(feature_names)) if i not in indices_to_remove]
    
    X_train_clean = X_train[:, indices_to_keep]
    X_test_clean = X_test[:, indices_to_keep]
    feature_names_clean = [feature_names[i] for i in indices_to_keep]
    
    print(f"   Final features: {len(feature_names_clean)}")
    print(f"   Reduction: {removed_count} features removed ({removed_count/len(feature_names)*100:.1f}%)")
    
    # Save cleaned data
    cleaned_data = {
        'X_train': X_train_clean,
        'X_test': X_test_clean,
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'label_encoder': data['label_encoder'],
        'scaler': data['scaler'], 
        'feature_names': feature_names_clean,
        'num_classes': data['num_classes'],
        'class_names': data['class_names'],
        'removed_features': features_to_remove,
        'original_feature_count': len(feature_names)
    }
    
    output_file = processed_dir / "processed_data_no_multicollinearity.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(cleaned_data, f)
    
    print(f"   Saved cleaned dataset: {output_file}")
    
    return X_train_clean, X_test_clean, feature_names_clean

if __name__ == "__main__":
    remove_correlated_features()

#!/usr/bin/env python3
"""
Display multicollinearity analysis results and visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import pandas as pd

def display_multicollinearity_results():
    """Display the multicollinearity analysis results"""
    
    viz_dir = Path("visualizations")
    
    print("📊 MULTICOLLINEARITY ANALYSIS RESULTS")
    print("=" * 60)
    
    # Read the summary report
    report_file = viz_dir / "multicollinearity_report.txt"
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract key statistics
        for line in lines[:20]:
            if "High Correlation Pairs Found:" in line:
                high_corr = line.split(":")[1].strip()
                print(f"🔍 High Correlation Pairs (|r| > 0.8): {high_corr}")
            elif "Extreme Correlation Pairs Found:" in line:
                extreme_corr = line.split(":")[1].strip()
                print(f"⚠️  Extreme Correlation Pairs (|r| > 0.95): {extreme_corr}")
            elif "High VIF Features:" in line:
                high_vif = line.split(":")[1].strip()
                print(f"📈 High VIF Features (>10): {high_vif}")
    
    print(f"\n🎯 RISK ASSESSMENT: CRITICAL")
    print(f"   Your dataset has significant multicollinearity issues!")
    
    # Key multicollinearity visualizations
    multicollinearity_files = [
        ("multicollinearity_summary.png", "Complete Multicollinearity Analysis Summary"),
        ("correlation_heatmap.png", "Full Feature Correlation Heatmap"),
        ("high_correlation_focus.png", "Focused View of Problematic Features")
    ]
    
    print(f"\n🖼️  Displaying Multicollinearity Visualizations:")
    print("-" * 50)
    
    for filename, title in multicollinearity_files:
        file_path = viz_dir / filename
        
        if file_path.exists():
            print(f"\n📊 {title}")
            print(f"   File: {filename}")
            
            try:
                # Load and display the image
                img = mpimg.imread(file_path)
                
                plt.figure(figsize=(16, 12))
                plt.imshow(img)
                plt.axis('off')
                plt.title(title, fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.show()
                
                print(f"   ✅ Displayed: {title}")
                
            except Exception as e:
                print(f"   ❌ Error displaying {filename}: {e}")
        else:
            print(f"\n❌ File not found: {filename}")
    
    # Display key findings
    print(f"\n🔍 KEY FINDINGS:")
    print(f"=" * 30)
    
    # Perfect correlations (r = 1.0)
    perfect_correlations = [
        "Total Fwd Packets ↔ Subflow Fwd Packets",
        "Total Backward Packets ↔ Subflow Bwd Packets", 
        "Total Length of Fwd Packets ↔ Subflow Fwd Bytes",
        "Total Length of Bwd Packets ↔ Subflow Bwd Bytes",
        "Fwd Packet Length Mean ↔ Avg Fwd Segment Size",
        "Bwd Packet Length Mean ↔ Avg Bwd Segment Size",
        "Fwd PSH Flags ↔ SYN Flag Count",
        "Fwd Header Length ↔ Fwd Header Length.1"
    ]
    
    print(f"\n🔴 CRITICAL: 8 Perfect Correlations (r = 1.0)")
    print(f"   These are essentially duplicate features:")
    for i, corr in enumerate(perfect_correlations, 1):
        print(f"   {i}. {corr}")
    
    # Feature removal recommendations
    print(f"\n💡 IMMEDIATE RECOMMENDATIONS:")
    print(f"-" * 35)
    print(f"🗑️  Remove 21 highly redundant features:")
    
    features_to_remove = [
        "Subflow Fwd Packets", "Subflow Bwd Packets", "Subflow Fwd Bytes", 
        "Subflow Bwd Bytes", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "SYN Flag Count", "Fwd Header Length.1", "Average Packet Size",
        "Fwd IAT Total", "Flow IAT Max", "Bwd Packet Length Std",
        "Packet Length Std", "Fwd IAT Max", "Idle Mean", "Bwd Packet Length Max",
        "Total Backward Packets", "Total Length of Fwd Packets", 
        "Total Length of Bwd Packets", "Fwd Packet Length Mean", "Bwd Packet Length Mean"
    ]
    
    for i, feature in enumerate(features_to_remove, 1):
        print(f"   {i:2d}. {feature}")
    
    print(f"\n🛠️  IMPACT OF REMOVING REDUNDANT FEATURES:")
    print(f"   • Original features: 78")
    print(f"   • After removal: 57 features (26.9% reduction)")
    print(f"   • Eliminates perfect correlations")
    print(f"   • Reduces model complexity")
    print(f"   • Improves model stability")
    print(f"   • Prevents overfitting")
    
    print(f"\n⚠️  MODEL IMPLICATIONS:")
    print(f"   • Current multicollinearity will cause:")
    print(f"     - Unstable coefficient estimates")
    print(f"     - High variance in predictions") 
    print(f"     - Difficulty interpreting feature importance")
    print(f"     - Poor generalization performance")
    
    print(f"\n✅ RECOMMENDED NEXT STEPS:")
    print(f"   1. Create filtered dataset without redundant features")
    print(f"   2. Verify correlation matrix after feature removal")
    print(f"   3. Re-train models with cleaned feature set")
    print(f"   4. Compare model performance before/after cleaning")
    print(f"   5. Use regularization (Ridge/Lasso) for remaining correlations")
    
    print(f"\n📁 Generated Files:")
    print(f"   📊 multicollinearity_summary.png - Complete analysis overview")
    print(f"   🔥 correlation_heatmap.png - Full 78x78 correlation matrix")
    print(f"   🎯 high_correlation_focus.png - Focus on problematic features")
    print(f"   📄 multicollinearity_report.txt - Detailed text report")

def create_feature_removal_script():
    """Create a script to remove highly correlated features"""
    
    script_content = '''#!/usr/bin/env python3
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
    
    print(f"🗑️  Removing {len(features_to_remove)} highly correlated features...")
    
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
    for feature in features_to_remove:
        if feature in feature_names:
            indices_to_remove.append(feature_names.index(feature))
            print(f"   ✓ Removing: {feature}")
        else:
            print(f"   ⚠️  Not found: {feature}")
    
    # Remove features
    indices_to_keep = [i for i in range(len(feature_names)) if i not in indices_to_remove]
    
    X_train_clean = X_train[:, indices_to_keep]
    X_test_clean = X_test[:, indices_to_keep]
    feature_names_clean = [feature_names[i] for i in indices_to_keep]
    
    print(f"   Final features: {len(feature_names_clean)}")
    print(f"   Reduction: {len(indices_to_remove)} features removed ({len(indices_to_remove)/len(feature_names)*100:.1f}%)")
    
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
    
    print(f"   ✅ Saved cleaned dataset: {output_file}")
    
    return X_train_clean, X_test_clean, feature_names_clean

if __name__ == "__main__":
    remove_correlated_features()
'''
    
    # Save the script
    with open("remove_correlated_features.py", 'w') as f:
        f.write(script_content)
    
    print(f"\n✅ Created feature removal script: remove_correlated_features.py")
    print(f"   Run this script to create a cleaned dataset without multicollinearity")

if __name__ == "__main__":
    display_multicollinearity_results()
    create_feature_removal_script()
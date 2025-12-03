#!/usr/bin/env python3
"""
Display the generated imputation and preprocessing visualization graphs
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def display_preprocessing_visualizations():
    """Display the key preprocessing visualization graphs"""
    
    viz_dir = Path("visualizations")
    
    # Key visualization files
    key_visualizations = [
        ("comprehensive_before_after_preprocessing.png", "Comprehensive Before vs After Preprocessing"),
        ("imputation_validation_detailed.png", "Detailed Imputation Validation"),
        ("feature_transformation_summary.png", "Feature Transformation Summary"),
        ("raw_data_overview.png", "Raw Data Overview"),
        ("preprocessed_data_overview.png", "Preprocessed Data Overview")
    ]
    
    print("🖼️  Displaying Preprocessing Visualization Graphs")
    print("=" * 60)
    
    for filename, title in key_visualizations:
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
    
    print(f"\n📁 All visualization files are saved in: {viz_dir.absolute()}")
    
    # Summary of what each visualization shows
    print(f"\n📋 Visualization Summary:")
    print(f"   1. comprehensive_before_after_preprocessing.png:")
    print(f"      - Class distribution before/after balancing")
    print(f"      - Missing values before/after imputation")
    print(f"      - Feature distributions raw vs normalized")
    print(f"      - Statistical summaries")
    
    print(f"   2. imputation_validation_detailed.png:")
    print(f"      - Comparison of original vs imputed data distributions")
    print(f"      - Validates that imputation preserved data characteristics")
    
    print(f"   3. feature_transformation_summary.png:")
    print(f"      - Distribution of missing value percentages across features")
    print(f"      - Normalization effectiveness (σ ≈ 1)")
    print(f"      - Mean centering effectiveness (μ ≈ 0)")
    print(f"      - Overall transformation statistics")

if __name__ == "__main__":
    display_preprocessing_visualizations()
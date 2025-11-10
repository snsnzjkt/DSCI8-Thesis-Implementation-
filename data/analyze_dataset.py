"""
Dataset Statistics and Analysis
Provides overview of the CICIDS2017 dataset composition and characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import List, Dict, Any

def load_dataset():
    """Load all CSV files from the dataset"""
    print("\n📊 Loading CICIDS2017 dataset...")
    data_dir = Path("data/raw")
    all_data = []
    
    for file in data_dir.glob("*.csv"):
        print(f"Reading {file.name}")
        df = pd.read_csv(file)
        all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total samples loaded: {len(combined_data):,}")
    return combined_data

def analyze_class_distribution(data: pd.DataFrame):
    """Analyze and visualize class distribution"""
    print("\n📊 Analyzing class distribution...")
    
    # Get class counts
    class_counts = data[' Label'].value_counts()
    total_samples = len(data)
    
    print("\nClass Distribution:")
    print("-" * 50)
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{class_name.strip():25} {count:8,} samples ({percentage:5.2f}%)")
    
    # Visualize class distribution
    plt.figure(figsize=(15, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Attack Class Distribution", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("results/class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_statistics(data: pd.DataFrame):
    """Analyze feature statistics and distributions"""
    print("\n📊 Feature Statistics:")
    print("-" * 50)
    
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    # Remove label column if present
    label_cols = [col for col in numeric_cols if 'label' in col.lower() or 'class' in col.lower()]
    if label_cols:
        numeric_cols = numeric_cols.drop(label_cols)
    
    # Calculate basic statistics
    stats = data[numeric_cols].describe()
    
    print(f"\nTotal Features: {len(numeric_cols)}")
    print(f"Numeric Features: {len(numeric_cols)}")
    
    # Check for missing values
    missing = data.isnull().sum()
    features_with_missing = missing[missing > 0]
    if len(features_with_missing) > 0:
        print("\nFeatures with missing values:")
        for feature, count in features_with_missing.items():
            print(f"{feature:30} {count:8,} missing values")
    else:
        print("\nNo missing values found in the dataset")
    
    # Create feature distribution plots
    print("\nGenerating feature distribution visualizations...")
    
    # Select top 15 most varying features
    variances = data[numeric_cols].var()
    top_features = variances.nlargest(15).index
    
    # Distribution plots
    output_dir = Path("results/feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for feature in top_features:
        plt.figure(figsize=(15, 6))
        
        # Distribution plot with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(data=data, x=feature, stat='density', kde=True, bins=50)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        
        # Boxplot for outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(y=data[feature])
        plt.title(f'Boxplot of {feature}')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{feature}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create correlation heatmap
    print("Generating correlation heatmap...")
    plt.figure(figsize=(15, 12))
    correlation_matrix = data[numeric_cols].corr()
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig("results/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Convert Label to numeric for correlation analysis
    # Map Labels to numbers
    label_mapping = {label: idx for idx, label in enumerate(data[' Label'].unique())}
    numeric_labels = data[' Label'].map(label_mapping)
    
    # Feature importance based on correlation with numeric Label
    label_correlations = abs(data[numeric_cols].corrwith(numeric_labels))
    top_correlations = label_correlations.nlargest(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_correlations.values, y=top_correlations.index)
    plt.title('Top 10 Features Correlated with Attack Labels')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print correlation values
    print("\nTop 10 Features by Correlation with Attack Labels:")
    print("-" * 50)
    for feature, corr in top_correlations.items():
        print(f"{feature:30} {corr:.4f}")

def analyze_data_split():
    """Analyze train/test split statistics"""
    print("\n📊 Dataset Split Analysis:")
    print("-" * 50)
    
    # Load split data if available
    data_dir = Path("data/processed")
    
    # Check if processed directory exists
    if not data_dir.exists():
        print("\nProcessed data directory not found.")
        print("Please run preprocessing first to generate train/test splits.")
        return
        
    # Look for train/test files
    train_files = list(data_dir.glob("*train*.pkl"))
    test_files = list(data_dir.glob("*test*.pkl"))
    
    if not train_files or not test_files:
        print("\nProcessed train/test split files not found.")
        print("Please run preprocessing first to generate the splits.")
        return
        
    try:
        with open(train_files[0], "rb") as f:
            import pickle
            train_data = pickle.load(f)
        with open(test_files[0], "rb") as f:
            test_data = pickle.load(f)
            
        if isinstance(train_data, tuple):
            X_train, y_train = train_data
            X_test, y_test = test_data
        else:
            X_train = train_data
            X_test = test_data
            y_train = train_data.get('labels', [])
            y_test = test_data.get('labels', [])
        
        print(f"\nTraining Set: {len(X_train):,} samples")
        print(f"Testing Set:  {len(X_test):,} samples")
        
        # Class distribution in splits
        print("\nClass Distribution in Splits:")
        if y_train is not None and len(y_train) > 0:
            train_dist = pd.Series(y_train).value_counts()
            test_dist = pd.Series(y_test).value_counts()
            
            for class_idx in sorted(set(y_train)):
                train_count = train_dist.get(class_idx, 0)
                test_count = test_dist.get(class_idx, 0)
                train_pct = (train_count / len(y_train)) * 100
                test_pct = (test_count / len(y_test)) * 100
                class_name = f"Class {class_idx}" if isinstance(class_idx, (int, float)) else class_idx
                print(f"\n{class_name}:")
                print(f"  Train: {train_count:6,} ({train_pct:.2f}%)")
                print(f"  Test:  {test_count:6,} ({test_pct:.2f}%)")
        else:
            print("\nNo label information found in the processed data.")
            
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"\nError reading processed data files: {str(e)}")
        print("The files may be corrupted or in an unexpected format.")
    except Exception as e:
        print(f"\nUnexpected error analyzing data split: {str(e)}")
        print("Please check the processed data files format.")

def main():
    """Main analysis function"""
    print("\n🔍 CICIDS2017 Dataset Analysis")
    print("=" * 50)
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Load and analyze raw data
    data = load_dataset()
    
    # Analyze class distribution
    analyze_class_distribution(data)
    
    # Analyze feature statistics
    analyze_feature_statistics(data)
    
    # Analyze train/test split
    analyze_data_split()
    
    print("\n✅ Analysis complete! Visualizations saved in 'results' directory.")

if __name__ == "__main__":
    main()
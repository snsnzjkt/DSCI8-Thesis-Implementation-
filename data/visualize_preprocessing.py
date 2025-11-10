"""
Visualization of network traffic data before and after preprocessing
Generates comprehensive visualizations to understand data distribution and preprocessing effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_raw_data():
    """Load raw network traffic data"""
    print("Loading raw data...")
    data_dir = Path('data/raw')
    dfs = []
    
    for file in data_dir.glob('*.csv'):
        print(f"Reading {file.name}")
        df = pd.read_csv(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def create_feature_distributions(data, title, output_path):
    """Create distribution plots for numeric features"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    n_features = len(numeric_cols)
    n_rows = (n_features + 2) // 3  # Ceiling division
    
    plt.figure(figsize=(15, 5 * n_rows))
    plt.suptitle(f"{title} Feature Distributions", fontsize=16, y=0.95)
    
    for idx, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, 3, idx)
        sns.histplot(data[col], kde=True)
        plt.title(f"{col} Distribution")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(data, title, output_path):
    """Create correlation heatmap for numeric features"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr = data[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title(f"{title} Feature Correlations", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_attack_distribution(data, title, output_path):
    """Create visualization of attack type distribution"""
    plt.figure(figsize=(12, 6))
    label_col = ' Label' if ' Label' in data.columns else 'Label'  # Handle both possible column names
    attack_counts = data[label_col].value_counts()
    
    # Plot attack distribution
    sns.barplot(x=attack_counts.index, y=attack_counts.values)
    plt.title(f"{title} Attack Type Distribution", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    plt.yscale('log')  # Use log scale for better visualization
    
    # Add count labels on top of bars
    for i, v in enumerate(attack_counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance(data, title, output_path):
    """Create feature importance visualization based on variance"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    variances = data[numeric_cols].var().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=variances.index, y=variances.values)
    plt.title(f"{title} Feature Importance (Variance)", fontsize=14)
    plt.xticks(rotation=90)
    plt.ylabel('Variance')
    plt.yscale('log')  # Use log scale for better visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_preprocessing_effects():
    """Create visualizations for data before and after preprocessing"""
    print("\n📊 Creating data visualizations...")
    
    # Create output directory
    output_dir = Path('results/data_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    raw_data = load_raw_data()
    print(f"Loaded {len(raw_data):,} samples")
    
    # Create visualizations for raw data
    print("\nGenerating raw data visualizations...")
    create_feature_distributions(
        raw_data, 
        "Raw Data",
        output_dir / "raw_feature_distributions.png"
    )
    create_correlation_heatmap(
        raw_data,
        "Raw Data",
        output_dir / "raw_correlation_heatmap.png"
    )
    create_attack_distribution(
        raw_data,
        "Raw Data",
        output_dir / "raw_attack_distribution.png"
    )
    create_feature_importance(
        raw_data,
        "Raw Data",
        output_dir / "raw_feature_importance.png"
    )
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    try:
        with open('data/processed/train_data.pkl', 'rb') as f:
            X_train, y_train = pickle.load(f)
        with open('data/processed/test_data.pkl', 'rb') as f:
            X_test, y_test = pickle.load(f)
            
        # Convert to DataFrame for visualization
        preprocessed_data = pd.DataFrame(
            np.vstack([X_train, X_test]),
            columns=[f'Feature_{i+1}' for i in range(X_train.shape[1])]
        )
        preprocessed_data['Label'] = np.concatenate([y_train, y_test])
        
        # Create visualizations for preprocessed data
        print("\nGenerating preprocessed data visualizations...")
        create_feature_distributions(
            preprocessed_data,
            "Preprocessed Data",
            output_dir / "preprocessed_feature_distributions.png"
        )
        create_correlation_heatmap(
            preprocessed_data,
            "Preprocessed Data",
            output_dir / "preprocessed_correlation_heatmap.png"
        )
        create_attack_distribution(
            preprocessed_data,
            "Preprocessed Data",
            output_dir / "preprocessed_attack_distribution.png"
        )
        create_feature_importance(
            preprocessed_data,
            "Preprocessed Data",
            output_dir / "preprocessed_feature_importance.png"
        )
        
        # Create before-after comparison plots
        print("\nGenerating comparison visualizations...")
        
        # Feature statistics comparison
        plt.figure(figsize=(15, 6))
        
        # Raw data statistics
        plt.subplot(1, 2, 1)
        raw_stats = raw_data.select_dtypes(include=[np.number]).describe()
        sns.boxplot(data=raw_data.select_dtypes(include=[np.number]))
        plt.title("Raw Data Feature Statistics")
        plt.xticks(rotation=90)
        
        # Preprocessed data statistics
        plt.subplot(1, 2, 2)
        preprocessed_stats = preprocessed_data.select_dtypes(include=[np.number]).describe()
        sns.boxplot(data=preprocessed_data.drop('Label', axis=1))
        plt.title("Preprocessed Data Feature Statistics")
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_statistics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary statistics
        with open(output_dir / "data_statistics_summary.txt", 'w') as f:
            f.write("Raw Data Statistics:\n")
            f.write("===================\n")
            f.write(str(raw_stats))
            f.write("\n\nPreprocessed Data Statistics:\n")
            f.write("===========================\n")
            f.write(str(preprocessed_stats))
        
        print("\n✅ Visualizations completed!")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error loading preprocessed data: {e}")
        print("Please run the preprocessing script first.")

if __name__ == '__main__':
    visualize_preprocessing_effects()
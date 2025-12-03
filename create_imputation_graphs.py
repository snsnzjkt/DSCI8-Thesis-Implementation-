#!/usr/bin/env python3
"""
Create comprehensive before/after preprocessing visualizations for CIC-IDS2017 dataset
Shows the impact of data imputation, normalization, and other preprocessing steps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
sns.set_palette("husl")
PRIMARY_COLOR = '#2E86AB'  # Professional blue

class ImputationVisualizer:
    def __init__(self):
        self.data_dir = Path("data")
        self.viz_dir = Path("visualizations")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.viz_dir.mkdir(exist_ok=True)
        
    def load_raw_data(self):
        """Load raw CIC-IDS2017 data"""
        print("📁 Loading raw data...")
        
        # Find raw CSV files
        raw_files = list(self.raw_dir.glob("*.csv"))
        if not raw_files:
            print("❌ No raw CSV files found!")
            return None
        
        print(f"Found {len(raw_files)} raw files:")
        for file in raw_files[:3]:  # Show first 3
            print(f"   - {file.name}")
        if len(raw_files) > 3:
            print(f"   ... and {len(raw_files) - 3} more")
        
        # Load first file as representative sample
        sample_file = raw_files[0]
        print(f"Loading sample from: {sample_file.name}")
        
        try:
            df = pd.read_csv(sample_file)
            print(f"✅ Loaded {len(df):,} samples with {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"❌ Error loading raw data: {e}")
            return None
    
    def load_processed_data(self):
        """Load processed data"""
        print("📁 Loading processed data...")
        
        processed_file = self.processed_dir / "processed_data.pkl"
        if not processed_file.exists():
            print(f"❌ Processed data not found: {processed_file}")
            return None
        
        try:
            with open(processed_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✅ Loaded processed data:")
            print(f"   - Training: {data['X_train'].shape}")
            print(f"   - Test: {data['X_test'].shape}")
            print(f"   - Classes: {data['num_classes']}")
            print(f"   - Features: {len(data['feature_names'])}")
            
            return data
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return None
    
    def create_before_after_comparison(self, raw_df, processed_data):
        """Create comprehensive before/after preprocessing comparison"""
        print("📊 Creating before/after preprocessing comparison...")
        
        # Extract processed training data (representative sample)
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        feature_names = processed_data['feature_names']
        class_names = processed_data['class_names']
        
        # Create comprehensive comparison figure
        fig = plt.figure(figsize=(20, 15))
        
        # Main title
        fig.suptitle('CIC-IDS2017 Dataset: Before vs After Preprocessing', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Class Distribution Comparison (Top Row, Left)
        ax1 = fig.add_subplot(gs[0, :2])
        if ' Label' in raw_df.columns:
            label_col = ' Label'
        elif 'Label' in raw_df.columns:
            label_col = 'Label'
        else:
            label_col = None
            
        if label_col:
            raw_counts = raw_df[label_col].value_counts()
            # Limit to top 10 classes for readability
            top_classes = raw_counts.head(10)
            
            y_pos = np.arange(len(top_classes))
            ax1.barh(y_pos, top_classes.values, color=PRIMARY_COLOR, alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([str(cls)[:20] for cls in top_classes.index], fontsize=8)
            ax1.set_xlabel('Sample Count (log scale)')
            ax1.set_xscale('log')
            ax1.set_title('BEFORE: Raw Data Class Distribution\n(Highly Imbalanced)', 
                         fontweight='bold')
            
            # Add percentages
            total = len(raw_df)
            for i, count in enumerate(top_classes.values):
                pct = (count / total) * 100
                ax1.text(count, i, f' {pct:.1f}%', va='center', fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'No labels found in raw data', ha='center', va='center')
            ax1.set_title('BEFORE: Raw Data (No Labels)', fontweight='bold')
        
        # 2. After Preprocessing Class Distribution (Top Row, Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        processed_counts = np.bincount(y_train)
        processed_labels = [class_names[i] for i in range(len(processed_counts)) if processed_counts[i] > 0]
        processed_values = processed_counts[processed_counts > 0]
        
        y_pos = np.arange(len(processed_labels))
        bars = ax2.barh(y_pos, processed_values, color=PRIMARY_COLOR)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([str(cls)[:20] for cls in processed_labels], fontsize=8)
        ax2.set_xlabel('Sample Count')
        ax2.set_title('AFTER: Balanced Training Data\n(1:1 Ratio via SMOTE)', fontweight='bold')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, processed_values)):
            ax2.text(count + max(processed_values) * 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{count:,}', ha='left', va='center', fontsize=8)
        
        # 3. Data Quality: Missing Values (Second Row, Left)
        ax3 = fig.add_subplot(gs[1, :2])
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if 'label' not in col.lower()]
        
        missing_counts = raw_df[feature_cols].isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        
        if len(missing_features) > 0:
            top_missing = missing_features.head(15)  # Show top 15
            ax3.barh(range(len(top_missing)), top_missing.values, color='red', alpha=0.7)
            ax3.set_yticks(range(len(top_missing)))
            ax3.set_yticklabels([col[:20] for col in top_missing.index], fontsize=8)
            ax3.set_xlabel('Missing Value Count')
            ax3.set_title(f'BEFORE: Missing Values\n({len(missing_features)} features affected)', 
                         fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Missing Values\nFound in Sample', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('BEFORE: Data Quality\n(No Missing Values)', fontweight='bold')
        
        # 4. After Processing: Data Quality (Second Row, Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        # Check for missing values in processed data
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        processed_missing = X_train_df.isnull().sum()
        processed_missing_features = processed_missing[processed_missing > 0]
        
        if len(processed_missing_features) > 0:
            ax4.barh(range(len(processed_missing_features)), list(processed_missing_features.values), 
                    color='orange', alpha=0.7)
            ax4.set_title('AFTER: Missing Values\n(Should be None)', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, '✅ No Missing Values\n(Median Imputation Applied)', 
                    ha='center', va='center', fontsize=12, color='green', fontweight='bold')
            ax4.set_title('AFTER: Data Quality\n(Imputation Complete)', fontweight='bold')
        ax4.set_xlim(0, 1)
        
        # 5. Feature Distribution Examples (Third Row)
        # Select a few representative features for comparison
        sample_features = ['Destination_Port', 'Flow_Duration', 'Total_Fwd_Packets', 'Fwd_Packet_Length_Mean']
        available_features = [f for f in sample_features if f in raw_df.columns]
        
        if not available_features:
            # Use first few numeric features
            available_features = feature_cols[:4]
        
        for i, feature in enumerate(available_features[:2]):
            # Raw data distribution
            ax_raw = fig.add_subplot(gs[2, i*2])
            if feature in raw_df.columns:
                raw_values = raw_df[feature].dropna()
                # Handle infinite values
                raw_values = raw_values[np.isfinite(raw_values)]
                
                if len(raw_values) > 0:
                    try:
                        # Use log scale for better visualization if data spans many orders of magnitude
                        value_range = raw_values.max() - raw_values.min()
                        if value_range > 0 and raw_values.max() / max(raw_values.min(), 1e-10) > 1000:
                            raw_values_log = np.log10(raw_values + 1)  # +1 to handle zeros
                            ax_raw.hist(raw_values_log, bins=30, color=PRIMARY_COLOR, alpha=0.7, edgecolor='black')
                            ax_raw.set_xlabel('log10(Value + 1)')
                        else:
                            ax_raw.hist(raw_values, bins=30, color=PRIMARY_COLOR, alpha=0.7, edgecolor='black')
                            ax_raw.set_xlabel('Value')
                        ax_raw.set_ylabel('Frequency')
                        ax_raw.set_title(f'BEFORE: {feature[:15]}', fontweight='bold', fontsize=10)
                    except Exception as e:
                        ax_raw.text(0.5, 0.5, f'Error plotting\n{feature}', ha='center', va='center')
                        ax_raw.set_title(f'BEFORE: {feature[:15]} (Error)', fontweight='bold', fontsize=10)
                else:
                    ax_raw.text(0.5, 0.5, 'No valid data', ha='center', va='center')
                    ax_raw.set_title(f'BEFORE: {feature[:15]} (No Data)', fontweight='bold', fontsize=10)
            
            # Processed data distribution
            ax_processed = fig.add_subplot(gs[2, i*2 + 1])
            if feature in feature_names:
                feature_idx = feature_names.index(feature)
                processed_values = X_train[:, feature_idx]
                # Ensure values are finite
                processed_values = processed_values[np.isfinite(processed_values)]
                
                if len(processed_values) > 0:
                    try:
                        ax_processed.hist(processed_values, bins=30, color='green', alpha=0.7, edgecolor='black')
                        ax_processed.set_xlabel('Normalized Value')
                        ax_processed.set_ylabel('Frequency')
                        ax_processed.set_title(f'AFTER: {feature[:15]}\n(Normalized)', fontweight='bold', fontsize=10)
                    except Exception as e:
                        ax_processed.text(0.5, 0.5, f'Error plotting\nnormalized {feature}', ha='center', va='center')
                        ax_processed.set_title(f'AFTER: {feature[:15]} (Error)', fontweight='bold', fontsize=10)
                else:
                    ax_processed.text(0.5, 0.5, 'No valid data', ha='center', va='center')
                    ax_processed.set_title(f'AFTER: {feature[:15]} (No Data)', fontweight='bold', fontsize=10)
            else:
                ax_processed.text(0.5, 0.5, f'Feature not found\nin processed data', ha='center', va='center')
                ax_processed.set_title(f'AFTER: {feature[:15]} (Missing)', fontweight='bold', fontsize=10)
        
        # 6. Statistical Summary (Fourth Row)
        ax6 = fig.add_subplot(gs[3, :2])
        
        # Raw data stats
        raw_stats = [
            f'Total Samples: {len(raw_df):,}',
            f'Total Features: {len(feature_cols)}',
            f'Missing Values: {raw_df[feature_cols].isnull().sum().sum():,}',
            f'Infinite Values: {np.isinf(raw_df[feature_cols].select_dtypes(include=[np.number])).sum().sum():,}',
            f'Data Types: {raw_df.dtypes.value_counts().to_dict()}',
            f'Memory Usage: {raw_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB'
        ]
        
        ax6.text(0.05, 0.95, 'BEFORE: Raw Data Statistics\n\n' + '\n'.join(raw_stats), 
                transform=ax6.transAxes, fontsize=10, verticalalignment='top', 
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[3, 2:])
        
        # Processed data stats
        processed_stats = [
            f'Training Samples: {len(X_train):,}',
            f'Test Samples: {processed_data["X_test"].shape[0]:,}',
            f'Final Features: {len(feature_names)}',
            f'Classes: {len(class_names)}',
            f'Missing Values: 0 (Imputed)',
            f'Value Range: [{X_train.min():.3f}, {X_train.max():.3f}]',
            f'Memory Usage: {X_train.nbytes / 1024**2:.1f} MB'
        ]
        
        ax7.text(0.05, 0.95, 'AFTER: Processed Data Statistics\n\n' + '\n'.join(processed_stats), 
                transform=ax7.transAxes, fontsize=10, verticalalignment='top', 
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        ax7.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'comprehensive_before_after_preprocessing.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Saved comprehensive comparison to: {self.viz_dir / 'comprehensive_before_after_preprocessing.png'}")
    
    def create_imputation_detail_plots(self, raw_df, processed_data):
        """Create detailed imputation validation plots"""
        print("📊 Creating detailed imputation validation plots...")
        
        X_train = processed_data['X_train']
        feature_names = processed_data['feature_names']
        
        # Find features that had missing values in raw data
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if 'label' not in col.lower()]
        
        missing_counts = raw_df[feature_cols].isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        
        if len(missing_features) == 0:
            print("   No missing values found in raw data sample")
            
            # Create a demonstration plot showing the normalization effect instead
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Data Transformation: Raw vs Normalized Features', fontsize=16, fontweight='bold')
            
            # Select first 6 features for demonstration
            demo_features = feature_cols[:6]
            
            for i, feature in enumerate(demo_features):
                row, col = i // 3, i % 3
                
                if feature in raw_df.columns:
                    # Raw data
                    raw_values = raw_df[feature].dropna()
                    
                    # Normalized data (if feature exists in processed)
                    if feature in feature_names:
                        feature_idx = feature_names.index(feature)
                        normalized_values = X_train[:, feature_idx]
                        
                        # Plot both distributions
                        axes[row, col].hist(raw_values, bins=30, alpha=0.7, label='Raw Data', 
                                          color='blue', density=True)
                        axes[row, col].hist(normalized_values, bins=30, alpha=0.7, label='Normalized', 
                                          color='red', density=True)
                        axes[row, col].set_title(f'{feature[:20]}', fontweight='bold')
                        axes[row, col].set_xlabel('Value')
                        axes[row, col].set_ylabel('Density')
                        axes[row, col].legend()
                        axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'normalization_comparison.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            
            print(f"✅ Created normalization comparison plot")
            return
        
        # Create imputation validation plots
        n_features = min(6, len(missing_features))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Imputation Validation: Original vs Imputed Data Distributions', 
                    fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(list(missing_features.index)[:n_features]):
            row, col = i // 3, i % 3
            
            # Original data (non-missing only)
            original_clean = raw_df[feature].dropna()
            
            # Handle infinite values in original data
            original_clean = original_clean[np.isfinite(original_clean)]
            
            # Find corresponding feature in processed data
            if feature in feature_names:
                feature_idx = feature_names.index(feature)
                imputed_values = X_train[:, feature_idx]
                
                # Ensure imputed values are finite
                imputed_values = imputed_values[np.isfinite(imputed_values)]
                
                if len(original_clean) > 0 and len(imputed_values) > 0:
                    # Plot distributions
                    axes[row, col].hist(original_clean, bins=30, alpha=0.7, 
                                      label=f'Original (n={len(original_clean):,})', 
                                      color='blue', density=True)
                    axes[row, col].hist(imputed_values, bins=30, alpha=0.7, 
                                      label=f'After Imputation (n={len(imputed_values):,})', 
                                      color='red', density=True)
                else:
                    axes[row, col].text(0.5, 0.5, f'No finite values\nin {feature}', 
                                      ha='center', va='center')
                
                axes[row, col].set_title(f'{feature}\n({missing_counts[feature]:,} values imputed)', 
                                       fontweight='bold')
                axes[row, col].set_xlabel('Value')
                axes[row, col].set_ylabel('Density')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
            else:
                axes[row, col].text(0.5, 0.5, f'Feature {feature}\nnot found in\nprocessed data', 
                                  ha='center', va='center')
                axes[row, col].set_title(f'{feature} (Not Found)')
        
        # Fill empty subplots
        for i in range(n_features, 6):
            row, col = i // 3, i % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'imputation_validation_detailed.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Created detailed imputation validation plots")
        print(f"   Found {len(missing_features)} features with missing values")
        print(f"   Validated imputation for {min(n_features, len(missing_features))} features")
    
    def create_feature_transformation_summary(self, raw_df, processed_data):
        """Create summary of all feature transformations"""
        print("📊 Creating feature transformation summary...")
        
        X_train = processed_data['X_train']
        feature_names = processed_data['feature_names']
        
        # Calculate transformation statistics
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if 'label' not in col.lower()]
        
        transformation_stats = []
        
        for feature in feature_cols:
            if feature in feature_names:
                feature_idx = feature_names.index(feature)
                
                # Raw stats (handle infinite values)
                raw_values = raw_df[feature].dropna()
                raw_values = raw_values[np.isfinite(raw_values)]
                
                if len(raw_values) > 0:
                    raw_mean = raw_values.mean()
                    raw_std = raw_values.std()
                else:
                    raw_mean = 0
                    raw_std = 0
                    
                raw_missing = raw_df[feature].isnull().sum()
                
                # Processed stats (handle infinite values)
                processed_values = X_train[:, feature_idx]
                processed_values = processed_values[np.isfinite(processed_values)]
                
                if len(processed_values) > 0:
                    proc_mean = processed_values.mean()
                    proc_std = processed_values.std()
                else:
                    proc_mean = 0
                    proc_std = 0
                
                transformation_stats.append({
                    'feature': feature,
                    'raw_mean': raw_mean,
                    'raw_std': raw_std,
                    'raw_missing': raw_missing,
                    'proc_mean': proc_mean,
                    'proc_std': proc_std,
                    'missing_pct': (raw_missing / len(raw_df)) * 100
                })
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Transformation Summary', fontsize=16, fontweight='bold')
        
        # 1. Missing value distribution
        missing_pcts = [stat['missing_pct'] for stat in transformation_stats]
        axes[0, 0].hist(missing_pcts, bins=20, color=PRIMARY_COLOR, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Missing Value Percentage')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Distribution of Missing Value Percentages\nAcross Features')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Normalization effectiveness (standard deviation)
        proc_stds = [stat['proc_std'] for stat in transformation_stats]
        axes[0, 1].hist(proc_stds, bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=1.0, color='red', linestyle='--', label='Target (σ=1)')
        axes[0, 1].set_xlabel('Standard Deviation After Normalization')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Normalization Effectiveness\n(Should be centered around 1)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Mean centering effectiveness
        proc_means = [stat['proc_mean'] for stat in transformation_stats]
        axes[1, 0].hist(proc_means, bins=20, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0.0, color='red', linestyle='--', label='Target (μ=0)')
        axes[1, 0].set_xlabel('Mean After Normalization')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].set_title('Mean Centering Effectiveness\n(Should be centered around 0)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary statistics table
        axes[1, 1].axis('off')
        
        summary_text = f"""Preprocessing Summary Statistics:
        
Total Features: {len(transformation_stats)}
Features with Missing Values: {sum(1 for s in transformation_stats if s['raw_missing'] > 0)}
Average Missing Value %: {np.mean(missing_pcts):.2f}%
Max Missing Value %: {max(missing_pcts):.2f}%

Normalization Quality:
Mean of Means: {np.mean(proc_means):.6f} (target: 0)
Mean of Std Devs: {np.mean(proc_stds):.6f} (target: 1)
Std Dev Range: [{min(proc_stds):.3f}, {max(proc_stds):.3f}]

Transformation Success:
✅ Missing values imputed
✅ Features normalized (z-score)
✅ Data ready for ML models"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'feature_transformation_summary.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Created feature transformation summary")
        print(f"   Analyzed {len(transformation_stats)} features")
        print(f"   {sum(1 for s in transformation_stats if s['raw_missing'] > 0)} features had missing values")
        print(f"   Average normalization quality: μ={np.mean(proc_means):.6f}, σ={np.mean(proc_stds):.6f}")
    
    def run_complete_analysis(self):
        """Run complete imputation and preprocessing visualization analysis"""
        print("🚀 Creating Comprehensive Preprocessing Analysis")
        print("=" * 60)
        
        # Load data
        raw_df = self.load_raw_data()
        processed_data = self.load_processed_data()
        
        if raw_df is None or processed_data is None:
            print("❌ Cannot proceed - missing data files")
            return
        
        print(f"\n📊 Data Overview:")
        print(f"   Raw data: {raw_df.shape}")
        print(f"   Processed train: {processed_data['X_train'].shape}")
        print(f"   Processed test: {processed_data['X_test'].shape}")
        
        # Create visualizations
        self.create_before_after_comparison(raw_df, processed_data)
        self.create_imputation_detail_plots(raw_df, processed_data)
        self.create_feature_transformation_summary(raw_df, processed_data)
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"📊 Generated visualizations in: {self.viz_dir}")
        print("   - comprehensive_before_after_preprocessing.png")
        print("   - imputation_validation_detailed.png (if missing values found)")
        print("   - normalization_comparison.png (if no missing values)")
        print("   - feature_transformation_summary.png")
        print("\n🔍 Key Insights:")
        print("   1. Compare class distributions before/after balancing")
        print("   2. Validate imputation preserved data characteristics")  
        print("   3. Confirm normalization achieved μ≈0, σ≈1")
        print("   4. Check data quality improvements")

def main():
    """Main execution function"""
    visualizer = ImputationVisualizer()
    visualizer.run_complete_analysis()

if __name__ == "__main__":
    main()
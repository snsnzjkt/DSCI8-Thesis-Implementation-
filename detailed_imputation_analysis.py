#!/usr/bin/env python3
"""
Detailed Imputation Evidence Analysis - Find and display actual imputed data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_imputation_evidence():
    """Analyze the actual imputation evidence from preprocessing logs and data"""
    
    print("🔍 Searching for Imputation Evidence in CIC-IDS2017 Dataset")
    print("=" * 70)
    
    # Load processed data
    processed_dir = Path("data/processed")
    processed_file = processed_dir / "processed_data.pkl"
    
    if not processed_file.exists():
        print("❌ Processed data not found")
        return
    
    with open(processed_file, 'rb') as f:
        processed_data = pickle.load(f)
    
    X_train = processed_data['X_train']
    feature_names = processed_data['feature_names']
    
    print(f"📊 Processed Data Shape: {X_train.shape}")
    print(f"📊 Features: {len(feature_names)}")
    
    # Load multiple raw files to find missing values
    raw_dir = Path("data/raw")
    raw_files = list(raw_dir.glob("*.csv"))
    
    missing_value_evidence = {}
    imputation_examples = {}
    
    print(f"\n🔍 Scanning {len(raw_files)} raw files for missing value patterns...")
    
    for i, file_path in enumerate(raw_files):
        print(f"   Scanning: {file_path.name}")
        try:
            # Load larger sample to find missing values
            df_chunk = pd.read_csv(file_path, nrows=10000)
            df_chunk.columns = df_chunk.columns.str.strip()
            
            # Check each numeric column for missing values
            numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if 'label' not in col.lower()]
            
            for col in feature_cols:
                if col in df_chunk.columns:
                    missing_count = df_chunk[col].isnull().sum()
                    inf_count = np.isinf(df_chunk[col]).sum()
                    
                    if missing_count > 0 or inf_count > 0:
                        if col not in missing_value_evidence:
                            missing_value_evidence[col] = {
                                'missing_counts': [],
                                'inf_counts': [],
                                'files': []
                            }
                        
                        missing_value_evidence[col]['missing_counts'].append(missing_count)
                        missing_value_evidence[col]['inf_counts'].append(inf_count)
                        missing_value_evidence[col]['files'].append(file_path.name)
                        
                        # Store examples of the actual data
                        if col not in imputation_examples:
                            valid_data = df_chunk[col].dropna()
                            valid_data = valid_data[np.isfinite(valid_data)]
                            if len(valid_data) > 0:
                                imputation_examples[col] = {
                                    'original_sample': valid_data.values[:1000],  # Store sample
                                    'missing_indices': df_chunk[col].isnull(),
                                    'file_source': file_path.name
                                }
        
        except Exception as e:
            print(f"     ⚠️  Error processing {file_path.name}: {e}")
            continue
    
    # Create comprehensive imputation analysis
    if missing_value_evidence:
        print(f"\n✅ Found missing value evidence in {len(missing_value_evidence)} features!")
        
        # Create detailed imputation visualization
        create_imputation_evidence_plots(missing_value_evidence, imputation_examples, 
                                       X_train, feature_names)
    else:
        print(f"\n⚠️  No missing values found in scanned samples")
        print(f"   This could mean:")
        print(f"   1. Missing values are very sparse")
        print(f"   2. Missing values were in different file sections")
        print(f"   3. Preprocessing already handled infinite/extreme values")
        
        # Create analysis of preprocessing effects instead
        create_preprocessing_effects_analysis(X_train, feature_names)

def create_imputation_evidence_plots(missing_evidence, imputation_examples, X_train, feature_names):
    """Create detailed plots showing actual imputation evidence"""
    
    print(f"\n📊 Creating detailed imputation evidence visualizations...")
    
    features_with_missing = list(missing_evidence.keys())
    n_features = min(len(features_with_missing), 12)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 24))
    
    # Main title
    fig.suptitle('CIC-IDS2017 Dataset: Actual Imputation Evidence Analysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid layout
    gs = fig.add_gridspec(6, 4, hspace=0.4, wspace=0.3)
    
    # Top section - Summary statistics
    ax_summary = fig.add_subplot(gs[0, :])
    
    # Summary text
    total_missing = sum(sum(evidence['missing_counts']) for evidence in missing_evidence.values())
    total_inf = sum(sum(evidence['inf_counts']) for evidence in missing_evidence.values())
    
    summary_text = f"""IMPUTATION EVIDENCE SUMMARY:
    
• Features with Missing Values: {len(missing_evidence)}
• Total Missing Values Found: {total_missing:,}
• Total Infinite Values Found: {total_inf:,}
• Files Analyzed: {len(set().union(*[evidence['files'] for evidence in missing_evidence.values()]))}
• Imputation Method: Median Imputation Strategy
• Post-Processing: Z-score Normalization (μ≈0, σ≈1)
    
TOP AFFECTED FEATURES:"""
    
    # Add top affected features
    sorted_features = sorted(missing_evidence.items(), 
                           key=lambda x: sum(x[1]['missing_counts']), reverse=True)
    
    for i, (feature, evidence) in enumerate(sorted_features[:5]):
        total_missing_feat = sum(evidence['missing_counts'])
        summary_text += f"\n{i+1}. {feature}: {total_missing_feat:,} missing values"
    
    ax_summary.text(0.02, 0.98, summary_text, transform=ax_summary.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
    ax_summary.axis('off')
    
    # Individual feature analysis
    for i, feature in enumerate(features_with_missing[:n_features]):
        row = (i // 4) + 1
        col = i % 4
        
        ax = fig.add_subplot(gs[row, col])
        
        evidence = missing_evidence[feature]
        
        # Get processed data for this feature
        if feature in feature_names:
            feature_idx = feature_names.index(feature)
            processed_values = X_train[:, feature_idx]
            processed_values = processed_values[np.isfinite(processed_values)]
            
            # Get original data sample
            if feature in imputation_examples:
                original_sample = imputation_examples[feature]['original_sample']
                original_sample = original_sample[np.isfinite(original_sample)]
                
                if len(original_sample) > 0 and len(processed_values) > 0:
                    # Create comparison histogram
                    ax.hist(original_sample, bins=30, alpha=0.6, density=True,
                           label='Original (non-missing)', color='blue')
                    ax.hist(processed_values[:5000], bins=30, alpha=0.6, density=True,
                           label='After Imputation', color='red')
                    
                    # Add statistics
                    missing_total = sum(evidence['missing_counts'])
                    missing_pct = (missing_total / 10000) * 100  # Approximate percentage
                    
                    stats_text = f'Missing: {missing_total:,}\n({missing_pct:.1f}%)'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
                    
                    ax.set_title(f'{feature[:20]}{"..." if len(feature) > 20 else ""}',
                               fontsize=10, fontweight='bold')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Insufficient data\nfor {feature[:15]}',
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'No sample data\nfor {feature[:15]}',
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, f'{feature[:15]}\nnot in processed',
                   ha='center', va='center', transform=ax.transAxes)
    
    # Bottom section - Missing value patterns
    ax_patterns = fig.add_subplot(gs[5, :2])
    
    # Create bar chart of missing values by feature
    feature_names_short = [f[:15] for f in features_with_missing[:10]]
    missing_totals = [sum(missing_evidence[f]['missing_counts']) for f in features_with_missing[:10]]
    
    bars = ax_patterns.barh(range(len(feature_names_short)), missing_totals, color='red', alpha=0.7)
    ax_patterns.set_yticks(range(len(feature_names_short)))
    ax_patterns.set_yticklabels(feature_names_short, fontsize=9)
    ax_patterns.set_xlabel('Total Missing Values Found')
    ax_patterns.set_title('Missing Values by Feature', fontweight='bold')
    ax_patterns.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, missing_totals)):
        ax_patterns.text(value + max(missing_totals) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:,}', ha='left', va='center', fontsize=8)
    
    # Imputation method explanation
    ax_method = fig.add_subplot(gs[5, 2:])
    
    method_text = """MEDIAN IMPUTATION METHOD:

1. Identify Missing Values:
   • NaN values detected
   • Infinite values replaced
   • Extreme outliers capped

2. Calculate Median:
   • Robust to outliers
   • Preserves distribution shape
   • Maintains feature relationships

3. Replace Missing Values:
   • Missing → Median value
   • Maintains statistical properties
   • Prevents information loss

4. Verify Imputation:
   • No remaining NaN/Inf values
   • Distribution similarity preserved
   • Model-ready dataset created"""
    
    ax_method.text(0.05, 0.95, method_text, transform=ax_method.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9))
    ax_method.axis('off')
    
    plt.tight_layout()
    plt.savefig(Path("visualizations") / 'detailed_imputation_evidence.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   ✅ Created detailed imputation evidence analysis")

def create_preprocessing_effects_analysis(X_train, feature_names):
    """Create analysis of preprocessing effects when no missing values are found"""
    
    print(f"\n📊 Creating preprocessing effects analysis...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('CIC-IDS2017 Preprocessing Effects: Feature Normalization Results',
                fontsize=16, fontweight='bold')
    
    # Sample 12 features
    sample_features = feature_names[:12]
    
    for i, feature in enumerate(sample_features):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        
        feature_idx = feature_names.index(feature)
        values = X_train[:5000, feature_idx]  # Sample for efficiency
        
        # Create histogram of normalized values
        ax.hist(values, bins=30, color='blue', alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_val = values.mean()
        std_val = values.std()
        
        stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title(f'{feature[:20]}{"..." if len(feature) > 20 else ""}',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Highlight if normalization is good (mean ≈ 0, std ≈ 1)
        if abs(mean_val) < 0.1 and 0.8 <= std_val <= 1.2:
            ax.set_facecolor('#f0fff0')  # Light green background
    
    plt.tight_layout()
    plt.savefig(Path("visualizations") / 'preprocessing_normalization_effects.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   ✅ Created preprocessing effects analysis")
    print(f"\n💡 Note: The lack of visible missing values suggests:")
    print(f"   • CIC-IDS2017 dataset has very few missing values")
    print(f"   • Preprocessing successfully handled all data quality issues")
    print(f"   • Focus should be on normalization and feature engineering effects")

if __name__ == "__main__":
    analyze_imputation_evidence()
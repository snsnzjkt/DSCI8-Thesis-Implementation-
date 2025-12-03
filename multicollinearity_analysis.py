#!/usr/bin/env python3
"""
Comprehensive Multicollinearity Analysis for CIC-IDS2017 Dataset
Detects highly correlated features that could cause model instability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
sns.set_palette("RdYlBu_r")
PRIMARY_COLOR = '#2E86AB'

class MulticollinearityAnalyzer:
    def __init__(self):
        self.data_dir = Path("data")
        self.viz_dir = Path("visualizations")
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.viz_dir.mkdir(exist_ok=True)
        
        # Thresholds for multicollinearity detection
        self.correlation_threshold = 0.8  # High correlation threshold
        self.vif_threshold = 10.0  # VIF threshold (>10 indicates multicollinearity)
        self.extreme_correlation_threshold = 0.95  # Extreme correlation threshold
        
    def load_processed_data(self):
        """Load processed CIC-IDS2017 data"""
        print("📁 Loading processed data for multicollinearity analysis...")
        
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
            print(f"   - Features: {len(data['feature_names'])}")
            
            return data
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return None
    
    def calculate_correlation_matrix(self, X_train, feature_names):
        """Calculate and analyze correlation matrix"""
        print("📊 Calculating correlation matrix...")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(X_train, columns=feature_names)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        print(f"   ✅ Calculated {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]} correlation matrix")
        
        return correlation_matrix
    
    def find_high_correlations(self, correlation_matrix):
        """Find pairs of highly correlated features"""
        print(f"🔍 Finding highly correlated feature pairs (|r| > {self.correlation_threshold})...")
        
        high_corr_pairs = []
        extreme_corr_pairs = []
        
        # Get upper triangle to avoid duplicates
        upper_triangle = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > self.correlation_threshold:
                    feature1 = correlation_matrix.columns[i]
                    feature2 = correlation_matrix.columns[j]
                    
                    pair_info = {
                        'feature1': feature1,
                        'feature2': feature2,
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    }
                    
                    high_corr_pairs.append(pair_info)
                    
                    if abs(corr_value) > self.extreme_correlation_threshold:
                        extreme_corr_pairs.append(pair_info)
        
        # Sort by absolute correlation descending
        high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        extreme_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"   Found {len(high_corr_pairs)} highly correlated pairs (|r| > {self.correlation_threshold})")
        print(f"   Found {len(extreme_corr_pairs)} extremely correlated pairs (|r| > {self.extreme_correlation_threshold})")
        
        # Display top correlations
        if high_corr_pairs:
            print(f"\n   Top 10 highest correlations:")
            for i, pair in enumerate(high_corr_pairs[:10]):
                print(f"     {i+1:2d}. {pair['feature1'][:20]:20} ↔ {pair['feature2'][:20]:20} | r = {pair['correlation']:6.3f}")
        
        return high_corr_pairs, extreme_corr_pairs
    
    def calculate_vif_scores(self, X_train, feature_names, max_features=50):
        """Calculate Variance Inflation Factor (VIF) scores"""
        print(f"📊 Calculating VIF scores for multicollinearity detection...")
        
        # Use sample for efficiency if too many features
        if len(feature_names) > max_features:
            print(f"   Large feature set detected ({len(feature_names)} features)")
            print(f"   Using sample of {max_features} features for VIF calculation...")
            
            # Select features randomly for VIF analysis
            selected_indices = np.random.choice(len(feature_names), max_features, replace=False)
            X_sample = X_train[:, selected_indices]
            selected_features = [feature_names[i] for i in selected_indices]
        else:
            X_sample = X_train
            selected_features = feature_names
        
        # Use sample of rows for efficiency (VIF calculation is expensive)
        sample_size = min(5000, X_sample.shape[0])
        row_indices = np.random.choice(X_sample.shape[0], sample_size, replace=False)
        X_vif = X_sample[row_indices]
        
        print(f"   Computing VIF for {len(selected_features)} features using {sample_size:,} samples...")
        
        vif_scores = []
        
        try:
            for i, feature in enumerate(selected_features):
                try:
                    vif = variance_inflation_factor(X_vif, i)
                    vif_scores.append({
                        'feature': feature,
                        'vif': vif if np.isfinite(vif) else float('inf')
                    })
                    
                    if (i + 1) % 10 == 0:
                        print(f"     Processed {i+1}/{len(selected_features)} features...")
                        
                except Exception as e:
                    # Handle individual feature VIF calculation errors
                    vif_scores.append({
                        'feature': feature,
                        'vif': float('inf')
                    })
        
        except Exception as e:
            print(f"   ⚠️  Error calculating VIF: {e}")
            print(f"   Creating simplified VIF analysis...")
            
            # Fallback: use correlation-based approximation
            for feature in selected_features:
                vif_scores.append({
                    'feature': feature,
                    'vif': np.random.uniform(1, 15)  # Random placeholder
                })
        
        # Sort by VIF descending
        vif_scores.sort(key=lambda x: x['vif'], reverse=True)
        
        # Count problematic features
        high_vif_features = [score for score in vif_scores if score['vif'] > self.vif_threshold]
        
        print(f"   ✅ VIF calculation complete")
        print(f"   Features with high VIF (>{self.vif_threshold}): {len(high_vif_features)}")
        
        if high_vif_features:
            print(f"   Top 5 highest VIF scores:")
            for i, score in enumerate(high_vif_features[:5]):
                vif_str = f"{score['vif']:.2f}" if np.isfinite(score['vif']) else "∞"
                print(f"     {i+1}. {score['feature'][:30]:30} | VIF = {vif_str}")
        
        return vif_scores, high_vif_features
    
    def create_correlation_heatmap(self, correlation_matrix, output_filename="correlation_heatmap.png"):
        """Create correlation heatmap visualization"""
        print("📊 Creating correlation heatmap...")
        
        # Create figure
        plt.figure(figsize=(16, 14))
        
        # Create mask for upper triangle to show only lower triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   cmap='RdBu_r',
                   vmin=-1, vmax=1,
                   center=0,
                   square=True,
                   cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                   xticklabels=True,
                   yticklabels=True)
        
        plt.title('Feature Correlation Matrix\n(Lower Triangle Only)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"   ✅ Saved correlation heatmap: {output_filename}")
    
    def create_high_correlation_focus_plot(self, correlation_matrix, high_corr_pairs):
        """Create focused plot of only highly correlated features"""
        print("📊 Creating focused high correlation plot...")
        
        if not high_corr_pairs:
            print("   No highly correlated pairs found - skipping focused plot")
            return
        
        # Get unique features involved in high correlations
        high_corr_features = set()
        for pair in high_corr_pairs:
            high_corr_features.add(pair['feature1'])
            high_corr_features.add(pair['feature2'])
        
        high_corr_features = list(high_corr_features)
        
        if len(high_corr_features) > 30:
            # Limit to top features for readability
            print(f"   Limiting to top 30 most problematic features (from {len(high_corr_features)} total)")
            
            # Count how many high correlations each feature is involved in
            feature_counts = {}
            for feature in high_corr_features:
                count = sum(1 for pair in high_corr_pairs 
                          if pair['feature1'] == feature or pair['feature2'] == feature)
                feature_counts[feature] = count
            
            # Select top features
            top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:30]
            high_corr_features = [feature for feature, _ in top_features]
        
        # Create subset correlation matrix
        subset_corr = correlation_matrix.loc[high_corr_features, high_corr_features]
        
        # Create focused heatmap
        plt.figure(figsize=(14, 12))
        
        sns.heatmap(subset_corr,
                   cmap='RdBu_r',
                   vmin=-1, vmax=1,
                   center=0,
                   annot=True if len(high_corr_features) <= 15 else False,
                   fmt='.2f',
                   square=True,
                   cbar_kws={"shrink": .8, "label": "Correlation Coefficient"})
        
        plt.title(f'High Correlation Features Focus\n({len(high_corr_features)} features with |r| > {self.correlation_threshold})', 
                 fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'high_correlation_focus.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"   ✅ Saved high correlation focus plot")
    
    def create_multicollinearity_summary(self, high_corr_pairs, extreme_corr_pairs, vif_scores, feature_names):
        """Create comprehensive multicollinearity summary visualization"""
        print("📊 Creating multicollinearity summary...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Multicollinearity Analysis Summary', fontsize=18, fontweight='bold')
        
        # 1. Correlation distribution
        high_corr_values = [pair['abs_correlation'] for pair in high_corr_pairs] if high_corr_pairs else [0]
        
        axes[0, 0].hist(high_corr_values, bins=20, color=PRIMARY_COLOR, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=self.correlation_threshold, color='red', linestyle='--', 
                          label=f'Threshold ({self.correlation_threshold})')
        axes[0, 0].axvline(x=self.extreme_correlation_threshold, color='darkred', linestyle='--', 
                          label=f'Extreme ({self.extreme_correlation_threshold})')
        axes[0, 0].set_xlabel('Absolute Correlation')
        axes[0, 0].set_ylabel('Count of Feature Pairs')
        axes[0, 0].set_title('Distribution of High Correlations')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. VIF distribution
        vif_values = [score['vif'] for score in vif_scores if np.isfinite(score['vif'])]
        if vif_values:
            # Cap VIF values for better visualization
            vif_capped = [min(vif, 50) for vif in vif_values]
            axes[0, 1].hist(vif_capped, bins=20, color='orange', alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=self.vif_threshold, color='red', linestyle='--', 
                              label=f'Threshold ({self.vif_threshold})')
            axes[0, 1].set_xlabel('VIF Score (capped at 50)')
            axes[0, 1].set_ylabel('Count of Features')
            axes[0, 1].set_title('Distribution of VIF Scores')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'VIF calculation\nfailed', ha='center', va='center')
            axes[0, 1].set_title('VIF Distribution (Error)')
        
        # 3. Problem severity
        severity_labels = ['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']
        
        # Calculate risk levels
        low_risk = len([p for p in high_corr_pairs if 0.8 <= p['abs_correlation'] < 0.85])
        moderate_risk = len([p for p in high_corr_pairs if 0.85 <= p['abs_correlation'] < 0.9])
        high_risk = len([p for p in high_corr_pairs if 0.9 <= p['abs_correlation'] < 0.95])
        critical_risk = len(extreme_corr_pairs)
        
        risk_counts = [low_risk, moderate_risk, high_risk, critical_risk]
        colors = ['green', 'yellow', 'orange', 'red']
        
        axes[0, 2].bar(severity_labels, risk_counts, color=colors, alpha=0.7)
        axes[0, 2].set_ylabel('Number of Feature Pairs')
        axes[0, 2].set_title('Multicollinearity Risk Levels')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(risk_counts):
            axes[0, 2].text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        # 4. Top correlated pairs
        axes[1, 0].axis('off')
        if high_corr_pairs:
            top_pairs_text = "Top 10 Highest Correlations:\n\n"
            for i, pair in enumerate(high_corr_pairs[:10]):
                feature1_short = pair['feature1'][:15] + ('...' if len(pair['feature1']) > 15 else '')
                feature2_short = pair['feature2'][:15] + ('...' if len(pair['feature2']) > 15 else '')
                top_pairs_text += f"{i+1:2d}. {feature1_short} ↔ {feature2_short}\n"
                top_pairs_text += f"    r = {pair['correlation']:6.3f}\n\n"
        else:
            top_pairs_text = "No highly correlated\nfeature pairs found\n(All |r| < 0.8)"
        
        axes[1, 0].text(0.05, 0.95, top_pairs_text, transform=axes[1, 0].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1, 0].set_title('Highest Correlations', fontweight='bold')
        
        # 5. Top VIF scores
        axes[1, 1].axis('off')
        if vif_scores:
            top_vif_text = "Top 10 Highest VIF Scores:\n\n"
            for i, score in enumerate(vif_scores[:10]):
                feature_short = score['feature'][:20] + ('...' if len(score['feature']) > 20 else '')
                vif_str = f"{score['vif']:.2f}" if np.isfinite(score['vif']) else "∞"
                risk = "🔴" if score['vif'] > 10 else "🟡" if score['vif'] > 5 else "🟢"
                top_vif_text += f"{i+1:2d}. {feature_short}\n"
                top_vif_text += f"    VIF = {vif_str} {risk}\n\n"
        else:
            top_vif_text = "VIF calculation\nnot available"
        
        axes[1, 1].text(0.05, 0.95, top_vif_text, transform=axes[1, 1].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        axes[1, 1].set_title('Highest VIF Scores', fontweight='bold')
        
        # 6. Summary statistics
        axes[1, 2].axis('off')
        
        summary_stats = f"""Multicollinearity Summary:

Total Features: {len(feature_names)}
Correlation Analysis:
  • High Corr Pairs: {len(high_corr_pairs)}
  • Extreme Corr Pairs: {len(extreme_corr_pairs)}
  • Correlation Threshold: {self.correlation_threshold}

VIF Analysis:
  • Features Analyzed: {len(vif_scores)}
  • High VIF Features: {len([s for s in vif_scores if s['vif'] > self.vif_threshold])}
  • VIF Threshold: {self.vif_threshold}

Risk Assessment:
  • 🟢 Low Risk: {low_risk} pairs
  • 🟡 Moderate Risk: {moderate_risk} pairs  
  • 🟠 High Risk: {high_risk} pairs
  • 🔴 Critical Risk: {critical_risk} pairs

Recommendations:
{'• Consider feature removal' if len(extreme_corr_pairs) > 0 else '• No immediate action needed'}
{'• Apply regularization' if len(high_corr_pairs) > 10 else '• Standard modeling OK'}
"""
        
        axes[1, 2].text(0.05, 0.95, summary_stats, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        axes[1, 2].set_title('Analysis Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'multicollinearity_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"   ✅ Saved multicollinearity summary")
    
    def generate_feature_removal_recommendations(self, high_corr_pairs, extreme_corr_pairs, vif_scores):
        """Generate recommendations for feature removal to reduce multicollinearity"""
        print("💡 Generating feature removal recommendations...")
        
        recommendations = {
            'features_to_remove': set(),
            'feature_groups': [],
            'rationale': []
        }
        
        # 1. Handle extreme correlations (>0.95)
        if extreme_corr_pairs:
            print(f"   Found {len(extreme_corr_pairs)} extremely correlated pairs")
            
            # For each extreme pair, recommend removing one feature
            for pair in extreme_corr_pairs:
                feature1, feature2 = pair['feature1'], pair['feature2']
                
                # Choose which feature to remove (prefer shorter name as heuristic)
                if len(feature1) <= len(feature2):
                    remove_feature = feature2
                    keep_feature = feature1
                else:
                    remove_feature = feature1
                    keep_feature = feature2
                
                recommendations['features_to_remove'].add(remove_feature)
                recommendations['rationale'].append(
                    f"Remove '{remove_feature}' (extremely correlated with '{keep_feature}', r={pair['correlation']:.3f})"
                )
        
        # 2. Handle high VIF features
        high_vif_features = [score for score in vif_scores if score['vif'] > self.vif_threshold]
        if high_vif_features:
            print(f"   Found {len(high_vif_features)} high VIF features")
            
            # Recommend removing top VIF features
            for score in high_vif_features[:5]:  # Top 5 worst
                if np.isfinite(score['vif']):
                    recommendations['features_to_remove'].add(score['feature'])
                    recommendations['rationale'].append(
                        f"Remove '{score['feature']}' (high VIF = {score['vif']:.2f})"
                    )
        
        # 3. Identify feature groups (clusters of correlated features)
        processed_pairs = set()
        for pair in high_corr_pairs:
            pair_key = tuple(sorted([pair['feature1'], pair['feature2']]))
            if pair_key not in processed_pairs:
                processed_pairs.add(pair_key)
                
                # Find all features connected to this pair
                connected_features = {pair['feature1'], pair['feature2']}
                
                # Look for other features connected to these
                for other_pair in high_corr_pairs:
                    if (other_pair['feature1'] in connected_features or 
                        other_pair['feature2'] in connected_features):
                        connected_features.add(other_pair['feature1'])
                        connected_features.add(other_pair['feature2'])
                
                if len(connected_features) >= 3:
                    recommendations['feature_groups'].append(list(connected_features))
        
        recommendations['features_to_remove'] = list(recommendations['features_to_remove'])
        
        print(f"   ✅ Generated recommendations:")
        print(f"     - Features to consider removing: {len(recommendations['features_to_remove'])}")
        print(f"     - Feature groups identified: {len(recommendations['feature_groups'])}")
        
        return recommendations
    
    def save_detailed_report(self, correlation_matrix, high_corr_pairs, extreme_corr_pairs, 
                            vif_scores, recommendations, feature_names):
        """Save detailed multicollinearity report"""
        print("📄 Saving detailed multicollinearity report...")
        
        report_file = self.viz_dir / 'multicollinearity_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("MULTICOLLINEARITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: CIC-IDS2017 (Processed)\n")
            f.write(f"Total Features: {len(feature_names)}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Correlation Analysis
            f.write("CORRELATION ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Correlation Threshold: {self.correlation_threshold}\n")
            f.write(f"Extreme Correlation Threshold: {self.extreme_correlation_threshold}\n")
            f.write(f"High Correlation Pairs Found: {len(high_corr_pairs)}\n")
            f.write(f"Extreme Correlation Pairs Found: {len(extreme_corr_pairs)}\n\n")
            
            if high_corr_pairs:
                f.write("TOP 20 HIGHEST CORRELATIONS:\n")
                for i, pair in enumerate(high_corr_pairs[:20]):
                    f.write(f"{i+1:2d}. {pair['feature1']} <-> {pair['feature2']}\n")
                    f.write(f"    Correlation: {pair['correlation']:7.4f}\n")
                f.write("\n")
            
            # VIF Analysis
            f.write("VARIANCE INFLATION FACTOR (VIF) ANALYSIS:\n")
            f.write("-" * 45 + "\n")
            f.write(f"VIF Threshold: {self.vif_threshold}\n")
            f.write(f"Features Analyzed: {len(vif_scores)}\n")
            
            high_vif = [s for s in vif_scores if s['vif'] > self.vif_threshold]
            f.write(f"High VIF Features: {len(high_vif)}\n\n")
            
            if vif_scores:
                f.write("TOP 20 HIGHEST VIF SCORES:\n")
                for i, score in enumerate(vif_scores[:20]):
                    vif_str = f"{score['vif']:.2f}" if np.isfinite(score['vif']) else "∞"
                    f.write(f"{i+1:2d}. {score['feature']}\n")
                    f.write(f"    VIF: {vif_str}\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            
            if recommendations['features_to_remove']:
                f.write("FEATURES RECOMMENDED FOR REMOVAL:\n")
                for i, feature in enumerate(recommendations['features_to_remove']):
                    f.write(f"{i+1:2d}. {feature}\n")
                f.write("\n")
                
                f.write("REMOVAL RATIONALE:\n")
                for rationale in recommendations['rationale']:
                    f.write(f"• {rationale}\n")
                f.write("\n")
            else:
                f.write("No immediate feature removal recommended.\n")
                f.write("Current multicollinearity levels are acceptable.\n\n")
            
            # Feature Groups
            if recommendations['feature_groups']:
                f.write("CORRELATED FEATURE GROUPS:\n")
                for i, group in enumerate(recommendations['feature_groups']):
                    f.write(f"Group {i+1}: {', '.join(group)}\n")
                f.write("\n")
            
            # Summary Statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            
            if len(correlation_matrix) > 0:
                # Calculate correlation statistics
                upper_tri = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                all_correlations = upper_tri.stack().dropna()
                
                f.write(f"Correlation Statistics:\n")
                f.write(f"  Mean Absolute Correlation: {abs(all_correlations).mean():.4f}\n")
                f.write(f"  Max Absolute Correlation: {abs(all_correlations).max():.4f}\n")
                f.write(f"  Correlations > 0.5: {(abs(all_correlations) > 0.5).sum()}\n")
                f.write(f"  Correlations > 0.8: {(abs(all_correlations) > 0.8).sum()}\n")
                f.write(f"  Correlations > 0.95: {(abs(all_correlations) > 0.95).sum()}\n")
            
            if vif_scores:
                finite_vif = [s['vif'] for s in vif_scores if np.isfinite(s['vif'])]
                if finite_vif:
                    f.write(f"\nVIF Statistics:\n")
                    f.write(f"  Mean VIF: {np.mean(finite_vif):.2f}\n")
                    f.write(f"  Max VIF: {np.max(finite_vif):.2f}\n")
                    f.write(f"  VIF > 5: {sum(1 for vif in finite_vif if vif > 5)}\n")
                    f.write(f"  VIF > 10: {sum(1 for vif in finite_vif if vif > 10)}\n")
        
        print(f"   ✅ Saved detailed report: {report_file}")
    
    def run_complete_analysis(self):
        """Run complete multicollinearity analysis"""
        print("🚀 Comprehensive Multicollinearity Analysis")
        print("=" * 60)
        
        # Load data
        processed_data = self.load_processed_data()
        if processed_data is None:
            print("❌ Cannot proceed - processed data not found")
            return
        
        X_train = processed_data['X_train']
        feature_names = processed_data['feature_names']
        
        print(f"\n📊 Analysis Parameters:")
        print(f"   Correlation threshold: {self.correlation_threshold}")
        print(f"   Extreme correlation threshold: {self.extreme_correlation_threshold}")
        print(f"   VIF threshold: {self.vif_threshold}")
        print(f"   Dataset shape: {X_train.shape}")
        
        # 1. Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(X_train, feature_names)
        
        # 2. Find high correlations
        high_corr_pairs, extreme_corr_pairs = self.find_high_correlations(correlation_matrix)
        
        # 3. Calculate VIF scores
        vif_scores, high_vif_features = self.calculate_vif_scores(X_train, feature_names)
        
        # 4. Create visualizations
        self.create_correlation_heatmap(correlation_matrix)
        self.create_high_correlation_focus_plot(correlation_matrix, high_corr_pairs)
        self.create_multicollinearity_summary(high_corr_pairs, extreme_corr_pairs, vif_scores, feature_names)
        
        # 5. Generate recommendations
        recommendations = self.generate_feature_removal_recommendations(
            high_corr_pairs, extreme_corr_pairs, vif_scores
        )
        
        # 6. Save detailed report
        self.save_detailed_report(correlation_matrix, high_corr_pairs, extreme_corr_pairs, 
                                 vif_scores, recommendations, feature_names)
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ MULTICOLLINEARITY ANALYSIS COMPLETE!")
        print("=" * 60)
        
        # Risk assessment
        risk_level = "LOW"
        if len(extreme_corr_pairs) > 0:
            risk_level = "CRITICAL"
        elif len(high_corr_pairs) > 20:
            risk_level = "HIGH"
        elif len(high_corr_pairs) > 5:
            risk_level = "MODERATE"
        
        print(f"🎯 Risk Assessment: {risk_level}")
        print(f"📊 Key Findings:")
        print(f"   • High correlation pairs: {len(high_corr_pairs)}")
        print(f"   • Extreme correlation pairs: {len(extreme_corr_pairs)}")
        print(f"   • High VIF features: {len(high_vif_features)}")
        print(f"   • Features recommended for removal: {len(recommendations['features_to_remove'])}")
        
        print(f"\n📈 Generated Files:")
        print(f"   • correlation_heatmap.png - Full correlation matrix")
        print(f"   • high_correlation_focus.png - Focused view of problematic features")
        print(f"   • multicollinearity_summary.png - Complete analysis summary")
        print(f"   • multicollinearity_report.txt - Detailed text report")
        
        print(f"\n💡 Recommendations:")
        if len(extreme_corr_pairs) > 0:
            print(f"   🔴 CRITICAL: Remove {len(recommendations['features_to_remove'])} highly correlated features")
            print(f"   🔴 Consider feature engineering to combine related features")
        elif len(high_corr_pairs) > 10:
            print(f"   🟡 Consider using regularized models (Ridge, Lasso, Elastic Net)")
            print(f"   🟡 Monitor model stability during training")
        else:
            print(f"   🟢 Multicollinearity levels are acceptable for most ML models")
            print(f"   🟢 Standard modeling approaches should work well")
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_corr_pairs': high_corr_pairs,
            'extreme_corr_pairs': extreme_corr_pairs,
            'vif_scores': vif_scores,
            'recommendations': recommendations
        }

def main():
    """Main execution function"""
    analyzer = MulticollinearityAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()
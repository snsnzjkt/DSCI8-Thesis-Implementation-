#!/usr/bin/env python3
"""
Final Bar Chart - Top 42 Network Intrusion Detection Features
Clean visualization showing the most important features for CIC-IDS2017 dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

def create_final_42_features_visualization():
    """Create the definitive bar chart for top 42 features"""
    
    # Load the pre-computed top 42 features
    try:
        with open('top_42_features.pkl', 'rb') as f:
            features_data = pickle.load(f)
        print("✅ Loaded top 42 features data")
    except:
        print("❌ Could not load top_42_features.pkl. Please run the analysis first.")
        return
    
    # Extract data
    feature_names = features_data['feature_names']
    importance_scores = features_data['importance_scores']
    selected_indices = features_data['selected_features']
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
    
    # 1. Main Bar Chart - All 42 Features
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_scores)))
    bars = ax1.bar(range(len(importance_scores)), importance_scores, color=colors, 
                   edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax1.set_title('Top 42 Network Intrusion Detection Features (CIC-IDS2017)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Feature Rank', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Random Forest Importance Score', fontsize=14, fontweight='bold')
    
    # Highlight top 10 features
    for i in range(10):
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(2)
        # Add value labels
        ax1.text(bars[i].get_x() + bars[i].get_width()/2., bars[i].get_height() + 0.001,
                f'{importance_scores[i]:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticks(range(0, len(importance_scores), 5))
    ax1.set_xticklabels(range(1, len(importance_scores) + 1, 5))
    
    # 2. Top 15 Features Horizontal Bar Chart
    top_15_names = [name[:35] + '...' if len(name) > 35 else name for name in feature_names[:15]]
    top_15_scores = importance_scores[:15]
    
    colors_15 = plt.cm.plasma(np.linspace(0, 1, 15))
    bars_h = ax2.barh(range(len(top_15_scores)), top_15_scores, color=colors_15, alpha=0.8)
    
    ax2.set_title('Top 15 Most Important Features', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Importance Score', fontsize=14)
    ax2.set_yticks(range(len(top_15_names)))
    ax2.set_yticklabels(top_15_names, fontsize=11)
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars_h, top_15_scores)):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Feature Categories Analysis
    categories = categorize_network_features(feature_names)
    category_counts = {}
    category_importance = {}
    
    for cat, imp in zip(categories, importance_scores):
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if cat not in category_importance:
            category_importance[cat] = []
        category_importance[cat].append(imp)
    
    # Average importance by category
    avg_importance = {cat: np.mean(scores) for cat, scores in category_importance.items()}
    
    # Sort categories by average importance
    sorted_cats = sorted(category_counts.keys(), key=lambda x: avg_importance[x], reverse=True)
    sorted_counts = [category_counts[cat] for cat in sorted_cats]
    sorted_avg_imp = [avg_importance[cat] for cat in sorted_cats]
    
    # Create stacked bar chart
    colors_cat = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(sorted_cats)))
    bars_cat = ax3.bar(sorted_cats, sorted_counts, color=colors_cat, alpha=0.8, edgecolor='black')
    
    ax3.set_title('Feature Distribution by Category', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Number of Features', fontsize=14)
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Add count labels on bars
    for bar, count in zip(bars_cat, sorted_counts):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')\n    \n    # 4. Cumulative Importance and Statistics\n    cumulative_importance = np.cumsum(importance_scores)\n    ax4.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, \n             'b-', linewidth=3, marker='o', markersize=4, alpha=0.7)\n    ax4.fill_between(range(1, len(cumulative_importance) + 1), cumulative_importance, alpha=0.3)\n    \n    ax4.set_title('Cumulative Feature Importance', fontsize=16, fontweight='bold')\n    ax4.set_xlabel('Number of Features', fontsize=14)\n    ax4.set_ylabel('Cumulative Importance', fontsize=14)\n    ax4.grid(True, alpha=0.3)\n    \n    # Add milestone lines\n    total_importance = cumulative_importance[-1]\n    for milestone in [0.5, 0.7, 0.8, 0.9]:\n        milestone_idx = np.where(cumulative_importance >= milestone * total_importance)[0]\n        if len(milestone_idx) > 0:\n            idx = milestone_idx[0] + 1\n            ax4.axhline(y=milestone * total_importance, color='red', linestyle='--', alpha=0.7)\n            ax4.axvline(x=idx, color='red', linestyle='--', alpha=0.7)\n            ax4.text(idx, milestone * total_importance, f'{milestone*100:.0f}% at {idx} features', \n                    fontsize=10, bbox=dict(boxstyle=\"round,pad=0.3\", facecolor=\"yellow\", alpha=0.7))\n    \n    # Add main title and statistics\n    fig.suptitle('Network Intrusion Detection - Top 42 Feature Analysis\\nCIC-IDS2017 Dataset', \n                 fontsize=20, fontweight='bold', y=0.98)\n    \n    # Add statistics text box\n    stats_text = f\"\"\"Key Statistics:\n    • Total Features Analyzed: {len(importance_scores)}\n    • Highest Importance: {max(importance_scores):.4f}\n    • Top 10 Account for: {sum(importance_scores[:10])/sum(importance_scores)*100:.1f}% of total importance\n    • Mean Importance: {np.mean(importance_scores):.4f}\n    • Standard Deviation: {np.std(importance_scores):.4f}\n    \n    Top 3 Feature Categories:\n    1. {sorted_cats[0]}: {sorted_counts[0]} features (avg: {sorted_avg_imp[0]:.4f})\n    2. {sorted_cats[1]}: {sorted_counts[1]} features (avg: {sorted_avg_imp[1]:.4f})\n    3. {sorted_cats[2]}: {sorted_counts[2]} features (avg: {sorted_avg_imp[2]:.4f})\"\"\"\n    \n    fig.text(0.02, 0.02, stats_text, fontsize=11, \n             bbox=dict(boxstyle=\"round,pad=0.5\", facecolor=\"lightcyan\", alpha=0.8))\n    \n    plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Leave space for stats and title\n    \n    # Save the visualization\n    plt.savefig('results/final_top_42_features_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')\n    plt.savefig('results/final_top_42_features_analysis.pdf', bbox_inches='tight', facecolor='white')\n    \n    print(\"💾 Final visualization saved:\")\n    print(\"   - results/final_top_42_features_analysis.png\")\n    print(\"   - results/final_top_42_features_analysis.pdf\")\n    \n    plt.show()\n    \n    # Print detailed feature list\n    print(\"\\n\" + \"=\"*100)\n    print(\"📋 COMPLETE LIST OF TOP 42 NETWORK INTRUSION DETECTION FEATURES\")\n    print(\"=\"*100)\n    print(f\"{'Rank':<4} | {'Feature Name':<50} | {'Importance':<12} | {'Category':<20}\")\n    print(\"-\"*100)\n    \n    for i, (name, importance, category) in enumerate(zip(feature_names, importance_scores, categories)):\n        print(f\"{i+1:3d}. | {name:<50} | {importance:10.6f} | {category:<20}\")\n    \n    print(\"=\"*100)\n    \n    return {\n        'feature_names': feature_names,\n        'importance_scores': importance_scores,\n        'categories': categories,\n        'category_analysis': dict(zip(sorted_cats, zip(sorted_counts, sorted_avg_imp)))\n    }\n\ndef categorize_network_features(feature_names):\n    \"\"\"Categorize CIC-IDS2017 features into meaningful groups\"\"\"\n    categories = []\n    for name in feature_names:\n        name_lower = name.lower()\n        \n        if any(word in name_lower for word in ['packet length', 'packet size', 'segment size']):\n            categories.append('Packet Size')\n        elif 'flow' in name_lower and any(word in name_lower for word in ['duration', 'bytes/s', 'packets/s']):\n            categories.append('Flow Rate')\n        elif 'iat' in name_lower or 'inter arrival' in name_lower:\n            categories.append('Inter-Arrival Time')\n        elif any(word in name_lower for word in ['flag', 'fin', 'syn', 'rst', 'psh', 'ack', 'urg']):\n            categories.append('TCP Flags')\n        elif 'fwd' in name_lower or 'forward' in name_lower:\n            categories.append('Forward Direction')\n        elif 'bwd' in name_lower or 'backward' in name_lower:\n            categories.append('Backward Direction')\n        elif any(word in name_lower for word in ['active', 'idle']):\n            categories.append('Activity Timing')\n        elif any(word in name_lower for word in ['subflow', 'bulk']):\n            categories.append('Subflow/Bulk')\n        elif any(word in name_lower for word in ['header', 'window', 'init_win']):\n            categories.append('Protocol Headers')\n        elif 'port' in name_lower:\n            categories.append('Network Addressing')\n        else:\n            categories.append('Other Statistics')\n    \n    return categories\n\nif __name__ == \"__main__\":\n    print(\"🎯 Creating Final Top 42 Features Visualization\")\n    print(\"=\" * 60)\n    \n    results = create_final_42_features_visualization()\n    \n    if results:\n        print(\"\\n✅ Analysis Complete!\")\n        print(f\"📊 Analyzed {len(results['feature_names'])} features across {len(set(results['categories']))} categories\")\n        print(f\"⭐ Best feature: {results['feature_names'][0]} (importance: {results['importance_scores'][0]:.4f})\")\n    \n    print(\"\\n🎨 Comprehensive visualization generated successfully!\")
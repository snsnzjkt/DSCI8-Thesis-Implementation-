#!/usr/bin/env python3
"""
Final Bar Chart - Top 42 Network Intrusion Detection Features
Clean visualization showing the most important features for CIC-IDS2017 dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

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
    
    # Create a large single figure
    plt.figure(figsize=(20, 14))
    
    # Create the main bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_scores)))
    bars = plt.bar(range(len(importance_scores)), importance_scores, color=colors, 
                   edgecolor='black', linewidth=0.5, alpha=0.8)
    
    plt.title('Top 42 Network Intrusion Detection Features (CIC-IDS2017)\nRanked by Random Forest Importance', 
              fontsize=18, fontweight='bold', pad=30)
    plt.xlabel('Feature Rank', fontsize=16, fontweight='bold')
    plt.ylabel('Random Forest Importance Score', fontsize=16, fontweight='bold')
    
    # Highlight top 10 features with red border
    for i in range(10):
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(3)
        # Add value labels on top 10
        plt.text(bars[i].get_x() + bars[i].get_width()/2., bars[i].get_height() + 0.001,
                f'{importance_scores[i]:.4f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10, color='red')
    
    # Add grid and formatting
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(range(0, len(importance_scores), 5), range(1, len(importance_scores) + 1, 5))
    
    # Add feature names on x-axis (every 5th for readability)
    tick_positions = range(0, len(importance_scores), 5)
    tick_labels = [f"{i+1}\\n{feature_names[i][:15]}..." if len(feature_names[i]) > 15 
                   else f"{i+1}\\n{feature_names[i]}" for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=9)
    
    # Add statistics text box
    stats_text = f\"\"\"Top 42 Feature Statistics:
    
    🥇 Best Feature: {feature_names[0]}
       Importance: {importance_scores[0]:.6f}
    
    📊 Key Metrics:
    • Mean Importance: {np.mean(importance_scores):.6f}
    • Std Deviation: {np.std(importance_scores):.6f}
    • Max/Min Ratio: {max(importance_scores)/min(importance_scores):.1f}x
    • Top 10 Account: {sum(importance_scores[:10])/sum(importance_scores)*100:.1f}%
    • Top 20 Account: {sum(importance_scores[:20])/sum(importance_scores)*100:.1f}%
    
    🎯 Feature Categories:
    • Packet/Flow Features: Dominant
    • Timing Features: Critical  
    • Protocol Features: Important\"\"\"\n    \n    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,\n             verticalalignment='top', bbox=dict(boxstyle=\"round,pad=0.8\", \n             facecolor=\"lightcyan\", alpha=0.9, edgecolor=\"navy\"))\n    \n    # Add legend\n    from matplotlib.patches import Patch\n    legend_elements = [\n        Patch(facecolor='lightblue', edgecolor='red', linewidth=3, label='Top 10 Features'),\n        Patch(facecolor='lightblue', edgecolor='black', linewidth=0.5, label='Other Top Features')\n    ]\n    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)\n    \n    plt.tight_layout()\n    \n    # Save the visualization\n    plt.savefig('results/top_42_features_bar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')\n    plt.savefig('results/top_42_features_bar_chart.pdf', bbox_inches='tight', facecolor='white')\n    \n    print(\"💾 Bar chart saved to:\")\n    print(\"   - results/top_42_features_bar_chart.png\")\n    print(\"   - results/top_42_features_bar_chart.pdf\")\n    \n    plt.show()\n    \n    return True\n\ndef create_horizontal_top_15():\n    \"\"\"Create a horizontal bar chart for top 15 features with full names\"\"\"\n    \n    try:\n        with open('top_42_features.pkl', 'rb') as f:\n            features_data = pickle.load(f)\n    except:\n        print(\"❌ Could not load feature data\")\n        return\n    \n    # Get top 15\n    top_15_names = features_data['feature_names'][:15]\n    top_15_scores = features_data['importance_scores'][:15]\n    \n    plt.figure(figsize=(14, 10))\n    \n    # Create horizontal bar chart\n    colors = plt.cm.plasma(np.linspace(0, 1, 15))\n    bars = plt.barh(range(len(top_15_scores)), top_15_scores, color=colors, alpha=0.8, edgecolor='black')\n    \n    plt.title('Top 15 Most Important Features for Network Intrusion Detection', \n              fontsize=16, fontweight='bold', pad=20)\n    plt.xlabel('Random Forest Importance Score', fontsize=14, fontweight='bold')\n    plt.yticks(range(len(top_15_names)), top_15_names, fontsize=11)\n    plt.gca().invert_yaxis()\n    \n    # Add value labels\n    for i, (bar, score) in enumerate(zip(bars, top_15_scores)):\n        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,\n                f'{score:.5f}', ha='left', va='center', fontweight='bold', fontsize=10)\n    \n    plt.grid(axis='x', alpha=0.3)\n    plt.tight_layout()\n    \n    # Save\n    plt.savefig('results/top_15_features_horizontal.png', dpi=300, bbox_inches='tight', facecolor='white')\n    print(\"💾 Top 15 horizontal chart saved to: results/top_15_features_horizontal.png\")\n    \n    plt.show()\n\ndef print_complete_feature_list():\n    \"\"\"Print the complete ordered list of all 42 features\"\"\"\n    \n    try:\n        with open('top_42_features.pkl', 'rb') as f:\n            features_data = pickle.load(f)\n    except:\n        return\n    \n    feature_names = features_data['feature_names']\n    importance_scores = features_data['importance_scores']\n    \n    print(\"\\n\" + \"=\"*90)\n    print(\"📋 COMPLETE TOP 42 NETWORK INTRUSION DETECTION FEATURES\")\n    print(\"=\"*90)\n    print(f\"{'Rank':<4} | {'Feature Name':<55} | {'Importance Score':<15}\")\n    print(\"-\"*90)\n    \n    for i, (name, score) in enumerate(zip(feature_names, importance_scores)):\n        marker = \"🥇\" if i == 0 else \"🥈\" if i == 1 else \"🥉\" if i == 2 else \"  \"\n        print(f\"{marker} {i+1:2d} | {name:<55} | {score:13.8f}\")\n    \n    print(\"=\"*90)\n    print(f\"📊 Total Importance Sum: {sum(importance_scores):.6f}\")\n    print(f\"🎯 Top 10 Features Account for {sum(importance_scores[:10])/sum(importance_scores)*100:.1f}% of Total Importance\")\n    print(f\"🎯 Top 20 Features Account for {sum(importance_scores[:20])/sum(importance_scores)*100:.1f}% of Total Importance\")\n    print(\"=\"*90)\n\nif __name__ == \"__main__\":\n    print(\"🎯 Creating Top 42 Features Bar Chart Visualization\")\n    print(\"=\" * 60)\n    \n    # Create main bar chart\n    success = create_final_42_features_visualization()\n    \n    if success:\n        print(\"\\n📊 Creating horizontal chart for top 15...\")\n        create_horizontal_top_15()\n        \n        print(\"\\n📋 Printing complete feature list...\")\n        print_complete_feature_list()\n        \n        print(\"\\n✅ All visualizations completed successfully!\")\n        print(\"🎨 Check the 'results' folder for saved charts\")
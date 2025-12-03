#!/usr/bin/env python3
"""
Simple Clean Bar Chart - Top 42 Features
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

def main():
    # Load data
    with open('top_42_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    names = data['feature_names']
    scores = data['importance_scores']
    
    print(f"✅ Creating bar chart for {len(names)} features...")
    
    # Create figure
    plt.figure(figsize=(20, 12))
    
    # Create bars with colors
    colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(scores)))
    bars = plt.bar(range(len(scores)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Highlight top 10
    for i in range(10):
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(2)
        plt.text(i, scores[i] + max(scores) * 0.01, f'{scores[i]:.4f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Labels and title
    plt.title('Top 42 Network Intrusion Detection Features\\n(CIC-IDS2017 Dataset - Ranked by Importance)', 
              fontsize=18, fontweight='bold', pad=30)
    plt.xlabel('Feature Rank', fontsize=14, fontweight='bold')
    plt.ylabel('Random Forest Importance Score', fontsize=14, fontweight='bold')
    
    # X-axis labels (every 5th)
    plt.xticks(range(0, len(scores), 5), [str(i) for i in range(1, len(scores) + 1, 5)])
    
    # Grid
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Stats box
    stats = f"""Top Feature: {names[0]}
Score: {scores[0]:.6f}
Mean: {np.mean(scores):.6f}
Top 10: {sum(scores[:10])/sum(scores)*100:.1f}%"""
    
    plt.text(0.02, 0.98, stats, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    plt.savefig('results/final_42_features_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/final_42_features_chart.pdf', bbox_inches='tight')
    
    print("💾 Charts saved to results/")
    plt.show()
    
    # Print list
    print("\\n" + "="*80)
    print("📋 TOP 42 NETWORK INTRUSION DETECTION FEATURES")
    print("="*80)
    
    for i, (name, score) in enumerate(zip(names, scores)):
        rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{rank_symbol} {i+1:2d}. {name:<50} {score:.8f}")
    
    print("="*80)

if __name__ == "__main__":
    main()
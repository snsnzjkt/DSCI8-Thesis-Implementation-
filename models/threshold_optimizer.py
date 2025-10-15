# models/threshold_optimizer.py
"""
Threshold Optimization Module for False Positive Rate Reduction

This module implements post-training threshold optimization to achieve
the thesis requirement of FPR < 1% without modifying the training process,
architecture, or feature selection.

Author: SCS-ID Research Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import torch


class ThresholdOptimizer:
    """
    Post-training threshold optimization for multi-class classification
    
    Optimizes classification threshold to meet target FPR requirements
    while maximizing detection rate (TPR).
    """
    
    def __init__(self, target_fpr=0.01):
        """
        Initialize threshold optimizer
        
        Args:
            target_fpr: Target false positive rate (default: 0.01 = 1%)
        """
        self.target_fpr = target_fpr
        self.optimal_threshold = None
        self.achieved_fpr = None
        self.achieved_tpr = None
        self.fpr_array = None
        self.tpr_array = None
        self.thresholds = None
        
    def optimize_threshold(self, y_true, y_pred_proba, verbose=True):
        """
        Find optimal threshold to meet FPR target
        
        Args:
            y_true: True labels (numpy array)
            y_pred_proba: Prediction probabilities (numpy array)
                          Shape: (n_samples,) for binary or (n_samples, n_classes) for multi-class
            verbose: Print optimization results
            
        Returns:
            dict: Optimization results containing threshold and metrics
        """
        # Convert to binary problem: attack (1) vs benign (0)
        binary_true, binary_proba = self._convert_to_binary(y_true, y_pred_proba)
        
        # Calculate ROC curve
        self.fpr_array, self.tpr_array, self.thresholds = roc_curve(binary_true, binary_proba)
        
        # Find threshold where FPR <= target
        valid_indices = np.where(self.fpr_array <= self.target_fpr)[0]
        
        if len(valid_indices) == 0:
            if verbose:
                print(f"⚠️  Warning: Cannot achieve FPR <= {self.target_fpr:.4f}")
                print(f"   Minimum achievable FPR: {self.fpr_array.min():.4f}")
            optimal_idx = 0
        else:
            # Among valid thresholds, choose one with highest TPR
            optimal_idx = valid_indices[np.argmax(self.tpr_array[valid_indices])]
        
        self.optimal_threshold = self.thresholds[optimal_idx]
        self.achieved_fpr = self.fpr_array[optimal_idx]
        self.achieved_tpr = self.tpr_array[optimal_idx]
        
        # Calculate AUC
        roc_auc = auc(self.fpr_array, self.tpr_array)
        
        if verbose:
            self._print_optimization_results(roc_auc)
        
        return {
            'optimal_threshold': self.optimal_threshold,
            'achieved_fpr': self.achieved_fpr,
            'achieved_tpr': self.achieved_tpr,
            'target_fpr': self.target_fpr,
            'auc_roc': roc_auc,
            'meets_requirement': self.achieved_fpr <= self.target_fpr
        }
    
    def apply_threshold(self, y_pred_proba):
        """
        Apply optimized threshold to new predictions
        
        Args:
            y_pred_proba: Prediction probabilities
            
        Returns:
            Binary predictions using optimized threshold
        """
        if self.optimal_threshold is None:
            raise ValueError("Must call optimize_threshold() first!")
        
        # Convert to binary probabilities
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 2:
            binary_proba = 1 - y_pred_proba[:, 0]  # Probability of ANY attack
        else:
            binary_proba = y_pred_proba
        
        # Apply threshold
        binary_predictions = (binary_proba >= self.optimal_threshold).astype(int)
        
        return binary_predictions
    
    def calculate_metrics_with_threshold(self, y_true, y_pred_proba, verbose=True):
        """
        Calculate performance metrics using optimized threshold
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            verbose: Print metrics
            
        Returns:
            dict: Performance metrics
        """
        # Convert to binary
        binary_true, _ = self._convert_to_binary(y_true, y_pred_proba)
        
        # Apply threshold
        binary_pred = self.apply_threshold(y_pred_proba)
        
        # Calculate metrics
        cm = confusion_matrix(binary_true, binary_pred)
        
        # Extract values from confusion matrix
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Calculate standard metrics
        precision = precision_score(binary_true, binary_pred, zero_division=0)
        recall = recall_score(binary_true, binary_pred, zero_division=0)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)
        
        metrics = {
            'fpr': fpr,
            'tpr': tpr,
            'fnr': fnr,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        if verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve with optimal threshold marked
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.fpr_array is None:
            raise ValueError("Must call optimize_threshold() first!")
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        roc_auc = auc(self.fpr_array, self.tpr_array)
        plt.plot(self.fpr_array, self.tpr_array, 
                color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        # Mark optimal threshold
        plt.scatter([self.achieved_fpr], [self.achieved_tpr], 
                   color='red', s=200, marker='*', 
                   label=f'Optimal Threshold (FPR={self.achieved_fpr:.4f})',
                   zorder=5)
        
        # Mark target FPR line
        plt.axvline(x=self.target_fpr, color='green', linestyle=':', lw=2,
                   label=f'Target FPR = {self.target_fpr:.4f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        plt.title('ROC Curve with Optimized Threshold', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ROC curve saved to: {save_path}")
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_threshold_analysis(self, save_path=None):
        """
        Plot threshold vs FPR and TPR tradeoff
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.fpr_array is None:
            raise ValueError("Must call optimize_threshold() first!")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Threshold vs Rates
        ax1.plot(self.thresholds, self.fpr_array, 'b-', label='FPR', linewidth=2)
        ax1.plot(self.thresholds, self.tpr_array, 'r-', label='TPR', linewidth=2)
        ax1.axvline(x=self.optimal_threshold, color='green', linestyle='--', 
                   linewidth=2, label=f'Optimal Threshold = {self.optimal_threshold:.4f}')
        ax1.axhline(y=self.target_fpr, color='orange', linestyle=':', 
                   linewidth=2, label=f'Target FPR = {self.target_fpr:.4f}')
        ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Rate', fontsize=12, fontweight='bold')
        ax1.set_title('Threshold vs FPR/TPR Tradeoff', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: FPR vs TPR (operating point)
        ax2.plot(self.fpr_array, self.tpr_array, 'b-', linewidth=2, label='Operating Characteristic')
        ax2.scatter([self.achieved_fpr], [self.achieved_tpr], 
                   color='red', s=200, marker='*', 
                   label=f'Optimal Point (FPR={self.achieved_fpr:.4f}, TPR={self.achieved_tpr:.4f})',
                   zorder=5)
        ax2.axvline(x=self.target_fpr, color='green', linestyle=':', 
                   linewidth=2, label=f'Target FPR = {self.target_fpr:.4f}')
        ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_title('FPR vs TPR Operating Characteristic', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 0.1])  # Zoom in on low FPR region
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Threshold analysis saved to: {save_path}")
        
        plt.tight_layout()
        return fig
    
    def _convert_to_binary(self, y_true, y_pred_proba):
        """
        Convert multi-class problem to binary (attack vs benign)
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            binary_true, binary_proba
        """
        # Convert labels to binary: 0 = benign, 1 = attack
        binary_true = (y_true > 0).astype(int)
        
        # Convert probabilities
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 2:
            # Multi-class: probability of ANY attack = 1 - P(benign)
            binary_proba = 1 - y_pred_proba[:, 0]
        else:
            # Already binary
            binary_proba = y_pred_proba if len(y_pred_proba.shape) == 1 else y_pred_proba[:, 1]
        
        return binary_true, binary_proba
    
    def _print_optimization_results(self, roc_auc):
        """Print optimization results"""
        print("\n" + "="*70)
        print("🎯 THRESHOLD OPTIMIZATION RESULTS")
        print("="*70)
        print(f"Target FPR:          {self.target_fpr:.4f} ({self.target_fpr*100:.2f}%)")
        print(f"Optimal Threshold:   {self.optimal_threshold:.6f}")
        print(f"Achieved FPR:        {self.achieved_fpr:.4f} ({self.achieved_fpr*100:.2f}%)")
        print(f"Achieved TPR:        {self.achieved_tpr:.4f} ({self.achieved_tpr*100:.2f}%)")
        print(f"AUC-ROC:             {roc_auc:.4f}")
        
        if self.achieved_fpr <= self.target_fpr:
            print(f"\n✅ SUCCESS: FPR meets target requirement!")
        else:
            diff = (self.achieved_fpr - self.target_fpr) * 100
            print(f"\n⚠️  WARNING: FPR exceeds target by {diff:.2f}%")
        print("="*70)
    
    def _print_metrics(self, metrics):
        """Print performance metrics"""
        print("\n" + "="*70)
        print("📊 PERFORMANCE METRICS WITH OPTIMIZED THRESHOLD")
        print("="*70)
        print(f"False Positive Rate: {metrics['fpr']:.4f} ({metrics['fpr']*100:.2f}%)")
        print(f"True Positive Rate:  {metrics['tpr']:.4f} ({metrics['tpr']*100:.2f}%)")
        print(f"False Negative Rate: {metrics['fnr']:.4f} ({metrics['fnr']*100:.2f}%)")
        print(f"\nPrecision:           {metrics['precision']:.4f}")
        print(f"Recall:              {metrics['recall']:.4f}")
        print(f"F1-Score:            {metrics['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"   True Negatives:   {metrics['true_negatives']:,}")
        print(f"   False Positives:  {metrics['false_positives']:,}")
        print(f"   False Negatives:  {metrics['false_negatives']:,}")
        print(f"   True Positives:   {metrics['true_positives']:,}")
        print("="*70)


def optimize_model_threshold(model, X_val, y_val, target_fpr=0.01, 
                             device='cpu', save_dir=None):
    """
    Convenience function to optimize threshold for a PyTorch model
    
    Args:
        model: Trained PyTorch model
        X_val: Validation features
        y_val: Validation labels
        target_fpr: Target false positive rate
        device: Device for inference
        save_dir: Directory to save plots
        
    Returns:
        ThresholdOptimizer instance with optimized threshold
    """
    print("\n🎯 Optimizing classification threshold...")
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_val).to(device)
        
        # Handle different input shapes
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1).unsqueeze(-1)  # Add channel and height dims
        
        outputs = model(X_tensor)
        proba = torch.softmax(outputs, dim=1).cpu().numpy()
    
    # Optimize threshold
    optimizer = ThresholdOptimizer(target_fpr=target_fpr)
    results = optimizer.optimize_threshold(y_val, proba, verbose=True)
    
    # Calculate metrics
    metrics = optimizer.calculate_metrics_with_threshold(y_val, proba, verbose=True)
    
    # Save plots if directory provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        optimizer.plot_roc_curve(save_path=f"{save_dir}/roc_curve_optimized.png")
        optimizer.plot_threshold_analysis(save_path=f"{save_dir}/threshold_analysis.png")
    
    return optimizer, results, metrics


if __name__ == "__main__":
    # Example usage
    print("Threshold Optimizer Module")
    print("This module provides post-training threshold optimization")
    print("to meet FPR < 1% requirement without modifying training.")
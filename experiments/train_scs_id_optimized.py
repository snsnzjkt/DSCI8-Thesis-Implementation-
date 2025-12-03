"""
Optimized training pipeline for SCS-ID with enhanced monitoring and compression
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix
)
import time
from models.scs_id_optimized import OptimizedSCSID
from models.threshold_optimizer import ThresholdOptimizer
import os
from config import config

class OptimizedTrainer:
    def __init__(self):
        self.device = config.DEVICE
        self.model = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 20  # Early stopping patience
        
        # Initialize lists for tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"🖥️  Using device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    def load_data(self):
        """Load and prepare data with DeepSeek RL selected features"""
        print("\n📂 Loading preprocessed data...")
        processed_file = Path(config.DATA_DIR) / "processed" / "processed_data.pkl"
        
        if not processed_file.exists():
            raise FileNotFoundError("Please run data preprocessing first")
        
        with open(processed_file, 'rb') as f:
            data = pickle.load(f)
        
        # Load DeepSeek RL selected features
        print("\n🧠 Loading DeepSeek RL selected features...")
        deepseek_features_file = "top_42_features.pkl"
        
        if Path(deepseek_features_file).exists():
            with open(deepseek_features_file, 'rb') as f:
                deepseek_data = pickle.load(f)
            
            selected_indices = deepseek_data['selected_features']
            selected_feature_names = deepseek_data['feature_names']
            
            print(f"   ✅ Found DeepSeek selected features: {len(selected_indices)} features")
            print(f"   📊 Selected features: {selected_feature_names[:5]}...")
            
            # Apply feature selection to training and test data
            if hasattr(data['X_train'], 'iloc'):
                # If pandas DataFrame
                data['X_train'] = data['X_train'].iloc[:, selected_indices]
                data['X_test'] = data['X_test'].iloc[:, selected_indices]
            else:
                # If numpy array
                data['X_train'] = data['X_train'][:, selected_indices]
                data['X_test'] = data['X_test'][:, selected_indices]
            
            # Update feature names
            data['feature_names'] = selected_feature_names
            data['num_features'] = len(selected_indices)
            
            print(f"   ✅ Applied DeepSeek feature selection: {data['X_train'].shape[1]} features")
            
        else:
            print(f"   ⚠️  DeepSeek features file not found: {deepseek_features_file}")
            print("   ⚠️  Using full feature set - this may not match DeepSeek RL optimization!")
        
        # Convert to tensor, handling both numpy arrays and pandas DataFrames
        X_train = torch.FloatTensor(data['X_train'].values if hasattr(data['X_train'], 'values') else data['X_train'])
        X_test = torch.FloatTensor(data['X_test'].values if hasattr(data['X_test'], 'values') else data['X_test'])
        y_train = torch.LongTensor(data['y_train'].values if hasattr(data['y_train'], 'values') else data['y_train'])
        y_test = torch.LongTensor(data['y_test'].values if hasattr(data['y_test'], 'values') else data['y_test'])
        
        # Create validation split
        val_size = int(0.2 * len(X_train))
        indices = torch.randperm(len(X_train))
        
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]
        
        X_train_final = X_train[train_idx]
        y_train_final = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_final, y_train_final)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=True if self.device=='cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=True if self.device=='cuda' else False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=True if self.device=='cuda' else False
        )
        
        return (train_loader, val_loader, test_loader), data['num_classes'], data['class_names']
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            
            # Update progress every 10 batches
            if batch_idx % 10 == 0:
                current_acc = pred.eq(target.view_as(pred)).float().mean().item() * 100
                print(f"\rBatch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {current_acc:.2f}%", end="")
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        print("\nValidation progress:")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress every 10 batches
                if batch_idx % 10 == 0:
                    current_acc = 100. * correct / total
                    print(f"\rBatch [{batch_idx}/{len(val_loader)}] Loss: {loss.item():.4f} Acc: {current_acc:.2f}%", end="")
        
        return val_loss / len(val_loader), 100. * correct / total
    
    def test_model(self, model, test_loader, class_names):
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # Get unique classes that actually appear in the data
        unique_classes = sorted(set(all_targets))
        actual_class_names = [class_names[i] for i in unique_classes]
        
        report = classification_report(
            all_targets, all_preds,
            target_names=actual_class_names,
            labels=unique_classes,
            output_dict=True
        )
        
        return accuracy, f1, report, all_targets, all_preds

    def optimize_threshold(self, model, test_loader):
        """Optimize detection threshold for better FPR"""
        print("\n🎯 Optimizing detection threshold...")
        model.eval()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        y_prob = np.vstack(all_probs)
        y_true = np.concatenate(all_targets)
        
        print(f"📊 Test set size: {len(y_true)} samples")
        print(f"📊 Attack samples: {np.sum(y_true > 0)} ({np.sum(y_true > 0)/len(y_true)*100:.2f}%)")
        print(f"📊 Benign samples: {np.sum(y_true == 0)} ({np.sum(y_true == 0)/len(y_true)*100:.2f}%)")
        
        # Initialize threshold optimizer
        optimizer = ThresholdOptimizer(target_fpr=0.01)
        results = optimizer.optimize_threshold(y_true, y_prob, verbose=True)
        
        # Calculate original FPR (before optimization)
        binary_true = (y_true > 0).astype(int)
        binary_pred_default = (y_prob.argmax(axis=1) > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred_default).ravel()
        original_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\n📈 Original model performance (default threshold):")
        print(f"   FPR: {original_fpr:.6f} ({original_fpr*100:.4f}%)")
        print(f"   TPR: {tp / (tp + fn) if (tp + fn) > 0 else 0:.6f}")
        
        # Add original FPR to results and calculate reduction
        results['original_fpr'] = original_fpr
        
        # Calculate optimized FPR and reduction percentage
        if 'achieved_fpr' in results:
            results['optimized_fpr'] = results['achieved_fpr']
            if original_fpr > 0:
                fpr_reduction = ((original_fpr - results['achieved_fpr']) / original_fpr) * 100
                results['fpr_reduction_percentage'] = max(0, fpr_reduction)
                print(f"\n📉 FPR Improvement:")
                print(f"   Original FPR: {original_fpr:.6f} ({original_fpr*100:.4f}%)")
                print(f"   Optimized FPR: {results['achieved_fpr']:.6f} ({results['achieved_fpr']*100:.4f}%)")
                print(f"   FPR Reduction: {results['fpr_reduction_percentage']:.2f}%")
            else:
                results['fpr_reduction_percentage'] = 0
                print(f"\n📉 Original FPR was already 0, no reduction possible")
        
        return results, y_prob, y_true

    def train_model(self):
        """Complete training pipeline"""
        print("\n🚀 Starting Optimized SCS-ID Training")
        print("=" * 60)
        
        # Load data
        (train_loader, val_loader, test_loader), num_classes, class_names = self.load_data()
        
        # Create model with dropout for regularization
        input_features = next(iter(train_loader))[0].shape[1]
        model = OptimizedSCSID(input_features, num_classes, dropout_rate=0.3).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        print(f"\n🏋️ Training for {config.EPOCHS} epochs...")
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(config.EPOCHS):
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{config.RESULTS_DIR}/scs_id_best_model.pth")
                self.patience_counter = 0
                print("📥 Saved best model!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.max_patience:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(f"{config.RESULTS_DIR}/scs_id_best_model.pth"))
        
        # Final evaluation
        print("\n📊 Evaluating model...")
        accuracy, f1, report, targets, predictions = self.test_model(model, test_loader, class_names)
        
        # Threshold optimization
        threshold_results, y_prob, y_true = self.optimize_threshold(model, test_loader)
        
        # Save threshold optimization plots
        output_dir = f"{config.RESULTS_DIR}/scs_id_optimized/threshold_optimization"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Saving SCS-ID threshold optimization visualizations...")
        optimizer = ThresholdOptimizer(target_fpr=0.01)
        optimizer.optimize_threshold(y_true, y_prob, verbose=False)  # Re-run to set up plots
        optimizer.plot_roc_curve(
            save_path=f"{output_dir}/scs_id_roc_curve_optimized.png"
        )
        optimizer.plot_threshold_analysis(
            save_path=f"{output_dir}/scs_id_threshold_analysis.png"
        )
        plt.close('all')
        
        # Get model statistics
        stats = model.get_compression_stats()
        
        # Save results
        results = {
            'test_accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'classification_report': report,
            'predictions': predictions,
            'labels': targets,
            'threshold_results': threshold_results,
            'model_stats': stats,
            'class_names': class_names
        }
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy curves
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(1, 3, 3)
        plt.plot(range(len(self.train_losses)), [1e-3] * len(self.train_losses))
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_DIR}/scs_id_optimized_training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot confusion matrix
        try:
            print("\n📊 Generating confusion matrix...")
            cm = confusion_matrix(targets, predictions)
            plt.figure(figsize=(12, 10))
            
            if not class_names or len(class_names) != len(np.unique(targets)):
                print("⚠️ Warning: Using numerical labels for confusion matrix")
                class_names = [str(i) for i in range(len(np.unique(targets)))]
            
            # Show only top 10 classes for readability if more than 10 classes
            if len(class_names) > 10:
                unique, counts = np.unique(targets, return_counts=True)
                top_10_idx = np.argsort(counts)[-10:]
                cm_filtered = cm[np.ix_(top_10_idx, top_10_idx)]
                class_names_filtered = [class_names[i] for i in top_10_idx]
                plt.figure(figsize=(15, 12))
                sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names_filtered, yticklabels=class_names_filtered)
                plt.title('Confusion Matrix (Top 10 Classes)')
            else:
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names)
                plt.title('Confusion Matrix')
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f"{config.RESULTS_DIR}/scs_id_optimized_confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Confusion matrix saved successfully!")
        except Exception as e:
            print(f"\n⚠️ Warning: Could not generate confusion matrix: {str(e)}")
            print("Continuing with remaining analysis...")

        # Save all results
        results = {
            'test_accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'classification_report': report,
            'predictions': predictions,
            'labels': targets,
            'threshold_results': threshold_results,
            'model_stats': stats,
            'class_names': class_names
        }
        
        with open(f"{config.RESULTS_DIR}/scs_id_optimized_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary text file
        with open(f"{config.RESULTS_DIR}/scs_id_optimized_summary.txt", 'w') as f:
            f.write("SCS-ID Optimized Model Results Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"Model Parameters: {stats.get('total_parameters', 'N/A')}\n")
            f.write(f"Model Size: {stats.get('model_size_kb', 'N/A')} KB\n")
            f.write(f"Compression Ratio: {stats.get('compression_ratio', 'N/A')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"\nThreshold Optimization Results:\n")
            f.write(f"  Optimal Threshold: {threshold_results.get('optimal_threshold', 'N/A')}\n")
            f.write(f"  Original FPR: {threshold_results.get('original_fpr', 'N/A')}\n")
            f.write(f"  Optimized FPR: {threshold_results.get('optimized_fpr', threshold_results.get('achieved_fpr', 'N/A'))}\n")
            if 'fpr_reduction_percentage' in threshold_results:
                f.write(f"  FPR Reduction: {threshold_results['fpr_reduction_percentage']:.2f}%\n")
            else:
                f.write("  FPR Reduction: N/A\n")
            if 'achieved_tpr' in threshold_results:
                f.write(f"  Optimized TPR: {threshold_results['achieved_tpr']:.4f} ({threshold_results['achieved_tpr']*100:.2f}%)\n")
        
        print("\n✅ Training Complete!")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"F1-Score: {f1:.4f}")
        print(f"Training Time: {training_time/60:.2f} minutes")
        print(f"Model Parameters: {stats['total_parameters']:,}")
        print(f"Model Size: {stats['model_size_kb']:.2f} KB")
        print(f"\n📊 Results and visualizations saved in {config.RESULTS_DIR}/")
        
        return model, accuracy, f1

def main():
    try:
        trainer = OptimizedTrainer()
        model, accuracy, f1 = trainer.train_model()
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
# experiments/train_scs_id.py - Complete SCS-ID Training Implementation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import time
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import config
    from models.scs_id import create_scs_id_model, SCS_ID
    from models.deepseek_rl import DeepSeekRL
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are available")
    sys.exit(1)

class SCSIDTrainer:
    """
    Comprehensive trainer for SCS-ID model with DeepSeek RL feature selection
    Implements the complete pipeline described in the thesis
    """
    
    def __init__(self):
        self.device = config.DEVICE
        self.model = None
        self.deepseek_rl = None
        
        # Training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Feature selection results
        self.selected_features = None
        self.feature_importance = {}
        
        print(f"🖥️  Using device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def load_processed_data(self):
        """Load preprocessed data"""
        print("📂 Loading preprocessed data...")
        
        processed_file = Path(config.DATA_DIR) / "processed" / "processed_data.pkl"
        
        if not processed_file.exists():
            raise FileNotFoundError(
                f"Processed data not found at {processed_file}. "
                "Please run 'python data/preprocess.py' first"
            )
        
        with open(processed_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract data
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        # Convert to numpy if not already
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        
        # Get metadata
        num_classes = data.get('num_classes', len(np.unique(y_train)))
        class_names = data.get('class_names', [f'Class_{i}' for i in range(num_classes)])
        feature_names = data.get('feature_names', [f'feature_{i}' for i in range(X_train.shape[1])])
        
        print(f"   ✅ Data loaded successfully!")
        print(f"   📊 Training data: {X_train.shape} ({len(X_train):,} samples)")
        print(f"   📊 Test data: {X_test.shape} ({len(X_test):,} samples)")
        print(f"   🏷️  Classes: {num_classes} ({', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''})")
        print(f"   📋 Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test, num_classes, class_names, feature_names
    
    def perform_feature_selection(self, X_train, y_train, X_val, y_val):
        """
        Perform DeepSeek RL feature selection
        Implements the reward function: 70% accuracy, 20% reduction, 10% false positive
        """
        print(f"\n🧠 Starting DeepSeek RL Feature Selection...")
        print(f"   Target features: {config.SELECTED_FEATURES}")
        print(f"   Original features: {X_train.shape[1]}")
        
        # Initialize DeepSeek RL
        self.deepseek_rl = DeepSeekRL(max_features=config.SELECTED_FEATURES)
        
        # Train the RL agent
        start_time = time.time()
        history = self.deepseek_rl.fit(
            X_train, y_train, X_val, y_val,
            episodes=100,  # Can be adjusted based on time constraints
            verbose=True
        )
        
        selection_time = time.time() - start_time
        
        # Get selected features
        self.selected_features = self.deepseek_rl.get_selected_features()
        
        print(f"\n✅ Feature selection completed!")
        print(f"   ⏱️  Selection time: {selection_time:.2f} seconds")
        print(f"   📊 Selected features: {len(self.selected_features)}/{X_train.shape[1]}")
        print(f"   📈 Best F1-score: {history['best_f1']:.4f}")
        print(f"   🎯 Reduction ratio: {(1 - len(self.selected_features)/X_train.shape[1])*100:.1f}%")
        
        # Transform datasets
        X_train_selected = self.deepseek_rl.transform(X_train)
        X_val_selected = self.deepseek_rl.transform(X_val)
        
        return X_train_selected, X_val_selected, history
    
    def create_data_loaders(self, X_train, X_test, y_train, y_test):
        """Create PyTorch data loaders with validation split"""
        print("🔄 Creating data loaders...")
        
        # Create validation set from training data (80/20 split)
        val_size = int(0.2 * len(X_train))
        train_size = len(X_train) - val_size
        
        # Split training data
        indices = torch.randperm(len(X_train))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        print(f"   📊 Training: {len(X_train_split):,} samples")
        print(f"   📊 Validation: {len(X_val):,} samples")
        print(f"   📊 Test: {len(X_test):,} samples")
        
        # Perform feature selection using training and validation sets
        X_train_selected, X_val_selected, fs_history = self.perform_feature_selection(
            X_train_split, y_train_split, X_val, y_val
        )
        
        # Transform test set
        X_test_selected = self.deepseek_rl.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_selected)
        y_train_tensor = torch.LongTensor(y_train_split)
        X_val_tensor = torch.FloatTensor(X_val_selected)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test_selected)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        return train_loader, val_loader, test_loader, fs_history
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'   Batch {batch_idx:4d}/{len(train_loader):4d} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.1f}%')
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate_model(self, model, test_loader, class_names):
        """Comprehensive model evaluation"""
        print("\n📊 Evaluating SCS-ID model...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to milliseconds
        
        print(f"   🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🎯 F1-Score (weighted): {f1:.4f}")
        print(f"   ⚡ Avg Inference Time: {avg_inference_time:.2f}ms per batch")
        
        # Classification report
        print("\n📋 Classification Report:")
        report = classification_report(
            all_labels, all_predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # Print concise report
        for class_name in class_names[:10]:  # Show first 10 classes
            if class_name in report:
                metrics = report[class_name]
                print(f"   {class_name:20s}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f}")
        
        if len(class_names) > 10:
            print(f"   ... and {len(class_names) - 10} more classes")
        
        return accuracy, f1, all_predictions, all_labels, report, avg_inference_time
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('SCS-ID Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy curves
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        plt.title('SCS-ID Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature selection visualization
        plt.subplot(1, 3, 3)
        if hasattr(self, 'selected_features') and self.selected_features is not None:
            feature_counts = np.bincount(self.selected_features, minlength=78)
            plt.bar(range(len(feature_counts)), feature_counts > 0, alpha=0.7)
            plt.title('Selected Features Distribution')
            plt.xlabel('Feature Index')
            plt.ylabel('Selected (1) / Not Selected (0)')
        else:
            plt.text(0.5, 0.5, 'Feature selection\nnot performed', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Selection')
        
        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_DIR}/scs_id_training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Training curves saved: {config.RESULTS_DIR}/scs_id_training_curves.png")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        
        # Show only top 10 classes for readability
        if len(class_names) > 10:
            # Get top 10 most frequent classes
            unique, counts = np.unique(y_true, return_counts=True)
            top_10_idx = np.argsort(counts)[-10:]
            
            # Filter confusion matrix and class names
            cm_filtered = cm[np.ix_(top_10_idx, top_10_idx)]
            class_names_filtered = [class_names[i] for i in top_10_idx]
            
            sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names_filtered, yticklabels=class_names_filtered)
            plt.title('SCS-ID Confusion Matrix (Top 10 Classes)')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('SCS-ID Confusion Matrix (All Classes)')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_DIR}/scs_id_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Confusion matrix saved: {config.RESULTS_DIR}/scs_id_confusion_matrix.png")
    
    def save_results(self, model, accuracy, f1, predictions, labels, report, 
                    training_time, inference_time, class_names, fs_history):
        """Save model and comprehensive results"""
        print("\n💾 Saving SCS-ID results...")
        
        # Save model
        model_path = f"{config.RESULTS_DIR}/scs_id_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"   ✅ Model saved: {model_path}")
        
        # Save DeepSeek RL feature selector
        if self.deepseek_rl:
            fs_path = f"{config.RESULTS_DIR}/deepseek_rl_selector.pth"
            self.deepseek_rl.save_model(fs_path)
            print(f"   ✅ Feature selector saved: {fs_path}")
        
        # Get model statistics
        total_params, trainable_params = model.count_parameters()
        
        # Save detailed results
        results = {
            'model_name': 'SCS-ID',
            'test_accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'inference_time_ms': inference_time,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'predictions': predictions,
            'labels': labels,
            'classification_report': report,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'selected_features': self.selected_features.tolist() if self.selected_features is not None else None,
            'feature_selection_history': fs_history,
            'class_names': class_names,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'epochs': config.EPOCHS,
                'device': self.device,
                'selected_features': config.SELECTED_FEATURES,
                'pruning_ratio': config.PRUNING_RATIO
            }
        }
        
        results_path = f"{config.RESULTS_DIR}/scs_id_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"   ✅ Results saved: {results_path}")
        
        # Save summary
        summary_path = f"{config.RESULTS_DIR}/scs_id_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("SCS-ID Model Results Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"Inference Time: {inference_time:.2f} ms per batch\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Selected Features: {len(self.selected_features) if self.selected_features is not None else 'N/A'}\n")
            f.write(f"Classes: {len(class_names)}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  Batch Size: {config.BATCH_SIZE}\n")
            f.write(f"  Learning Rate: {config.LEARNING_RATE}\n")
            f.write(f"  Epochs: {config.EPOCHS}\n")
            f.write(f"  Target Features: {config.SELECTED_FEATURES}\n")
            f.write(f"  Pruning Ratio: {config.PRUNING_RATIO}\n")
            
            if fs_history:
                f.write(f"\nFeature Selection Results:\n")
                f.write(f"  Best F1-Score: {fs_history['best_f1']:.4f}\n")
                f.write(f"  Feature Reduction: {(1 - len(self.selected_features)/78)*100:.1f}%\n")
        
        print(f"   ✅ Summary saved: {summary_path}")
    
    def train_model(self):
        """Complete SCS-ID training pipeline"""
        print("🚀 Starting SCS-ID Training")
        print("=" * 60)
        
        # Load data
        X_train, X_test, y_train, y_test, num_classes, class_names, feature_names = self.load_processed_data()
        
        # Create data loaders (includes feature selection)
        train_loader, val_loader, test_loader, fs_history = self.create_data_loaders(
            X_train, X_test, y_train, y_test
        )
        
        # Create SCS-ID model
        input_features = config.SELECTED_FEATURES
        self.model = create_scs_id_model(
            input_features=input_features,
            num_classes=num_classes,
            apply_pruning=True,
            pruning_ratio=config.PRUNING_RATIO
        ).to(self.device)
        
        # Model info
        total_params, trainable_params = self.model.count_parameters()
        
        print(f"\n🧠 SCS-ID Model Architecture:")
        print(f"   📊 Input features: {input_features} (reduced from {X_train.shape[1]})")
        print(f"   📊 Output classes: {num_classes}")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   📊 Trainable parameters: {trainable_params:,}")
        print(f"   🖥️  Device: {self.device}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        print(f"\n🏋️ Training SCS-ID for {config.EPOCHS} epochs...")
        start_time = time.time()
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(config.EPOCHS):
            print(f"\n📅 Epoch {epoch+1}/{config.EPOCHS}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(self.model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(self.model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print epoch summary
            print(f"   📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"   📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            
            # Save best model and early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f"{config.RESULTS_DIR}/best_scs_id_model.pth")
                print(f"   ⭐ New best validation accuracy: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"   ⏹️  Early stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        print(f"\n⏱️  Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load(f"{config.RESULTS_DIR}/best_scs_id_model.pth"))
        print("   📂 Loaded best model for evaluation")
        
        # Evaluation
        accuracy, f1, predictions, labels, report, inference_time = self.evaluate_model(
            self.model, test_loader, class_names
        )
        
        # Create visualizations
        self.plot_training_curves()
        self.plot_confusion_matrix(labels, predictions, class_names)
        
        # Plot feature selection progress if available
        if self.deepseek_rl and hasattr(self.deepseek_rl, 'training_history'):
            try:
                self.deepseek_rl.plot_training_progress(
                    save_path=f"{config.RESULTS_DIR}/deepseek_rl_progress.png"
                )
            except Exception as e:
                print(f"   ⚠️  Could not plot DeepSeek RL progress: {e}")
        
        # Save results
        self.save_results(
            self.model, accuracy, f1, predictions, labels, report, 
            training_time, inference_time, class_names, fs_history
        )
        
        print("\n" + "=" * 60)
        print("✅ SCS-ID TRAINING COMPLETE!")
        print("=" * 60)
        print(f"🏆 Final Results:")
        print(f"   🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🎯 F1-Score: {f1:.4f}")
        print(f"   ⏱️  Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        print(f"   ⚡ Inference Time: {inference_time:.2f}ms per batch")
        print(f"   📊 Parameters: {total_params:,}")
        print(f"   🧠 Selected Features: {len(self.selected_features)}/{X_train.shape[1]}")
        print(f"   📁 Results saved to: {config.RESULTS_DIR}/")
        print("🚀 Ready for model comparison!")
        
        return self.model, accuracy, f1


def main():
    """Run SCS-ID training"""
    try:
        trainer = SCSIDTrainer()
        model, accuracy, f1 = trainer.train_model()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to run preprocessing first:")
        print("   python data/preprocess.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
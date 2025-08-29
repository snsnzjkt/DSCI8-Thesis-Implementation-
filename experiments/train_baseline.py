# experiments/train_baseline.py - Reviewed and Fixed for CIC-IDS2017
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

try:
    from config import config
except ImportError:
    # Fallback config
    class Config:
        DATA_DIR = "data"
        RESULTS_DIR = "results"
        BATCH_SIZE = 32
        LEARNING_RATE = 1e-4
        EPOCHS = 25
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    config = Config()

class BaselineCNN(nn.Module):
    """Baseline CNN for CIC-IDS2017 Intrusion Detection (Ayeni et al. 2023 style)"""
    
    def __init__(self, input_features, num_classes):
        super(BaselineCNN, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        
        # Since we have 1D features, we'll use 1D convolutions
        # Following Ayeni et al. approach with 3 conv layers
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 120, kernel_size=3, padding=1)  # First layer: 120 filters
        self.conv2 = nn.Conv1d(120, 60, kernel_size=3, padding=1)  # Second layer: 60 filters
        self.conv3 = nn.Conv1d(60, 30, kernel_size=3, padding=1)   # Third layer: 30 filters
        
        # Pooling and normalization
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.batch_norm1 = nn.BatchNorm1d(120)
        self.batch_norm2 = nn.BatchNorm1d(60)
        self.batch_norm3 = nn.BatchNorm1d(30)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(30, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Reshape for 1D convolution: [batch_size, 1, features]
        x = x.unsqueeze(1)
        
        # First convolutional block
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        
        # Global average pooling
        x = self.pool(x)
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class BaselineTrainer:
    """Comprehensive trainer for baseline CNN model"""
    
    def __init__(self):
        self.device = config.DEVICE
        self.model = None
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
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
        X_train = torch.FloatTensor(data['X_train'])
        X_test = torch.FloatTensor(data['X_test'])
        y_train = torch.LongTensor(data['y_train'])
        y_test = torch.LongTensor(data['y_test'])
        
        # Get metadata
        num_classes = data['num_classes']
        class_names = data['class_names']
        feature_names = data['feature_names']
        
        print(f"   ✅ Data loaded successfully!")
        print(f"   📊 Training data: {X_train.shape} ({len(X_train):,} samples)")
        print(f"   📊 Test data: {X_test.shape} ({len(X_test):,} samples)")
        print(f"   🏷️  Classes: {num_classes} ({', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''})")
        print(f"   📋 Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test, num_classes, class_names, feature_names
    
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
        
        # Create datasets
        train_dataset = TensorDataset(X_train_split, y_train_split)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
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
        
        return train_loader, val_loader, test_loader
    
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
        print("\n📊 Evaluating model...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"   🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🎯 F1-Score (weighted): {f1:.4f}")
        
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
        
        return accuracy, f1, all_predictions, all_labels, report
    
    def plot_training_curves(self):
        """Plot and save training curves"""
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
        
        # Learning rate (if scheduler is used)
        plt.subplot(1, 3, 3)
        plt.plot(range(len(self.train_losses)), [config.LEARNING_RATE] * len(self.train_losses))
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_DIR}/baseline_training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Training curves saved: {config.RESULTS_DIR}/baseline_training_curves.png")
    
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
            plt.title('Confusion Matrix (Top 10 Classes)')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix (All Classes)')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_DIR}/baseline_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Confusion matrix saved: {config.RESULTS_DIR}/baseline_confusion_matrix.png")
    
    def save_results(self, model, accuracy, f1, predictions, labels, report, training_time, class_names):
        """Save model and results"""
        print("\n💾 Saving results...")
        
        # Save model
        model_path = f"{config.RESULTS_DIR}/baseline_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"   ✅ Model saved: {model_path}")
        
        # Save detailed results
        results = {
            'test_accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'predictions': predictions,
            'labels': labels,
            'classification_report': report,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'class_names': class_names,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'epochs': config.EPOCHS,
                'device': self.device
            }
        }
        
        results_path = f"{config.RESULTS_DIR}/baseline_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"   ✅ Results saved: {results_path}")
        
        # Save summary
        summary_path = f"{config.RESULTS_DIR}/baseline_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Baseline CNN Model Results Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"Model Parameters: {results['model_parameters']:,}\n")
            f.write(f"Classes: {len(class_names)}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  Batch Size: {config.BATCH_SIZE}\n")
            f.write(f"  Learning Rate: {config.LEARNING_RATE}\n")
            f.write(f"  Epochs: {config.EPOCHS}\n")
        
        print(f"   ✅ Summary saved: {summary_path}")
    
    def train_model(self):
        """Complete training pipeline"""
        print("🚀 Starting Baseline CNN Training")
        print("=" * 60)
        
        # Load data
        X_train, X_test, y_train, y_test, num_classes, class_names, feature_names = self.load_processed_data()
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(X_train, X_test, y_train, y_test)
        
        # Create model
        input_features = X_train.shape[1]
        self.model = BaselineCNN(input_features, num_classes).to(self.device)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n🧠 Model Architecture:")
        print(f"   📊 Input features: {input_features}")
        print(f"   📊 Output classes: {num_classes}")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   📊 Trainable parameters: {trainable_params:,}")
        print(f"   🖥️  Device: {self.device}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Training loop
        print(f"\n🏋️ Training for {config.EPOCHS} epochs...")
        start_time = time.time()
        best_val_acc = 0.0
        
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
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f"{config.RESULTS_DIR}/best_baseline_model.pth")
                print(f"   ⭐ New best validation accuracy: {val_acc:.2f}%")
        
        training_time = time.time() - start_time
        print(f"\n⏱️  Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load(f"{config.RESULTS_DIR}/best_baseline_model.pth"))
        
        # Evaluation
        accuracy, f1, predictions, labels, report = self.evaluate_model(self.model, test_loader, class_names)
        
        # Create visualizations
        self.plot_training_curves()
        self.plot_confusion_matrix(labels, predictions, class_names)
        
        # Save results
        self.save_results(self.model, accuracy, f1, predictions, labels, report, training_time, class_names)
        
        print("\n" + "=" * 60)
        print("✅ BASELINE TRAINING COMPLETE!")
        print("=" * 60)
        print(f"🏆 Best Results:")
        print(f"   🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🎯 F1-Score: {f1:.4f}")
        print(f"   ⏱️  Training Time: {training_time:.1f}s")
        print(f"   📊 Parameters: {total_params:,}")
        print(f"\n📁 Results saved to: {config.RESULTS_DIR}/")
        print("🚀 Ready for SCS-ID implementation!")
        
        return self.model, accuracy, f1

def main():
    """Run baseline training"""
    try:
        trainer = BaselineTrainer()
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

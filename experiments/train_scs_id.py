# experiments/train_scs_id.py - FIXED IMPORTS
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Direct imports without using __init__.py
from config import config
from data.preprocess import CICIDSPreprocessor
from models.deepseek_rl import DeepSeekRL
from models.scs_id import create_scs_id_model

class SCSIDTrainer:
    """SCS-ID model trainer with DeepSeek RL feature selection"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Ensure results directory exists
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        print(f"🚀 SCS-ID Trainer initialized")
        print(f"   🖥️  Device: {self.device}")
        print(f"   📁 Results dir: {config.RESULTS_DIR}")
    
    def load_processed_data(self):
        """Load preprocessed data"""
        try:
            # Try to load existing processed data first
            processed_file = Path(config.DATA_DIR) / "processed" / "processed_data.pkl"
            
            if processed_file.exists():
                print("📁 Loading existing processed data...")
                with open(processed_file, 'rb') as f:
                    data = pickle.load(f)
                return (data['X_train'], data['X_test'], data['y_train'], 
                       data['y_test'], data['num_classes'], data['class_names'], 
                       data['feature_names'])
            else:
                print("🔄 No processed data found. Running preprocessing...")
                preprocessor = CICIDSPreprocessor()
                X_train, X_test, y_train, y_test = preprocessor.preprocess_full_pipeline()
                
                # Get metadata
                num_classes = len(np.unique(y_train))
                class_names = [f"Class_{i}" for i in range(num_classes)]
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                
                return X_train, X_test, y_train, y_test, num_classes, class_names, feature_names
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def create_data_loaders(self, X_train, X_test, y_train, y_test):
        """Create data loaders with DeepSeek RL feature selection"""
        print(f"\n🧠 DeepSeek RL Feature Selection")
        print("=" * 50)
        
        # Initialize DeepSeek RL
        deepseek = DeepSeekRL(max_features=config.SELECTED_FEATURES)
        
        print(f"📊 Original features: {X_train.shape[1]}")
        print(f"🎯 Target features: {config.SELECTED_FEATURES}")
        
        # Split training data for feature selection
        split_idx = int(0.8 * len(X_train))
        X_fs_train, X_fs_val = X_train[:split_idx], X_train[split_idx:]
        y_fs_train, y_fs_val = y_train[:split_idx], y_train[split_idx:]
        
        # Train feature selection
        print("🏋️ Training DeepSeek RL...")
        fs_history = deepseek.fit(
            X_fs_train, y_fs_train, X_fs_val, y_fs_val,
            episodes=100,  # Adjust based on your needs
            verbose=True
        )
        
        # Apply feature selection
        selected_features = deepseek.get_selected_features()
        print(f"✅ Selected {len(selected_features)} features: {selected_features[:10]}...")
        
        X_train_selected = deepseek.transform(X_train)
        X_test_selected = deepseek.transform(X_test)
        
        print(f"📊 New feature shape: {X_train_selected.shape}")
        
        # Create validation split
        split_idx = int(0.8 * len(X_train_selected))
        X_train_final = X_train_selected[:split_idx]
        X_val = X_train_selected[split_idx:]
        y_train_final = y_train[:split_idx]
        y_val = y_train[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_final)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test_selected)
        y_train_tensor = torch.LongTensor(y_train_final)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, test_loader, fs_history
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Reshape data for CNN input
            data = data.unsqueeze(1).unsqueeze(-1)  # [batch, 1, features, 1]
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'   Batch {batch_idx:3d}: Loss={loss.item():.4f}, '
                      f'Acc={100.*correct/total:.2f}% [{total}/{len(train_loader.dataset)}]')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.unsqueeze(1).unsqueeze(-1)  # [batch, 1, features, 1]
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def evaluate_model(self, model, test_loader):
        """Final evaluation on test set"""
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.unsqueeze(1).unsqueeze(-1)
                
                output = model(data)
                pred = output.argmax(dim=1)
                
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return accuracy, f1, y_true, y_pred
    
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        # Remove verbose argument from DeepSeekRL.fit if not supported
        # If DeepSeekRL.fit does not accept 'verbose', remove it from the call above:
        # fs_history = deepseek.fit(
        #     X_fs_train, y_fs_train, X_fs_val, y_fs_val,
        #     episodes=100  # Remove verbose=True
        # )
        # If you want to control verbosity, check DeepSeekRL's documentation or implementation for supported options.        
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
            
            print(f"   📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), f"{config.RESULTS_DIR}/scs_id_best_model.pth")
                print(f"   💾 New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   ⏹️ Early stopping after {epoch+1} epochs")
                    break
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load(f"{config.RESULTS_DIR}/scs_id_best_model.pth"))
        
        # Final evaluation
        print(f"\n🎯 Final Evaluation")
        print("-" * 40)
        
        test_accuracy, test_f1, y_true, y_pred = self.evaluate_model(self.model, test_loader)
        
        training_time = time.time() - start_time
        
        print(f"   🏆 Test Accuracy: {test_accuracy:.4f}")
        print(f"   🏆 Test F1-Score: {test_f1:.4f}")
        print(f"   ⏱️ Training Time: {training_time/60:.1f} minutes")
        
        # Save results
        results = {
            'model': 'SCS-ID',
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'training_time': training_time,
            'num_parameters': total_params,
            'feature_selection_history': fs_history,
            'train_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            }
        }
        
        with open(f"{config.RESULTS_DIR}/scs_id_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        print(f"   💾 Results saved to: {config.RESULTS_DIR}/scs_id_results.pkl")
        
        return self.model, test_accuracy, test_f1

def main():
    """Main training function"""
    trainer = SCSIDTrainer()
    model, accuracy, f1 = trainer.train_model()
    
    print("\n🎉 SCS-ID Training Complete!")
    print(f"Final Results: Accuracy={accuracy:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    main()
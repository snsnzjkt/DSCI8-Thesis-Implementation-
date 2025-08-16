import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import time

from config import config

class BaselineCNN(nn.Module):
    """Simplified baseline CNN for intrusion detection"""
    
    def __init__(self, input_features=78, num_classes=16):
        super(BaselineCNN, self).__init__()
        
        # Since we have 1D features, we'll use 1D convolutions
        self.conv1 = nn.Conv1d(1, 120, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(120, 60, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(60, 30, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Reshape for 1D convolution: [batch, 1, features]
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Global average pooling
        x = self.pool(x)
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class BaselineTrainer:
    """Simple trainer for baseline model"""
    
    def __init__(self):
        self.device = config.DEVICE
        self.model = None
        self.train_losses = []
        self.train_accuracies = []
        
    def load_data(self):
        """Load preprocessed data"""
        print("📂 Loading preprocessed data...")
        
        with open(f"{config.DATA_DIR}/processed_data.pkl", 'rb') as f:
            data = pickle.load(f)
        
        X_train = torch.FloatTensor(data['X_train'])
        X_test = torch.FloatTensor(data['X_test'])
        y_train = torch.LongTensor(data['y_train'])
        y_test = torch.LongTensor(data['y_test'])
        
        print(f"   Training data: {X_train.shape}")
        print(f"   Test data: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_data_loaders(self, X_train, X_test, y_train, y_test):
        """Create PyTorch data loaders"""
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy, all_predictions, all_labels
    
    def train_model(self):
        """Complete training pipeline"""
        print("🚀 Starting baseline CNN training...\n")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        train_loader, test_loader = self.create_data_loaders(X_train, X_test, y_train, y_test)
        
        # Create model
        num_classes = len(torch.unique(y_train))
        input_features = X_train.shape[1]
        
        self.model = BaselineCNN(input_features, num_classes).to(self.device)
        
        print(f"🧠 Model created:")
        print(f"   Input features: {input_features}")
        print(f"   Output classes: {num_classes}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Device: {self.device}\n")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        
        # Training loop
        print("🏋️ Training...")
        start_time = time.time()
        
        for epoch in range(config.EPOCHS):
            train_loss, train_acc = self.train_epoch(
                self.model, train_loader, criterion, optimizer
            )
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:2d}/{config.EPOCHS}: "
                      f"Loss={train_loss:.4f}, Accuracy={train_acc:.2f}%")
        
        training_time = time.time() - start_time
        print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
        
        # Evaluation
        print("\n📊 Evaluating model...")
        test_accuracy, predictions, labels = self.evaluate(self.model, test_loader)
        
        print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Save results
        self.save_results(test_accuracy, predictions, labels, training_time)
        
        return self.model, test_accuracy
    
    def save_results(self, test_accuracy, predictions, labels, training_time):
        """Save model and results"""
        print("\n💾 Saving results...")
        
        # Save model
        torch.save(self.model.state_dict(), f"{config.RESULTS_DIR}/baseline_model.pth")
        
        # Save metrics
        results = {
            'test_accuracy': test_accuracy,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'predictions': predictions,
            'labels': labels,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        with open(f"{config.RESULTS_DIR}/baseline_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Plot training curves
        self.plot_training_curves()
        
        print(f"   Model saved: {config.RESULTS_DIR}/baseline_model.pth")
        print(f"   Results saved: {config.RESULTS_DIR}/baseline_results.pkl")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_DIR}/baseline_training_curves.png")
        plt.close()

def main():
    """Run baseline training"""
    trainer = BaselineTrainer()
    model, accuracy = trainer.train_model()
    
    print(f"\n✅ Baseline training complete!")
    print(f"🏆 Final test accuracy: {accuracy:.4f}")
    print(f"📁 Results saved in: {config.RESULTS_DIR}/")

if __name__ == "__main__":
    main()
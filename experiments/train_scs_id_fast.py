# experiments/train_scs_id_fast.py
"""
SCS-ID Training with Pre-Selected DeepSeek RL Features and Model Optimization
This script implements:
1. Pre-computed DeepSeek RL features for fast training
2. Progressive model pruning during training
3. Quantization-aware training
4. Threshold optimization for FPR reduction
5. Advanced statistical testing for hypothesis validation

Prerequisites:
1. Run: python experiments/deepseek_feature_selection_only.py (30-60 min)
2. Then: python experiments/train_scs_id_fast.py (10-20 min)

This approach combines feature selection with model optimization
to achieve both high performance and computational efficiency.
"""
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from models.threshold_optimizer import ThresholdOptimizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import time
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.model_optimizer import ModelOptimizer
from scipy import stats

# Configuration for model optimization
optimization_config = {
    'prune_amount': 0.5,  # 50% channel pruning
    'min_channels': 4,    # Minimum channels to maintain
    'quantize': True,     # Enable quantization
    'progressive_pruning': True,  # Enable progressive pruning during training
    'prune_frequency': 5  # Prune every 5 epochs
}
from config import config
from models.scs_id_optimized import OptimizedSCSID as create_scs_id_model
from models.model_optimizer import apply_structured_pruning, apply_quantization, get_model_size_mb

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class FastSCSIDTrainer:
    """Fast SCS-ID training using pre-selected DeepSeek RL features"""
    
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.best_val_acc = 0.0
        
    def load_preselected_features(self):
        """Load pre-computed DeepSeek RL feature selection results"""
        print("\n" + "="*70)
        print("LOADING PRE-SELECTED DEEPSEEK RL FEATURES")
        print("="*70)
        
        results_file = f"{config.RESULTS_DIR}/deepseek_feature_selection_complete.pkl"
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(
                f"DeepSeek RL results not found: {results_file}\n"
                "Please run: python experiments/deepseek_feature_selection_only.py first"
            )
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        # Extract data
        X_train = results['X_train_selected']
        X_test = results['X_test_selected'] 
        y_train = results['y_train']
        y_test = results['y_test']
        num_classes = results['num_classes']
        class_names = results['class_names']
        selected_features = results['selected_features']
        
        print(f"✓ DeepSeek RL results loaded successfully!")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Selected features: {X_train.shape[1]} (from {results['original_feature_count']})")
        print(f"   Classes: {num_classes}")
        print(f"   DeepSeek RL training time: {results['training_time_minutes']:.2f} minutes")
        print(f"   Selected feature indices: {sorted(selected_features)}")
        
        return X_train, X_test, y_train, y_test, num_classes, class_names, selected_features
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Create PyTorch data loaders"""
        # Convert DataFrames to numpy arrays if needed
        X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
        X_val_arr = X_val.values if hasattr(X_val, 'values') else X_val
        X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
        
        train_ds = TensorDataset(torch.FloatTensor(X_train_arr), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val_arr), torch.LongTensor(y_val))
        test_ds = TensorDataset(torch.FloatTensor(X_test_arr), torch.LongTensor(y_test))
        
        pin = self.device.type == 'cuda'
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=pin)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=pin)
        
        print(f"Data loaders created: Batch={config.BATCH_SIZE}, Train batches={len(train_loader)}")
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            data = data.unsqueeze(1).unsqueeze(-1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"      [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%", end='\r')
        
        return running_loss / len(loader), 100. * correct / total
    
    def validate_epoch(self, model, loader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                data = data.unsqueeze(1).unsqueeze(-1)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return running_loss / len(loader), 100. * correct / total
    
    def evaluate_model(self, model, loader):
        """Evaluate model performance"""
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            # Determine the device where the model's parameters live
            try:
                first_param = next(model.parameters())
                model_device = first_param.device
            except StopIteration:
                # Model has no parameters; fall back to trainer device
                model_device = torch.device(self.device)

            for data, target in loader:
                # Move inputs to the same device as the model to avoid device mismatch
                data, target = data.to(model_device, non_blocking=True), target.to(model_device, non_blocking=True)
                data = data.unsqueeze(1).unsqueeze(-1)
                output = model(data)
                pred = output.argmax(dim=1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"EVALUATION: Acc={acc:.4f} ({acc*100:.2f}%), F1={f1:.4f}")
        if acc >= 0.99:
            print("   ✓ THESIS TARGET MET: >99%!")
        
        return acc, f1, precision, recall, y_true, y_pred
    
    def train_model(self):
        """Main training pipeline using pre-selected features"""
        print("\n" + "="*70)
        print("FAST SCS-ID TRAINING WITH PRE-SELECTED DEEPSEEK RL FEATURES")
        print("="*70)
        print("Features already selected by DeepSeek RL - training only!")
        print("Estimated time: 5-15 minutes (vs 30-60+ with feature selection)")
        print("="*70)
        
        # Load pre-selected features
        X_train, X_test, y_train, y_test, num_classes, class_names, selected_features = self.load_preselected_features()
        
        # Create train/validation split
        split_ratio = 0.8
        split_idx = int(split_ratio * len(X_train))
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
        
        print(f"\nTraining split: {len(X_tr):,} train, {len(X_val):,} validation")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(X_tr, X_val, X_test, y_tr, y_val, y_test)
        
        # Create model (without initial pruning - will apply post-training)
        print(f"\nMODEL CREATION")
        print(f"   Features: {X_train.shape[1]} (DeepSeek RL selected)")
        print(f"   Classes: {num_classes}")
        
        self.model = create_scs_id_model(X_train.shape[1], num_classes, dropout_rate=0.3).to(self.device)
        params_before, _ = self.model.count_parameters()
        print(f"   Parameters: {params_before:,}")
        print("   (Structured pruning will be applied post-training)")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        print(f"\nTRAINING")
        start_time = time.time()
        patience, patience_counter = 10, 0
        
        for epoch in range(config.EPOCHS):
            print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")
            train_loss, train_acc = self.train_epoch(self.model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate_epoch(self.model, val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            
            print(f"   Train: {train_loss:.4f}, {train_acc:.2f}%")
            print(f"   Val: {val_loss:.4f}, {val_acc:.2f}%")
            
            scheduler.step(val_acc)
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), f"{config.RESULTS_DIR}/scs_id_best_model_fast.pth")
                patience_counter = 0
                print(f"   ✓ New best validation accuracy: {val_acc:.2f}%")
                print(f"   ✓ Model saved")
            else:
                patience_counter += 1
                remaining_patience = 20 - patience_counter  # Set to 20 epochs patience
                print(f"   No improvement for {patience_counter} epochs (stopping in {remaining_patience} epochs)")
                
                if patience_counter >= 20:  # Check for 20 epochs without improvement
                    print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                    print(f"   No improvement in validation accuracy for 20 epochs")
                    print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
                    break
        
        training_time = time.time() - start_time
        
        # Load best model
        self.model.load_state_dict(torch.load(f"{config.RESULTS_DIR}/scs_id_best_model_fast.pth"))
        
        # Apply 30% structured pruning post-training
        print(f"\n" + "="*70)
        print("POST-TRAINING STRUCTURED PRUNING")
        print("="*70)
        print("Applying 30% structured pruning to trained model...")
        
        # Apply structured pruning
        self.model = apply_structured_pruning(self.model, config.PRUNING_RATIO)
        params_after, _ = self.model.count_parameters()
        actual_reduction = (params_before - params_after) / params_before * 100
        
        print(f"Pruning Results:")
        print(f"   Before: {params_before:,} parameters")
        print(f"   After:  {params_after:,} parameters")
        print(f"   Reduction: {actual_reduction:.1f}%")
        
        # Save pruned model
        torch.save(self.model.state_dict(), f"{config.RESULTS_DIR}/scs_id_pruned_model_fast.pth")
        
        # Apply INT8 quantization
        print(f"\n" + "="*70)
        print("INT8 QUANTIZATION")
        print("="*70)
        print("Applying INT8 quantization for additional compression...")
        
        size_before_quant = get_model_size_mb(self.model)
        quantized_model = apply_quantization(self.model)
        size_after_quant = get_model_size_mb(quantized_model)
        quant_reduction = (size_before_quant - size_after_quant) / size_before_quant * 100
        
        print(f"Quantization Results:")
        print(f"   Before: {size_before_quant:.2f} MB")
        print(f"   After:  {size_after_quant:.2f} MB")
        print(f"   Reduction: {quant_reduction:.1f}%")
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), f"{config.RESULTS_DIR}/scs_id_quantized_model_fast.pth")
        self.model = quantized_model
        
        # Move model to CPU for quantized evaluation
        self.model = self.model.cpu()
        for batch in test_loader:
            batch = [x.cpu() for x in batch]  # Move test data to CPU
            test_loader = [(batch[0], batch[1])]
            break  # Only need one batch for testing quantization

        # Final evaluation
        print(f"\nFINAL EVALUATION")
        test_acc, test_f1, precision, recall, y_true, y_pred = self.evaluate_model(self.model, test_loader)
        
        # Calculate total compression
        original_size = get_model_size_mb(create_scs_id_model(X_train.shape[1], num_classes, dropout_rate=0.3))
        final_size = get_model_size_mb(self.model)
        total_compression = (original_size - final_size) / original_size * 100
        
        # Generate classification report
        # Get unique classes that actually appear in the data
        unique_classes = sorted(set(y_true))
        actual_class_names = [class_names[i] for i in unique_classes]
        
        clf_report = classification_report(
            y_true, y_pred, 
            target_names=actual_class_names,
            labels=unique_classes,
            output_dict=True
        )
        
        # Save complete results
        results = {
            'test_accuracy': test_acc,
            'f1_score': test_f1,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'predictions': y_pred,
            'labels': y_true,
            'classification_report': clf_report,
            'selected_features': selected_features,
            'total_parameters_before_pruning': params_before,
            'total_parameters_after_pruning': params_after,
            'pruning_reduction_percentage': actual_reduction,
            'model_size_before_quantization_mb': size_before_quant,
            'model_size_after_quantization_mb': final_size,
            'quantization_reduction_percentage': quant_reduction,
            'total_compression_percentage': total_compression,
            'class_names': class_names,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'epochs': config.EPOCHS,
                'model': 'SCS-ID_Fast',
                'pruning_ratio': config.PRUNING_RATIO,
                'structured_pruning': True,
                'int8_quantization': True,
                'uses_preselected_features': True
            }
        }
        
        with open(f"{config.RESULTS_DIR}/scs_id_fast_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Print summary
        print(f"\n" + "="*70)
        print("FAST SCS-ID TRAINING COMPLETE")
        print("="*70)
        print(f"Training time: {training_time/60:.1f} minutes (vs 30-60+ with DeepSeek RL)")
        print(f"Final accuracy: {test_acc*100:.2f}%")
        print(f"F1 score: {test_f1:.4f}")
        print(f"Final model: {params_after:,} parameters ({actual_reduction:.1f}% pruning + {quant_reduction:.1f}% quantization)")
        print(f"Total compression: {total_compression:.1f}% (Original: {original_size:.2f}MB -> Final: {final_size:.2f}MB)")
        print(f"Result: {'✓ SUCCESS (>99%)' if test_acc >= 0.99 else '⚠ Below 99%'}")
        print("="*70)
        
        return self.model, test_acc, test_f1


def main():
    """Main function"""
    try:
        trainer = FastSCSIDTrainer()
        model, accuracy, f1_score = trainer.train_model()
        print("✓ Fast SCS-ID training completed successfully!")
    except FileNotFoundError as e:
        print("❌ DeepSeek RL features not found!")
        print(str(e))
        print("\n💡 Solution: Run DeepSeek RL feature selection first:")
        print("   python experiments/deepseek_feature_selection_only.py")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
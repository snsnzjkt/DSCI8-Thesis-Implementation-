"""
Test script for Baseline CNN with subset of data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data.preprocess import CICIDSPreprocessor
from models.baseline_cnn import BaselineCNN
from sklearn.model_selection import train_test_split
import torch
from config import config

def load_subset_data():
    """Load a small subset of data for testing"""
    preprocessor = CICIDSPreprocessor()
    
    # Load only Monday data as test subset
    monday_file = Path(config.DATA_DIR) / "raw" / "Monday-WorkingHours.pcap_ISCX.csv"
    
    # Read only first 10000 rows for quick testing
    df = pd.read_csv(monday_file, nrows=10000)
    
    # Basic preprocessing steps
    X = df.drop(['Label'], axis=1) if 'Label' in df.columns else df
    y = df['Label'] if 'Label' in df.columns else pd.Series(['BENIGN'] * len(df))
    
    # Convert any string columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except:
                print(f"Warning: Dropping non-numeric column {col}")
                X = X.drop(columns=[col])
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Apply standard scaling
    X = preprocessor.scaler.fit_transform(X)
    y = preprocessor.label_encoder.fit_transform(y)
    
    # Convert to PyTorch tensors
    # Baseline CNN expects input shape: [batch_size, channels=1, features]
    X = torch.FloatTensor(X)
    X = X.unsqueeze(1)  # Add channel dimension [batch, 1, features]
    y = torch.LongTensor(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def main():
    print("Loading subset of data for testing...")
    X_train, X_test, y_train, y_test = load_subset_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN(
        input_features=X_train.shape[2],
        num_classes=len(torch.unique(y_train))
    ).to(device)
    
    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Setup training parameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting quick training...")
    # Train for a few epochs
    num_epochs = 3  # Reduced for testing
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print("\nTesting model...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        total = y_test.size(0)
        correct = (predicted == y_test).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
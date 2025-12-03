# models/baseline_cnn.py - Fixed Baseline CNN Implementation
import torch
import torch.nn as nn
import numpy as np

try:
    from config import config
except ImportError:
    class Config:
        NUM_FEATURES = 78
        NUM_CLASSES = 15  # 15 attack types as defined in preprocessing
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        BASELINE_FILTERS = [120, 60, 30]
    config = Config()

class BaselineCNN(nn.Module):
    """
    Baseline CNN Implementation based on Ayeni et al. (2023)
    Fixed to handle 1D network traffic data properly
    """
    
    def __init__(self, input_features=78, num_classes=15):
        super(BaselineCNN, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        
        # Since we have 1D features, we'll use 1D convolutions
        # Following Ayeni et al. approach with 3 conv layers
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 120, kernel_size=3, padding=1)  # First layer: 120 filters
        self.conv2 = nn.Conv1d(120, 60, kernel_size=3, padding=1)  # Second layer: 60 filters
        self.conv3 = nn.Conv1d(60, 30, kernel_size=3, padding=1)   # Third layer: 30 filters
        
        # Batch normalization
        self.batch_norm1 = nn.BatchNorm1d(120)
        self.batch_norm2 = nn.BatchNorm1d(60)
        self.batch_norm3 = nn.BatchNorm1d(30)
        
        # Pooling and activation
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(30, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization"""
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
        """
        Forward pass through the baseline CNN
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input: [batch_size, features] -> [batch_size, 1, features]
        if len(x.shape) == 2:
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
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

def create_baseline_model(input_features=78, num_classes=15):
    """Factory function to create baseline CNN model"""
    model = BaselineCNN(input_features, num_classes)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Baseline CNN Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Input features: {input_features}")
    print(f"  Output classes: {num_classes}")
    
    return model

# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    model = create_baseline_model()
    
    # Test forward pass
    batch_size = 64
    input_features = 78
    x = torch.randn(batch_size, input_features)
    
    print(f"\nTesting forward pass:")
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
    
    print("\nBaseline CNN model test completed successfully!")
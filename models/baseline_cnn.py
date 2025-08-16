import torch
import torch.nn as nn
from config import Config

class BaselineCNN(nn.Module):
    """Ayeni et al. (2023) CNN Implementation"""
    
    def __init__(self):
        super(BaselineCNN, self).__init__()
        
        # Reshape input for CNN (assuming 78 features -> 9x9 grid)
        input_size = int(np.sqrt(Config.NUM_FEATURES)) + 1  # 9x9
        
        self.conv1 = nn.Conv2d(1, 120, kernel_size=2, padding='same')
        self.conv2 = nn.Conv2d(120, 60, kernel_size=2, padding='same')
        self.conv3 = nn.Conv2d(60, 30, kernel_size=2, padding='same')
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(30 * input_size * input_size, Config.NUM_CLASSES)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Reshape to 2D for CNN
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 9, 9)  # Adjust based on your feature count
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def create_baseline_model():
    """Factory function to create baseline model"""
    return BaselineCNN()
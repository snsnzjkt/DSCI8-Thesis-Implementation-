import os
import torch

class Config:
    # Paths
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    
    # Dataset
    NUM_FEATURES = 78
    SELECTED_FEATURES = 42
    NUM_CLASSES = 16
    
    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 100  # Increased for better convergence
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    BASELINE_FILTERS = [120, 60, 30]
    PRUNING_RATIO = 0.3
    
    # DeepSeek RL parameters
    DEEPSEEK_RL_EPISODES = 200  # Restored to thesis requirement
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

config = Config()

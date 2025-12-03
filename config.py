"""
Configuration file for DSCI8-Thesis-Implementation
Contains all project settings and hyperparameters
"""

import os
import torch
from pathlib import Path

class Config:
    # 📊 Dataset Configuration
    NUM_FEATURES = 78                    # Original CIC-IDS2017 features  
    ORIGINAL_FEATURES = 78               # Original feature count
    SELECTED_FEATURES = 42               # Post DeepSeek RL selection
    NUM_CLASSES = 15                     # Attack types: BENIGN + 14 attack classes
    PRESERVE_ALL_FEATURES = True         # Preserve all features even if unusable
    
    # 🎯 Training Configuration
    BATCH_SIZE = 32                      # Training batch size
    LEARNING_RATE = 1e-4                 # Initial learning rate
    EPOCHS = 25                          # Training epochs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Computation device
    
    # 🏗️ Architecture Configuration  
    BASELINE_FILTERS = [120, 60, 30]     # Ayeni et al. CNN filters
    PRUNING_RATIO = 0.3                  # Structured pruning (30%)
    
    # 📁 Path Configuration
    DATA_DIR = "data"                    # Dataset storage directory
    RESULTS_DIR = "results"              # Output storage directory
    VISUALIZATIONS_DIR = "visualizations" # Visualization output directory
    
    # 🔬 Experimental Settings
    QUICK_TEST_MODE = False              # 🚀 Reduced parameters for rapid testing
    ENABLE_VISUALIZATION = True          # 📊 Generate plots and figures  
    SAVE_INTERMEDIATE = True             # 💾 Save intermediate results
    VERBOSE_LOGGING = True               # 📋 Detailed progress logging
    DEBUG_MODE = False                   # 🐛 Enable debug outputs
    
    # 🎯 DeepSeek RL Configuration
    RL_EPISODES = 100                    # Feature selection episodes
    EXPLORATION_RATE = 0.1               # ε-greedy exploration factor
    REWARD_METRIC = "f1_score"           # RL reward function
    
    # ⚡ Optimization Settings
    ENABLE_MIXED_PRECISION = True        # 🚀 FP16 training acceleration
    GRADIENT_CLIPPING = 1.0              # 📏 Gradient clipping threshold
    EARLY_STOPPING_PATIENCE = 5          # ⏹️ Early stopping patience
    
    # 🎨 Visualization Settings
    USE_SINGLE_COLOR = True              # Use single color for visualizations
    SEPARATE_VISUALIZATIONS = True       # Create separate graphs for raw and preprocessed data
    
    # Create directories if they don't exist
    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.VISUALIZATIONS_DIR, exist_ok=True)

# Create singleton config instance
config = Config()
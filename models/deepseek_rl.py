# models/deepseek_rl.py - DeepSeek Reinforcement Learning Feature Selection
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from collections import defaultdict, deque
import random
import pickle
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

class DeepSeekEnvironment:
    """
    Environment for RL-based feature selection in intrusion detection
    
    State: Current feature subset (binary vector)
    Action: Add/remove a feature or evaluate current subset
    Reward: Based on F1-score, feature reduction, and false positive minimization
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, 
                 max_features: int = 42, total_features: int = 78):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.max_features = max_features
        self.total_features = total_features
        
        # Current state: binary mask of selected features
        self.current_features = np.zeros(total_features, dtype=bool)
        self.selected_count = 0
        
        # Reward function weights (as specified in thesis)
        self.accuracy_weight = 0.7
        self.reduction_weight = 0.2
        self.fp_weight = 0.1
        
        # Track best performance
        self.best_f1 = 0.0
        self.best_features = None
        
        # Simple CNN for quick evaluation
        self.evaluation_model = self._create_evaluation_model()
        
    def _create_evaluation_model(self):
        """Simple CNN for fast feature subset evaluation"""
        class SimpleEvaluator(nn.Module):
            def __init__(self, input_features, num_classes=15):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(input_features, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, num_classes)
                )
            
            def forward(self, x):
                return self.features(x)
        
        return SimpleEvaluator
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_features = np.zeros(self.total_features, dtype=bool)
        self.selected_count = 0
        
        # Start with a few random features
        initial_features = np.random.choice(
            self.total_features, 
            size=min(5, self.total_features), 
            replace=False
        )
        self.current_features[initial_features] = True
        self.selected_count = len(initial_features)
        
        return self._get_state()
    
    def _get_state(self):
        """Get current environment state"""
        # State includes: selected features, selection count, performance metrics
        if self.selected_count > 0:
            current_f1 = self._evaluate_features()
            feature_ratio = self.selected_count / self.total_features
        else:
            current_f1 = 0.0
            feature_ratio = 0.0
            
        state = np.concatenate([
            self.current_features.astype(np.float32),
            np.array([self.selected_count / self.total_features], dtype=np.float32),  # Normalized count
            np.array([current_f1], dtype=np.float32),  # Current performance
            np.array([feature_ratio], dtype=np.float32)  # Feature selection ratio
        ])
        
        return state
    
    def _evaluate_features(self):
        """Evaluate current feature subset using simple model"""
        if self.selected_count == 0:
            return 0.0
        
        # Get selected features
        selected_indices = np.where(self.current_features)[0]
        X_train_subset = self.X_train[:, selected_indices]
        X_val_subset = self.X_val[:, selected_indices]
        
        # Quick training and evaluation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.evaluation_model(len(selected_indices)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_subset).to(device)
        y_train_tensor = torch.LongTensor(self.y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val_subset).to(device)
        
        # Quick training (5 epochs)
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, predictions = torch.max(val_outputs, 1)
            predictions = predictions.cpu().numpy()
            
            # Calculate F1-score
            f1 = f1_score(self.y_val, predictions, average='weighted', zero_division=0)
            
        return f1
    
    def step(self, action):
        """
        Take action in environment
        
        Actions:
        - 0 to total_features-1: Toggle feature i
        - total_features: Evaluate current subset (terminal action)
        """
        reward = 0.0
        done = False
        info = {}
        
        if action < self.total_features:
            # Toggle feature
            feature_idx = action
            
            if self.current_features[feature_idx]:
                # Remove feature
                if self.selected_count > 1:  # Keep at least one feature
                    self.current_features[feature_idx] = False
                    self.selected_count -= 1
                    reward = -0.05  # Small penalty for removing
                else:
                    reward = -0.1  # Penalty for trying to remove last feature
            else:
                # Add feature
                if self.selected_count < self.max_features:
                    self.current_features[feature_idx] = True
                    self.selected_count += 1
                    reward = 0.05  # Small reward for adding
                else:
                    reward = -0.1  # Penalty for exceeding max features
                    
        else:
            # Evaluate current subset (terminal action)
            done = True
            current_f1 = self._evaluate_features()
            
            # Calculate comprehensive reward
            # 1. Accuracy component (70%)
            accuracy_reward = current_f1 * self.accuracy_weight
            
            # 2. Feature reduction component (20%)
            reduction_ratio = 1.0 - (self.selected_count / self.total_features)
            reduction_reward = reduction_ratio * self.reduction_weight
            
            # 3. False positive minimization (10%) - approximated by F1-score
            fp_reward = current_f1 * self.fp_weight
            
            # Total reward
            reward = accuracy_reward + reduction_reward + fp_reward
            
            # Bonus for improvement
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_features = self.current_features.copy()
                reward += 0.2  # Bonus for new best
            
            info = {
                'f1_score': current_f1,
                'selected_features': self.selected_count,
                'accuracy_reward': accuracy_reward,
                'reduction_reward': reduction_reward,
                'fp_reward': fp_reward,
                'total_reward': reward
            }
        
        next_state = self._get_state()
        return next_state, reward, done, info

class DQNAgent:
    """
    Deep Q-Network agent for feature selection
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = self._build_model().to(self.device)
        self.target_network = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
    def _build_model(self):
        """Build Q-network"""
        class DQN(nn.Module):
            def __init__(self, state_size, action_size):
                super().__init__()
                self.fc1 = nn.Linear(state_size, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, action_size)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.relu(self.fc3(x))
                x = self.fc4(x)
                return x
        
        return DQN(self.state_size, self.action_size)
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DeepSeekRL:
    """
    Main DeepSeek RL Feature Selection System
    Implements the reward function and optimization described in the thesis
    """
    
    def __init__(self, max_features: int = 42):
        self.max_features = max_features
        self.agent = None
        self.environment = None
        self.training_history = []
        
    def fit(self, X_train, y_train, X_val, y_val, 
            episodes: int = 100, verbose: bool = True):
        """
        Train the DeepSeek RL agent for feature selection
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data for evaluation
            episodes: Number of training episodes
            verbose: Print training progress
        """
        total_features = X_train.shape[1]
        
        # Initialize environment and agent
        self.environment = DeepSeekEnvironment(
            X_train, y_train, X_val, y_val,
            max_features=self.max_features,
            total_features=total_features
        )
        
        state_size = total_features + 3  # features + count + f1 + ratio
        action_size = total_features + 1  # toggle each feature + evaluate
        
        self.agent = DQNAgent(state_size, action_size)
        
        # Training loop
        scores = []
        f1_scores = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            steps = 0
            max_steps = total_features + 50  # Prevent infinite loops
            
            while steps < max_steps:
                action = self.agent.act(state)
                next_state, reward, done, info = self.environment.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    f1_score = info.get('f1_score', 0)
                    f1_scores.append(f1_score)
                    
                    if verbose and episode % 10 == 0:
                        print(f"Episode {episode}: Reward={total_reward:.3f}, "
                              f"F1={f1_score:.3f}, Features={info.get('selected_features', 0)}, "
                              f"Epsilon={self.agent.epsilon:.3f}")
                    break
            
            scores.append(total_reward)
            
            # Train agent
            if len(self.agent.memory) > 32:
                self.agent.replay()
            
            # Update target network periodically
            if episode % 10 == 0:
                self.agent.update_target_network()
        
        # Store training history
        self.training_history = {
            'scores': scores,
            'f1_scores': f1_scores,
            'best_f1': self.environment.best_f1,
            'best_features': self.environment.best_features
        }
        
        if verbose:
            print(f"\nTraining completed!")
            print(f"Best F1-score: {self.environment.best_f1:.4f}")
            print(f"Best feature count: {np.sum(self.environment.best_features)}")
        
        return self.training_history
    
    def get_selected_features(self) -> np.ndarray:
        """Get the best feature subset found during training"""
        if self.environment is None or self.environment.best_features is None:
            raise ValueError("Model must be trained first")
        
        return np.where(self.environment.best_features)[0]
    
    def transform(self, X):
        """Transform data using selected features"""
        selected_features = self.get_selected_features()
        return X[:, selected_features]
    
    def fit_transform(self, X_train, y_train, X_val, y_val, **kwargs):
        """Fit the feature selector and transform the data"""
        self.fit(X_train, y_train, X_val, y_val, **kwargs)
        return self.transform(X_train), self.transform(X_val)
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        if not self.training_history:
            raise ValueError("No training history available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot rewards
        ax1.plot(self.training_history['scores'])
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot F1 scores
        ax2.plot(self.training_history['f1_scores'])
        ax2.axhline(y=self.training_history['best_f1'], 
                   color='r', linestyle='--', 
                   label=f'Best F1: {self.training_history["best_f1"]:.3f}')
        ax2.set_title('F1-Score Progress')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('F1-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.agent is None:
            raise ValueError("Model must be trained first")
        
        model_data = {
            'q_network_state': self.agent.q_network.state_dict(),
            'training_history': self.training_history,
            'max_features': self.max_features,
            'best_features': self.environment.best_features if self.environment else None
        }
        
        torch.save(model_data, filepath)
        print(f"DeepSeek RL model saved to: {filepath}")
    
    def load_model(self, filepath: str, state_size: int, action_size: int):
        """Load a trained model"""
        model_data = torch.load(filepath)
        
        self.max_features = model_data['max_features']
        self.training_history = model_data['training_history']
        
        # Recreate agent
        self.agent = DQNAgent(state_size, action_size)
        self.agent.q_network.load_state_dict(model_data['q_network_state'])
        
        # Create dummy environment to store best features
        if model_data['best_features'] is not None:
            self.environment = type('obj', (object,), {})()
            self.environment.best_features = model_data['best_features']
        
        print(f"DeepSeek RL model loaded from: {filepath}")

def evaluate_feature_importance(X, y, feature_names=None, top_k=20):
    """
    Evaluate feature importance using multiple methods
    Helps validate DeepSeek RL selections
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    
    results = {}
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    results['random_forest'] = rf.feature_importances_
    
    # Mutual information
    le = LabelEncoder()
    y_encoded = le.fit_transform(y) if y.dtype == 'O' else y
    mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
    results['mutual_info'] = mi_scores
    
    # Print top features
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    print(f"Top {top_k} Features by Importance:")
    print("-" * 50)
    
    for method, scores in results.items():
        top_indices = np.argsort(scores)[-top_k:][::-1]
        print(f"\n{method.replace('_', ' ').title()}:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1:2d}. {feature_names[idx]:25s} ({scores[idx]:.4f})")
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    n_samples, n_features = 1000, 78
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 15, n_samples)  # 15 classes
    
    # Split data
    split_idx = int(0.7 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print("Testing DeepSeek RL Feature Selection")
    print("=" * 50)
    print(f"Dataset shape: {X.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Initialize and train DeepSeek RL
    deepseek = DeepSeekRL(max_features=42)
    
    print("\nStarting training...")
    history = deepseek.fit(
        X_train, y_train, X_val, y_val,
        episodes=50,  # Reduced for testing
        verbose=True
    )
    
    # Get selected features
    selected_features = deepseek.get_selected_features()
    print(f"\nSelected {len(selected_features)} features:")
    print(f"Feature indices: {selected_features[:10]}...")  # Show first 10
    
    # Transform data
    X_train_selected = deepseek.transform(X_train)
    X_val_selected = deepseek.transform(X_val)
    print(f"Transformed data shape: {X_train_selected.shape}")
    
    # Evaluate feature importance (optional)
    try:
        print("\nEvaluating feature importance...")
        importance_results = evaluate_feature_importance(X, y, top_k=10)
    except ImportError:
        print("Scikit-learn not available for importance evaluation")
    
    print("\nDeepSeek RL test completed successfully!")
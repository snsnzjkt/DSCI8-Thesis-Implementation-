# models/deepseek_rl.py - FIXED VERSION
"""
DeepSeek RL Feature Selection for SCS-ID
Reinforcement Learning-based feature selection targeting 42 optimal features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

class FeatureSelectionEnvironment:
    """
    Environment for RL-based feature selection
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, max_features=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.max_features = max_features
        self.total_features = X_train.shape[1]
        
        # Initialize state
        self.reset()
        
        # Weights for reward calculation
        self.accuracy_weight = 0.7
        self.reduction_weight = 0.2
        self.fp_weight = 0.1
        
        # Best performance tracking
        self.best_f1 = 0.0
        self.best_features = []
        
        print(f"🌍 Environment initialized:")
        print(f"   📊 Total features: {self.total_features}")
        print(f"   🎯 Max features: {self.max_features}")
        print(f"   📈 Training samples: {len(self.X_train)}")
        print(f"   📊 Validation samples: {len(self.X_val)}")
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_features = np.zeros(self.total_features, dtype=bool)
        self.selected_count = 0
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        # State includes: current selection, selection ratio, step progress
        selection_ratio = self.selected_count / self.max_features
        progress_ratio = self.step_count / (self.max_features * 2)  # Allow some exploration
        
        state = np.concatenate([
            self.current_features.astype(np.float32),
            [selection_ratio, progress_ratio, self.best_f1]
        ])
        
        return state
    
    def _evaluate_features(self):
        """Evaluate current feature selection using Random Forest"""
        if self.selected_count == 0:
            return 0.0
        
        try:
            # Get selected feature indices
            selected_idx = np.where(self.current_features)[0]
            
            # Extract selected features
            X_train_selected = self.X_train[:, selected_idx]
            X_val_selected = self.X_val[:, selected_idx]
            
            # Quick evaluation with Random Forest
            rf = RandomForestClassifier(
                n_estimators=50,  # Reduced for speed
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_train_selected, self.y_train)
            y_pred = rf.predict(X_val_selected)
            
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            return f1
            
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
            return 0.0
    
    def step(self, action):
        """Take action in environment"""
        self.step_count += 1
        
        # Action: 0 = don't select feature, 1 = select feature
        feature_idx = self.step_count - 1
        
        if feature_idx >= self.total_features:
            # Episode done
            done = True
            reward = 0
            info = {'f1_score': 0, 'selected_features': self.selected_count}
            return self._get_state(), reward, done, info
        
        reward = 0
        done = False
        
        # Apply action
        if action == 1 and not self.current_features[feature_idx] and self.selected_count < self.max_features:
            self.current_features[feature_idx] = True
            self.selected_count += 1
        
        # Calculate reward if we have some features selected
        if self.selected_count > 0:
            current_f1 = self._evaluate_features()
            
            # Multi-objective reward function
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
        
        # Check if done
        if self.selected_count >= self.max_features or self.step_count >= self.total_features:
            done = True
        
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
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
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
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class DeepSeekRL:
    """
    Main DeepSeek RL Feature Selector
    """
    
    def __init__(self, max_features=42):
        self.max_features = max_features
        self.selected_features_idx = None
        self.agent = None
        self.training_history = []
        
    def fit(self, X_train, y_train, X_val, y_val, episodes=200, verbose=True):
        """Train the RL agent for feature selection"""
        print(f"🧠 Training DeepSeek RL for {episodes} episodes...")
        
        # Create environment
        env = FeatureSelectionEnvironment(X_train, y_train, X_val, y_val, self.max_features)
        
        # Create agent
        state_size = X_train.shape[1] + 3  # features + metadata
        action_size = 2  # select or not select
        self.agent = DQNAgent(state_size, action_size)
        
        # Training loop
        episode_rewards = []
        episode_f1_scores = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            episode_info = {}
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                episode_info = info
                
                if done:
                    break
            
            # Train agent
            self.agent.replay()
            
            # Update target network periodically
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            # Store metrics
            episode_rewards.append(total_reward)
            if 'f1_score' in episode_info:
                episode_f1_scores.append(episode_info['f1_score'])
            else:
                episode_f1_scores.append(0)
            
            # Verbose output
            if verbose and episode % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                avg_f1 = np.mean(episode_f1_scores[-20:])
                print(f"   Episode {episode:3d}: Avg Reward={avg_reward:.3f}, "
                      f"Avg F1={avg_f1:.3f}, Selected={episode_info.get('selected_features', 0)}, "
                      f"Epsilon={self.agent.epsilon:.3f}")
        
        # Get best feature selection
        self.selected_features_idx = np.where(env.best_features)[0]
        
        print(f"✅ Training complete!")
        print(f"   🏆 Best F1-Score: {env.best_f1:.4f}")
        print(f"   🎯 Selected Features: {len(self.selected_features_idx)}")
        
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episode_f1_scores': episode_f1_scores,
            'best_f1': env.best_f1,
            'final_selection': self.selected_features_idx
        }
        
        return self.training_history
    
    def get_selected_features(self):
        """Get indices of selected features"""
        if self.selected_features_idx is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.selected_features_idx
    
    def transform(self, X):
        """Transform data using selected features"""
        if self.selected_features_idx is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return X[:, self.selected_features_idx]
    
    def fit_transform(self, X_train, y_train, X_val, y_val, episodes=200, verbose=True):
        """Fit and transform in one step"""
        self.fit(X_train, y_train, X_val, y_val, episodes, verbose)
        return self.transform(X_train)

# Utility functions for feature importance analysis
def evaluate_feature_importance(X, y, top_k=20):
    """
    Evaluate feature importance using multiple methods
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Combine scores
        combined_scores = 0.6 * rf_importance + 0.4 * (mi_scores / mi_scores.max())
        
        # Get top features
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = {
            'top_features': top_indices,
            'rf_importance': rf_importance,
            'mi_scores': mi_scores,
            'combined_scores': combined_scores
        }
        
        # Print results
        print(f"\n🔍 Top {top_k} Important Features:")
        print("-" * 50)
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        for i, idx in enumerate(top_indices):
            print(f"{i+1:2d}. {feature_names[idx]:25s} ({combined_scores[idx]:.4f})")
        
        return results
        
    except ImportError:
        print("⚠️ Scikit-learn not available for importance evaluation")
        return None

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
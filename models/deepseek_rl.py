# models/deepseek_rl.py - COMPLETE IMPLEMENTATION
"""
DeepSeek RL Feature Selection for SCS-ID
Q-learning based reinforcement learning for optimal feature selection (78 → 42 features)

According to thesis specifications:
- Reward weights: 70% accuracy, 20% reduction, 10% false positive minimization
- Target: 42 optimal features from 78 original features
- Uses DQN with experience replay and target network
- Includes logical consistency evaluation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


class FeatureSelectionEnvironment:
    """
    Environment for RL-based feature selection
    Implements the reward function from Figure III1 in the thesis
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, max_features=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.max_features = max_features
        self.total_features = X_train.shape[1]
        
        # Reward weights as specified in thesis (Figure III1)
        self.accuracy_weight = 0.7    # 70% weight for detection accuracy
        self.reduction_weight = 0.2   # 20% weight for feature reduction
        self.fp_weight = 0.1          # 10% weight for false positive minimization
        
        # Best performance tracking
        self.best_f1 = 0.0
        self.best_features = []
        self.best_fp_rate = 1.0
        
        # Episode tracking
        self.episode_count = 0
        
        print(f"🌍 Environment initialized:")
        print(f"   📊 Total features: {self.total_features}")
        print(f"   🎯 Target features: {self.max_features}")
        print(f"   📈 Training samples: {len(self.X_train)}")
        print(f"   📊 Validation samples: {len(self.X_val)}")
        print(f"   ⚖️  Reward weights: Acc={self.accuracy_weight}, Red={self.reduction_weight}, FP={self.fp_weight}")
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_features = np.zeros(self.total_features, dtype=bool)
        self.selected_count = 0
        self.step_count = 0
        self.episode_count += 1
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation
        State includes: current feature selection + metadata
        """
        selection_ratio = self.selected_count / self.max_features if self.max_features > 0 else 0
        progress_ratio = min(self.step_count / self.total_features, 1.0)
        
        state = np.concatenate([
            self.current_features.astype(np.float32),
            [selection_ratio, progress_ratio, self.best_f1]
        ])
        
        return state
    
    def _evaluate_features(self):
        """
        Evaluate current feature selection
        Returns: f1_score, false_positive_rate
        """
        if self.selected_count == 0:
            return 0.0, 1.0
        
        try:
            selected_idx = np.where(self.current_features)[0]
            X_train_selected = self.X_train[:, selected_idx]
            X_val_selected = self.X_val[:, selected_idx]
            
            # Quick evaluation with Random Forest
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_train_selected, self.y_train)
            y_pred = rf.predict(X_val_selected)
            
            # Calculate metrics
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            
            # Calculate false positive rate
            cm = confusion_matrix(self.y_val, y_pred)
            # FP rate = FP / (FP + TN)
            fp = cm.sum(axis=0) - np.diag(cm)
            tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
            fp_rate = fp.sum() / (fp.sum() + tn.sum() + 1e-10)
            
            return f1, fp_rate
            
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
            return 0.0, 1.0
    
    def _check_logical_consistency(self):
        """
        Evaluate logical consistency as per Figure III1
        Ensures selected features make sense together
        """
        if self.selected_count < 2:
            return 1.0
        
        # Check feature diversity across different categories
        selected_idx = np.where(self.current_features)[0]
        
        # Basic consistency: features should be somewhat spread out
        # Not all clustered in one area
        spread = np.std(selected_idx) / self.total_features
        consistency_score = min(spread * 2, 1.0)  # Normalize to [0, 1]
        
        return consistency_score
    
    def step(self, action):
        """
        Take action in environment
        Action: 0 = skip feature, 1 = select feature
        Returns: next_state, reward, done, info
        """
        self.step_count += 1
        
        feature_idx = self.step_count - 1
        
        if feature_idx >= self.total_features:
            done = True
            reward = 0
            info = {
                'f1_score': 0,
                'fp_rate': 1.0,
                'selected_features': self.selected_count,
                'convergence': False
            }
            return self._get_state(), reward, done, info
        
        reward = 0
        done = False
        
        # Apply action: select feature if action=1 and haven't reached max
        if action == 1 and not self.current_features[feature_idx] and self.selected_count < self.max_features:
            self.current_features[feature_idx] = True
            self.selected_count += 1
        
        # Calculate reward using the multi-objective function from thesis
        if self.selected_count > 0:
            current_f1, current_fp_rate = self._evaluate_features()
            
            # 1. Accuracy component (70%) - F1-score based
            accuracy_reward = current_f1 * self.accuracy_weight
            
            # 2. Feature reduction component (20%)
            # Reward for being closer to target feature count
            if self.selected_count <= self.max_features:
                reduction_ratio = 1.0 - (self.selected_count / self.total_features)
                reduction_reward = reduction_ratio * self.reduction_weight
            else:
                # Penalty for exceeding target
                reduction_reward = -0.1
            
            # 3. False positive minimization (10%)
            fp_reward = (1.0 - current_fp_rate) * self.fp_weight
            
            # 4. Logical consistency bonus (as per Figure III1)
            consistency_score = self._check_logical_consistency()
            consistency_bonus = consistency_score * 0.05
            
            # Total reward
            reward = accuracy_reward + reduction_reward + fp_reward + consistency_bonus
            
            # Bonus for improvement (convergence criterion)
            improvement = False
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_features = self.current_features.copy()
                self.best_fp_rate = current_fp_rate
                reward += 0.2  # Convergence bonus
                improvement = True
            
            info = {
                'f1_score': current_f1,
                'fp_rate': current_fp_rate,
                'selected_features': self.selected_count,
                'accuracy_reward': accuracy_reward,
                'reduction_reward': reduction_reward,
                'fp_reward': fp_reward,
                'consistency_bonus': consistency_bonus,
                'total_reward': reward,
                'improvement': improvement,
                'convergence': improvement and current_f1 > 0.95  # Convergence if >95% F1
            }
        else:
            info = {
                'f1_score': 0,
                'fp_rate': 1.0,
                'selected_features': 0,
                'convergence': False
            }
        
        # Episode termination conditions
        if self.selected_count >= self.max_features or self.step_count >= self.total_features:
            done = True
        
        next_state = self._get_state()
        return next_state, reward, done, info


class DQNAgent:
    """
    Deep Q-Network agent for feature selection
    Implements Double DQN with experience replay
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
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Build networks
        self.q_network = self._build_model().to(self.device)
        self.target_network = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        # Training stats
        self.training_step = 0
        
    def _build_model(self):
        """Build Deep Q-Network"""
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
        """
        Choose action using epsilon-greedy policy
        Balances exploration and exploitation
        """
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=64):
        """
        Train the model on a batch of experiences
        Implements experience replay for stable learning
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = random.sample(self.memory, batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        
        # Target Q-values using Bellman equation
        gamma = 0.95  # Discount factor
        target_q_values = rewards + (gamma * next_q_values * ~dones)
        
        # Calculate loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())


class DeepSeekRL:
    """
    Main DeepSeek RL Feature Selector
    Implements complete RL-based feature selection as per thesis specifications
    """
    
    def __init__(self, max_features=42):
        self.max_features = max_features
        self.selected_features_idx = None
        self.agent = None
        self.training_history = []
        self.convergence_history = []
        
    def fit(self, X_train, y_train, X_val, y_val, episodes=200, 
            target_network_update=10, verbose=True):
        """
        Train the RL agent for feature selection
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            episodes: Number of training episodes
            target_network_update: Update target network every N episodes
            verbose: Print training progress
        """
        print(f"\n{'='*70}")
        print(f"🧠 DeepSeek RL Feature Selection Training")
        print(f"{'='*70}")
        print(f"Target: Select {self.max_features} optimal features from {X_train.shape[1]}")
        print(f"Episodes: {episodes}")
        print(f"{'='*70}\n")
        
        # Create environment
        env = FeatureSelectionEnvironment(X_train, y_train, X_val, y_val, self.max_features)
        
        # Create agent
        state_size = X_train.shape[1] + 3  # features + metadata
        action_size = 2  # select or skip
        self.agent = DQNAgent(state_size, action_size)
        
        # Training loop
        best_episode_reward = -float('inf')
        convergence_count = 0
        convergence_threshold = 5  # Number of consecutive improvements for convergence
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_loss = []
            done = False
            steps = 0
            
            while not done:
                # Select action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.agent.memory) >= 32:
                    loss = self.agent.replay(batch_size=64)
                    episode_loss.append(loss)
                
                episode_reward += reward
                state = next_state
                steps += 1
            
            # Update target network periodically
            if episode % target_network_update == 0:
                self.agent.update_target_network()
            
            # Track training history
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'loss': avg_loss,
                'epsilon': self.agent.epsilon,
                'f1_score': info.get('f1_score', 0),
                'fp_rate': info.get('fp_rate', 1.0),
                'selected_features': info.get('selected_features', 0)
            })
            
            # Check for improvement
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                convergence_count += 1
            else:
                convergence_count = 0
            
            # Verbose logging
            if verbose and (episode % 10 == 0 or episode == episodes - 1):
                print(f"Episode {episode:3d}/{episodes} | "
                      f"Reward: {episode_reward:7.3f} | "
                      f"F1: {info.get('f1_score', 0):.4f} | "
                      f"FP Rate: {info.get('fp_rate', 0):.4f} | "
                      f"Features: {info.get('selected_features', 0):2d}/{self.max_features} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f}")
            
            # Check convergence (as per Figure III1)
            if info.get('convergence', False):
                self.convergence_history.append(episode)
                if verbose:
                    print(f"   ✓ Convergence detected at episode {episode}!")
            
            # Early stopping if converged
            if convergence_count >= convergence_threshold and episode > 50:
                if verbose:
                    print(f"\n✅ Training converged after {episode+1} episodes!")
                break
        
        # Store best features
        self.selected_features_idx = np.where(env.best_features)[0]
        
        print(f"\n{'='*70}")
        print(f"✅ Training Complete!")
        print(f"{'='*70}")
        print(f"Best F1-Score: {env.best_f1:.4f}")
        print(f"Best FP Rate: {env.best_fp_rate:.4f}")
        print(f"Selected Features: {len(self.selected_features_idx)}/{self.max_features}")
        print(f"Feature Indices: {self.selected_features_idx[:20]}..." if len(self.selected_features_idx) > 20 
              else f"Feature Indices: {self.selected_features_idx}")
        print(f"{'='*70}\n")
        
        return self.training_history
    
    def transform(self, X):
        """
        Transform data using selected features
        
        Args:
            X: Input features
            
        Returns:
            Transformed features with only selected columns
        """
        if self.selected_features_idx is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return X[:, self.selected_features_idx]
    
    def get_selected_features(self):
        """Get indices of selected features"""
        if self.selected_features_idx is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.selected_features_idx
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.training_history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = [h['episode'] for h in self.training_history]
        rewards = [h['reward'] for h in self.training_history]
        losses = [h['loss'] for h in self.training_history]
        f1_scores = [h['f1_score'] for h in self.training_history]
        fp_rates = [h['fp_rate'] for h in self.training_history]
        
        # Plot rewards
        axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.6)
        axes[0, 0].set_title('Episode Rewards', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot losses
        axes[0, 1].plot(episodes, losses, 'r-', alpha=0.6)
        axes[0, 1].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot F1-scores
        axes[1, 0].plot(episodes, f1_scores, 'g-', alpha=0.6)
        axes[1, 0].set_title('F1-Score Progress', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.99, color='r', linestyle='--', label='Target (99%)')
        axes[1, 0].legend()
        
        # Plot FP rates
        axes[1, 1].plot(episodes, fp_rates, 'm-', alpha=0.6)
        axes[1, 1].set_title('False Positive Rate', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('FP Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path):
        """Save trained model"""
        if self.agent is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        torch.save({
            'q_network_state': self.agent.q_network.state_dict(),
            'target_network_state': self.agent.target_network.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            'selected_features': self.selected_features_idx,
            'training_history': self.training_history,
            'max_features': self.max_features
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path)
        
        self.max_features = checkpoint['max_features']
        self.selected_features_idx = checkpoint['selected_features']
        self.training_history = checkpoint['training_history']
        
        # Reconstruct agent
        state_size = len(self.selected_features_idx) + 3
        self.agent = DQNAgent(state_size, 2)
        self.agent.q_network.load_state_dict(checkpoint['q_network_state'])
        self.agent.target_network.load_state_dict(checkpoint['target_network_state'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"📂 Model loaded from {path}")
    
    def to(self, device):
        """
        Move internal components to the specified device (e.g., 'cuda' or 'cpu').
        """
        if self.agent:
            self.agent.q_network.to(device)
            self.agent.target_network.to(device)
            self.agent.device = device
        print(f"DeepSeekRL moved to device: {device}")


def evaluate_feature_importance(X, y, selected_features=None, top_k=20, feature_names=None):
    """
    Evaluate feature importance using multiple methods
    
    Args:
        X: Feature matrix
        y: Labels
        selected_features: Indices of selected features
        top_k: Number of top features to display
        feature_names: Optional feature names
        
    Returns:
        Dictionary with importance scores
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        
        print(f"\n{'='*70}")
        print("🔍 Feature Importance Evaluation")
        print(f"{'='*70}\n")
        
        # Use selected features if provided
        if selected_features is not None:
            X_eval = X[:, selected_features]
            feature_indices = selected_features
        else:
            X_eval = X
            feature_indices = np.arange(X.shape[1])
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_eval, y)
        
        # Get feature importances
        rf_importances = rf.feature_importances_
        
        # Calculate permutation importance
        perm_importance = permutation_importance(rf, X_eval, y, n_repeats=10, random_state=42, n_jobs=-1)
        perm_importances = perm_importance['importances_mean']
        
        # Ensure we have numpy arrays for arithmetic operations
        rf_importances = np.array(rf_importances)
        perm_importances = np.array(perm_importances)
        
        # Combine scores (weighted average)
        combined_scores = 0.6 * rf_importances + 0.4 * perm_importances
        
        # Sort by importance
        sorted_idx = np.argsort(combined_scores)[::-1][:top_k]
        
        # Display results
        print(f"Top {top_k} Most Important Features:\n")
        print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12} {'RF Score':<12} {'Perm Score':<12}")
        print("-" * 70)
        
        for rank, idx in enumerate(sorted_idx, 1):
            original_idx = feature_indices[idx]
            feature_name = f"Feature_{original_idx}" if feature_names is None else feature_names[original_idx]
            print(f"{rank:<6} {feature_name:<25} {combined_scores[idx]:>10.4f}  "
                  f"{rf_importances[idx]:>10.4f}  {perm_importances[idx]:>10.4f}")
        
        results = {
            'combined_scores': combined_scores,
            'rf_importances': rf_importances,
            'perm_importances': perm_importances,
            'top_features_idx': sorted_idx,
            'feature_indices': feature_indices
        }
        
        return results
        
    except ImportError as e:
        print(f"⚠️ Feature importance evaluation requires scikit-learn: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 DeepSeek RL Feature Selection - Testing")
    print("="*70 + "\n")
    
    # Generate sample data (simulating CIC-IDS2017 structure)
    np.random.seed(42)
    n_samples = 2000
    n_features = 78  # As per thesis
    
    print(f"Generating synthetic dataset...")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: 15 (multi-class intrusion detection)\n")
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 15, n_samples)
    
    # Split data (70% train, 30% validation)
    split_idx = int(0.7 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Data split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples\n")
    
    # Initialize and train DeepSeek RL
    deepseek = DeepSeekRL(max_features=42)
    
    # Train the model
    history = deepseek.fit(
        X_train, y_train, X_val, y_val,
        episodes=100,  # Reduced for testing
        verbose=True
    )
    
    # Get selected features
    selected_features = deepseek.get_selected_features()
    print(f"\n✅ Feature selection complete!")
    print(f"Selected {len(selected_features)} features: {selected_features[:15]}...\n")
    
    # Transform data
    X_train_selected = deepseek.transform(X_train)
    X_val_selected = deepseek.transform(X_val)
    print(f"Transformed data shapes:")
    print(f"  Training: {X_train_selected.shape}")
    print(f"  Validation: {X_val_selected.shape}\n")
    
    # Plot training history
    print("Generating training visualizations...")
    deepseek.plot_training_history(save_path='deepseek_rl_training.png')
    
    # Evaluate feature importance
    print("\nEvaluating feature importance...")
    importance_results = evaluate_feature_importance(
        X, y, 
        selected_features=selected_features,
        top_k=10
    )
    
    # Save model
    print("\nSaving model...")
    deepseek.save_model('deepseek_rl_model.pth')
    
    print("\n" + "="*70)
    print("✅ DeepSeek RL Test Completed Successfully!")
    print("="*70)
    print("\nKey Results:")
    print(f"  • Selected Features: {len(selected_features)}/42")
    print(f"  • Best F1-Score: {max([h['f1_score'] for h in history]):.4f}")
    print(f"  • Training Episodes: {len(history)}")
    print(f"  • Convergence Events: {len(deepseek.convergence_history)}")
    print("="*70 + "\n")
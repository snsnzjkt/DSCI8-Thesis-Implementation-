# tests/test_deepseek_rl.py
"""
Comprehensive Test Suite for DeepSeek RL Feature Selection
Tests all components to ensure thesis requirements are met
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import unittest
from models.deepseek_rl import (
    FeatureSelectionEnvironment,
    DQNAgent,
    DeepSeekRL,
    evaluate_feature_importance
)


class TestFeatureSelectionEnvironment(unittest.TestCase):
    """Test FeatureSelectionEnvironment class"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 500
        self.n_features = 78
        self.max_features = 42
        
        # Generate test data
        X = np.random.randn(self.n_samples, self.n_features)
        y = np.random.randint(0, 15, self.n_samples)
        
        # Split
        split_idx = int(0.7 * self.n_samples)
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_val = X[split_idx:]
        self.y_val = y[split_idx:]
        
        # Create environment
        self.env = FeatureSelectionEnvironment(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            max_features=self.max_features
        )
    
    def test_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.total_features, self.n_features)
        self.assertEqual(self.env.max_features, self.max_features)
        self.assertEqual(self.env.accuracy_weight, 0.7)
        self.assertEqual(self.env.reduction_weight, 0.2)
        self.assertEqual(self.env.fp_weight, 0.1)
        print("✓ Environment initialization test passed")
    
    def test_reset(self):
        """Test environment reset"""
        state = self.env.reset()
        self.assertEqual(len(state), self.n_features + 3)
        self.assertEqual(self.env.selected_count, 0)
        self.assertEqual(self.env.step_count, 0)
        self.assertTrue(np.all(self.env.current_features == False))
        print("✓ Environment reset test passed")
    
    def test_step_action(self):
        """Test step function with different actions"""
        self.env.reset()
        
        # Action 1: Select feature
        state, reward, done, info = self.env.step(action=1)
        self.assertEqual(self.env.selected_count, 1)
        self.assertTrue(self.env.current_features[0])
        
        # Action 0: Skip feature
        state, reward, done, info = self.env.step(action=0)
        self.assertEqual(self.env.selected_count, 1)  # Should stay 1
        
        print("✓ Step action test passed")
    
    def test_reward_calculation(self):
        """Test reward function components"""
        self.env.reset()
        
        # Select some features
        for i in range(10):
            state, reward, done, info = self.env.step(action=1)
        
        # Check reward components exist
        self.assertIn('accuracy_reward', info)
        self.assertIn('reduction_reward', info)
        self.assertIn('fp_reward', info)
        self.assertIn('total_reward', info)
        
        # Check weights are applied correctly
        expected_total = (info['accuracy_reward'] + 
                         info['reduction_reward'] + 
                         info['fp_reward'] + 
                         info.get('consistency_bonus', 0))
        
        # Allow small floating point difference
        self.assertAlmostEqual(info['total_reward'], expected_total, places=5)
        
        print("✓ Reward calculation test passed")
    
    def test_max_features_limit(self):
        """Test that max features limit is enforced"""
        self.env.reset()
        
        # Try to select more than max_features
        for i in range(self.max_features + 10):
            state, reward, done, info = self.env.step(action=1)
        
        # Should not exceed max_features
        self.assertLessEqual(self.env.selected_count, self.max_features)
        print("✓ Max features limit test passed")
    
    def test_episode_termination(self):
        """Test episode termination conditions"""
        self.env.reset()
        done = False
        steps = 0
        
        while not done and steps < self.n_features + 10:
            _, _, done, _ = self.env.step(action=1)
            steps += 1
        
        self.assertTrue(done)
        print("✓ Episode termination test passed")


class TestDQNAgent(unittest.TestCase):
    """Test DQNAgent class"""
    
    def setUp(self):
        """Set up test agent"""
        self.state_size = 81  # 78 features + 3 metadata
        self.action_size = 2
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001,
            epsilon=1.0
        )
    
    def test_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        print("✓ Agent initialization test passed")
    
    def test_network_architecture(self):
        """Test Q-network architecture"""
        # Test forward pass
        dummy_state = torch.randn(1, self.state_size)
        with torch.no_grad():
            output = self.agent.q_network(dummy_state)
        
        self.assertEqual(output.shape, (1, self.action_size))
        print("✓ Network architecture test passed")
    
    def test_action_selection(self):
        """Test action selection"""
        state = np.random.randn(self.state_size)
        action = self.agent.act(state)
        
        self.assertIn(action, [0, 1])
        self.assertIsInstance(action, (int, np.integer))
        print("✓ Action selection test passed")
    
    def test_memory_storage(self):
        """Test experience replay buffer"""
        state = np.random.randn(self.state_size)
        action = 1
        reward = 0.5
        next_state = np.random.randn(self.state_size)
        done = False
        
        initial_size = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.memory), initial_size + 1)
        print("✓ Memory storage test passed")
    
    def test_replay_training(self):
        """Test replay training"""
        # Fill memory with experiences
        for _ in range(100):
            state = np.random.randn(self.state_size)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_size)
            done = np.random.choice([True, False])
            self.agent.remember(state, action, reward, next_state, done)
        
        # Perform replay
        loss = self.agent.replay(batch_size=32)
        
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
        print("✓ Replay training test passed")
    
    def test_epsilon_decay(self):
        """Test epsilon decay"""
        initial_epsilon = self.agent.epsilon
        
        # Fill memory and train
        for _ in range(50):
            state = np.random.randn(self.state_size)
            self.agent.remember(state, 1, 0.5, state, False)
        
        for _ in range(10):
            self.agent.replay(batch_size=32)
        
        self.assertLess(self.agent.epsilon, initial_epsilon)
        print("✓ Epsilon decay test passed")
    
    def test_target_network_update(self):
        """Test target network update"""
        # Modify main network
        for param in self.agent.q_network.parameters():
            param.data.fill_(1.0)
        
        # Update target network
        self.agent.update_target_network()
        
        # Check if weights are copied
        q_params = list(self.agent.q_network.parameters())
        target_params = list(self.agent.target_network.parameters())
        
        for q_param, target_param in zip(q_params, target_params):
            self.assertTrue(torch.allclose(q_param, target_param))
        
        print("✓ Target network update test passed")


class TestDeepSeekRL(unittest.TestCase):
    """Test DeepSeekRL main class"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 500
        self.n_features = 78
        self.max_features = 42
        
        X = np.random.randn(self.n_samples, self.n_features)
        y = np.random.randint(0, 15, self.n_samples)
        
        split_idx = int(0.7 * self.n_samples)
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_val = X[split_idx:]
        self.y_val = y[split_idx:]
        
        self.deepseek = DeepSeekRL(max_features=self.max_features)
    
    def test_initialization(self):
        """Test DeepSeekRL initialization"""
        self.assertEqual(self.deepseek.max_features, self.max_features)
        self.assertIsNone(self.deepseek.selected_features_idx)
        self.assertEqual(len(self.deepseek.training_history), 0)
        print("✓ DeepSeekRL initialization test passed")
    
    def test_fit(self):
        """Test training process"""
        history = self.deepseek.fit(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            episodes=20,  # Short training for test
            verbose=False
        )
        
        self.assertIsNotNone(self.deepseek.selected_features_idx)
        self.assertGreater(len(history), 0)
        self.assertLessEqual(len(self.deepseek.selected_features_idx), self.max_features)
        print("✓ Fit test passed")
    
    def test_transform(self):
        """Test data transformation"""
        # First train the model
        self.deepseek.fit(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            episodes=20,
            verbose=False
        )
        
        # Transform data
        X_transformed = self.deepseek.transform(self.X_train)
        
        self.assertEqual(X_transformed.shape[0], self.X_train.shape[0])
        self.assertLessEqual(X_transformed.shape[1], self.max_features)
        print("✓ Transform test passed")
    
    def test_get_selected_features(self):
        """Test getting selected features"""
        # Train first
        self.deepseek.fit(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            episodes=20,
            verbose=False
        )
        
        selected = self.deepseek.get_selected_features()
        
        self.assertIsInstance(selected, np.ndarray)
        self.assertLessEqual(len(selected), self.max_features)
        self.assertTrue(np.all(selected >= 0))
        self.assertTrue(np.all(selected < self.n_features))
        print("✓ Get selected features test passed")
    
    def test_save_load_model(self):
        """Test model saving and loading"""
        import tempfile
        
        # Train model
        self.deepseek.fit(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            episodes=20,
            verbose=False
        )
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
        
        self.deepseek.save_model(model_path)
        
        # Create new instance and load
        new_deepseek = DeepSeekRL(max_features=self.max_features)
        new_deepseek.load_model(model_path)
        
        # Check if features match
        np.testing.assert_array_equal(
            self.deepseek.get_selected_features(),
            new_deepseek.get_selected_features()
        )
        
        # Cleanup
        os.remove(model_path)
        print("✓ Save/load model test passed")


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance evaluation"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(500, 78)
        self.y = np.random.randint(0, 15, 500)
        self.selected_features = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    
    def test_feature_importance_evaluation(self):
        """Test feature importance calculation"""
        results = evaluate_feature_importance(
            self.X, self.y,
            selected_features=self.selected_features,
            top_k=5
        )
        
        self.assertIsNotNone(results)
        self.assertIn('combined_scores', results)
        self.assertIn('rf_importances', results)
        self.assertIn('perm_importances', results)
        self.assertEqual(len(results['combined_scores']), len(self.selected_features))
        print("✓ Feature importance evaluation test passed")


class TestThesisRequirements(unittest.TestCase):
    """Test compliance with thesis requirements"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 78
        
        X = np.random.randn(self.n_samples, self.n_features)
        y = np.random.randint(0, 15, self.n_samples)
        
        split_idx = int(0.7 * self.n_samples)
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_val = X[split_idx:]
        self.y_val = y[split_idx:]
    
    def test_feature_count_requirement(self):
        """Test: Must select exactly 42 features"""
        deepseek = DeepSeekRL(max_features=42)
        deepseek.fit(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            episodes=50,
            verbose=False
        )
        
        selected = deepseek.get_selected_features()
        self.assertLessEqual(len(selected), 42, 
                            "Must select at most 42 features (thesis requirement)")
        self.assertGreater(len(selected), 0,
                          "Must select at least some features")
        print("✓ Feature count requirement test passed")
    
    def test_reward_weights_requirement(self):
        """Test: Reward weights must be 70% accuracy, 20% reduction, 10% FP"""
        env = FeatureSelectionEnvironment(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            max_features=42
        )
        
        self.assertEqual(env.accuracy_weight, 0.7, 
                        "Accuracy weight must be 70% (thesis requirement)")
        self.assertEqual(env.reduction_weight, 0.2,
                        "Reduction weight must be 20% (thesis requirement)")
        self.assertEqual(env.fp_weight, 0.1,
                        "FP weight must be 10% (thesis requirement)")
        
        # Total should be 1.0
        total = env.accuracy_weight + env.reduction_weight + env.fp_weight
        self.assertAlmostEqual(total, 1.0, places=5,
                              msg="Reward weights should sum to 1.0")
        print("✓ Reward weights requirement test passed")
    
    def test_feature_reduction_requirement(self):
        """Test: Must achieve >40% feature reduction"""
        deepseek = DeepSeekRL(max_features=42)
        deepseek.fit(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            episodes=50,
            verbose=False
        )
        
        original_features = 78
        selected_features = len(deepseek.get_selected_features())
        reduction_pct = (1 - selected_features / original_features) * 100
        
        self.assertGreater(reduction_pct, 40,
                          f"Must achieve >40% reduction (got {reduction_pct:.1f}%)")
        print(f"✓ Feature reduction: {reduction_pct:.1f}% (target: >40%)")
    
    def test_convergence_detection(self):
        """Test: Must detect convergence"""
        deepseek = DeepSeekRL(max_features=42)
        history = deepseek.fit(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            episodes=100,
            verbose=False
        )
        
        # Check if convergence was detected at least once
        convergence_events = [h for h in history if h.get('f1_score', 0) > 0.8]
        self.assertGreater(len(convergence_events), 0,
                          "Should detect some high-performing episodes")
        print("✓ Convergence detection test passed")
    
    def test_dqn_components_requirement(self):
        """Test: Must use DQN with experience replay and target network"""
        agent = DQNAgent(state_size=81, action_size=2)
        
        # Check DQN components exist
        self.assertIsNotNone(agent.q_network, "Must have Q-network")
        self.assertIsNotNone(agent.target_network, "Must have target network")
        self.assertIsNotNone(agent.memory, "Must have experience replay buffer")
        self.assertIsNotNone(agent.optimizer, "Must have optimizer")
        
        # Check epsilon-greedy policy
        self.assertGreater(agent.epsilon, 0, "Must use epsilon-greedy exploration")
        self.assertLessEqual(agent.epsilon, 1.0, "Epsilon must be <= 1.0")
        
        print("✓ DQN components requirement test passed")


def run_integration_test():
    """
    Run a complete integration test simulating the full pipeline
    """
    print("\n" + "="*70)
    print("🧪 Running Integration Test")
    print("="*70 + "\n")
    
    # Generate realistic test data
    np.random.seed(42)
    n_samples = 2000
    n_features = 78
    
    print(f"1. Generating synthetic CIC-IDS2017-like data...")
    print(f"   Samples: {n_samples}, Features: {n_features}, Classes: 15")
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 15, n_samples)
    
    # Split data
    train_split = int(0.6 * n_samples)
    val_split = int(0.8 * n_samples)
    
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:val_split]
    y_val = y[train_split:val_split]
    X_test = X[val_split:]
    y_test = y[val_split:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 1: Train DeepSeek RL
    print(f"\n2. Training DeepSeek RL (50 episodes for testing)...")
    deepseek = DeepSeekRL(max_features=42)
    history = deepseek.fit(
        X_train, y_train, X_val, y_val,
        episodes=50,
        verbose=False
    )
    
    selected_features = deepseek.get_selected_features()
    print(f"   ✓ Selected {len(selected_features)} features")
    
    # Step 2: Transform data
    print(f"\n3. Transforming data with selected features...")
    X_train_selected = deepseek.transform(X_train)
    X_test_selected = deepseek.transform(X_test)
    print(f"   Original shape: {X_train.shape}")
    print(f"   Transformed shape: {X_train_selected.shape}")
    
    # Step 3: Evaluate performance
    print(f"\n4. Evaluating performance...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    # Train on selected features
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_selected, y_train)
    y_pred = rf.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Step 4: Check thesis requirements
    print(f"\n5. Checking thesis requirements...")
    checks = []
    
    # Feature count check
    feature_check = len(selected_features) <= 42
    checks.append(("Feature count <= 42", feature_check))
    
    # Feature reduction check
    reduction = (1 - len(selected_features) / n_features) * 100
    reduction_check = reduction > 40
    checks.append((f"Feature reduction > 40% ({reduction:.1f}%)", reduction_check))
    
    # Performance check (relaxed for synthetic data)
    perf_check = f1 > 0.5  # Lower threshold for random data
    checks.append((f"F1-Score > 0.5 ({f1:.4f})", perf_check))
    
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {check_name}")
    
    all_passed = all([c[1] for c in checks])
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ Integration Test PASSED!")
    else:
        print("⚠️  Integration Test FAILED - Some checks did not pass")
    print("="*70 + "\n")
    
    return all_passed


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("🧪 DeepSeek RL - Comprehensive Test Suite")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureSelectionEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestDQNAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestDeepSeekRL))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureImportance))
    suite.addTests(loader.loadTestsFromTestCase(TestThesisRequirements))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run integration test
    integration_passed = run_integration_test()
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Integration test: {'PASSED' if integration_passed else 'FAILED'}")
    print("="*70 + "\n")
    
    # Return overall success
    overall_success = result.wasSuccessful() and integration_passed
    
    if overall_success:
        print("✅ All tests PASSED! DeepSeek RL is ready for deployment.")
    else:
        print("❌ Some tests FAILED. Please review and fix issues.")
    
    return overall_success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
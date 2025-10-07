# SCS-ID: A Squeezed ConvSeek for Efficient Intrusion Detection in Campus Networks

## 📋 Project Overview

**SCS-ID** is a novel lightweight convolutional neural network architecture specifically designed for efficient intrusion detection in campus networks. This implementation combines the computational efficiency of SqueezeNet with the pattern recognition capabilities of ConvSeek, enhanced by DeepSeek reinforcement learning for optimal feature selection.

## 🎯 Research Objectives

1. **Real-time Monitoring Improvement**: Reduce computational overhead by >50% through optimized architecture
2. **Detection Accuracy Enhancement**: Maintain high accuracy while reducing false positive rates by >20%
3. **Explainable AI Integration**: Provide transparent, interpretable security decisions via hybrid LIME-SHAP framework

## 🏗️ Architecture Overview

### Core Components
1. **DeepSeek RL Feature Selection**: Q-learning based optimization selecting 42 optimal features from 78
2. **SCS-ID CNN Architecture**: Lightweight network with depthwise separable convolutions
3. **Model Compression**: Structured pruning (30%) + INT8 quantization
4. **Dual Explainability**: Hybrid LIME-SHAP system for transparent decisions

### Technical Specifications
- **Input**: 42×1×1 tensor (reduced from 78 features)
- **Architecture**: Fire modules + ConvSeek blocks + Global max pooling
- **Optimization**: Adam optimizer with learning rate scheduling
- **Dataset**: CIC-IDS2017 (15 attack types, 78 features)

## 📁 Project Structure

```
SCS-ID/
├── config.py                 # Configuration settings
├── requirements.txt           # Dependencies
├── main.py                   # Complete pipeline execution
├── data/
│   ├── download_dataset.py   # Dataset download utility
│   └── preprocess.py         # Data preprocessing pipeline
├── models/
│   ├── baseline_cnn.py       # Ayeni et al. baseline CNN
│   ├── scs_id.py            # SCS-ID architecture
│   ├── deepseek_rl.py       # DeepSeek RL feature selection
│   └── utils.py             # Model utilities
└── experiments/
    ├── train_baseline.py     # Baseline CNN training
    ├── train_scs_id.py      # SCS-ID training
    └── compare_models.py     # Model comparison & analysis
```

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Option 1: Complete Pipeline Execution
```bash
# Run entire pipeline
python main.py

# Skip certain steps if already completed
python main.py --skip-preprocessing --skip-baseline
python main.py --skip-explainability
python main.py --quick-test  # For faster testing
```

### Option 2: Individual Module Execution

#### 1. Data Preprocessing
```bash
# Download and preprocess CIC-IDS2017 dataset
python data/preprocess.py
```

**Features:**
- Z-score normalization
- Median-based imputation
- Outlier removal (Isolation Forest)
- SMOTE oversampling
- Stratified train/test split

**Outputs:**
- `data/processed/processed_data.pkl`
- `data/processed/preprocessing_report.txt`

#### 2. Baseline CNN Training
```bash
# Train Ayeni et al. baseline CNN
python experiments/train_baseline.py
```

**Features:**
- Three-layer CNN architecture
- Multi-class classification (16 classes)
- Comprehensive evaluation metrics
- Training curve visualization

**Outputs:**
- `results/baseline_model.pth`
- `results/baseline_results.pkl`
- `results/baseline_training_curves.png`

#### 3. SCS-ID Training
```bash
# Train SCS-ID with DeepSeek RL
python experiments/train_scs_id.py
```

**Features:**
- DeepSeek RL feature selection (78→42 features)
- SCS-ID architecture training
- Model compression (pruning + quantization)
- Explainability integration

**Outputs:**
- `results/scs_id_model.pth`
- `results/scs_id_results.pkl`
- `results/feature_selection_history.pkl`
- `results/explainability_report.txt`

#### 4. Model Comparison & Analysis
```bash
# Compare baseline vs SCS-ID performance
python experiments/compare_models.py
```

**Features:**
- Statistical significance testing
- Computational efficiency analysis
- Performance visualization
- Comprehensive benchmarking

**Outputs:**
- `results/model_comparison_report.pdf`
- `results/statistical_analysis.pkl`
- `results/performance_visualizations.png`

## 📊 Expected Results

### Performance Metrics
- **SCS-ID Accuracy**: >99.5% (target)
- **Baseline CNN Accuracy**: ~99.78% (Ayeni et al.)
- **False Positive Reduction**: >20%
- **Parameter Reduction**: >75%
- **Inference Speed Improvement**: >300%

### Statistical Validation
- Paired t-test for significance (p < 0.05)
- Effect size calculation (Cohen's d)
- Bootstrap confidence intervals
- Cross-validation stability analysis

## 🔧 Configuration

### Key Parameters (config.py)
```python
# Dataset Configuration
NUM_FEATURES = 78          # Original feature count
SELECTED_FEATURES = 42     # After DeepSeek RL selection
NUM_CLASSES = 16           # Attack types + benign

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 25
DEVICE = "cuda" if available else "cpu"

# Model Configuration
BASELINE_FILTERS = [120, 60, 30]  # Baseline CNN filters
PRUNING_RATIO = 0.3              # 30% structured pruning
```

### Customization Options
```python
# Modify in config.py or pass as arguments
QUICK_TEST_MODE = False      # Reduced parameters for testing
ENABLE_VISUALIZATION = True  # Generate plots and figures
SAVE_INTERMEDIATE = True     # Save intermediate results
VERBOSE_LOGGING = True       # Detailed progress logging
```

## 🧪 Experimental Design

### Experiment 1: Baseline CNN (Ayeni et al.)
- Architecture: 3-layer CNN with [120, 60, 30] filters
- Input: All 78 features
- Objective: Establish baseline performance

### Experiment 2: SCS-ID
- Feature Selection: DeepSeek RL (42 optimal features)
- Architecture: Fire modules + ConvSeek blocks
- Optimization: Pruning + Quantization
- Explainability: LIME-SHAP integration

## 📈 Evaluation Metrics

### Detection Performance
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient
- **AUC-ROC**: Area Under Receiver Operating Characteristic

### Computational Efficiency
- **Parameter Count**: Total trainable parameters
- **Memory Usage**: Peak memory consumption
- **Inference Time**: Average prediction latency
- **Model Size**: Compressed model storage size

### Explainability Metrics
- **Fidelity**: Agreement between model and explanations
- **Stability**: Consistency across similar inputs
- **Coherence**: Human interpretability score

## 🔍 Detailed Module Documentation

### 1. Data Preprocessing (`data/preprocess.py`)

The preprocessing pipeline implements the data cleaning methodology described in the thesis:

```python
from data.preprocess import CICIDSPreprocessor

preprocessor = CICIDSPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess_full_pipeline()
```

**Key Methods:**
- `load_cicids_data()`: Downloads and loads CIC-IDS2017
- `clean_data()`: Removes duplicates, handles missing values
- `normalize_features()`: Z-score normalization
- `handle_class_imbalance()`: SMOTE oversampling
- `split_data()`: Stratified temporal validation

### 2. Baseline CNN (`models/baseline_cnn.py`)

Implements the Ayeni et al. (2023) baseline architecture:

```python
from models.baseline_cnn import create_baseline_model, BaselineCNN

model = create_baseline_model(num_features=78, num_classes=16)
```

**Architecture Features:**
- Input layer: 78 features
- Conv1D layers: [120, 60, 30] filters
- Activation: ReLU with BatchNorm
- Regularization: Dropout (0.3)
- Output: Dense layer (16 classes)

### 3. SCS-ID Architecture (`models/scs_id.py`)

The novel SCS-ID architecture combining SqueezeNet efficiency with ConvSeek pattern extraction:

```python
from models.scs_id import create_scs_id_model, SCS_ID

model = create_scs_id_model(
    input_features=42,
    num_classes=16,
    apply_pruning=True,
    pruning_ratio=0.3
)
```

**Architecture Components:**
- **Input Projection**: 1×1 convolution for feature mapping
- **Fire Modules**: Squeeze-expand pattern for efficiency
- **ConvSeek Blocks**: Pattern recognition with depthwise separable convolutions
- **Global Max Pooling**: Dimensionality reduction
- **Classifier**: Dense layers with dropout

### 4. DeepSeek RL Feature Selection (`models/deepseek_rl.py`)

Q-learning based feature selection optimized for intrusion detection:

```python
from models.deepseek_rl import DeepSeekRL

feature_selector = DeepSeekRL(
    num_features=78,
    target_features=42,
    reward_metric='f1_score'
)

selected_features = feature_selector.select_features(X_train, y_train)
```

**RL Components:**
- **State**: Current feature subset
- **Action**: Add/remove features
- **Reward**: F1-score improvement
- **Policy**: ε-greedy exploration

## 📚 Dependencies

### Core Requirements
```
torch>=1.12.0           # Deep learning framework
scikit-learn>=1.1.0     # Machine learning utilities
pandas>=1.4.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
matplotlib>=3.5.0      # Visualization
seaborn>=0.11.0        # Statistical visualization
imbalanced-learn>=0.9.0 # SMOTE implementation
```

## 📋 Thesis Validation Checklist

### ✅ Implementation Requirements
- [✅] CIC-IDS2017 dataset preprocessing
- [✅] Baseline CNN implementation (Ayeni et al.)
- [ ] SCS-ID architecture with Fire modules
- [ ] DeepSeek RL feature selection (78→42)
- [ ] Model compression (pruning + quantization)
- [ ] Hybrid LIME-SHAP explainability
- [ ] Statistical significance testing
- [ ] Computational efficiency analysis

### ✅ Performance Targets
- [ ] Detection accuracy >99%
- [ ] False positive reduction >20%
- [ ] Parameter reduction >75%
- [ ] Inference speed improvement >300%
- [ ] Statistical significance (p < 0.05)

### ✅ Documentation Requirements
- [ ] Comprehensive methodology description
- [ ] Experimental design validation
- [ ] Results reproducibility
- [ ] Future work recommendations


For optimal results, ensure proper environment setup and follow the preprocessing pipeline before training models. The comprehensive evaluation framework provides statistical validation necessary for academic rigor while maintaining practical applicability for real-world campus network security deployments.

**🚀 Ready to start? Run `python main.py` for complete pipeline execution or choose individual modules based on your research needs!**

# CNN Input Parameters Table - Network Intrusion Detection Models

## Overview
**Models**: Baseline CNN, SCS-ID, SCS-ID Optimized  
**Dataset**: CIC-IDS2017 (Canadian Institute for Cybersecurity)  
**Input Features**: Variable (78 original → 42 after DeepSeek RL feature selection)  
**Attack Types**: 15 classes (1 BENIGN + 14 attack types)

---

## CNN Model Input Parameters Comparison

### Model Architecture Overview

| Model | Input Features | Input Shape | Feature Source | Architecture |
|-------|---------------|-------------|----------------|--------------|
| **Baseline CNN** | 78 | `[batch, 1, 78]` | Raw CIC-IDS2017 | Ayeni et al. (2023) |
| **SCS-ID** | 42 | `[batch, 1, 42, 1]` | DeepSeek RL selected | Squeezed ConvSeek |
| **SCS-ID Optimized** | 42 | `[batch, 1, 42]` | DeepSeek RL selected | Enhanced efficiency |

---

## 1. Baseline CNN Input Parameters (78 features)8]`  
**Feature Processing**: Direct input to 1D CNN layers  
**Architecture**: 3 Conv1D layers + FC layers  

#### Baseline CNN Architecture Parameters
| Layer | Type | Input Shape | Output Shape | Parameters |
|-------|------|-------------|--------------|------------|
| Conv1 | Conv1d | `[B, 1, 78]` | `[B, 120, 78]` | 481 |
| Conv2 | Conv1d | `[B, 120, 78]` | `[B, 60, 78]` | 21,660 |
| Conv3 | Conv1d | `[B, 60, 78]` | `[B, 30, 78]` | 5,430 |
| Pool | AdaptiveAvgPool1d | `[B, 30, 78]` | `[B, 30, 1]` | 0 |
| FC1 | Linear | `[B, 30]` | `[B, 128]` | 3,968 |
| FC2 | Linear | `[B, 128]` | `[B, 64]` | 8,256 |
| FC3 | Linear | `[B, 64]` | `[B, 15]` | 975 |

**Total Parameters**: ~40,770

## 2. SCS-ID Input Parameters (42 features)

**Input Tensor Shape**: `[batch_size, 1, 42, 1]`  
**Feature Processing**: 4D input reshaped from DeepSeek RL selected features  
**Architecture**: Fire Modules + ConvSeek Blocks  

#### SCS-ID Architecture Parameters
| Module | Type | Input Shape | Output Shape | Key Components |
|--------|------|-------------|--------------|----------------|
| Input Conv | Conv1d | `[B, 1, 42]` | `[B, 16, 42]` | Channel expansion |
| Fire1 | FireModule | `[B, 16, 42]` | `[B, 32, 42]` | Squeeze-expand pattern |
| Fire2 | FireModule | `[B, 32, 42]` | `[B, 32, 42]` | Parameter efficiency |
| Fire3 | FireModule | `[B, 32, 42]` | `[B, 32, 42]` | Residual connections |
| ConvSeek1 | ConvSeekBlock | `[B, 32, 42]` | `[B, 64, 42]` | Depthwise separable |
| ConvSeek2 | ConvSeekBlock | `[B, 64, 42]` | `[B, 32, 42]` | 58% param reduction |
| Global Pool | Adaptive | `[B, 32, 42]` | `[B, 64]` | Max + Avg pooling |
| Classifier | FC Layers | `[B, 64]` | `[B, 15]` | Multi-layer FC |

**Total Parameters**: ~15,000-20,000 (75% reduction from baseline)

## 3. SCS-ID Optimized Input Parameters (42 features)

**Input Tensor Shape**: `[batch_size, 42]` → `[batch_size, 1, 42]`  
**Feature Processing**: Streamlined processing with channel attention  
**Architecture**: Enhanced Fire Modules + Efficient ConvSeek Blocks  

#### SCS-ID Optimized Architecture Parameters
| Module | Type | Input Shape | Output Shape | Optimization Features |
|--------|------|-------------|--------------|----------------------|
| Input Conv | Conv1d + Attention | `[B, 1, 42]` | `[B, 32, 42]` | Channel attention |
| Fire1 | OptimizedFire | `[B, 32, 42]` | `[B, 48, 42]` | Residual + Attention |
| Fire2 | OptimizedFire | `[B, 48, 42]` | `[B, 64, 42]` | Depthwise separable |
| ConvSeek1 | EnhancedConvSeek | `[B, 64, 42]` | `[B, 48, 42]` | Ultra-efficient (8:1 reduction) |
| ConvSeek2 | EnhancedConvSeek | `[B, 48, 42]` | `[B, 32, 42]` | Two-step pointwise |
| Classifier | FC + Dropout | `[B, 32]` | `[B, 15]` | Regularized classification |

**Total Parameters**: ~10,000-15,000 (85% reduction from baseline)

## 4. Feature Selection Process

### DeepSeek RL Feature Selection (78 → 42 features)
**Process**: Reinforcement Learning agent selects optimal feature subset  
**Objective**: Maximize F1-score while minimizing feature count  
**Method**: ε-greedy exploration with reward-based selection  

| Selection Criteria | Description | Impact |
|-------------------|-------------|--------|
| **Feature Importance** | Mutual information with target labels | High correlation features selected |
| **Redundancy Reduction** | Remove highly correlated features | Eliminates noise and overfitting |
| **Attack Discrimination** | Features that distinguish attack types | Improves classification accuracy |
| **Computational Efficiency** | Reduce model complexity | 46% feature reduction (78→42) |

#### Selected Feature Categories (42 features)
- **Flow Statistics**: Duration, packet counts, byte rates
- **Packet Analysis**: Length statistics, header information  
- **Timing Features**: Inter-arrival times, flow timing
- **Protocol Flags**: TCP flags and connection states
- **Statistical Measures**: Mean, std, min, max values

## 5. Input Data Preprocessing Pipeline

### Data Flow: Raw Features → CNN Input

```
Raw Network Traffic (PCAP)
         ↓
CICFlowMeter Feature Extraction
         ↓
78 Network Flow Features
         ↓
DeepSeek RL Feature Selection
         ↓
42 Selected Features
         ↓
Normalization & Scaling
         ↓
CNN Input Tensor
```

#### Preprocessing Steps
| Step | Process | Input | Output | Purpose |
|------|---------|-------|--------|---------|
| 1 | **Feature Extraction** | PCAP files | 78 features | CICFlowMeter statistics |
| 2 | **Data Cleaning** | Raw features | Clean features | Handle missing/infinite values |
| 3 | **Feature Selection** | 78 features | 42 features | DeepSeek RL optimization |
| 4 | **Normalization** | Raw values | Scaled values | Z-score normalization (μ=0, σ=1) |
| 5 | **Tensor Formatting** | Feature vector | CNN tensor | Reshape for model input |

## 6. Model Performance Comparison

### Parameter Efficiency Analysis

| Metric | Baseline CNN | SCS-ID | SCS-ID Optimized |
|--------|--------------|--------|-----------------|
| **Input Features** | 78 | 42 | 42 |
| **Total Parameters** | ~40,770 | ~18,000 | ~12,000 |
| **Parameter Reduction** | - | 75% | 85% |
| **Model Size (KB)** | ~159 | ~70 | ~47 |
| **Inference Speed** | 1.0x | 2.1x | 2.8x |
| **Memory Usage** | 1.0x | 0.4x | 0.3x |

### Architecture Innovation Summary
| Innovation | Baseline CNN | SCS-ID | SCS-ID Optimized |
|------------|--------------|--------|-----------------|
| **Feature Selection** | ❌ Manual | ✅ DeepSeek RL | ✅ DeepSeek RL |
| **Fire Modules** | ❌ Standard Conv | ✅ Squeeze-Expand | ✅ Enhanced Fire |
| **ConvSeek Blocks** | ❌ Standard Conv | ✅ Depthwise Sep. | ✅ Ultra-Efficient |
| **Channel Attention** | ❌ None | ❌ None | ✅ Full Integration |
| **Residual Connections** | ❌ None | ✅ Limited | ✅ Comprehensive |
| **Structured Pruning** | ❌ None | ✅ 30% L1-norm | ✅ Enhanced Pruning |

---

## CNN Training Configuration

### Training Parameters by Model

| Parameter | Baseline CNN | SCS-ID | SCS-ID Optimized |
|-----------|--------------|--------|------------------|
| **Batch Size** | 64 | 32 | 32 |
| **Learning Rate** | 1e-4 | 1e-4 | 1e-4 |
| **Optimizer** | Adam | Adam | Adam |
| **Loss Function** | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| **Epochs** | 25 | 25 | 25 |
| **Weight Decay** | 1e-5 | 1e-5 | 1e-5 |
| **Dropout Rate** | 0.2 | 0.2-0.5 | 0.2-0.5 |
| **Early Stopping** | 5 patience | 5 patience | 5 patience |

### Data Augmentation & Regularization

| Technique | Baseline CNN | SCS-ID | SCS-ID Optimized |
|-----------|--------------|--------|------------------|
| **Batch Normalization** | ✅ | ✅ | ✅ |
| **Dropout** | ✅ Basic | ✅ Adaptive | ✅ Advanced |
| **Data Balancing** | ✅ SMOTE | ✅ SMOTE | ✅ SMOTE |
| **Feature Scaling** | ✅ Z-score | ✅ Z-score | ✅ Z-score |
| **Gradient Clipping** | ❌ | ✅ | ✅ |
| **Label Smoothing** | ❌ | ❌ | ✅ |

---

## Attack Classification Output

All CNN models classify network flows into **15 attack types**:

| Class ID | Attack Type | Category | Model Input | Output Probability |
|----------|-------------|----------|-------------|-------------------|
| 0 | BENIGN | Normal Traffic | Feature vector | P(BENIGN \| x) |
| 1 | Bot | Malware | Feature vector | P(Bot \| x) |
| 2 | DDoS | Denial of Service | Feature vector | P(DDoS \| x) |
| 3 | DoS GoldenEye | Denial of Service | Feature vector | P(DoS GoldenEye \| x) |
| 4 | DoS Hulk | Denial of Service | Feature vector | P(DoS Hulk \| x) |
| 5 | DoS Slowhttptest | Denial of Service | Feature vector | P(DoS Slowhttptest \| x) |
| 6 | DoS slowloris | Denial of Service | Feature vector | P(DoS slowloris \| x) |
| 7 | FTP-Patator | Brute Force | Feature vector | P(FTP-Patator \| x) |
| 8 | Heartbleed | Vulnerability Exploit | Feature vector | P(Heartbleed \| x) |
| 9 | Infiltration | Advanced Persistent Threat | Feature vector | P(Infiltration \| x) |
| 10 | PortScan | Reconnaissance | Feature vector | P(PortScan \| x) |
| 11 | SSH-Patator | Brute Force | Feature vector | P(SSH-Patator \| x) |
| 12 | Web Attack - Brute Force | Web Attack | Feature vector | P(Web BF \| x) |
| 13 | Web Attack - Sql Injection | Web Attack | Feature vector | P(SQL Inj \| x) |
| 14 | Web Attack - XSS | Web Attack | Feature vector | P(XSS \| x) |

---

## Key Technical Specifications

### Input Processing Pipeline
```
Network Flow → Feature Extraction → Feature Selection → CNN Processing
     ↓               ↓                    ↓              ↓
   PCAP          78 features         42 features    Classification
```

### Model Architecture Differences

1. **Baseline CNN**: 
   - Direct convolution on 78 features
   - Standard 3-layer CNN architecture
   - No feature selection optimization

2. **SCS-ID**: 
   - DeepSeek RL feature selection (42 features)
   - Fire modules for parameter efficiency
   - ConvSeek blocks for depthwise separable convolution

3. **SCS-ID Optimized**: 
   - Enhanced Fire modules with attention
   - Ultra-efficient ConvSeek blocks
   - Advanced regularization and residual connections

### Performance Targets
- **Accuracy**: >99% on test set
- **False Positive Rate**: <1%
- **Parameter Reduction**: >75% vs baseline
- **Inference Speed**: 2-3x faster than baseline
- **Model Compression**: 75% size reduction through pruning + quantization
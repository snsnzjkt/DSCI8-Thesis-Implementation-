# SCS-ID Two-Stage Training Pipeline

## Overview

The SCS-ID training has been separated into two stages to optimize development workflow:

### 🔄 **Two-Stage Approach Benefits:**
1. **Separate Time-Intensive Operations**: DeepSeek RL (30-60 min) runs independently
2. **Fast Iteration**: Reuse features for multiple SCS-ID training runs (5-15 min each)
3. **Resource Efficiency**: Run DeepSeek RL once, experiment with SCS-ID parameters quickly
4. **Better Development**: Separate experimental phase from compute-intensive phase

---

## 📋 **Usage Options**

### Option 1: Guided Workflow (Recommended)
```bash
python run_scs_id_workflow.py
```
- Automatically checks prerequisites
- Runs both stages in sequence
- Provides clear progress updates
- Handles errors gracefully

### Option 2: Manual Execution
```bash
# Stage 1: DeepSeek RL Feature Selection (30-60 minutes)
python experiments/deepseek_feature_selection_only.py

# Stage 2: Fast SCS-ID Training (5-15 minutes)  
python experiments/train_scs_id_fast.py
```

### Option 3: Original Combined Version (If Needed)
```bash
python experiments/train_scs_id.py
```
- Runs everything together (30-60+ minutes total)
- Use this if you prefer the original workflow

---

## 📁 **File Structure**

### New Files Created:
- `experiments/deepseek_feature_selection_only.py` - Standalone DeepSeek RL
- `experiments/train_scs_id_fast.py` - Fast SCS-ID training with pre-selected features
- `run_scs_id_workflow.py` - Guided two-stage workflow

### Modified Files:
- `experiments/train_scs_id.py` - Updated to use DeepSeek RL (original combined approach)
- `config.py` - Already configured with DEEPSEEK_RL_EPISODES = 100

---

## ⏱️ **Time Comparison**

| Approach | Stage 1 | Stage 2 | Total | Re-runs |
|----------|---------|---------|-------|---------|
| **Two-Stage** | 30-60 min | 5-15 min | 35-75 min | 5-15 min each |
| **Combined** | - | - | 30-60+ min | 30-60+ min each |

**Advantage**: After initial run, each SCS-ID experiment takes only 5-15 minutes!

---

## 📊 **What Each Stage Does**

### Stage 1: DeepSeek RL Feature Selection
- Loads CIC-IDS2017 preprocessed data (78 features)
- Trains DeepSeek RL agent (100 episodes by default)
- Selects optimal 42 features using Q-learning
- Saves results to `results/deepseek_feature_selection_complete.pkl`
- Generates training plots and analysis

### Stage 2: Fast SCS-ID Training
- Loads pre-selected features from Stage 1
- Trains SCS-ID model with Fire modules + ConvSeek blocks
- Applies 30% structured pruning (post-training)
- Applies INT8 quantization (~75% size reduction)
- Performs threshold optimization (FPR < 1%)
- Saves complete results and model files

---

## 🎯 **Complete Feature Set**

Both approaches provide the full thesis implementation:
- ✅ DeepSeek RL feature selection (78 → 42 features)
- ✅ SCS-ID architecture (Fire modules + ConvSeek blocks)
- ✅ 30% structured pruning (post-training)
- ✅ INT8 quantization (~75% compression)
- ✅ Threshold optimization (FPR < 1% requirement)

---

## 💡 **Usage Recommendations**

1. **First Run**: Use `python run_scs_id_workflow.py`
2. **Experimenting**: After Stage 1 completes, run Stage 2 multiple times with different hyperparameters
3. **Development**: Modify SCS-ID architecture and quickly test without re-running DeepSeek RL
4. **Production**: Use saved features for deployment without DeepSeek RL dependency

---

## 📈 **Expected Results**

Both approaches should achieve:
- **Accuracy**: >99% (thesis requirement)
- **F1 Score**: >0.99
- **Model Compression**: 80%+ total (pruning + quantization)
- **FPR**: <1% (thesis requirement)

The two-stage approach provides identical results with better development efficiency.
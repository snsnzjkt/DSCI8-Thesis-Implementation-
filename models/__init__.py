"""
SCS-ID Neural Network Models and Optimization
"""

from .scs_id import SCS_ID, create_scs_id_model, fix_batchnorm_after_pruning
from .baseline_cnn import BaselineCNN

try:
    from .deepseek_rl import DeepSeekRL
except ImportError:
    DeepSeekRL = None

try:
    from .lime_shap_explainer import HybridLIMESHAPExplainer
except ImportError:
    HybridLIMESHAPExplainer = None

__all__ = [
    'SCS_ID',
    'create_scs_id_model',
    'fix_batchnorm_after_pruning',
    'BaselineCNN',
    'DeepSeekRL',
    'HybridLIMESHAPExplainer'
]




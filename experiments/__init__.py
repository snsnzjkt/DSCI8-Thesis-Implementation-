"""
Training and evaluation experiments
"""

try:
    from .train_baseline import BaselineTrainer
    from .train_scs_id import SCSIDTrainer
    from .compare_models import ModelComparator
except ImportError:
    pass

__all__ = []
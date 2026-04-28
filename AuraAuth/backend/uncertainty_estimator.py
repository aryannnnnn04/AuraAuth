"""
Uncertainty Estimator Module - backward-compatibility shim.

The canonical implementation lives in backend.core.uncertainty_estimator.
This module re-exports the core class so that any legacy imports still work.
"""

from backend.core.uncertainty_estimator import UncertaintyEstimator  # noqa: F401


__all__ = ["UncertaintyEstimator"]

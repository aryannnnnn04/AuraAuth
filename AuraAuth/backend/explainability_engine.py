"""
Explainability Engine Module - backward-compatibility shim.

The canonical implementation lives in backend.core.explainability_engine.
This module re-exports the core class so that any legacy imports still work.
"""

from backend.core.explainability_engine import ExplainabilityEngine  # noqa: F401


__all__ = ["ExplainabilityEngine"]

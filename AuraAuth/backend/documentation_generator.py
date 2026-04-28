"""
Documentation Generator Module — backward-compatibility shim.

The canonical implementation lives in backend.core.documentation_generator.
This module re-exports the core class so that any legacy imports still work.
"""

from backend.core.documentation_generator import DocumentationGenerator  # noqa: F401


__all__ = ["DocumentationGenerator"]

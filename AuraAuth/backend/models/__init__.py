"""
Models module: Centralized location for trained models and model wrappers.

This module stores:
- Serialized trained models
- Model metadata
- Preprocessing pipelines
- Fitted transformers (scalers, encoders, etc.)
"""

# Models will be stored here during training
# Format: pickle or joblib for sklearn models, native for XGB/LGB

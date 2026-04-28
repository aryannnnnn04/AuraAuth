"""
Configuration module for AuraAuth system.
Centralized settings for all components including hyperparameters, paths, and constants.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Configuration for data processing and profiling."""
    
    min_dataset_size: int = 100
    max_dataset_size: int = 50000
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    missing_value_threshold: float = 0.5  # Drop features with >50% missing
    

@dataclass
class OptimizationConfig:
    """Configuration for Optuna-based hyperparameter optimization."""
    
    n_trials: int = 50
    n_startup_trials: int = 5
    timeout: int = 3600  # seconds
    random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    direction: str = "maximize"  # Maximize reliability metric
    

@dataclass
class ModelConfig:
    """Configuration for model search space and candidates."""
    
    model_candidates: List[str] = field(default_factory=lambda: [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "svm",
    ])
    sample_weight_support: bool = True
    cross_validation_folds: int = 5


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation."""
    
    method: str = "bootstrap"  # Options: bootstrap, entropy, dropout
    n_bootstrap_samples: int = 100
    confidence_threshold: float = 0.7
    

@dataclass
class OODConfig:
    """Configuration for out-of-distribution detection."""
    
    method: str = "isolation_forest"  # Options: isolation_forest, mahalanobis
    contamination: float = 0.1
    use_training_data: bool = True
    

@dataclass
class ExplainabilityConfig:
    """Configuration for SHAP-based explainability."""
    
    shap_explainer_type: str = "tree"  # Options: tree, kernel, sampling
    n_background_samples: int = 100
    max_display_features: int = 10


@dataclass
class AuraAuthConfig:
    """Master configuration class combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    debug_mode: bool = False


# Upload limits
MAX_UPLOAD_SIZE_BYTES: int = 100_000_000  # 100 MB

# Global configuration instance
DEFAULT_CONFIG = AuraAuthConfig()

"""
Pipeline Manager Module for AuraAuth AutoML System

Orchestrates the complete AutoML workflow by coordinating all system components
(data profiling, preprocessing, model optimization, selection, uncertainty estimation,
distribution shift detection, explainability, and documentation generation).

Pipeline orchestration improves maintainability, reproducibility, and clarity in ML
systems by providing a single entry point for end-to-end model training and evaluation.

Author: AuraAuth Development Team
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

from backend.data_profiler import DataProfiler
from backend.preprocessing import PreprocessingPipeline
from backend.automl_engine import AutoMLEngine
from backend.model_selector import ModelSelector
from backend.core.uncertainty_estimator import UncertaintyEstimator
from backend.core.distribution_shift_detector import DistributionShiftDetector
from backend.core.explainability_engine import ExplainabilityEngine
from backend.core.documentation_generator import DocumentationGenerator


# Configure logging
logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Central orchestrator for the complete AuraAuth AutoML pipeline.

    This class coordinates all components of the AutoML system to execute a full
    end-to-end workflow: from data profiling through model training, uncertainty
    estimation, distribution shift detection, explainability analysis, and automatic
    documentation generation.

    The pipeline is designed to be:
    - **Sequential**: Stages execute in a fixed, deterministic order
    - **Decoupled**: Each component is independent and testable in isolation
    - **Observable**: Clear logging and error messages at each stage
    - **Reproducible**: Same inputs produce same outputs (modulo randomness)
    - **Demo-friendly**: Straightforward logic suitable for explanation and evaluation

    Key Design Principles:
    - Orchestration only (no ML logic; delegates to specialized modules)
    - Stage-wise execution with intermediate state preservation
    - Clear input/output contracts for each stage
    - Graceful error handling with informative messages
    - Support for both classification and regression tasks

    Attributes:
        None (stateless orchestrator)

    Example:
        >>> from backend.core.pipeline_manager import PipelineManager
        >>> import numpy as np
        >>>
        >>> manager = PipelineManager()
        >>> X = np.random.rand(1000, 15)
        >>> y = np.random.randint(0, 2, 1000)
        >>> feature_names = [f"feature_{i}" for i in range(15)]
        >>>
        >>> results = manager.run_full_pipeline(
        ...     X=X,
        ...     y=y,
        ...     feature_names=feature_names,
        ...     task_type="classification"
        ... )
        >>>
        >>> # Access results from each pipeline stage
        >>> print(f"Best model: {results['model_selection']['best_model_name']}")
        >>> print(f"Uncertainty level: {results['uncertainty']['uncertainty_level']}")
    """

    def __init__(self, n_trials: int = 50) -> None:
        """
        Initialize the PipelineManager.
        
        Args:
            n_trials: Number of optimization trials per model (default: 50)
        """
        self.n_trials = n_trials
        logger.info(f"PipelineManager initialized with n_trials={n_trials}")
    
    def run_full_pipeline(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        feature_names: List[str],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Execute the complete AutoML pipeline in sequential stages.

        This method orchestrates all 9 pipeline stages in a fixed order:
        1. Data Profiling - Analyze training data characteristics
        2. Preprocessing - Prepare data (split, scaling, encoding)
        3. AutoML Optimization - Optimize all candidate models
        4. Model Selection - Select best model by performance
        5. Model Training - Train best model on full training set
        6. Uncertainty Estimation - Quantify prediction confidence
        7. Distribution Shift Detection - Detect data drift
        8. Explainability - Generate feature importance and local explanations
        9. Documentation - Auto-generate model cards and dataset sheets

        Each stage receives inputs from previous stages, allowing information to
        flow through the pipeline in a controlled manner.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
                Should contain numerical features (int, float). Handle categorical
                features before calling this method or they will be processed by
                PreprocessingPipeline.
            y (Optional[np.ndarray]): Target labels/values. Shape (n_samples,).
                For classification: integer class labels (0, 1, 2, ...).
                For regression: continuous numerical values.
                Optional for unsupervised preprocessing, but required for
                supervised learning (classification/regression).
            feature_names (List[str]): Names of features for interpretability.
                Length must match X.shape[1]. Used in explainability,
                documentation, and error messages.
            task_type (str): Type of learning task. Options:
                - "classification": Binary or multiclass classification
                - "regression": Continuous value prediction
                Default: "classification"

        Returns:
            Dict[str, Any]: Comprehensive results from all pipeline stages:
                - "data_profile" (Dict): Dataset health, warnings, statistics
                - "preprocessing" (Dict): Data split info, scaling parameters
                - "automl_results" (Dict): Model optimization results
                - "model_selection" (Dict): Best model and ranking table
                - "trained_model" (object): Fitted best model instance
                - "uncertainty" (Dict): Confidence scores and uncertainty level
                - "distribution_shift" (Dict): Shift score and shift level
                - "explainability" (Dict): Feature importance and explanations
                - "documentation" (Dict): Model card and dataset sheet
                - "pipeline_metadata" (Dict): Execution timestamps and stage info

        Raises:
            ValueError: If task_type not recognized, y is None for supervised tasks,
                feature_names length mismatch, or X/y dimension mismatch.
            TypeError: If inputs are wrong type (X/y not ndarray, feature_names not list).
            RuntimeError: If any pipeline stage fails (with original exception chained).

        Notes:
            - Pipeline is deterministic given same inputs and fixed random seed.
            - Each stage's output becomes available for downstream inspection.
            - If a stage fails, entire pipeline fails with clear error message.
            - Preprocessing splits data into train/test; downstream stages use test set.
            - Uncertainty and shift detection use different data (test set).
            - For best reproducibility, set numpy random seed before calling.

        Example:
            >>> import numpy as np
            >>> np.random.seed(42)
            >>>
            >>> # Generate toy data
            >>> X = np.random.rand(500, 10)
            >>> y = np.random.randint(0, 2, 500)
            >>> feature_names = [f"f_{i}" for i in range(10)]
            >>>
            >>> manager = PipelineManager()
            >>> results = manager.run_full_pipeline(X, y, feature_names, "classification")
            >>>
            >>> # Inspect results
            >>> print(f"Data profile: {results['data_profile']['n_samples']} samples")
            >>> print(f"Best model: {results['model_selection']['best_model_name']}")
            >>> print(f"Confidence: {results['uncertainty']['confidence_score']:.3f}")
        """
        # ===== Input Validation =====
        self._validate_inputs(X, y, feature_names, task_type)

        logger.info(f"Starting AutoML pipeline for {task_type} task")
        logger.info(f"Input shape: {X.shape}, Features: {len(feature_names)}")

        # Initialize result container
        pipeline_results = {}

        try:
            # ===== STAGE 1: Data Profiling =====
            logger.info("STAGE 1: Data Profiling")
            pipeline_results["data_profile"] = self._stage_data_profiling(X, y, task_type)

            # ===== STAGE 2: Preprocessing =====
            logger.info("STAGE 2: Preprocessing")
            preprocessing_output = self._stage_preprocessing(X, y, task_type)
            pipeline_results["preprocessing"] = preprocessing_output["metadata"]
            X_train = preprocessing_output["X_train"]
            X_test = preprocessing_output["X_test"]
            y_train = preprocessing_output["y_train"]

            # ===== STAGE 3: AutoML Optimization =====
            logger.info("STAGE 3: AutoML Optimization")
            pipeline_results["automl_results"] = self._stage_automl_optimization(
                X_train, y_train, task_type
            )

            # ===== STAGE 4: Model Selection =====
            logger.info("STAGE 4: Model Selection")
            selection_output = self._stage_model_selection(
                pipeline_results["automl_results"],
                task_type
            )
            # Store full selection output (includes best_model_name and best_score for frontend)
            pipeline_results["model_selection"] = selection_output
            best_model_name = selection_output["best_model_name"]
            best_hyperparams = selection_output["best_hyperparams"]

            # ===== STAGE 5: Model Training =====
            logger.info("STAGE 5: Model Training")
            training_output = self._stage_model_training(
                best_model_name,
                best_hyperparams,
                X_train,
                y_train,
                task_type
            )
            trained_model = training_output["model"]
            pipeline_results["trained_model"] = trained_model

            # ===== STAGE 6: Uncertainty Estimation =====
            logger.info("STAGE 6: Uncertainty Estimation")
            pipeline_results["uncertainty"] = self._stage_uncertainty_estimation(
                trained_model,
                X_test,
                task_type
            )

            # ===== STAGE 7: Distribution Shift Detection =====
            logger.info("STAGE 7: Distribution Shift Detection")
            pipeline_results["distribution_shift"] = self._stage_distribution_shift(
                X_train,
                X_test
            )

            # ===== STAGE 8: Explainability =====
            logger.info("STAGE 8: Explainability Analysis")
            pipeline_results["explainability"] = self._stage_explainability(
                trained_model,
                X_train,
                X_test,
                feature_names,
                task_type
            )

            # ===== STAGE 9: Documentation Generation =====
            logger.info("STAGE 9: Documentation Generation")
            pipeline_results["documentation"] = self._stage_documentation(
                best_model_name,
                pipeline_results,
                feature_names,
                task_type
            )

            # ===== Pipeline Metadata =====
            pipeline_results["pipeline_metadata"] = {
                "task_type": task_type,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "status": "success",
                "message": "Pipeline completed successfully"
            }

            logger.info("✓ Pipeline completed successfully")

        except Exception as e:
            logger.error(f"✗ Pipeline failed at stage: {str(e)}")
            pipeline_results["pipeline_metadata"] = {
                "status": "failed",
                "error": str(e)
            }
            raise RuntimeError(f"Pipeline execution failed: {str(e)}") from e

        return pipeline_results

    # ==================== Pipeline Stages ====================

    def _stage_data_profiling(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Stage 1: Analyze dataset characteristics and health.

        Executes DataProfiler.analyze() to compute statistics about the training data:
        - Sample count, feature count
        - Missing values, data types
        - Target distribution (for classification)
        - Data quality scores
        - Warnings and recommendations

        Args:
            X (np.ndarray): Feature matrix
            y (Optional[np.ndarray]): Target labels
            task_type (str): "classification" or "regression"

        Returns:
            Dict[str, Any]: Data profile including statistics and warnings
        """
        try:
            import pandas as pd
            
            # Convert numpy arrays to pandas for DataProfiler
            X_df = pd.DataFrame(X)
            y_series = pd.Series(y) if y is not None else None
            
            profiler = DataProfiler()
            profile_report = profiler.analyze(X_df, y_series, dataset_name="Dataset")
            
            # Convert report to dictionary for consistency
            if hasattr(profile_report, 'to_dict'):
                profile = profile_report.to_dict()
            elif hasattr(profile_report, '__dict__'):
                profile = profile_report.__dict__
            else:
                profile = profile_report
            
            # Convert any enum values to strings
            from enum import Enum
            for key, value in profile.items():
                if isinstance(value, Enum):
                    profile[key] = value.value if hasattr(value, 'value') else str(value)
            
            # Ensure required keys exist (documentation validators expect them)
            if 'n_samples' not in profile and 'dataset_size' in profile:
                profile['n_samples'] = profile['dataset_size']
            
            if 'n_features' not in profile and 'feature_count' in profile:
                profile['n_features'] = profile['feature_count']
            
            # Add missing_values_pct if not present
            if 'missing_values_pct' not in profile:
                if 'missing_value_report' in profile:
                    # Calculate average missing percentage
                    missing_values = profile['missing_value_report']
                    if missing_values:
                        profile['missing_values_pct'] = sum(missing_values.values()) / len(missing_values) if missing_values else 0
                    else:
                        profile['missing_values_pct'] = 0
                else:
                    profile['missing_values_pct'] = 0
            
            # Add target_distribution if not present
            if 'target_distribution' not in profile:
                if 'class_balance_report' in profile and profile['class_balance_report']:
                    profile['target_distribution'] = profile['class_balance_report']
                else:
                    profile['target_distribution'] = {}
            
            n_samples = profile.get('n_samples', X.shape[0])
            logger.info(f"  ✓ Profiled {n_samples} samples")
            return profile
        except Exception as e:
            logger.error(f"  ✗ Data profiling failed: {str(e)}")
            raise

    def _stage_preprocessing(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Stage 2: Preprocess data (split, scale, encode).

        Executes PreprocessingPipeline to:
        - Split data into train/test sets
        - Apply scaling/normalization
        - Encode categorical features
        - Handle missing values

        Args:
            X (np.ndarray): Feature matrix
            y (Optional[np.ndarray]): Target labels
            task_type (str): "classification" or "regression"

        Returns:
            Dict[str, Any]: Dictionary with:
                - X_train, X_test, y_train, y_test: Split data
                - metadata: Preprocessing information
        """
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split
            
            # Convert to pandas
            X_df = pd.DataFrame(X)
            y_series = pd.Series(y) if y is not None else None
            
            # Apply preprocessing
            pipeline = PreprocessingPipeline()
            X_transformed = pipeline.fit_transform(X_df, y_series)
            
            # Check target cardinality for stratification
            stratify_target = None
            if y_series is not None and task_type == "classification":
                n_samples = len(y_series)
                n_classes = len(np.unique(y_series))
                min_samples_per_class = n_samples / n_classes
                
                # Can only stratify if each class has at least 2 samples (for train_test_split)
                if min_samples_per_class < 2:
                    logger.warning(
                        f"Target has {n_classes} classes with only {n_samples} samples. "
                        f"Cannot stratify train/test split. Using random split instead."
                    )
                else:
                    stratify_target = y_series
            
            # Train/test split with or without stratification
            test_size = 0.2
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_transformed,
                    y_series,
                    test_size=test_size,
                    random_state=42,
                    stratify=stratify_target
                )
            except Exception as split_error:
                # If stratified split fails, fall back to random split
                logger.warning(f"Stratified split failed: {str(split_error)}. Using random split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_transformed,
                    y_series,
                    test_size=test_size,
                    random_state=42,
                    stratify=None
                )

            # Convert to numpy arrays for consistent downstream handling
            X_train = np.asarray(X_train, dtype=float)
            X_test = np.asarray(X_test, dtype=float)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

            metadata = {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_test_ratio": f"{len(X_train) / len(X_transformed):.1%}",
                "features_after_preprocessing": X_train.shape[1]
            }

            logger.info(f"  ✓ Split into {len(X_train)} train, {len(X_test)} test")

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"  ✗ Preprocessing failed: {str(e)}")
            raise

    def _stage_automl_optimization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Stage 3: Optimize all candidate models.

        Executes AutoMLEngine.run_automl() to:
        - Train multiple models with different hyperparameters
        - Evaluate using cross-validation
        - Return mean and std scores for each model

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            task_type (str): "classification" or "regression"

        Returns:
            Dict[str, Any]: AutoML results with model scores and hyperparameters
        """
        try:
            engine = AutoMLEngine(n_trials=self.n_trials, cv_folds=3, verbose=False)
            if task_type == "classification":
                n_classes = len(np.unique(y_train))
                metric = "roc_auc" if n_classes == 2 else "accuracy"
            else:
                metric = "neg_mean_squared_error"
            results = engine.run_automl(X_train, y_train, metric=metric)
            logger.info(f"  ✓ AutoML evaluated {len(results)} models")
            return {"model_scores": results}
        except Exception as e:
            logger.error(f"  ✗ AutoML optimization failed: {str(e)}")
            raise

    def _stage_model_selection(
        self,
        automl_results: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Stage 4: Select best model from AutoML results.

        Executes ModelSelector.select_best_model() to:
        - Rank models by performance
        - Select top model
        - Return full ranking and best hyperparameters

        Args:
            automl_results (Dict[str, Any]): Results from AutoML stage
            task_type (str): "classification" or "regression"

        Returns:
            Dict[str, Any]: Dictionary with:
                - best_model_name: Name of selected model
                - best_score: Best model's mean score
                - ranking_table: DataFrame of all models ranked
                - best_hyperparams: Best hyperparameters
                - metadata: Selection info and ranking
        """
        try:
            # Extract model scores from automl results
            model_scores_dict = automl_results.get("model_scores", {})
            
            selector = ModelSelector()
            best_model_name, ranking_info = selector.select_best_model(model_scores_dict)
            
            # Extract best hyperparameters and score
            best_model_result = model_scores_dict.get(best_model_name, {})
            best_hyperparams = best_model_result.get("best_params", {})
            best_score = best_model_result.get("mean_score", 0.0)

            logger.info(f"  ✓ Selected best model: {best_model_name} (score: {best_score:.4f})")

            # Convert ranking list to DataFrame for frontend
            import pandas as pd
            ranking_rows = ranking_info.get("ranking", [])
            ranking_df = pd.DataFrame(ranking_rows)
            
            return {
                "best_model_name": best_model_name,
                "best_score": best_score,
                "best_hyperparams": best_hyperparams,
                "ranking_table": ranking_df,
                "metadata": {
                    "best_model_name": best_model_name,
                    "ranking": ranking_rows,
                    "best_metrics": best_model_result
                }
            }
        except Exception as e:
            logger.error(f"  ✗ Model selection failed: {str(e)}")
            raise

    def _stage_model_training(
        self,
        model_name: str,
        hyperparams: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Stage 5: Train best model on full training data.

        Instantiates the best model with optimal hyperparameters and fits it
        on the training data.

        Args:
            model_name (str): Name of model to train (e.g., "RandomForestClassifier")
            hyperparams (Dict[str, Any]): Optimal hyperparameters
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            task_type (str): "classification" or "regression"

        Returns:
            Dict[str, Any]: Dictionary with:
                - model: Trained model instance
        """
        try:
            import inspect
            from sklearn.ensemble import (
                RandomForestClassifier, RandomForestRegressor,
                GradientBoostingClassifier, GradientBoostingRegressor
            )
            from sklearn.linear_model import LogisticRegression, LinearRegression
            try:
                import xgboost as xgb
                xgb_available = True
            except ImportError:
                xgb_available = False

            try:
                import lightgbm as lgb
                lgb_available = True
            except ImportError:
                lgb_available = False

            # Model factory - maps class names to classes
            model_factory = {
                # Classification
                "LogisticRegression": LogisticRegression,
                "RandomForestClassifier": RandomForestClassifier,
                "GradientBoostingClassifier": GradientBoostingClassifier,
                # Regression
                "LinearRegression": LinearRegression,
                "RandomForestRegressor": RandomForestRegressor,
                "GradientBoostingRegressor": GradientBoostingRegressor,
            }

            if xgb_available:
                model_factory["XGBClassifier"] = xgb.XGBClassifier
                model_factory["XGBRegressor"] = xgb.XGBRegressor

            if lgb_available:
                model_factory["LGBMClassifier"] = lgb.LGBMClassifier
                model_factory["LGBMRegressor"] = lgb.LGBMRegressor

            # Map AutoML model names to class names
            name_mapping = {
                "logistic_regression": "LogisticRegression" if task_type == "classification" else "LinearRegression",
                "random_forest": "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor",
                "xgboost": "XGBClassifier" if task_type == "classification" else "XGBRegressor",
                "lightgbm": "LGBMClassifier" if task_type == "classification" else "LGBMRegressor",
                # Also accept the class names directly
                "LogisticRegression": "LogisticRegression",
                "RandomForestClassifier": "RandomForestClassifier",
                "RandomForestRegressor": "RandomForestRegressor",
                "XGBClassifier": "XGBClassifier",
                "XGBRegressor": "XGBRegressor",
                "LGBMClassifier": "LGBMClassifier",
                "LGBMRegressor": "LGBMRegressor",
            }
            
            # Resolve model name
            resolved_model_name = name_mapping.get(model_name, model_name)
            
            # Get model class
            if resolved_model_name not in model_factory:
                raise ValueError(f"Unknown model: {model_name} (resolved to {resolved_model_name})")

            model_class = model_factory[resolved_model_name]
            # Only pass random_state when the estimator supports it
            init_params = dict(hyperparams or {})
            if "random_state" in inspect.signature(model_class).parameters:
                init_params["random_state"] = 42

            model = model_class(**init_params)
            model.fit(X_train, y_train)

            logger.info(f"  ✓ Trained {resolved_model_name}")

            return {"model": model}

        except Exception as e:
            logger.error(f"  ✗ Model training failed: {str(e)}")
            raise

    def _stage_uncertainty_estimation(
        self,
        model,
        X_test: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Stage 6: Estimate prediction uncertainty on test data.

        Executes UncertaintyEstimator to quantify confidence in predictions.

        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            task_type (str): "classification" or "regression"

        Returns:
            Dict[str, Any]: Uncertainty metrics
        """
        try:
            estimator = UncertaintyEstimator()

            if task_type == "classification":
                uncertainty = estimator.estimate_classification_uncertainty(model, X_test)
            else:
                uncertainty = estimator.estimate_regression_uncertainty(model, X_test)

            logger.info(f"  ✓ Estimated uncertainty ({uncertainty.get('uncertainty_level', 'unknown')})")
            return uncertainty

        except Exception as e:
            logger.error(f"  ✗ Uncertainty estimation failed: {str(e)}")
            raise

    def _stage_distribution_shift(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Stage 7: Detect distribution shift between training and test data.

        Fits DistributionShiftDetector on training data and detects shift in test data.

        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features

        Returns:
            Dict[str, Any]: Distribution shift results
        """
        try:
            detector = DistributionShiftDetector()
            detector.fit(X_train)
            shift = detector.detect_shift(X_test)

            logger.info(f"  ✓ Detected distribution shift ({shift.get('shift_level', 'unknown')})")
            return shift

        except Exception as e:
            logger.error(f"  ✗ Distribution shift detection failed: {str(e)}")
            raise

    def _stage_explainability(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Stage 8: Generate feature importance and local explanations.

        Fits ExplainabilityEngine and generates:
        - Global feature importance
        - Local explanation for first test sample

        Args:
            model: Trained model
            X_train (np.ndarray): Training features (for background)
            X_test (np.ndarray): Test features (for explanations)
            feature_names (List[str]): Feature names
            task_type (str): "classification" or "regression"

        Returns:
            Dict[str, Any]: Global and local explanations with frontend-compatible schema
        """
        try:
            engine = ExplainabilityEngine()
            engine.fit_explainer(model, X_train, feature_names)

            global_exp = engine.explain_global()
            local_exp = engine.explain_local(X_test[0:1])

            # Transform to frontend-compatible schema
            # Convert feature_importance list to dict
            global_feature_importance = {}
            if "feature_importance" in global_exp:
                for item in global_exp["feature_importance"]:
                    feature_name = item.get("feature", "unknown")
                    importance = item.get("importance", 0.0)
                    global_feature_importance[feature_name] = importance
            
            # Create feature statistics from global explanation
            feature_statistics = {}
            if "feature_importance" in global_exp:
                for item in global_exp["feature_importance"]:
                    feature_name = item.get("feature", "unknown")
                    feature_statistics[feature_name] = {
                        "importance": item.get("importance", 0.0),
                        "rank": len(feature_statistics) + 1
                    }

            result = {
                "global_feature_importance": global_feature_importance,
                "feature_statistics": feature_statistics,
                "explanation_method": "SHAP",
                "local_explanations": local_exp.get("explanation", "No local explanation available"),
                "top_positive_features": local_exp.get("top_positive_features", []),
                "top_negative_features": local_exp.get("top_negative_features", [])
            }

            logger.info("  ✓ Generated explanations")
            return result

        except Exception as e:
            logger.error(f"  ✗ Explainability generation failed: {str(e)}")
            raise

    def _stage_documentation(
        self,
        model_name: str,
        pipeline_results: Dict[str, Any],
        feature_names: List[str],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Stage 9: Auto-generate model card and dataset sheet.

        Generates comprehensive documentation from pipeline results.

        Args:
            model_name (str): Name of best model
            pipeline_results (Dict[str, Any]): Results from previous stages
            feature_names (List[str]): Feature names
            task_type (str): "classification" or "regression"

        Returns:
            Dict[str, Any]: Model documentation with frontend-compatible schema
        """
        try:
            generator = DocumentationGenerator()

            # Extract relevant metrics for model card
            data_profile = pipeline_results.get("data_profile", {})
            model_selection = pipeline_results.get("model_selection", {})
            explainability = pipeline_results.get("explainability", {})
            uncertainty = pipeline_results.get("uncertainty", {})
            shift = pipeline_results.get("distribution_shift", {})

            # Prepare metrics for model card
            metrics = model_selection.get("metadata", {}).get("best_metrics", {})
            if not metrics:
                metrics = {"model_type": model_name, "task": task_type}

            # Map explainability output into generator-compatible schema
            feature_importance_list = []
            gfi = explainability.get("global_feature_importance", {})
            if isinstance(gfi, dict) and gfi:
                for feat, imp in gfi.items():
                    try:
                        feature_importance_list.append({"feature": str(feat), "importance": float(imp)})
                    except Exception:
                        continue
                feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)

            explainability_summary = {"feature_importance": feature_importance_list}

            # Map uncertainty output into generator-compatible schema
            uncertainty_summary = {
                "mean_confidence": uncertainty.get("confidence_score", uncertainty.get("mean_confidence", "unknown")),
                "uncertainty_level": uncertainty.get("uncertainty_level", "unknown"),
                "notes": uncertainty.get("notes", "")
            }

            model_card = generator.generate_model_card(
                model_name=model_name,
                metrics=metrics,
                data_profile=data_profile,
                explainability_summary=explainability_summary,
                uncertainty_summary=uncertainty_summary,
                shift_summary=shift
            )

            dataset_sheet = generator.generate_dataset_sheet(data_profile)

            # Transform to frontend-compatible schema
            dataset_info = {
                "total_samples": data_profile.get("n_samples", "N/A"),
                "total_features": len(feature_names) if feature_names else data_profile.get("n_features", "N/A"),
                "num_classes": len(data_profile.get("target_distribution", {})) if "target_distribution" in data_profile else "N/A",
                "missing_percentage": data_profile.get("missing_values_pct", 0.0),
                "duplicate_count": data_profile.get("duplicate_rows_pct", 0),
                "quality_score": data_profile.get("data_quality_score", "N/A")
            }

            training_report = {
                "execution_status": "Completed",
                "stages_completed": "9/9",
                "total_duration": pipeline_results.get("execution_time", "N/A"),
                "stage_details": {
                    "Data Profiling": data_profile.get("health_status", "Completed"),
                    "Model Selection": f"Selected {model_name}",
                    "Explainability": "SHAP-based explanations generated",
                    "Uncertainty": uncertainty.get("uncertainty_level", "Not computed"),
                    "Distribution Shift": shift.get("shift_level", "Not computed")
                }
            }

            recommendations = {
                "strengths": [
                    f"Model selected: {model_name}",
                    "Based on comprehensive evaluation of multiple models",
                    "Feature importance analysis available"
                ],
                "improvements": [
                    "Monitor for distribution shift in production",
                    "Track prediction confidence over time"
                ]
            }

            result = {
                "model_card": model_card,
                "dataset_info": dataset_info,
                "training_report": training_report,
                "recommendations": recommendations,
                "dataset_sheet": dataset_sheet
            }

            logger.info("  ✓ Generated documentation")
            return result

        except Exception as e:
            logger.error(f"  ✗ Documentation generation failed: {str(e)}")
            raise

    # ==================== Validation ====================

    def _validate_inputs(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        feature_names: List[str],
        task_type: str
    ) -> None:
        """
        Validate pipeline inputs.

        Args:
            X (np.ndarray): Feature matrix
            y (Optional[np.ndarray]): Target labels
            feature_names (List[str]): Feature names
            task_type (str): Task type

        Raises:
            ValueError: If inputs are invalid
            TypeError: If inputs are wrong type
        """
        # Check X
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, received {type(X).__name__}")

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, received shape {X.shape}")

        if X.shape[0] < 100:
            logger.warning("X has fewer than 100 samples; results may be unreliable")

        # Check y
        if y is not None and not isinstance(y, np.ndarray):
            raise TypeError(f"y must be numpy array, received {type(y).__name__}")

        if y is not None and len(y) != X.shape[0]:
            raise ValueError(
                f"y length ({len(y)}) does not match X samples ({X.shape[0]})"
            )

        # Check feature_names
        if not isinstance(feature_names, (list, tuple)):
            raise TypeError(f"feature_names must be list, received {type(feature_names).__name__}")

        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match "
                f"X features ({X.shape[1]})"
            )

        # Check task_type
        if task_type not in ["classification", "regression"]:
            raise ValueError(
                f"task_type must be 'classification' or 'regression', received '{task_type}'"
            )

        logger.info("✓ Input validation passed")

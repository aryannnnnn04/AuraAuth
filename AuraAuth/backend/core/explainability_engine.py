"""
Explainability Engine Module for AuraAuth AutoML System

Provides model-agnostic global and local explanations using SHAP (SHapley Additive
exPlanations) values, optimized for small tabular datasets in production ML systems.

Explainability is essential in high-stakes ML systems to ensure transparency and user
trust, particularly when making predictions that affect business or user outcomes. SHAP
values provide theoretically sound, game-theoretic explanations that are interpretable
and actionable.

Author: AuraAuth Development Team
"""

import numpy as np
import shap
from typing import Dict, Any, List


class ExplainabilityEngine:
    """
    Provides global and local model explanations using SHAP values.

    This engine automatically detects model type and applies the most appropriate SHAP
    explainer (TreeExplainer, LinearExplainer, or KernelExplainer). It is optimized for
    small tabular datasets where comprehensive explanations are critical for trust and
    deployment decisions.

    The engine computes:
    - Global feature importance (mean absolute SHAP values)
    - Local feature attributions (SHAP values for individual predictions)

    Outputs are returned as clean Python primitives (dicts, lists) suitable for
    visualization in Streamlit, web dashboards, or reports.

    Attributes:
        _explainer: Fitted SHAP explainer object (TreeExplainer, LinearExplainer, etc.)
        _is_fitted (bool): Whether fit_explainer() has been called.
        _feature_names (List[str]): Names of input features.
        _model_type (str): Type of explainer used ("tree", "linear", "kernel").
        _X_background (np.ndarray): Background dataset for explainer (stored for reference).
        _shap_values_global (Optional[np.ndarray]): Cached global SHAP values.

    Example:
        >>> from backend.core.explainability_engine import ExplainabilityEngine
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import numpy as np
        >>>
        >>> engine = ExplainabilityEngine()
        >>> model = RandomForestClassifier(n_estimators=100, random_state=42)
        >>> X_train = np.random.rand(500, 10)
        >>> y_train = np.random.randint(0, 2, 500)
        >>> model.fit(X_train, y_train)
        >>>
        >>> feature_names = [f"feature_{i}" for i in range(10)]
        >>> engine.fit_explainer(model, X_train, feature_names)
        >>>
        >>> # Global explanations
        >>> global_exp = engine.explain_global()
        >>> print(global_exp['feature_importance'][0])
        >>>
        >>> # Local explanation
        >>> X_test = np.random.rand(1, 10)
        >>> local_exp = engine.explain_local(X_test)
        >>> print(local_exp['explanation'])
    """

    def __init__(self) -> None:
        """Initialize the ExplainabilityEngine (unfitted state)."""
        self._explainer = None
        self._is_fitted = False
        self._feature_names = None
        self._model_type = None
        self._X_background = None
        self._shap_values_global = None

    def fit_explainer(
        self,
        model,
        X_background: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """
        Initialize and fit the SHAP explainer for a given model.

        This method automatically detects the model type and instantiates the appropriate
        SHAP explainer:
        - Tree-based models (RandomForest, XGBoost, LightGBM) → TreeExplainer
        - Linear models (LogisticRegression, LinearRegression) → LinearExplainer
        - Other models → KernelExplainer (slower but model-agnostic)

        Args:
            model: Fitted sklearn-style model (classifier or regressor).
            X_background (np.ndarray): Background dataset for SHAP explanation.
                Shape (n_samples, n_features). For small datasets, can use entire
                training set. For large datasets, use a representative sample (e.g., 100-200 rows).
            feature_names (List[str]): Names of features, must match X_background columns.
                Used for human-readable explanations.

        Raises:
            ValueError: If input shapes mismatch, feature_names length incorrect,
                or model is unsupported.
            TypeError: If X_background cannot be converted to numpy array or
                feature_names is not a list of strings.

        Notes:
            - Background dataset should be representative of the data distribution.
            - For small datasets (< 500 samples), entire training set is recommended.
            - TreeExplainer is significantly faster than KernelExplainer.
            - All outputs are computed lazily (only when requested).

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> engine = ExplainabilityEngine()
            >>> model = RandomForestClassifier()
            >>> X = np.random.rand(500, 10)
            >>> y = np.random.randint(0, 2, 500)
            >>> model.fit(X, y)
            >>> feature_names = [f"feat_{i}" for i in range(10)]
            >>> engine.fit_explainer(model, X, feature_names)
        """
        # Validate inputs
        self._validate_input(X_background, feature_names)

        # Convert to numpy if needed
        if not isinstance(X_background, np.ndarray):
            X_background = np.asarray(X_background)

        # Check feature names match data shape
        if len(feature_names) != X_background.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match "
                f"X_background columns ({X_background.shape[1]})"
            )

        # Store background data and feature names
        self._X_background = X_background
        self._feature_names = feature_names

        # Detect model type and create appropriate explainer
        explainer_type = self._get_explainer_type(model)
        self._model_type = explainer_type

        try:
            if explainer_type == "tree":
                self._explainer = shap.TreeExplainer(model)
            elif explainer_type == "linear":
                self._explainer = shap.LinearExplainer(model, X_background)
            elif explainer_type == "kernel":
                # KernelExplainer is slower but works with any model
                self._explainer = shap.KernelExplainer(
                    model.predict if hasattr(model, 'predict') else model,
                    X_background
                )
            else:
                raise ValueError(f"Unsupported explainer type: {explainer_type}")
        except Exception as e:
            raise RuntimeError(
                f"Error initializing {explainer_type} explainer: {str(e)}"
            )

        self._is_fitted = True

    def explain_global(self) -> Dict[str, Any]:
        """
        Explain model behavior globally using mean absolute SHAP values.

        This method computes feature importance as the mean absolute SHAP value across
        the background dataset. Features with higher mean |SHAP| values have greater
        impact on model predictions.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "feature_importance" (List[Dict]): Ranked list of features with importance:
                  * Each dict contains "feature" (str) and "importance" (float in [0, 1])
                  * Sorted by importance descending (most important first)
                - "notes" (str): Summary and interpretation guidance.

        Raises:
            ValueError: If explainer not fitted (fit_explainer must be called first).
            RuntimeError: If SHAP value computation fails.

        Notes:
            - Mean absolute SHAP values are normalized to [0, 1] for interpretability.
            - Features are ranked by importance (descending).
            - SHAP values are computed on the background dataset.
            - This is a model-agnostic global explanation.

        Examples:
            >>> engine = ExplainabilityEngine()
            >>> # ... fit_explainer called ...
            >>> global_exp = engine.explain_global()
            >>> for item in global_exp['feature_importance']:
            ...     print(f"{item['feature']}: {item['importance']:.3f}")
            >>> print(global_exp['notes'])
        """
        if not self._is_fitted:
            raise ValueError(
                "ExplainabilityEngine must be fitted using fit_explainer() "
                "before calling explain_global()"
            )

        try:
            # Compute SHAP values on background data
            shap_values = self._explainer.shap_values(self._X_background)

            # Handle different SHAP output formats
            if hasattr(shap_values, 'values'):
                # Modern SHAP Explanation object
                shap_values = shap_values.values
            if isinstance(shap_values, list):
                # List of arrays (one per class) — take positive class
                shap_values = np.asarray(shap_values[0])
            else:
                shap_values = np.asarray(shap_values)
            # If 3D (n_samples, n_features, n_classes), take positive class
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]
            # Ensure 2D (n_samples, n_features)
            if shap_values.ndim != 2:
                shap_values = shap_values.reshape(self._X_background.shape[0], -1)

            # Cache for potential reuse
            self._shap_values_global = shap_values

            # Compute mean absolute SHAP per feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            # Normalize to [0, 1]
            normalized_importance = self._normalize_importance(mean_abs_shap)

            # Create feature importance list
            feature_importance = []
            for feature_idx, feature_name in enumerate(self._feature_names):
                feature_importance.append({
                    "feature": feature_name,
                    "importance": float(normalized_importance[feature_idx])
                })

            # Sort by importance descending
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

            # Generate notes
            top_3_features = [x["feature"] for x in feature_importance[:3]]
            notes = (
                f"Global feature importance based on mean absolute SHAP values ({self._model_type} explainer). "
                f"Top 3 features: {', '.join(top_3_features)}. "
                f"Higher importance indicates stronger influence on model predictions."
            )

            return {
                "feature_importance": feature_importance,
                "notes": notes
            }

        except Exception as e:
            raise RuntimeError(
                f"Error computing global explanations: {str(e)}"
            )

    def explain_local(self, X_sample: np.ndarray) -> Dict[str, Any]:
        """
        Explain individual predictions using local SHAP values.

        This method computes SHAP values for one or more samples, identifying which
        features contributed positively (push prediction up) and negatively (push down)
        to the model's prediction.

        Args:
            X_sample (np.ndarray): Single sample or batch of samples.
                Shape (n_samples, n_features) or (n_features,) for single sample.
                Must have same number of features as background data.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "top_positive_features" (List[str]): Features that increased prediction
                  (positive SHAP values), sorted by magnitude.
                - "top_negative_features" (List[str]): Features that decreased prediction
                  (negative SHAP values), sorted by magnitude.
                - "explanation" (str): Human-readable summary of key drivers.

        Raises:
            ValueError: If explainer not fitted or X_sample shape mismatch.
            TypeError: If X_sample cannot be converted to numpy array.
            RuntimeError: If SHAP value computation fails.

        Notes:
            - For multi-sample input, explains first sample.
            - Top 3 positive and negative features are extracted.
            - All outputs use clean Python primitives (no raw SHAP objects).
            - Works with any sklearn-style model type.

        Examples:
            >>> engine = ExplainabilityEngine()
            >>> # ... fit_explainer called ...
            >>>
            >>> # Single sample
            >>> sample = np.array([[1.5, 2.3, 0.8, ...]])
            >>> local_exp = engine.explain_local(sample)
            >>> print(local_exp['explanation'])
            >>>
            >>> # Batch of samples
            >>> batch = np.random.rand(10, n_features)
            >>> local_exp = engine.explain_local(batch)
        """
        if not self._is_fitted:
            raise ValueError(
                "ExplainabilityEngine must be fitted using fit_explainer() "
                "before calling explain_local()"
            )

        # Validate and convert input
        if not isinstance(X_sample, np.ndarray):
            try:
                X_sample = np.asarray(X_sample)
            except Exception as e:
                raise TypeError(f"Cannot convert X_sample to numpy array: {str(e)}")

        # Handle 1D array (single sample)
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        if X_sample.ndim != 2:
            raise ValueError(
                f"X_sample must be 1D or 2D, received shape {X_sample.shape}"
            )

        if X_sample.shape[1] != len(self._feature_names):
            raise ValueError(
                f"X_sample has {X_sample.shape[1]} features, expected {len(self._feature_names)}"
            )

        try:
            # Compute SHAP values for sample(s)
            shap_values = self._explainer.shap_values(X_sample)

            # Handle different SHAP output formats
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values
            if isinstance(shap_values, list):
                shap_values = np.asarray(shap_values[0])
            else:
                shap_values = np.asarray(shap_values)
            # If 3D (n_samples, n_features, n_classes), take positive class
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]

            # Use first sample if batch
            if shap_values.ndim > 1:
                shap_values = shap_values[0]

            # Create feature contribution pairs
            contributions = [
                (self._feature_names[i], float(shap_values[i]))
                for i in range(len(self._feature_names))
            ]

            # Sort by absolute value (magnitude)
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            # Extract top positive and negative
            top_positive = [
                name for name, value in contributions if value > 0
            ][:3]

            top_negative = [
                name for name, value in contributions if value < 0
            ][:3]

            # Generate explanation text
            explanation = self._generate_explanation(
                top_positive,
                top_negative,
                contributions
            )

            return {
                "top_positive_features": top_positive,
                "top_negative_features": top_negative,
                "explanation": explanation
            }

        except Exception as e:
            raise RuntimeError(
                f"Error computing local explanations: {str(e)}"
            )

    # ==================== Private Helper Methods ====================

    def _get_explainer_type(self, model) -> str:
        """
        Detect and return the appropriate SHAP explainer type for the model.

        Strategy:
        1. Check for tree-based models (RandomForest, XGBoost, LightGBM, GradientBoosting)
        2. Check for linear models (LogisticRegression, LinearRegression)
        3. Fallback to KernelExplainer (works with any model but slower)

        Args:
            model: Fitted sklearn-style model.

        Returns:
            str: Explainer type ("tree", "linear", or "kernel").

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> model = RandomForestClassifier()
            >>> engine = ExplainabilityEngine()
            >>> explainer_type = engine._get_explainer_type(model)
            >>> print(explainer_type)  # "tree"
        """
        model_class_name = type(model).__name__

        # Tree-based models
        tree_models = [
            'RandomForestClassifier', 'RandomForestRegressor',
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'ExtraTreesClassifier', 'ExtraTreesRegressor'
        ]

        if model_class_name in tree_models:
            return "tree"

        # Linear models
        linear_models = [
            'LogisticRegression', 'LinearRegression',
            'Ridge', 'Lasso', 'ElasticNet'
        ]

        if model_class_name in linear_models:
            return "linear"

        # Fallback to kernel explainer
        return "kernel"

    def _normalize_importance(self, importance_values: np.ndarray) -> np.ndarray:
        """
        Normalize feature importance scores to [0, 1] range.

        Uses min-max scaling for interpretability. All importance values are
        scaled relative to the minimum (0) and maximum (1) observed values.

        Args:
            importance_values (np.ndarray): Raw importance scores.

        Returns:
            np.ndarray: Normalized importance scores in [0, 1].
        """
        importance_values = np.asarray(importance_values)

        if len(importance_values) == 0:
            return importance_values

        min_val = np.min(importance_values)
        max_val = np.max(importance_values)

        # Handle constant values (all same importance)
        if max_val == min_val:
            # All features equally important
            return np.ones_like(importance_values, dtype=float) * 0.5

        # Min-max scaling
        normalized = (importance_values - min_val) / (max_val - min_val)

        return normalized

    def _validate_input(
        self,
        X_background: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """
        Validate input to fit_explainer.

        Args:
            X_background (np.ndarray): Background dataset.
            feature_names (List[str]): Feature names.

        Raises:
            ValueError: If inputs are invalid.
            TypeError: If inputs are wrong type.
        """
        # Check feature_names type
        if not isinstance(feature_names, (list, tuple)):
            raise TypeError(
                f"feature_names must be list or tuple, received {type(feature_names).__name__}"
            )

        if not all(isinstance(name, str) for name in feature_names):
            raise TypeError("All feature_names must be strings")

        if len(feature_names) == 0:
            raise ValueError("feature_names cannot be empty")

        # Check X_background type and shape
        try:
            X_background = np.asarray(X_background)
        except Exception as e:
            raise TypeError(f"Cannot convert X_background to numpy array: {str(e)}")

        if X_background.ndim != 2:
            raise ValueError(
                f"X_background must be 2-dimensional, received shape {X_background.shape}"
            )

        if len(X_background) == 0:
            raise ValueError("X_background contains no samples")

    def _generate_explanation(
        self,
        top_positive: List[str],
        top_negative: List[str],
        all_contributions
    ) -> str:
        """
        Generate human-readable explanation text for a single prediction.

        Args:
            top_positive (List[str]): Top features pushing prediction up.
            top_negative (List[str]): Top features pushing prediction down.
            all_contributions: List of (feature, shap_value) tuples.

        Returns:
            str: Explanation text suitable for display.
        """
        explanation = "Key drivers for this prediction: "

        if top_positive:
            pos_text = ", ".join(top_positive[:2])
            explanation += f"Increased by {pos_text}. "
        else:
            explanation += "No strong positive drivers. "

        if top_negative:
            neg_text = ", ".join(top_negative[:2])
            explanation += f"Decreased by {neg_text}."
        else:
            explanation += "No strong negative drivers."

        return explanation

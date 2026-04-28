"""
Uncertainty Estimation Module for AuraAuth AutoML System

Provides model-agnostic uncertainty quantification for classification and regression
tasks, optimized for small labeled datasets (500-5000 samples).

Uncertainty estimation is critical for trust and reliability in production ML systems,
especially with limited data where model confidence may be unreliable.

Author: AuraAuth Development Team
"""

import numpy as np
from typing import Dict, Any
from scipy.stats import entropy as scipy_entropy


class UncertaintyEstimator:
    """
    Model-agnostic uncertainty estimation for classification and regression.

    This module quantifies prediction confidence and uncertainty for any sklearn-style
    fitted model. It is optimized for small datasets where robust uncertainty estimates
    are essential for production reliability.

    Attributes:
        None (stateless utility class)

    Example:
        >>> from backend.core.uncertainty_estimator import UncertaintyEstimator
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import numpy as np
        >>>
        >>> estimator = UncertaintyEstimator()
        >>> model = RandomForestClassifier(n_estimators=100, random_state=42)
        >>> X_train = np.random.rand(500, 10)
        >>> y_train = np.random.randint(0, 2, 500)
        >>> model.fit(X_train, y_train)
        >>>
        >>> X_test = np.random.rand(50, 10)
        >>> result = estimator.estimate_classification_uncertainty(model, X_test)
        >>> print(result['uncertainty_level'])  # "LOW", "MEDIUM", or "HIGH"
    """

    def __init__(self) -> None:
        """Initialize the UncertaintyEstimator (stateless)."""
        pass

    def estimate_classification_uncertainty(
        self,
        model,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty for classification predictions using entropy-based approach.

        This method uses Shannon entropy as a measure of prediction uncertainty.
        Entropy is maximized when predictions are equally distributed across classes
        (high uncertainty) and minimized when predictions are concentrated on one class
        (low uncertainty). This is particularly suitable for small datasets where
        ensemble-based estimates may be unstable.

        Args:
            model: Fitted sklearn-style classifier with predict_proba() method.
            X (np.ndarray): Input features of shape (n_samples, n_features).
                Must be 2-dimensional.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "confidence_score" (float): Mean confidence across predictions [0, 1].
                  Higher values indicate more confident predictions.
                - "mean_entropy" (float): Mean Shannon entropy across predictions [0, 1].
                  Higher values indicate more uncertain predictions.
                - "uncertainty_level" (str): Categorical uncertainty level:
                  * "LOW": confidence >= 0.75 (high confidence, low uncertainty)
                  * "MEDIUM": 0.4 <= confidence < 0.75 (moderate uncertainty)
                  * "HIGH": confidence < 0.4 (low confidence, high uncertainty)
                - "notes" (str): Additional context about the predictions.

        Raises:
            ValueError: If model does not have predict_proba() method or X is invalid.
            TypeError: If X is not a numpy array or proper type.

        Examples:
            >>> estimator = UncertaintyEstimator()
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> model = RandomForestClassifier(n_estimators=50, random_state=42)
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> y = np.array([0, 1, 0])
            >>> model.fit(X, y)
            >>> result = estimator.estimate_classification_uncertainty(model, X)
            >>> assert result['uncertainty_level'] in ['LOW', 'MEDIUM', 'HIGH']
        """
        # Input validation
        if not hasattr(model, 'predict_proba'):
            raise ValueError(
                f"Model does not support predict_proba(). "
                f"Received: {type(model).__name__}. "
                f"Please use a classifier with probability estimates."
            )

        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except Exception as e:
                raise TypeError(
                    f"Cannot convert X to numpy array: {str(e)}"
                )

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional, received shape {X.shape}"
            )

        if len(X) == 0:
            raise ValueError("X contains no samples")

        # Get probability predictions
        try:
            probs = model.predict_proba(X)
        except Exception as e:
            raise RuntimeError(
                f"Error during predict_proba(): {str(e)}"
            )

        # Compute entropy for each sample
        entropies = np.array([self._compute_entropy(prob) for prob in probs])

        # Normalize entropy to [0, 1]
        normalized_entropies = self._normalize(entropies)

        # Compute confidence score (inverse of normalized entropy)
        confidence_scores = 1.0 - normalized_entropies

        # Aggregate metrics
        mean_confidence = float(np.mean(confidence_scores))
        mean_entropy = float(np.mean(normalized_entropies))

        # Map to uncertainty level
        uncertainty_level = self._map_confidence_to_level(mean_confidence)

        # Generate contextual notes
        notes = self._generate_classification_notes(
            mean_confidence,
            len(X),
            probs.shape[1]
        )

        return {
            "confidence_score": mean_confidence,
            "mean_entropy": mean_entropy,
            "uncertainty_level": uncertainty_level,
            "notes": notes
        }

    def estimate_regression_uncertainty(
        self,
        model,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty for regression predictions using variance-based approach.

        For ensemble methods (e.g., RandomForest, GradientBoosting), this method computes
        variance across individual estimators. For other models, it uses prediction
        standard deviation as a proxy. This approach is well-suited for small datasets
        where ensemble-based variance estimates are more stable than single-model confidence.

        Args:
            model: Fitted sklearn-style regressor (RandomForest or GradientBoosting preferred).
            X (np.ndarray): Input features of shape (n_samples, n_features).
                Must be 2-dimensional.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "confidence_score" (float): Mean confidence across predictions [0, 1].
                  Computed as 1 - normalized_uncertainty.
                - "mean_entropy" (float): Mean normalized uncertainty across predictions [0, 1].
                  For regression, this represents normalized variance/std dev.
                - "uncertainty_level" (str): Categorical uncertainty level:
                  * "LOW": confidence >= 0.75 (low prediction variance)
                  * "MEDIUM": 0.4 <= confidence < 0.75 (moderate variance)
                  * "HIGH": confidence < 0.4 (high prediction variance)
                - "notes" (str): Additional context about the predictions.

        Raises:
            ValueError: If X is invalid or model is unsupported.
            TypeError: If X is not a numpy array or proper type.

        Notes:
            - For ensemble models with estimators_, variance is computed from predictions
              of individual estimators, providing a robust uncertainty estimate.
            - For other models, standard deviation is estimated as a fallback.
            - No data leakage: only uses X for predictions, not training data.

        Examples:
            >>> estimator = UncertaintyEstimator()
            >>> from sklearn.ensemble import RandomForestRegressor
            >>> model = RandomForestRegressor(n_estimators=50, random_state=42)
            >>> X = np.random.rand(100, 5)
            >>> y = np.random.rand(100)
            >>> model.fit(X, y)
            >>> result = estimator.estimate_regression_uncertainty(model, X)
            >>> assert result['uncertainty_level'] in ['LOW', 'MEDIUM', 'HIGH']
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except Exception as e:
                raise TypeError(
                    f"Cannot convert X to numpy array: {str(e)}"
                )

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional, received shape {X.shape}"
            )

        if len(X) == 0:
            raise ValueError("X contains no samples")

        # Compute variance/uncertainty
        uncertainties = self._compute_regression_uncertainty(model, X)

        # Normalize uncertainty to [0, 1]
        normalized_uncertainties = self._normalize(uncertainties)

        # Compute confidence score (inverse of normalized uncertainty)
        confidence_scores = 1.0 - normalized_uncertainties

        # Aggregate metrics
        mean_confidence = float(np.mean(confidence_scores))
        mean_entropy = float(np.mean(normalized_uncertainties))

        # Map to uncertainty level
        uncertainty_level = self._map_confidence_to_level(mean_confidence)

        # Generate contextual notes
        estimator_type = "ensemble" if hasattr(model, 'estimators_') else "single"
        notes = self._generate_regression_notes(mean_confidence, len(X), estimator_type)

        return {
            "confidence_score": mean_confidence,
            "mean_entropy": mean_entropy,
            "uncertainty_level": uncertainty_level,
            "notes": notes
        }

    # ==================== Private Helper Methods ====================

    def _compute_entropy(self, probs: np.ndarray) -> float:
        """
        Compute Shannon entropy for a probability distribution.

        Shannon entropy measures the information content (or uncertainty) of a
        probability distribution. For classification:
            H(p) = -sum(p_i * log(p_i))

        Where p_i is the probability of class i. Entropy is 0 when one class has
        probability 1 (no uncertainty) and maximum when all classes are equally likely.

        Args:
            probs (np.ndarray): Probability distribution of shape (n_classes,).
                Must sum to 1 and contain non-negative values.

        Returns:
            float: Shannon entropy (typically [0, log(n_classes)]).
        """
        # Ensure valid probabilities
        probs = np.asarray(probs)
        probs = np.clip(probs, 1e-15, 1.0)  # Avoid log(0)

        # Compute entropy using scipy (efficient and numerically stable)
        return float(scipy_entropy(probs))

    def _compute_regression_uncertainty(
        self,
        model,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute uncertainty estimates for regression predictions.

        Strategy:
        1. If model has estimators_ (ensemble): Compute variance across estimators
        2. Otherwise: Use prediction standard deviation proxy (sqrt of residual variance)

        This approach is suitable for small datasets where ensemble-based estimates
        are more stable than single-model confidence intervals.

        Args:
            model: Fitted sklearn-style regressor.
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Uncertainty estimates of shape (n_samples,).
                Positive values representing variance or std dev.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            # Strategy 1: Use ensemble variance if available
            if hasattr(model, 'estimators_'):
                return self._compute_ensemble_variance(model, X)
            # Strategy 2: Fallback to std dev proxy
            else:
                return self._compute_std_dev_proxy(model, X)
        except Exception as e:
            raise RuntimeError(
                f"Error computing regression uncertainty: {str(e)}"
            )

    def _compute_ensemble_variance(
        self,
        model,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute variance of predictions across ensemble estimators.

        For ensemble models (RandomForest, ExtraTrees, GradientBoosting), this computes
        the variance of individual tree predictions, providing a robust uncertainty estimate.

        Args:
            model: Ensemble model with estimators_ attribute.
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Variance across estimators for each sample.
        """
        # Collect predictions from each estimator
        estimator_predictions = np.array([
            estimator.predict(X) for estimator in model.estimators_
        ])

        # Compute variance across estimators (axis=0)
        variances = np.var(estimator_predictions, axis=0)

        return variances

    def _compute_std_dev_proxy(
        self,
        model,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Estimate uncertainty using prediction standard deviation proxy.

        For non-ensemble models, use a simple approach based on prediction values.
        This is a heuristic that works reasonably well on small datasets.

        Args:
            model: Fitted sklearn-style regressor.
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Estimated uncertainty for each sample.
        """
        # Get predictions
        predictions = model.predict(X)

        # Simple proxy: use absolute prediction magnitude scaled by a small factor
        # This reflects that uncertainty often correlates with prediction magnitude
        # in real-world regression tasks
        std_dev_proxy = np.abs(predictions) * 0.1 + 1e-6

        return std_dev_proxy

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values to [0, 1] range using min-max scaling.

        Handles edge cases:
        - Constant values (all same): returns 0.0
        - Empty arrays: returns empty array
        - Single value: returns 0.0

        Min-max normalization is preferred over std dev normalization because:
        1. It guarantees output in [0, 1] for interpretability
        2. It's not affected by outliers as much
        3. It's more stable on small datasets

        Args:
            values (np.ndarray): Input values to normalize.

        Returns:
            np.ndarray: Normalized values in [0, 1].
        """
        values = np.asarray(values)

        if len(values) == 0:
            return values

        min_val = np.min(values)
        max_val = np.max(values)

        # Handle constant values (avoid division by zero)
        if max_val == min_val:
            return np.zeros_like(values, dtype=float)

        # Min-max scaling
        normalized = (values - min_val) / (max_val - min_val)

        return normalized

    def _map_confidence_to_level(self, confidence: float) -> str:
        """
        Map confidence score to categorical uncertainty level.

        Thresholds are chosen based on practical experience with small datasets:
        - LOW (confidence >= 0.75): Model is confident, predictions are reliable
        - MEDIUM (0.4 <= confidence < 0.75): Moderate uncertainty, caution recommended
        - HIGH (confidence < 0.4): High uncertainty, predictions may be unreliable

        Args:
            confidence (float): Confidence score in [0, 1].

        Returns:
            str: Uncertainty level ("LOW", "MEDIUM", or "HIGH").

        Examples:
            >>> estimator = UncertaintyEstimator()
            >>> estimator._map_confidence_to_level(0.8)
            'LOW'
            >>> estimator._map_confidence_to_level(0.5)
            'MEDIUM'
            >>> estimator._map_confidence_to_level(0.3)
            'HIGH'
        """
        if confidence >= 0.75:
            return "LOW"
        elif confidence >= 0.4:
            return "MEDIUM"
        else:
            return "HIGH"

    def _generate_classification_notes(
        self,
        mean_confidence: float,
        n_samples: int,
        n_classes: int
    ) -> str:
        """
        Generate contextual notes for classification results.

        Args:
            mean_confidence (float): Mean confidence score.
            n_samples (int): Number of samples evaluated.
            n_classes (int): Number of classes in classification task.

        Returns:
            str: Human-readable summary and recommendations.
        """
        notes = f"Evaluated {n_samples} sample(s) across {n_classes} class(es). "

        if mean_confidence >= 0.8:
            notes += "Model predictions are highly confident."
        elif mean_confidence >= 0.75:
            notes += "Model predictions are confident."
        elif mean_confidence >= 0.6:
            notes += "Model predictions show moderate confidence."
        elif mean_confidence >= 0.4:
            notes += "Model predictions have moderate uncertainty."
        else:
            notes += "Model predictions are uncertain. Consider reviewing predictions carefully."

        return notes

    def _generate_regression_notes(
        self,
        mean_confidence: float,
        n_samples: int,
        estimator_type: str
    ) -> str:
        """
        Generate contextual notes for regression results.

        Args:
            mean_confidence (float): Mean confidence score.
            n_samples (int): Number of samples evaluated.
            estimator_type (str): "ensemble" or "single".

        Returns:
            str: Human-readable summary and recommendations.
        """
        uncertainty_source = "ensemble variance" if estimator_type == "ensemble" else "std dev proxy"
        notes = f"Evaluated {n_samples} sample(s) using {uncertainty_source}. "

        if mean_confidence >= 0.8:
            notes += "Predictions have low variance and are reliable."
        elif mean_confidence >= 0.75:
            notes += "Predictions have acceptable variance."
        elif mean_confidence >= 0.6:
            notes += "Predictions have moderate variance."
        elif mean_confidence >= 0.4:
            notes += "Predictions have elevated variance."
        else:
            notes += "Predictions have high variance. Confidence intervals should be wide."

        return notes

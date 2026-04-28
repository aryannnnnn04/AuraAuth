"""
Distribution Shift Detection Module for AuraAuth AutoML System

Detects whether incoming data distributions differ significantly from training data
distributions using classical statistical techniques optimized for small datasets.

Distribution shift detection is critical in real-world ML systems to prevent unreliable
predictions on unseen data that violates assumptions learned during training. This module
provides explainable, model-agnostic shift detection without requiring training labels.

Author: AuraAuth Development Team
"""

import numpy as np
from typing import Dict, Any
from scipy.stats import ks_2samp


class DistributionShiftDetector:
    """
    Detects significant distribution shifts in incoming data using statistical tests.

    This class provides model-agnostic, label-free distribution shift detection suitable
    for small datasets. It combines mean shift analysis (parametric) with Kolmogorov-Smirnov
    testing (non-parametric) to robustly detect distributional changes.

    The detector is fitted on training data and then used to assess whether new data comes
    from the same distribution. This is essential for production ML systems where data drift
    can cause performance degradation.

    Attributes:
        _is_fitted (bool): Whether fit() has been called.
        _train_mean (np.ndarray): Feature-wise mean of training data.
        _train_std (np.ndarray): Feature-wise standard deviation of training data.
        _train_data (np.ndarray): Training data samples (stored for KS test).
        _n_features (int): Number of features in training data.

    Example:
        >>> from backend.core.distribution_shift_detector import DistributionShiftDetector
        >>> import numpy as np
        >>>
        >>> detector = DistributionShiftDetector()
        >>> X_train = np.random.normal(0, 1, (500, 5))
        >>> detector.fit(X_train)
        >>>
        >>> X_new = np.random.normal(2, 1, (100, 5))  # Shifted mean
        >>> result = detector.detect_shift(X_new)
        >>> print(result['shift_level'])  # "HIGH" or "MEDIUM"
    """

    def __init__(self) -> None:
        """Initialize the DistributionShiftDetector (unfitted state)."""
        self._is_fitted = False
        self._train_mean = None
        self._train_std = None
        self._train_data = None
        self._n_features = None

    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the detector on training data by computing and storing feature statistics.

        This method stores training statistics (mean, std) and the training data itself
        (for non-parametric KS testing). This is the baseline distribution against which
        new data will be compared.

        Args:
            X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features).
                Must be 2-dimensional. Features should be numerical (int or float).

        Raises:
            ValueError: If X_train is not 2-dimensional or contains no samples.
            TypeError: If X_train cannot be converted to numpy array.

        Notes:
            - Training data is stored for KS testing; consider memory implications for
              very large datasets (though this module targets small datasets).
            - Features with zero variance are handled gracefully (std = 1e-10 used).

        Examples:
            >>> detector = DistributionShiftDetector()
            >>> X_train = np.random.randn(500, 10)
            >>> detector.fit(X_train)
            >>> # Now detector is ready to detect shifts
        """
        # Convert to numpy array if needed
        if not isinstance(X_train, np.ndarray):
            try:
                X_train = np.asarray(X_train)
            except Exception as e:
                raise TypeError(
                    f"Cannot convert X_train to numpy array: {str(e)}"
                )

        # Validate input shape
        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be 2-dimensional, received shape {X_train.shape}"
            )

        if len(X_train) == 0:
            raise ValueError("X_train contains no samples")

        # Store training statistics
        self._train_data = X_train.copy()
        self._train_mean = np.mean(X_train, axis=0)
        self._train_std = np.std(X_train, axis=0)

        # Handle zero-variance features (replace with small positive value)
        self._train_std = np.where(
            self._train_std < 1e-10,
            1e-10,
            self._train_std
        )

        self._n_features = X_train.shape[1]
        self._is_fitted = True

    def detect_shift(
        self,
        X_new: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect distribution shift in new data compared to training data.

        This method combines two statistical approaches:
        1. Parametric: Mean shift detection (assumes normality for interpretability)
        2. Non-parametric: Kolmogorov-Smirnov test (distribution-free, robust)

        The final shift score aggregates both measures, normalized to [0, 1].

        Args:
            X_new (np.ndarray): New feature matrix of shape (n_samples, n_features).
                Must have same number of features as training data.
            alpha (float): Significance level for KS test. Default 0.05.
                Lower values are more conservative (less likely to flag shift).

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "shift_score" (float): Aggregate shift score [0, 1].
                  Higher values indicate more significant shifts.
                - "shift_level" (str): Categorical severity:
                  * "LOW": shift_score < 0.3 (minimal distributional change)
                  * "MEDIUM": 0.3 <= shift_score < 0.6 (moderate change)
                  * "HIGH": shift_score >= 0.6 (significant distribution change)
                - "shifted_feature_ratio" (float): Proportion of features with significant
                  KS test p-values (p < alpha). Range [0, 1].
                - "notes" (str): Human-readable summary and recommendations.

        Raises:
            ValueError: If detector not fitted, X_new shape mismatch, or invalid input.
            TypeError: If X_new cannot be converted to numpy array.

        Notes:
            - The shift score is computed as weighted average of:
              * Mean shift magnitude (normalized)
              * Proportion of shifted features (KS test)
            - KS test p-values < alpha indicate features with shifted distributions.
            - This method requires no labels and is purely unsupervised.

        Examples:
            >>> detector = DistributionShiftDetector()
            >>> X_train = np.random.normal(0, 1, (500, 5))
            >>> detector.fit(X_train)
            >>>
            >>> # Small shift: slightly different mean
            >>> X_small_shift = np.random.normal(0.1, 1, (50, 5))
            >>> result1 = detector.detect_shift(X_small_shift)
            >>> print(result1['shift_level'])  # Likely "LOW"
            >>>
            >>> # Large shift: very different mean and variance
            >>> X_large_shift = np.random.normal(3, 2, (50, 5))
            >>> result2 = detector.detect_shift(X_large_shift)
            >>> print(result2['shift_level'])  # Likely "HIGH"
        """
        # Check if fitted
        if not self._is_fitted:
            raise ValueError(
                "DistributionShiftDetector must be fitted using fit() before calling detect_shift()"
            )

        # Validate input
        if not isinstance(X_new, np.ndarray):
            try:
                X_new = np.asarray(X_new)
            except Exception as e:
                raise TypeError(
                    f"Cannot convert X_new to numpy array: {str(e)}"
                )

        if X_new.ndim != 2:
            raise ValueError(
                f"X_new must be 2-dimensional, received shape {X_new.shape}"
            )

        if X_new.shape[1] != self._n_features:
            raise ValueError(
                f"X_new has {X_new.shape[1]} features, expected {self._n_features}"
            )

        if len(X_new) == 0:
            raise ValueError("X_new contains no samples")

        # ===== A. Mean Shift Detection =====
        mean_shift_score = self._compute_mean_shift_score(X_new)

        # ===== B. Kolmogorov-Smirnov Test =====
        shifted_feature_ratio = self._compute_ks_shift_ratio(X_new, alpha)

        # ===== C. Aggregate Shift Score =====
        # Weighted combination of mean shift and feature-level shifts
        # Weights: 50% mean shift, 50% KS feature ratio
        aggregate_score = 0.5 * mean_shift_score + 0.5 * shifted_feature_ratio

        # Normalize to [0, 1]
        aggregate_score = float(np.clip(aggregate_score, 0.0, 1.0))

        # ===== D. Map to Severity Level =====
        shift_level = self._map_shift_level(aggregate_score)

        # Generate notes
        notes = self._generate_notes(
            aggregate_score,
            shifted_feature_ratio,
            len(X_new)
        )

        return {
            "shift_score": aggregate_score,
            "shift_level": shift_level,
            "shifted_feature_ratio": shifted_feature_ratio,
            "notes": notes
        }

    # ==================== Private Helper Methods ====================

    def _compute_mean_shift_score(self, X_new: np.ndarray) -> float:
        """
        Compute normalized mean shift magnitude across all features.

        The mean shift for each feature is the absolute difference between new mean
        and training mean, normalized by the training standard deviation. This provides
        a scale-invariant measure of distributional change.

        Args:
            X_new (np.ndarray): New data of shape (n_samples, n_features).

        Returns:
            float: Mean shift score in [0, 1]. Higher values indicate larger shifts.
        """
        new_mean = np.mean(X_new, axis=0)

        # Compute feature-wise mean shift normalized by training std
        mean_shifts = np.abs(new_mean - self._train_mean) / self._train_std

        # Normalize across features: mean shift with clipping
        mean_shift_score = float(np.mean(np.clip(mean_shifts, 0.0, 1.0)))

        return mean_shift_score

    def _compute_ks_shift_ratio(
        self,
        X_new: np.ndarray,
        alpha: float
    ) -> float:
        """
        Compute proportion of features with significant KS test results.

        The Kolmogorov-Smirnov test is a non-parametric test that compares two
        distributions. For each feature, we test whether the distribution of the
        feature in training data differs from the distribution in new data.

        A feature is considered "shifted" if the KS test p-value is less than alpha.

        Args:
            X_new (np.ndarray): New data of shape (n_samples, n_features).
            alpha (float): Significance level for KS test.

        Returns:
            float: Proportion of features with significant shifts [0, 1].
        """
        n_shifted = 0

        for feature_idx in range(self._n_features):
            # Perform KS test
            statistic, p_value = ks_2samp(
                self._train_data[:, feature_idx],
                X_new[:, feature_idx]
            )

            # Count features with p-value < alpha
            if p_value < alpha:
                n_shifted += 1

        # Return ratio of shifted features
        shifted_ratio = float(n_shifted / self._n_features)

        return shifted_ratio

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values to [0, 1] range using min-max scaling.

        Handles edge cases such as constant values.

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

        # Handle constant values
        if max_val == min_val:
            return np.zeros_like(values, dtype=float)

        normalized = (values - min_val) / (max_val - min_val)

        return normalized

    def _map_shift_level(self, shift_score: float) -> str:
        """
        Map shift score to categorical severity level.

        Thresholds are chosen based on practical experience with small datasets:
        - LOW (score < 0.3): Minimal distributional change, predictions likely reliable
        - MEDIUM (0.3 <= score < 0.6): Moderate change, monitor performance
        - HIGH (score >= 0.6): Significant change, predictions may be unreliable

        Args:
            shift_score (float): Shift score in [0, 1].

        Returns:
            str: Severity level ("LOW", "MEDIUM", or "HIGH").

        Examples:
            >>> detector = DistributionShiftDetector()
            >>> detector._map_shift_level(0.2)
            'LOW'
            >>> detector._map_shift_level(0.45)
            'MEDIUM'
            >>> detector._map_shift_level(0.7)
            'HIGH'
        """
        if shift_score < 0.3:
            return "LOW"
        elif shift_score < 0.6:
            return "MEDIUM"
        else:
            return "HIGH"

    def _generate_notes(
        self,
        shift_score: float,
        shifted_feature_ratio: float,
        n_samples: int
    ) -> str:
        """
        Generate contextual notes and recommendations.

        Args:
            shift_score (float): Aggregate shift score [0, 1].
            shifted_feature_ratio (float): Proportion of shifted features [0, 1].
            n_samples (int): Number of new samples evaluated.

        Returns:
            str: Human-readable summary and actionable insights.
        """
        n_shifted_features = int(np.round(shifted_feature_ratio * self._n_features))

        notes = (
            f"Evaluated {n_samples} new sample(s) against {self._n_features} training feature(s). "
            f"KS test detected {n_shifted_features} shifted feature(s) ({shifted_feature_ratio:.1%}). "
        )

        if shift_score < 0.2:
            notes += "Distribution appears stable. Predictions should be reliable."
        elif shift_score < 0.3:
            notes += "Minimal shift detected. Continue monitoring."
        elif shift_score < 0.5:
            notes += "Moderate shift detected. Evaluate model performance on this data."
        elif shift_score < 0.6:
            notes += "Significant shift detected. Retraining may be beneficial."
        else:
            notes += "Major distribution shift detected. Model retraining is recommended."

        return notes

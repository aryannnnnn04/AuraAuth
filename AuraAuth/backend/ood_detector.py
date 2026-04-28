"""
Out-of-Distribution (OOD) Detector Module: Detect anomalous and novel inputs.

This module provides:
- Detection of out-of-distribution samples
- Anomaly detection in feature space
- Distance-based and density-based methods
- Integration with uncertainty estimates
- Handling of novel classes and extrapolation regions
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class OODDetector:
    """
    Detects out-of-distribution (OOD) and anomalous input samples.
    
    Critical for production ML systems, especially with small datasets:
    - Models trained on limited data may encounter novel inputs
    - OOD samples often cause degraded predictions
    - Enables rejection sampling for uncertain/anomalous predictions
    
    Methods:
    - Isolation Forest: Unsupervised anomaly detection
    - Mahalanobis Distance: Statistical distance in feature space
    - K-NN Distance: Distance to training data
    - Statistical: Detect samples far from training distribution
    
    Attributes:
        method: OOD detection method to use
        contamination: Expected proportion of OOD samples in training data
        threshold: Threshold for OOD classification
    """
    
    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        random_state: int = 42,
        verbose: bool = False
    ):
        """
        Initialize OOD Detector.
        
        Args:
            method: Detection method
                Options: "isolation_forest", "mahalanobis", "knn_distance", "statistical"
            contamination: Expected OOD proportion in training data (0-1)
                          Used to set detection threshold
            random_state: Seed for reproducibility
            verbose: Print detection details
        """
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted detector model
        self.detector: Optional[BaseEstimator] = None
        
        # Training data statistics
        self.X_train: Optional[pd.DataFrame] = None
        self.train_feature_stats: Optional[Dict[str, Any]] = None
        self.threshold: Optional[float] = None
    
    def fit(
        self,
        X_train: pd.DataFrame
    ) -> 'OODDetector':
        """
        Fit distribution shift detector on training data.
        
        Args:
            X_train: Training feature matrix
            
        Returns:
            self (for method chaining)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        self.X_train = X_train.copy()
        
        if self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                verbose=self.verbose
            )
            self.detector.fit(X_train)
            # Set threshold based on contamination
            train_scores = self.detector.score_samples(X_train)
            self.threshold = np.percentile(train_scores, self.contamination * 100)
            
        elif self.method == "mahalanobis":
            # Compute training statistics
            self.train_feature_stats = {
                'mean': X_train.mean().values,
                'cov': np.cov(X_train.T),
                'cov_inv': np.linalg.pinv(np.cov(X_train.T))
            }
            # Threshold from chi-square distribution
            from scipy.stats import chi2
            df = X_train.shape[1]
            self.threshold = chi2.ppf(1 - self.contamination, df)
            
        elif self.method == "knn_distance":
            from sklearn.neighbors import KDTree
            self.detector = KDTree(X_train.values)
            # Find k-NN distances in training set
            distances, _ = self.detector.query(X_train.values, k=6)  # k+1 includes self
            self.threshold = np.percentile(distances[:, -1], (1 - self.contamination) * 100)
            
        elif self.method == "statistical":
            # Compute statistical bounds
            self.train_feature_stats = {
                'mean': X_train.mean().values,
                'std': X_train.std().values,
                'columns': X_train.columns.tolist()
            }
            from scipy.stats import norm
            self.threshold = norm.ppf(1 - (self.contamination / X_train.shape[1]))
        
        logger.info(f"OOD detector fitted with method='{self.method}'")
        return self
    
    def detect_ood(
        self,
        X: pd.DataFrame,
        return_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Detect out-of-distribution samples.
        
        Args:
            X: Feature matrix to analyze
            return_scores: Include OOD scores in output
            
        Returns:
            Dictionary with OOD detection results
        """
        if self.detector is None and self.train_feature_stats is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        if self.method == "isolation_forest":
            is_ood, scores = self._detect_ood_isolation_forest(X)
        elif self.method == "mahalanobis":
            is_ood, scores = self._detect_ood_mahalanobis(X)
        elif self.method == "knn_distance":
            is_ood, scores = self._detect_ood_knn_distance(X)
        elif self.method == "statistical":
            is_ood, scores = self._detect_ood_statistical(X)
        else:
            raise ValueError(f"Unknown OOD detection method: {self.method}")
        
        ood_percentage = (is_ood.sum() / len(X)) * 100 if len(X) > 0 else 0
        
        result = {
            "is_ood": is_ood,
            "ood_percentage": ood_percentage,
            "method_used": self.method,
            "threshold": self.threshold
        }
        
        if return_scores:
            result["ood_scores"] = scores
        
        return result
    
    def _detect_ood_isolation_forest(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Isolation Forest based OOD detection.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (is_ood_boolean, ood_scores)
        """
        scores = self.detector.score_samples(X)
        # Normalize to [0, 1] (lower score = more anomalous)
        scores_normalized = 1.0 / (1.0 + np.exp(scores))  # Sigmoid normalization
        is_ood = scores < self.threshold
        return is_ood, scores_normalized
    
    def _detect_ood_mahalanobis(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mahalanobis distance based OOD detection.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (is_ood_boolean, ood_scores)
        """
        mean = self.train_feature_stats['mean']
        cov_inv = self.train_feature_stats['cov_inv']
        
        # Center data
        X_centered = X.values - mean
        
        # Compute Mahalanobis distances
        distances = np.sqrt(np.diag(X_centered @ cov_inv @ X_centered.T))
        
        # Normalize to [0, 1]
        distances_normalized = 1.0 / (1.0 + np.exp(-distances))
        
        is_ood = distances > self.threshold
        return is_ood, distances_normalized
    
    def _detect_ood_knn_distance(
        self,
        X: pd.DataFrame,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-nearest neighbor distance based OOD detection.
        
        Args:
            X: Feature matrix
            k: Number of nearest neighbors to check
            
        Returns:
            Tuple of (is_ood_boolean, ood_scores)
        """
        distances, _ = self.detector.query(X.values, k=k)
        # Use distance to k-th neighbor
        knn_distances = distances[:, -1]
        
        # Normalize to [0, 1]
        max_dist = np.max(knn_distances) if len(knn_distances) > 0 else 1.0
        distances_normalized = knn_distances / (max_dist + 1e-10)
        
        is_ood = knn_distances > self.threshold
        return is_ood, distances_normalized
    
    def _detect_ood_statistical(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Statistical distribution based OOD detection.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (is_ood_boolean, ood_scores)
        """
        mean = self.train_feature_stats['mean']
        std = self.train_feature_stats['std']
        
        # Compute z-scores
        z_scores = np.abs((X.values - mean) / (std + 1e-10))
        
        # OOD score = maximum z-score per sample
        max_z_scores = np.max(z_scores, axis=1)
        
        # Normalize to [0, 1]
        max_z_normalized = 1.0 / (1.0 + np.exp(-max_z_scores))
        
        is_ood = max_z_scores > self.threshold
        return is_ood, max_z_normalized
    
    def integrate_with_uncertainty(
        self,
        confidence: np.ndarray,
        is_ood: np.ndarray
    ) -> np.ndarray:
        """
        Combine OOD detection with uncertainty estimates.
        
        Args:
            confidence: Confidence scores from UncertaintyEstimator
            is_ood: OOD boolean array from detect_ood
            
        Returns:
            Modified confidence scores, penalizing OOD samples
        """
        modified_confidence = confidence.copy()
        modified_confidence[is_ood] *= 0.5  # Reduce confidence for OOD samples
        return modified_confidence
    
    def get_ood_summary(
        self,
        is_ood: np.ndarray,
        ood_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate OOD detection summary.
        
        Args:
            is_ood: OOD boolean array
            ood_scores: Numerical OOD scores (optional)
            
        Returns:
            Dictionary with summary statistics
        """
        ood_count = is_ood.sum()
        ood_percentage = (ood_count / len(is_ood)) * 100
        ood_indices = np.where(is_ood)[0].tolist()
        
        summary = {
            "ood_count": int(ood_count),
            "ood_percentage": float(ood_percentage),
            "ood_sample_indices": ood_indices,
            "threshold_used": self.threshold
        }
        
        if ood_scores is not None:
            summary["score_distribution"] = {
                "mean": float(np.mean(ood_scores)),
                "std": float(np.std(ood_scores)),
                "min": float(np.min(ood_scores)),
                "max": float(np.max(ood_scores))
            }
        
        return summary
    
    def flag_unreliable_predictions(
        self,
        predictions: np.ndarray,
        confidence: np.ndarray,
        is_ood: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> np.ndarray:
        """
        Flag predictions that are unreliable due to OOD or low confidence.
        
        Args:
            predictions: Model predictions
            confidence: Confidence scores
            is_ood: OOD boolean array
            confidence_threshold: Confidence threshold
            
        Returns:
            Boolean array, True = unreliable prediction
        """
        unreliable = is_ood | (confidence < confidence_threshold)
        return unreliable
    
    def plot_ood_scores(
        self,
        ood_scores: np.ndarray,
        is_ood: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize OOD score distribution.
        
        Args:
            ood_scores: Numerical OOD scores
            is_ood: OOD boolean array
            save_path: Path to save figure, displays if None
            
        TODO: Create histogram showing:
        - Distribution of OOD scores for in-distribution samples
        - Distribution of OOD scores for OOD samples
        - Threshold line
        - Separation quality visualization
        """
        pass
    
    def export_detection_results(
        self,
        X: pd.DataFrame,
        is_ood: np.ndarray,
        ood_scores: Optional[np.ndarray] = None,
        filepath: str = "ood_results.csv"
    ) -> None:
        """
        Export OOD detection results to file.
        
        Args:
            X: Original feature matrix
            is_ood: OOD detection results
            ood_scores: OOD scores (optional)
            filepath: Path to save CSV
        """
        results_df = X.copy()
        results_df['is_ood'] = is_ood
        
        if ood_scores is not None:
            results_df['ood_score'] = ood_scores
        
        results_df.to_csv(filepath, index=False)
        print(f"OOD detection results saved to {filepath}")

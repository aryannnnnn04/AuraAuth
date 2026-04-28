"""
Model Selection Module: Reliability-aware model selection beyond accuracy.

This module provides:
- Multi-criteria model selection (accuracy, stability, variance)
- Stability and robustness assessment
- Ensemble strategies for improved reliability
- Model-specific strengths/weaknesses analysis
- Selection based on dataset characteristics
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass
class ModelMetrics:
    """
    Comprehensive metrics for model evaluation beyond accuracy.
    
    Attributes:
        accuracy: Classification accuracy or R² for regression
        cv_score_mean: Mean cross-validation score
        cv_score_std: Standard deviation of CV scores (stability metric)
        train_test_gap: Difference between train and test scores (overfitting indicator)
        prediction_variance: Model prediction variance (uncertainty)
        training_time: Time to train model (seconds)
        prediction_time: Time to make predictions (seconds)
        stability_score: Composite stability metric (0-1)
        reliability_score: Composite reliability metric (0-1)
    """
    model_name: str
    accuracy: float
    cv_score_mean: float
    cv_score_std: float
    train_test_gap: float
    prediction_variance: float
    training_time: float
    prediction_time: float
    stability_score: float = 0.0
    reliability_score: float = 0.0
    additional_metrics: Dict[str, float] = None


class ModelSelector:
    """
    Reliability-aware model selection engine.
    
    Selects optimal model based on multiple criteria beyond accuracy:
    - Model stability (low CV variance)
    - Generalization ability (small train-test gap)
    - Prediction confidence (low uncertainty)
    - Computational efficiency
    - Dataset characteristics alignment
    
    For small datasets, stability and variance are critical:
    - Models with high CV variance may not generalize well
    - Prefer simpler models that stabilize faster
    - Consider ensemble methods for robustness
    
    Attributes:
        weight_accuracy: Weight for accuracy in selection (0-1)
        weight_stability: Weight for stability in selection
        weight_efficiency: Weight for computational efficiency
    """
    
    def __init__(
        self,
        weight_accuracy: float = 0.4,
        weight_stability: float = 0.4,
        weight_efficiency: float = 0.2,
        min_stability_threshold: float = 0.5
    ):
        """
        Initialize Model Selector.
        
        Args:
            weight_accuracy: Importance of accuracy (0-1)
            weight_stability: Importance of stability/variance (0-1)
            weight_efficiency: Importance of speed (0-1)
            min_stability_threshold: Minimum acceptable stability score
            
        Notes:
            - Weights should sum to ~1.0 (or will be normalized)
            - For small datasets, increase weight_stability
        """
        # Normalize weights
        total = weight_accuracy + weight_stability + weight_efficiency
        self.weight_accuracy = weight_accuracy / total
        self.weight_stability = weight_stability / total
        self.weight_efficiency = weight_efficiency / total
        self.min_stability_threshold = min_stability_threshold
        
        self.model_metrics: Dict[str, ModelMetrics] = {}
    
    def evaluate_model(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_scores: np.ndarray,
        model_name: str,
        training_time: float = 0.0,
        prediction_time: float = 0.0
    ) -> ModelMetrics:
        """
        Comprehensively evaluate a trained model.
        
        Args:
            model: Trained model instance
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            cv_scores: Array of cross-validation scores
            model_name: Name of model for tracking
            training_time: Time to train (seconds)
            prediction_time: Time to predict (seconds)
            
        Returns:
            ModelMetrics object with comprehensive evaluation
        """
        from sklearn.metrics import accuracy_score
        
        # 1. Primary metrics
        cv_score_mean = np.mean(cv_scores) if len(cv_scores) > 0 else 0.0
        cv_score_std = np.std(cv_scores) if len(cv_scores) > 0 else 0.0
        
        # Get train and test scores
        train_preds = model.predict(X_train)
        train_score = accuracy_score(y_train, train_preds)
        
        test_preds = model.predict(X_test)
        test_score = accuracy_score(y_test, test_preds)
        
        # 2. Train-test gap (overfitting indicator)
        train_test_gap = train_score - test_score
        
        # 3. Stability score based on CV variance
        # Lower CV variance = higher stability (0 to 1 scale)
        # Max CV std ~0.2 indicates unstable model, 0.02 indicates very stable
        max_cv_std = 0.2
        normalized_cv_std = min(cv_score_std / max_cv_std, 1.0)
        stability_from_cv = 1.0 - normalized_cv_std
        
        # Penalize large train-test gap
        max_gap = 0.3
        normalized_gap = min(train_test_gap / max_gap, 1.0)
        stability_from_gap = 1.0 - normalized_gap
        
        # Combined stability score
        stability_score = 0.6 * stability_from_cv + 0.4 * stability_from_gap
        stability_score = np.clip(stability_score, 0.0, 1.0)
        
        # 4. Prediction variance (ensemble or bootstrap variance if available)
        # For now, use CV score variance as proxy
        prediction_variance = cv_score_std
        
        # 5. Composite reliability score
        # Prioritize: stability > accuracy > efficiency for small datasets
        reliability_score = (
            self.weight_accuracy * test_score +
            self.weight_stability * stability_score +
            self.weight_efficiency * (1.0 - min(training_time / 60.0, 1.0))  # Penalize long training
        )
        reliability_score = np.clip(reliability_score, 0.0, 1.0)
        
        # Create and store metrics
        metrics = ModelMetrics(
            model_name=model_name,
            accuracy=test_score,
            cv_score_mean=cv_score_mean,
            cv_score_std=cv_score_std,
            train_test_gap=train_test_gap,
            prediction_variance=prediction_variance,
            training_time=training_time,
            prediction_time=prediction_time,
            stability_score=stability_score,
            reliability_score=reliability_score,
            additional_metrics={
                "train_score": train_score,
                "test_score": test_score,
                "stability_from_cv": stability_from_cv,
                "stability_from_gap": stability_from_gap
            }
        )
        
        # Store metrics
        self.model_metrics[model_name] = metrics
        
        return metrics
    
    def select_best_model(
        self,
        results_dict: Dict[str, Dict[str, float]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select best model using multi-criteria weighted scoring.
        
        Args:
            results_dict: AutoMLEngine output with mean_score and std_score per model
                Format: {"model_name": {"mean_score": float, "std_score": float}}
                
        Returns:
            Tuple of (best_model_name, ranking_dict)
        """
        # Filter out models that failed (contain "error" key instead of scores)
        valid_results = {
            m: v for m, v in results_dict.items()
            if isinstance(v, dict) and "mean_score" in v
        }
        if not valid_results:
            raise ValueError("All models failed during AutoML optimization.")
        
        # Extract scores
        model_names = list(valid_results.keys())
        mean_scores = np.array([valid_results[m]["mean_score"] for m in model_names])
        std_scores = np.array([valid_results[m]["std_score"] for m in model_names])
        
        # Normalize scores
        norm_accuracy = self._normalize(mean_scores)
        # Stability: inverse of std (lower std = higher stability)
        # Convert to 0-1 scale: 1 - normalized(std)
        norm_stability = 1.0 - self._normalize(std_scores)
        
        # Efficiency: approximate based on model type (simpler models are faster)
        # Logistic Regression (fastest) -> Random Forest -> XGBoost/LightGBM
        efficiency_map = {
            "logistic_regression": 1.0,
            "random_forest": 0.7,
            "xgboost": 0.5,
            "lightgbm": 0.5
        }
        norm_efficiency = np.array([
            efficiency_map.get(m, 0.5) for m in model_names
        ])
        
        # Weighted composite score
        composite_scores = (
            self.weight_accuracy * norm_accuracy +
            self.weight_stability * norm_stability +
            self.weight_efficiency * norm_efficiency
        )
        
        # Create ranking
        ranking = []
        for i, model_name in enumerate(model_names):
            ranking.append({
                "rank": i + 1,
                "model_name": model_name,
                "mean_score": mean_scores[i],
                "std_score": std_scores[i],
                "norm_accuracy": norm_accuracy[i],
                "norm_stability": norm_stability[i],
                "norm_efficiency": norm_efficiency[i],
                "composite_score": composite_scores[i]
            })
        
        # Sort by composite score
        ranking.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Update ranks
        for i, item in enumerate(ranking):
            item["rank"] = i + 1
        
        best_model_name = ranking[0]["model_name"]
        
        return best_model_name, {
            "best_model": best_model_name,
            "ranking": ranking,
            "best_score": ranking[0]["composite_score"]
        }
    
    def rank_models(
        self,
        results_dict: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Rank all models by composite reliability score.
        
        Args:
            results_dict: AutoMLEngine output
                Format: {"model_name": {"mean_score": float, "std_score": float}}
                
        Returns:
            DataFrame with models ranked by various criteria
        """
        # Filter out models that failed
        valid_results = {
            m: v for m, v in results_dict.items()
            if isinstance(v, dict) and "mean_score" in v
        }
        if not valid_results:
            raise ValueError("All models failed during AutoML optimization.")
        
        # Extract scores
        model_names = list(valid_results.keys())
        mean_scores = np.array([valid_results[m]["mean_score"] for m in model_names])
        std_scores = np.array([valid_results[m]["std_score"] for m in model_names])
        
        # Normalize scores
        norm_accuracy = self._normalize(mean_scores)
        norm_stability = 1.0 - self._normalize(std_scores)
        
        # Efficiency approximation
        efficiency_map = {
            "logistic_regression": 1.0,
            "random_forest": 0.7,
            "xgboost": 0.5,
            "lightgbm": 0.5
        }
        norm_efficiency = np.array([
            efficiency_map.get(m, 0.5) for m in model_names
        ])
        
        # Composite scores
        composite_scores = (
            self.weight_accuracy * norm_accuracy +
            self.weight_stability * norm_stability +
            self.weight_efficiency * norm_efficiency
        )
        
        # Create ranking dataframe
        ranking_data = []
        for i, model_name in enumerate(model_names):
            ranking_data.append({
                "Model": model_name,
                "Mean_Score": f"{mean_scores[i]:.4f}",
                "Std_Score": f"{std_scores[i]:.4f}",
                "Accuracy_Norm": f"{norm_accuracy[i]:.3f}",
                "Stability_Norm": f"{norm_stability[i]:.3f}",
                "Efficiency_Norm": f"{norm_efficiency[i]:.3f}",
                "Composite_Score": f"{composite_scores[i]:.4f}",
                "Composite_Score_Value": composite_scores[i]
            })
        
        # Sort by composite score
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values("Composite_Score_Value", ascending=False).reset_index(drop=True)
        ranking_df["Rank"] = range(1, len(ranking_df) + 1)
        ranking_df = ranking_df[["Rank", "Model", "Mean_Score", "Std_Score", 
                                   "Accuracy_Norm", "Stability_Norm", "Efficiency_Norm", "Composite_Score"]]
        
        return ranking_df
    
    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values to 0-1 range using min-max scaling.
        
        Args:
            values: Array of numeric values
            
        Returns:
            Normalized array in range [0, 1]
        """
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val == min_val:
            # All values are the same
            return np.ones_like(values, dtype=float)
        
        return (values - min_val) / (max_val - min_val)
    
    def analyze_model_strengths(
        self,
        metrics: ModelMetrics,
        dataset_characteristics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze model strengths and weaknesses.

        Args:
            metrics: Model metrics to analyze
            dataset_characteristics: Optional dataset properties for context

        Returns:
            Dictionary with strengths, weaknesses, recommendations
        """
        strengths: List[str] = []
        weaknesses: List[str] = []
        recommendations: List[str] = []

        # Accuracy
        if metrics.accuracy >= 0.9:
            strengths.append("High accuracy (>= 0.90)")
        elif metrics.accuracy < 0.7:
            weaknesses.append("Low accuracy (< 0.70)")
            recommendations.append("Consider more complex models or additional features")

        # Stability
        if metrics.cv_score_std < 0.05:
            strengths.append("Stable cross-validation performance (low variance)")
        elif metrics.cv_score_std > 0.1:
            weaknesses.append("Unstable CV performance (high variance)")
            recommendations.append("Try regularization or simpler models")

        # Overfitting
        if metrics.train_test_gap < 0.05:
            strengths.append("Good generalization (small train-test gap)")
        elif metrics.train_test_gap > 0.15:
            weaknesses.append(f"Possible overfitting (train-test gap: {metrics.train_test_gap:.2f})")
            recommendations.append("Increase regularization or reduce model complexity")

        # Training time
        if metrics.training_time < 10:
            strengths.append("Fast training")
        elif metrics.training_time > 120:
            weaknesses.append("Slow training (> 2 min)")

        # Dataset-aware insights
        if dataset_characteristics:
            n_samples = dataset_characteristics.get("n_samples", 0)
            if n_samples < 500 and metrics.cv_score_std > 0.08:
                recommendations.append(
                    "Small dataset with high variance — prefer simpler models"
                )

        return {
            "model_name": metrics.model_name,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
        }
    
    def recommend_ensemble(
        self,
        top_models: List[Tuple[BaseEstimator, ModelMetrics]],
        ensemble_type: str = "voting"
    ) -> BaseEstimator:
        """
        Create an ensemble of the top models.

        Args:
            top_models: List of (model, metrics) tuples, ranked by quality
            ensemble_type: Type of ensemble ("voting", "stacking", "averaging")

        Returns:
            Ensemble model combining top performers

        Raises:
            ValueError: If fewer than 2 models provided
        """
        if len(top_models) < 2:
            raise ValueError("Need at least 2 models for an ensemble")

        from sklearn.ensemble import VotingClassifier

        estimators = [
            (m.model_name, model) for model, m in top_models
        ]

        if ensemble_type == "voting":
            return VotingClassifier(
                estimators=estimators,
                voting="soft",
            )

        # Default to soft-voting for unsupported types
        return VotingClassifier(
            estimators=estimators,
            voting="soft",
        )
    
    def get_model_comparison_report(self) -> str:
        """
        Generate human-readable comparison report of all evaluated models.

        Returns:
            Formatted text report comparing all evaluated models
        """
        if not self.model_metrics:
            return "No models have been evaluated yet."

        lines = [
            "=" * 60,
            "MODEL COMPARISON REPORT",
            "=" * 60,
            "",
        ]

        sorted_models = sorted(
            self.model_metrics.values(),
            key=lambda m: m.reliability_score,
            reverse=True,
        )

        for rank, m in enumerate(sorted_models, 1):
            lines.append(f"Rank {rank}: {m.model_name}")
            lines.append(f"  Accuracy:          {m.accuracy:.4f}")
            lines.append(f"  CV Mean:           {m.cv_score_mean:.4f} (+/- {m.cv_score_std:.4f})")
            lines.append(f"  Stability Score:   {m.stability_score:.4f}")
            lines.append(f"  Reliability Score: {m.reliability_score:.4f}")
            lines.append(f"  Train-Test Gap:    {m.train_test_gap:.4f}")
            lines.append(f"  Training Time:     {m.training_time:.2f}s")
            lines.append("")

        lines.append("=" * 60)
        best = sorted_models[0]
        lines.append(f"Best Model: {best.model_name} (reliability {best.reliability_score:.4f})")
        lines.append("=" * 60)

        return "\n".join(lines)
    
    def export_ranking(self, filepath: str) -> None:
        """
        Export model ranking and metrics to a file.

        Args:
            filepath: Path to save results (.csv, .json, or .txt)

        Raises:
            ValueError: If file extension is not supported
        """
        import json
        from datetime import datetime

        if not self.model_metrics:
            raise ValueError("No models have been evaluated yet.")

        sorted_models = sorted(
            self.model_metrics.values(),
            key=lambda m: m.reliability_score,
            reverse=True,
        )

        if filepath.endswith(".csv"):
            rows = []
            for rank, m in enumerate(sorted_models, 1):
                rows.append({
                    "rank": rank,
                    "model_name": m.model_name,
                    "accuracy": m.accuracy,
                    "cv_mean": m.cv_score_mean,
                    "cv_std": m.cv_score_std,
                    "stability_score": m.stability_score,
                    "reliability_score": m.reliability_score,
                    "training_time": m.training_time,
                })
            pd.DataFrame(rows).to_csv(filepath, index=False)

        elif filepath.endswith(".json"):
            data = {
                "timestamp": datetime.now().isoformat(),
                "ranking": [],
            }
            for rank, m in enumerate(sorted_models, 1):
                data["ranking"].append({
                    "rank": rank,
                    "model_name": m.model_name,
                    "accuracy": m.accuracy,
                    "cv_mean": m.cv_score_mean,
                    "cv_std": m.cv_score_std,
                    "stability_score": m.stability_score,
                    "reliability_score": m.reliability_score,
                    "training_time": m.training_time,
                })
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        elif filepath.endswith(".txt"):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.get_model_comparison_report())

        else:
            raise ValueError(
                f"Unsupported file extension: {filepath}. Use .csv, .json, or .txt"
            )

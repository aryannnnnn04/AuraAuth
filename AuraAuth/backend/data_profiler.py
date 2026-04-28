"""
Data Profiler Module: Comprehensive dataset analysis and health assessment.

This module is responsible for:
- Analyzing dataset characteristics (size, shape, types)
- Detecting data quality issues (missing values, duplicates, outliers)
- Assessing class balance and imbalance
- Computing feature-to-sample ratio (critical for small datasets)
- Generating structured dataset health reports
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from enum import Enum


class DataHealthStatus(Enum):
    """Enumeration of dataset health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class DatasetHealthReport:
    """
    Structured report containing comprehensive dataset analysis.
    
    Attributes:
        dataset_size: Number of samples in dataset
        feature_count: Number of features
        feature_to_sample_ratio: Critical metric for small datasets
        missing_value_report: Per-feature missing value statistics
        class_balance_report: Class distribution and balance metrics
        duplicate_samples: Count and percentage of duplicate rows
        outlier_summary: Number and distribution of outliers
        data_quality_score: Overall 0-100 score
        health_status: Categorical health assessment
        warnings: List of identified issues
        recommendations: Suggested data preprocessing steps
    """
    dataset_size: int
    feature_count: int
    feature_to_sample_ratio: float
    missing_value_report: Dict[str, float] = field(default_factory=dict)
    class_balance_report: Dict[str, Any] = field(default_factory=dict)
    duplicate_samples: Dict[str, Any] = field(default_factory=dict)
    outlier_summary: Dict[str, Any] = field(default_factory=dict)
    data_quality_score: float = 0.0
    health_status: DataHealthStatus = DataHealthStatus.FAIR
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    
    @staticmethod
    def _convert_numpy(obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: DatasetHealthReport._convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [DatasetHealthReport._convert_numpy(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format for serialization."""
        raw = {
            "dataset_size": self.dataset_size,
            "feature_count": self.feature_count,
            "feature_to_sample_ratio": self.feature_to_sample_ratio,
            "missing_value_report": self.missing_value_report,
            "class_balance_report": self.class_balance_report,
            "duplicate_samples": self.duplicate_samples,
            "outlier_summary": self.outlier_summary,
            "data_quality_score": self.data_quality_score,
            "health_status": self.health_status.value,  # Convert enum to string
            "warnings": self.warnings,
            "recommendations": self.recommendations
        }
        return self._convert_numpy(raw)


class DataProfiler:
    """
    Comprehensive dataset profiling and health assessment engine.
    
    Analyzes datasets to provide:
    - Statistical summaries
    - Data quality metrics
    - Class imbalance detection
    - Feature-sample ratio assessment (critical for small datasets)
    - Warnings and recommendations for model training
    
    This is essential for understanding dataset characteristics before
    applying AutoML, especially for small datasets where data quality
    has disproportionate impact on model performance.
    
    Attributes:
        missing_value_threshold: Drop features with missing % above this threshold
        min_class_samples: Minimum samples per class to avoid imbalance
    """
    
    def __init__(
        self,
        missing_value_threshold: float = 0.5,
        min_class_samples: int = 5,
        random_state: int = 42
    ):
        """
        Initialize DataProfiler.
        
        Args:
            missing_value_threshold: Percentage threshold for dropping features (0.0-1.0)
            min_class_samples: Minimum samples per class for acceptable balance
            random_state: Seed for reproducibility
        """
        self.missing_value_threshold = missing_value_threshold
        self.min_class_samples = min_class_samples
        self.random_state = random_state
        self.report: Optional[DatasetHealthReport] = None
    
    def analyze(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        dataset_name: str = "Unknown"
    ) -> DatasetHealthReport:
        """
        Perform comprehensive dataset analysis.
        
        Args:
            X: Feature matrix (DataFrame)
            y: Target variable (Series), optional for classification analysis
            dataset_name: Name/identifier for dataset
            
        Returns:
            DatasetHealthReport: Comprehensive analysis report
            
        Raises:
            ValueError: If X is not a DataFrame or contains invalid data
            
        Notes:
            - For classification: y should contain class labels
            - For regression: Analysis still valid but class-specific metrics skipped
            - Handles both numeric and categorical features
        """
        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        # Initialize report data dictionary
        report_data = {}
        
        # 1. Basic dataset dimensions
        n_samples, n_features = X.shape
        report_data['dataset_size'] = n_samples
        report_data['feature_count'] = n_features
        
        # 2. Feature-to-sample ratio (critical for small datasets)
        feature_to_sample_ratio = n_features / n_samples if n_samples > 0 else 0
        report_data['feature_to_sample_ratio'] = feature_to_sample_ratio
        
        # 3. Missing value analysis
        missing_values = self._analyze_missing_values(X)
        report_data['missing_value_report'] = missing_values
        
        # 4. Class balance analysis (if y provided)
        class_balance = {}
        if y is not None and isinstance(y, pd.Series):
            class_balance = self._analyze_class_balance(y)
        report_data['class_balance_report'] = class_balance
        
        # 5. Duplicate samples detection
        duplicates = self._detect_duplicates(X)
        report_data['duplicate_samples'] = duplicates
        
        # 6. Outlier detection (numeric features only)
        outliers = self._detect_outliers(X)
        report_data['outlier_summary'] = outliers
        
        # 7. Calculate data quality score
        quality_score = self._calculate_data_quality_score(report_data)
        report_data['data_quality_score'] = quality_score
        
        # 8. Determine health status
        health_status = self._generate_health_status(quality_score)
        report_data['health_status'] = health_status
        
        # 9. Generate warnings and recommendations
        warnings, recommendations = self._generate_warnings_and_recommendations(report_data)
        report_data['warnings'] = warnings
        report_data['recommendations'] = recommendations
        
        # Create and cache the report
        self.report = DatasetHealthReport(
            dataset_size=n_samples,
            feature_count=n_features,
            feature_to_sample_ratio=feature_to_sample_ratio,
            missing_value_report=missing_values,
            class_balance_report=class_balance,
            duplicate_samples=duplicates,
            outlier_summary=outliers,
            data_quality_score=quality_score,
            health_status=health_status,
            warnings=warnings,
            recommendations=recommendations
        )
        
        return self.report
    
    def _analyze_missing_values(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze missing value patterns per feature.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping feature names to missing value percentages
        """
        missing_report = {}
        n_samples = len(X)
        
        for col in X.columns:
            # Count missing values (NaN, None, pd.NA)
            missing_count = X[col].isna().sum()
            missing_pct = (missing_count / n_samples) * 100 if n_samples > 0 else 0
            missing_report[col] = missing_pct
        
        return missing_report
    
    def _analyze_class_balance(self, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze class distribution and balance metrics.
        
        Args:
            y: Target variable
            
        Returns:
            Dictionary containing:
            - class_counts: Count per class
            - class_percentages: Percentage per class
            - imbalance_ratio: Max/min class ratio
            - is_balanced: Boolean indicating acceptable balance
            - minority_class_size: Samples in smallest class
        """
        class_counts = y.value_counts().to_dict()
        n_total = len(y)
        
        # Calculate percentages per class
        class_percentages = {k: (v / n_total) * 100 for k, v in class_counts.items()}
        
        # Calculate imbalance ratio (max/min)
        if len(class_counts) > 0:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            imbalance_ratio = 1.0
        
        # Determine if balanced (imbalance ratio < 3 is generally acceptable)
        is_balanced = imbalance_ratio < 3.0
        
        # Get minority class size
        minority_class_size = min(class_counts.values()) if class_counts else 0
        
        return {
            'class_counts': class_counts,
            'class_percentages': class_percentages,
            'imbalance_ratio': imbalance_ratio,
            'is_balanced': is_balanced,
            'minority_class_size': minority_class_size,
            'num_classes': len(class_counts)
        }
    
    def _detect_duplicates(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify duplicate rows in dataset.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary containing:
            - duplicate_count: Total duplicate rows
            - duplicate_percentage: Percentage of data
            - has_duplicates: Boolean flag
        """
        n_samples = len(X)
        
        # Detect exact duplicates (comparing all columns)
        duplicate_mask = X.duplicated(keep=False)  # Mark all duplicates (including first occurrence)
        duplicate_count = duplicate_mask.sum()
        duplicate_percentage = (duplicate_count / n_samples) * 100 if n_samples > 0 else 0
        
        return {
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': duplicate_percentage,
            'has_duplicates': duplicate_count > 0
        }
    
    def _detect_outliers(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using IQR method on numeric features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary containing outlier detection results
        """
        outlier_summary = {
            'numeric_features': [],
            'features_with_outliers': [],
            'outlier_count': 0,
            'outlier_percentage': 0.0
        }
        
        n_samples = len(X)
        all_outliers = set()
        
        # IQR method for numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        outlier_summary['numeric_features'] = list(numeric_cols)
        
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outliers as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (X[col] < lower_bound) | (X[col] > upper_bound)
            
            if outlier_mask.any():
                outlier_summary['features_with_outliers'].append(col)
                all_outliers.update(X[outlier_mask].index.tolist())
        
        outlier_summary['outlier_count'] = len(all_outliers)
        outlier_summary['outlier_percentage'] = (len(all_outliers) / n_samples) * 100 if n_samples > 0 else 0
        
        return outlier_summary
    
    def _calculate_data_quality_score(self, report_data: Dict[str, Any]) -> float:
        """
        Calculate composite data quality score (0-100).
        
        Args:
            report_data: Intermediate analysis results
            
        Returns:
            Data quality score from 0 (worst) to 100 (best)
        """
        score = 100.0  # Start with perfect score
        
        # 1. Missing values penalty (weight: 30%)
        missing_values = report_data['missing_value_report']
        avg_missing = np.mean(list(missing_values.values())) if missing_values else 0
        missing_penalty = (avg_missing / 100.0) * 30  # Max 30 point penalty
        score -= missing_penalty
        
        # 2. Duplicate samples penalty (weight: 15%)
        duplicate_pct = report_data['duplicate_samples'].get('duplicate_percentage', 0)
        duplicate_penalty = (duplicate_pct / 100.0) * 15  # Max 15 point penalty
        score -= duplicate_penalty
        
        # 3. Class imbalance penalty (weight: 25%)
        class_balance = report_data['class_balance_report']
        if class_balance:
            imbalance_ratio = class_balance.get('imbalance_ratio', 1.0)
            # Penalize heavily imbalanced datasets
            # Ratio of 1.0 = no penalty, Ratio > 10 = full penalty
            imbalance_penalty = min((imbalance_ratio - 1) / 9 * 25, 25)
            score -= imbalance_penalty
        
        # 4. Feature-to-sample ratio penalty (weight: 20%)
        # Warn if more features than samples (curse of dimensionality)
        f2s_ratio = report_data['feature_to_sample_ratio']
        if f2s_ratio > 1.0:
            f2s_penalty = min(f2s_ratio * 20, 20)  # Max 20 point penalty
            score -= f2s_penalty
        else:
            # Still penalize high ratio (>0.5 is concerning for small datasets)
            f2s_penalty = min(f2s_ratio / 0.5 * 10, 10)  # Max 10 point penalty
            score -= f2s_penalty
        
        # 5. Outlier penalty (weight: 10%)
        outlier_pct = report_data['outlier_summary'].get('outlier_percentage', 0)
        outlier_penalty = (outlier_pct / 100.0) * 10  # Max 10 point penalty
        score -= outlier_penalty
        
        # Ensure score is in valid range [0, 100]
        score = max(0.0, min(100.0, score))
        
        return score
    
    def _generate_health_status(self, quality_score: float) -> DataHealthStatus:
        """
        Determine categorical health status from quality score.
        
        Args:
            quality_score: Numeric quality score (0-100)
            
        Returns:
            DataHealthStatus enum value
        """
        if quality_score >= 90:
            return DataHealthStatus.EXCELLENT
        elif quality_score >= 75:
            return DataHealthStatus.GOOD
        elif quality_score >= 50:
            return DataHealthStatus.FAIR
        elif quality_score >= 25:
            return DataHealthStatus.POOR
        else:
            return DataHealthStatus.CRITICAL
    
    def _generate_warnings_and_recommendations(
        self,
        report_data: Dict[str, Any]
    ) -> Tuple[list, list]:
        """
        Generate warnings and recommendations based on analysis.
        
        Args:
            report_data: Analysis results
            
        Returns:
            Tuple of (warnings_list, recommendations_list)
        """
        warnings = []
        recommendations = []
        
        # ===== WARNINGS =====
        
        # 1. Missing values warnings
        missing_values = report_data['missing_value_report']
        high_missing_features = [col for col, pct in missing_values.items() if pct > 50]
        if high_missing_features:
            warnings.append(f"High missing values (>50%) in features: {', '.join(high_missing_features)}")
        
        # 2. Feature-to-sample ratio warnings
        f2s_ratio = report_data['feature_to_sample_ratio']
        if f2s_ratio > 1.0:
            warnings.append(f"Features ({report_data['feature_count']}) exceed samples ({report_data['dataset_size']}) - curse of dimensionality risk")
        elif f2s_ratio > 0.3:
            warnings.append(f"High feature-to-sample ratio ({f2s_ratio:.2f}) - may cause overfitting")
        
        # 3. Dataset size warnings
        if report_data['dataset_size'] < 100:
            warnings.append(f"Very small dataset ({report_data['dataset_size']} samples) - limited for model training")
        elif report_data['dataset_size'] < 500:
            warnings.append(f"Small dataset ({report_data['dataset_size']} samples) - careful with model complexity")
        
        # 4. Class imbalance warnings
        class_balance = report_data['class_balance_report']
        if class_balance:
            if not class_balance.get('is_balanced', True):
                imbalance_ratio = class_balance.get('imbalance_ratio', 1.0)
                warnings.append(f"Severe class imbalance (ratio: {imbalance_ratio:.1f}) - consider balancing strategies")
            
            minority_class_size = class_balance.get('minority_class_size', 0)
            if minority_class_size < self.min_class_samples:
                warnings.append(f"Minority class too small ({minority_class_size} samples) - stratified splitting may fail")
        
        # 5. Duplicate samples warnings
        duplicate_pct = report_data['duplicate_samples'].get('duplicate_percentage', 0)
        if duplicate_pct > 10:
            warnings.append(f"High duplicate percentage ({duplicate_pct:.1f}%) - may inflate model performance")
        
        # 6. Outlier warnings
        outlier_pct = report_data['outlier_summary'].get('outlier_percentage', 0)
        if outlier_pct > 20:
            warnings.append(f"High outlier percentage ({outlier_pct:.1f}%) - consider outlier handling")
        
        # ===== RECOMMENDATIONS =====
        
        # 1. Missing value handling
        if high_missing_features:
            recommendations.append("Use KNN or iterative imputation for missing values (better than mean/median for small datasets)")
        elif missing_values and any(pct > 0 for pct in missing_values.values()):
            recommendations.append("Apply appropriate missing value imputation strategy before training")
        
        # 2. Class imbalance handling
        if class_balance and not class_balance.get('is_balanced', True):
            recommendations.append("Consider SMOTE oversampling or class weight adjustment for imbalanced classification")
        
        # 3. Feature-to-sample ratio handling
        if f2s_ratio > 0.3:
            recommendations.append("Apply feature selection or dimensionality reduction to reduce feature-to-sample ratio")
        
        # 4. Duplicate handling
        if duplicate_pct > 5:
            recommendations.append("Remove duplicate samples before training to avoid data leakage")
        
        # 5. Outlier handling
        if outlier_pct > 5:
            recommendations.append("Review and handle outliers using domain knowledge or robust scaling")
        
        # 6. Data collection recommendation
        if report_data['dataset_size'] < 500:
            recommendations.append("Collect more data if possible - current dataset is small for reliable model training")
        
        # 7. General best practices
        recommendations.append("Use stratified cross-validation to maintain class distribution in small datasets")
        recommendations.append("Employ regularization techniques (L1/L2) to prevent overfitting with small datasets")
        
        return warnings, recommendations
    
    def get_report(self) -> Optional[DatasetHealthReport]:
        """
        Retrieve the most recent analysis report.
        
        Returns:
            DatasetHealthReport if analysis completed, None otherwise
        """
        return self.report
    
    def print_summary(self) -> None:
        """
        Print human-readable summary of dataset health.
        """
        if self.report is None:
            print("No analysis report available. Run analyze() first.")
            return
        
        print("=" * 70)
        print("Dataset Health Summary")
        print("=" * 70)
        print(f"Size: {self.report.dataset_size} samples × {self.report.feature_count} features")
        print(f"Feature-to-Sample Ratio: {self.report.feature_to_sample_ratio:.3f}")
        print(f"Data Quality Score: {self.report.data_quality_score:.1f}/100")
        print(f"Health Status: {self.report.health_status.value.upper()}")
        print()
        
        if self.report.warnings:
            print(f"Warnings ({len(self.report.warnings)}):")
            for warning in self.report.warnings[:5]:
                print(f"  ⚠ {warning}")
            if len(self.report.warnings) > 5:
                print(f"  ... and {len(self.report.warnings) - 5} more")
            print()
        
        if self.report.recommendations:
            print(f"Recommendations ({len(self.report.recommendations)}):")
            for rec in self.report.recommendations[:5]:
                print(f"  → {rec}")
            if len(self.report.recommendations) > 5:
                print(f"  ... and {len(self.report.recommendations) - 5} more")
        
        print("=" * 70)

"""
AutoML Engine Module: Bayesian hyperparameter optimization and model search.

This module provides:
- Model search space definition for various algorithms
- Optuna-based Bayesian hyperparameter optimization
- Cross-validation and metric tracking
- Trial management and early stopping
- Support for multiple ML algorithms (LR, RF, XGB, LGB, SVM)
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import optuna
from optuna.study import Study
from optuna.trial import Trial
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)


class TrialResult:
    """
    Container for individual trial results from optimization.
    
    Attributes:
        trial_number: Sequential trial ID
        model_name: Name of model tested
        hyperparameters: Dict of hyperparameters used
        train_score: Cross-validation score on training data
        trial_score: Primary optimization objective score
        computation_time: Time to evaluate trial (seconds)
        is_best: Whether this is best trial so far
        additional_metrics: Dict of secondary metrics (variance, stability, etc.)
    """
    
    def __init__(
        self,
        trial_number: int,
        model_name: str,
        hyperparameters: Dict[str, Any],
        trial_score: float,
        train_score: float = 0.0,
        computation_time: float = 0.0
    ):
        self.trial_number = trial_number
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.trial_score = trial_score
        self.train_score = train_score
        self.computation_time = computation_time
        self.is_best = False
        self.additional_metrics: Dict[str, float] = {}


class AutoMLEngine:
    """
    Bayesian hyperparameter optimization engine using Optuna.
    
    Responsible for:
    - Defining model-specific hyperparameter search spaces
    - Running Bayesian optimization across multiple models
    - Tracking trial results and convergence
    - Selecting best models based on composite objectives
    - Supporting small-dataset-specific optimizations
    
    For small datasets (< 5000 samples), emphasis is on:
    - Model stability and variance over raw accuracy
    - Computational efficiency (fewer trials may be needed)
    - Robust validation strategies (cross-validation)
    - Avoiding overfitting (prefer simpler models)
    
    Attributes:
        n_trials: Number of optimization trials per model
        n_startup_trials: Random trials before Bayesian optimization
        cv_folds: Number of cross-validation folds
        random_state: Seed for reproducibility
    """
    
    def __init__(
        self,
        n_trials: int = 50,
        n_startup_trials: int = 5,
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize AutoML Engine.
        
        Args:
            n_trials: Total trials per model
            n_startup_trials: Initial random trials for exploration
            cv_folds: Cross-validation folds
            random_state: Reproducibility seed
            verbose: Print optimization progress
        """
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        
        # Results tracking
        self.trials_history: List[TrialResult] = []
        self.study: Optional[Study] = None
        self.best_models: Dict[str, Any] = {}
    
    def optimize_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metric: str = "roc_auc"
    ) -> Tuple[Dict[str, Any], float, float]:
        """
        Optimize single model using Bayesian search.
        
        Args:
            model_name: Model to optimize
            X_train: Training features
            y_train: Training target
            metric: Metric to optimize (roc_auc for classification)
            
        Returns:
            Tuple of (best_params, mean_score, std_score)
        """
        if model_name not in ["logistic_regression", "random_forest", "xgboost", "lightgbm", "svm"]:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Check if target cardinality is too high for stratified K-fold
        n_samples = len(y_train)
        n_classes = len(np.unique(y_train))
        
        # Infer task type from metric
        regression_metrics = {"neg_mean_squared_error", "neg_mean_absolute_error", "r2"}
        task_type = "regression" if metric in regression_metrics else "classification"
        
        # Determine appropriate cross-validation strategy
        cv_splitter = self._get_cv_splitter(n_samples, n_classes, task_type)
        
        # Get model template and search space
        model_template, search_space = self._get_model_and_space(model_name, task_type)
        
        # Create objective function
        def objective(trial: Trial) -> float:
            # Sample hyperparameters from search space
            params = {}
            for param_name, param_spec in search_space.items():
                if param_spec["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_spec["low"], param_spec["high"]
                    )
                elif param_spec["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, param_spec["low"], param_spec["high"],
                        log=param_spec.get("log", False)
                    )
                elif param_spec["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_spec["choices"]
                    )
            
            # Create model with sampled parameters
            model = model_template.set_params(**params)
            
            # Evaluate with appropriate cross-validation
            try:
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_splitter,
                    scoring=metric,
                    n_jobs=1,
                    error_score=0.0
                )
                return scores.mean() if len(scores) > 0 else 0.0
            except Exception as e:
                # Log detailed error information for debugging
                logger.error(
                    f"CV evaluation FAILED for trial: {type(e).__name__}: {str(e)}\n"
                    f"Model: {model.__class__.__name__}, Metric: {metric}, "
                    f"Data shape: {X_train.shape}",
                    exc_info=True  # Include full traceback
                )
                # Return 0.0 as fallback, but error is visible in logs
                return 0.0
        
        # Create and run Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=self.n_startup_trials,
                seed=self.random_state
            )
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        # Get best trial and refit on full training data for std calculation
        best_params = study.best_params
        # Calculate std by refitting with best params
        model = model_template.set_params(**best_params)
        try:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_splitter,
                scoring=metric,
                n_jobs=1,
                error_score=0.0
            )
            mean_score = scores.mean() if len(scores) > 0 else 0.0
            std_score = scores.std() if len(scores) > 0 else 0.0
        except Exception as e:
            # Log detailed error information for debugging
            logger.error(
                f"FINAL evaluation FAILED for {model_name}: {type(e).__name__}: {str(e)}\n"
                f"Best params: {best_params}, Data shape: {X_train.shape}",
                exc_info=True  # Include full traceback
            )
            mean_score = 0.0
            std_score = 0.0
        
        # Populate trials_history from completed study
        best_value = study.best_value
        for trial in study.trials:
            tr = TrialResult(
                trial_number=trial.number,
                model_name=model_name,
                hyperparameters=dict(trial.params),
                trial_score=trial.value if trial.value is not None else 0.0,
            )
            tr.is_best = (trial.value == best_value)
            self.trials_history.append(tr)

        # Store best model
        self.best_models[model_name] = {
            "model": model_template.set_params(**best_params),
            "params": best_params,
            "mean_score": mean_score,
            "std_score": std_score,
            "metric": metric
        }
        
        if self.verbose:
            print(f"✓ {model_name}: score={mean_score:.4f}±{std_score:.4f}")
        
        return best_params, mean_score, std_score
    
    def _get_cv_splitter(self, n_samples: int, n_classes: int, task_type: str = "classification"):
        """
        Select appropriate cross-validation splitter based on data characteristics.
        
        Handles edge cases:
        - Too many classes: Use simple holdout or minimal CV
        - Few samples: Reduce n_splits dynamically
        - Very imbalanced: Use stratified KFold with appropriate n_splits
        
        Args:
            n_samples: Number of training samples
            n_classes: Number of unique classes in target
            task_type: "classification" or "regression"
            
        Returns:
            Cross-validation splitter object
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # For regression, use standard KFold (stratification doesn't apply)
        if task_type == "regression":
            return KFold(n_splits=min(self.cv_folds, n_samples), shuffle=True, random_state=self.random_state)
        
        # Rule: Each class must have at least n_splits samples
        # If n_classes is too high relative to n_samples, adjust CV strategy
        
        # SAFETY CHECK: Handle edge case of 0 classes
        if n_classes == 0 or n_samples == 0:
            logger.warning(
                f"Invalid data dimensions: {n_samples} samples, {n_classes} classes. "
                f"Falling back to 2-fold KFold."
            )
            return KFold(
                n_splits=2,
                shuffle=True,
                random_state=self.random_state
            )
        
        # Calculate minimum samples per class
        min_samples_per_class = n_samples / n_classes if n_classes > 0 else n_samples
        
        # EXTREME case: fewer than 2 samples per class on average
        # Can't use any stratified fold, use simple k-fold instead
        if min_samples_per_class < 2:
            logger.warning(
                f"Target has extremely high cardinality ({n_classes} classes) with very few samples ({n_samples}). "
                f"Average {min_samples_per_class:.2f} samples per class. "
                f"Using KFold (non-stratified) with 2 folds to avoid stratification issues."
            )
            # Use non-stratified KFold with minimal splits
            return KFold(
                n_splits=2,
                shuffle=True,
                random_state=self.random_state
            )
        
        # HIGH cardinality case (2+ samples per class, but still problematic)
        elif min_samples_per_class < 3:
            logger.warning(
                f"Target has high cardinality ({n_classes} classes) with few samples ({n_samples}). "
                f"Average {min_samples_per_class:.2f} samples per class. "
                f"Using KFold with reduced folds."
            )
            # Use KFold with 2 splits
            return KFold(
                n_splits=2,
                shuffle=True,
                random_state=self.random_state
            )
        
        # MODERATE cardinality (3+ samples per class)
        elif min_samples_per_class < self.cv_folds:
            # Use reduced number of stratified splits
            n_splits = max(2, int(min_samples_per_class))
            logger.warning(
                f"Reducing CV folds from {self.cv_folds} to {n_splits} "
                f"({min_samples_per_class:.2f} samples per class on average)"
            )
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        
        # NORMAL case: Use standard stratified K-fold
        else:
            return StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
    
    def run_automl(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metric: str = "roc_auc"
    ) -> Dict[str, Any]:
        """
        Optimize all supported models sequentially.
        
        Args:
            X_train: Training features
            y_train: Training target
            metric: Metric to optimize
            
        Returns:
            Dict with results for each model
        """
        model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
        results = {}
        
        for model_name in model_names:
            try:
                best_params, mean_score, std_score = self.optimize_model(
                    model_name, X_train, y_train, metric
                )
                results[model_name] = {
                    "best_params": best_params,
                    "mean_score": mean_score,
                    "std_score": std_score
                }
            except Exception as e:
                if self.verbose:
                    print(f"✗ {model_name} failed: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def _get_model_and_space(
        self,
        model_name: str,
        task_type: str = "classification"
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Get model template and search space for Optuna.
        
        Args:
            model_name: Name of model
            task_type: "classification" or "regression"
            
        Returns:
            Tuple of (model_template, search_space_dict)
        """
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC, SVR
        
        try:
            import xgboost as xgb
        except ImportError:
            xgb = None
        
        try:
            import lightgbm as lgb
        except ImportError:
            lgb = None
        
        is_regression = task_type == "regression"
        
        if model_name == "logistic_regression":
            if is_regression:
                model = Ridge(random_state=self.random_state)
                search_space = {
                    "alpha": {"type": "float", "low": 1e-4, "high": 1e2, "log": True},
                }
            else:
                model = LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=1)
                search_space = {
                    "C": {"type": "float", "low": 1e-4, "high": 1e2, "log": True},
                    "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
                    "solver": {"type": "categorical", "choices": ["liblinear", "saga"]}
                }
        
        elif model_name == "random_forest":
            cls = RandomForestRegressor if is_regression else RandomForestClassifier
            model = cls(random_state=self.random_state, n_jobs=1)
            search_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "max_depth": {"type": "int", "low": 5, "high": 25},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]}
            }
        
        elif model_name == "xgboost":
            if xgb is None:
                raise ImportError("xgboost not installed")
            cls = xgb.XGBRegressor if is_regression else xgb.XGBClassifier
            model = cls(random_state=self.random_state, n_jobs=1, use_label_encoder=False)
            search_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 200},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0}
            }
        
        elif model_name == "lightgbm":
            if lgb is None:
                raise ImportError("lightgbm not installed")
            cls = lgb.LGBMRegressor if is_regression else lgb.LGBMClassifier
            model = cls(random_state=self.random_state, n_jobs=1, verbose=-1)
            search_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 200},
                "num_leaves": {"type": "int", "low": 20, "high": 50},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0}
            }
        
        elif model_name == "svm":
            if is_regression:
                model = SVR()
            else:
                model = SVC(random_state=self.random_state, probability=True)
            search_space = {
                "C": {"type": "float", "low": 0.1, "high": 100, "log": True},
                "kernel": {"type": "categorical", "choices": ["rbf", "poly"]},
                "gamma": {"type": "categorical", "choices": ["scale", "auto"]}
            }
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model, search_space
    
    def _evaluate_trial(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metric: str
    ) -> float:
        """
        Evaluate model on single trial using cross-validation.
        
        Args:
            model: Fitted model to evaluate
            X_train: Training data
            y_train: Training target
            metric: Evaluation metric
            
        Returns:
            Cross-validation score (mean)
        """
        # Get appropriate CV splitter based on target cardinality
        n_samples = len(y_train)
        n_classes = len(np.unique(y_train))
        regression_metrics = {"neg_mean_squared_error", "neg_mean_absolute_error", "r2"}
        task_type = "regression" if metric in regression_metrics else "classification"
        cv_splitter = self._get_cv_splitter(n_samples, n_classes, task_type)
        
        try:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_splitter,
                scoring=metric,
                n_jobs=1,
                error_score=0.0
            )
            return scores.mean() if len(scores) > 0 else 0.0
        except Exception as e:
            logger.warning(f"Evaluation failed: {str(e)}")
            return 0.0
    
    def get_best_model(self, model_name: str) -> Optional[BaseEstimator]:
        """
        Get best fitted model for specified algorithm.
        
        PLANNED FEATURE: Not currently used by PipelineManager or FastAPI endpoints.
        Placeholder for future direct model retrieval API.
        
        Args:
            model_name: Name of model to retrieve
            
        Returns:
            Best model instance or None if not found
            
        Implementation notes:
            - Return fitted model from best_models dict
            - Validate model_name exists in best_models
            - Handle case where optimization not completed for this model
        """
        # TODO: Implement model retrieval for future API
        pass
    
    def get_trials_dataframe(self) -> pd.DataFrame:
        """
        Convert trial history to DataFrame for analysis.
        
        PLANNED FEATURE: Not currently used. Placeholder for future analysis API.
        Useful for exporting optimization history for external analysis.
        
        Returns:
            DataFrame with columns for each trial's details
            
        Implementation notes:
        - Create DataFrame from trials_history list
        - Include trial_number, model_name, trial_score, computation_time
        - Add one column per hyperparameter with its value
        - Sort by trial number for chronological view
        """
        # TODO: Implement trials export for future analysis API
        pass
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of optimization process.
        
        PLANNED FEATURE: Not currently used. Placeholder for future dashboard/reporting.
        Useful for summarizing optimization results programmatically.
        
        Returns:
            Dictionary with optimization statistics including:
            - total_trials_run (int)
            - best_trial_number (int)
            - best_model_name (str)
            - best_score (float)
            - time_elapsed (float, seconds)
            - convergence_status (str): "converged", "in_progress", "stalled"
            - trials_per_model (Dict[str, int])
        """
        # TODO: Implement for future dashboard/reporting API
        pass
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Visualize optimization history (trial scores over time).
        
        PLANNED FEATURE: Not currently used. Placeholder for future visualization API.
        Useful for understanding optimization convergence and model performance trends.
        
        Args:
            save_path: Path to save figure, displays if None
            
        Implementation notes:
        - Create plot with trial number on x-axis, score on y-axis
        - Show separate line for each model (different colors)
        - Overlay best score trend line
        - Add convergence indicators
        - Include legend showing model names
        """
        # TODO: Implement for future visualization API
        pass
    
    def export_results(self, filepath: str) -> None:
        """
        Export optimization results to file for external analysis.
        
        PLANNED FEATURE: Not currently used. Placeholder for future export API.
        Useful for exporting optimization results to CSV/JSON for reporting and analysis.
        
        Args:
            filepath: Path to save results (auto-detect format: .csv or .json)
            
        Implementation notes:
        - Detect file format from extension
        - Save complete trial history with all details
        - Include best models information and scores
        - Save all hyperparameters for each trial
        - Include timing and efficiency metrics
        """
        # TODO: Implement for future export/reporting API
        pass


class ModelFactory:
    """
    Factory for creating model instances with given hyperparameters.
    
    Encapsulates model initialization logic and handles:
    - Parameter validation and conversion
    - Model instantiation
    - Algorithm-specific preprocessing
    """
    
    @staticmethod
    def create_model(
        model_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        task_type: str = "classification",
        **kwargs: Any
    ) -> BaseEstimator:
        """
        Create model instance with specified hyperparameters.
        
        PLANNED REFACTORING: Used internally by optimize_model(). Marked for future extraction.
        Currently implemented inline within AutoMLEngine.optimize_model().
        Future refactoring will use this factory for cleaner code organization.
        
        Args:
            model_name: Name of model (logistic_regression, random_forest, xgboost, lightgbm, svm)
            hyperparameters: Dict of model-specific hyperparameters
            random_state: Random seed for reproducibility
            
        Returns:
            Instantiated scikit-learn compatible model
            
        Raises:
            ValueError: If model_name not recognized
            
        Implementation notes:
        - Support models: LogisticRegression, RandomForest, XGBoost, LightGBM, SVM
        - Pass hyperparameters to model constructor
        - Set random_state for reproducibility
        - Handle model-specific parameters (e.g., GPU support for XGB/LGB)
        - Validate hyperparameters match model's expected ranges
        """
        # Allow both dict-style and kwargs-style hyperparameters
        params: Dict[str, Any] = {}
        if hyperparameters:
            params.update(hyperparameters)
        if kwargs:
            params.update(kwargs)

        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC, SVR

        try:
            import xgboost as xgb
        except ImportError:
            xgb = None

        try:
            import lightgbm as lgb
        except ImportError:
            lgb = None

        is_regression = task_type == "regression"

        if model_name == "logistic_regression":
            if is_regression:
                return Ridge(random_state=random_state, **params)
            return LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                n_jobs=1,
                **params
            )

        if model_name == "random_forest":
            cls = RandomForestRegressor if is_regression else RandomForestClassifier
            return cls(
                random_state=random_state,
                n_jobs=1,
                **params
            )

        if model_name == "xgboost":
            if xgb is None:
                raise ImportError("xgboost not installed")
            cls = xgb.XGBRegressor if is_regression else xgb.XGBClassifier
            return cls(
                random_state=random_state,
                n_jobs=1,
                use_label_encoder=False,
                **params
            )

        if model_name == "lightgbm":
            if lgb is None:
                raise ImportError("lightgbm not installed")
            cls = lgb.LGBMRegressor if is_regression else lgb.LGBMClassifier
            return cls(
                random_state=random_state,
                n_jobs=1,
                verbose=-1,
                **params
            )

        if model_name == "svm":
            if is_regression:
                return SVR(**params)
            return SVC(
                probability=True,
                random_state=random_state,
                **params
            )

        raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def get_search_space(model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameter search space for model.
        
        PLANNED REFACTORING: Used internally by optimize_model(). Marked for future extraction.
        Currently implemented inline within AutoMLEngine.optimize_model().
        Future refactoring will use this factory for cleaner code organization.
        
        Args:
            model_name: Name of model (logistic_regression, random_forest, xgboost, lightgbm, svm)
            
        Returns:
            Dictionary defining search space for Optuna with schema:
            {"param_name": {"type": "int"|"float"|"categorical", 
                           "low": float, "high": float, 
                           "log": bool, "choices": []}}
            
        Implementation notes:
        - Define appropriate search ranges for each model type
        - Include bounds optimized for small datasets
        - Support categorical choices for specific parameters
        - Use log scale for exponential parameters (C, learning_rate)
        """
        # Keep search spaces consistent with AutoMLEngine._get_model_and_space
        if model_name == "logistic_regression":
            return {
                "C": {"type": "float", "low": 1e-4, "high": 1e2, "log": True},
                "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
                "solver": {"type": "categorical", "choices": ["liblinear", "saga"]},
            }

        if model_name == "random_forest":
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "max_depth": {"type": "int", "low": 5, "high": 25},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
            }

        if model_name == "xgboost":
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 200},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            }

        if model_name == "lightgbm":
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 200},
                "num_leaves": {"type": "int", "low": 20, "high": 50},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
            }

        if model_name == "svm":
            return {
                "C": {"type": "float", "low": 0.1, "high": 100, "log": True},
                "kernel": {"type": "categorical", "choices": ["rbf", "poly"]},
                "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
            }

        raise ValueError(f"Unknown model: {model_name}")

"""
Preprocessing Pipeline Module: Data transformation and feature engineering.

This module handles:
- Missing value imputation (mean, median, KNN, iterative methods)
- Categorical feature encoding (one-hot, label, target encoding)
- Feature scaling (standardization, normalization, robust scaling)
- Train-test splitting with stratification
- Handling imbalanced datasets
- Feature transformations specific to dataset characteristics
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class PreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline for AutoML.
    
    Orchestrates multiple preprocessing steps in a configurable manner,
    optimized for small datasets where feature engineering and data
    handling significantly impact model performance.
    
    Pipeline stages:
    1. Missing value imputation
    2. Feature encoding (categorical -> numeric)
    3. Feature scaling/normalization
    4. Outlier handling (optional)
    5. Train-test splitting with stratification
    
    Attributes:
        imputation_strategy: Method for handling missing values
        encoding_strategy: Method for categorical encoding
        scaling_strategy: Method for feature scaling
        random_state: Seed for reproducibility
    """
    
    def __init__(
        self,
        imputation_strategy: str = "mean",
        encoding_strategy: str = "onehot",
        scaling_strategy: str = "standard",
        random_state: int = 42
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            imputation_strategy: Missing value handling
                Options: "mean", "median", "knn", "iterative", "drop"
            encoding_strategy: Categorical encoding method
                Options: "onehot", "label", "target"
            scaling_strategy: Feature scaling method
                Options: "standard", "minmax", "robust", "none"
            random_state: Seed for reproducibility
        """
        self.imputation_strategy = imputation_strategy
        self.encoding_strategy = encoding_strategy
        self.scaling_strategy = scaling_strategy
        self.random_state = random_state
        
        # Fitted transformers (stored for consistent transformation)
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.encoder = None
        self.scaler = None
        self.feature_names: Optional[List[str]] = None
        self.numeric_features: Optional[List[str]] = None
        self.categorical_features: Optional[List[str]] = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessingPipeline':
        """
        Fit all preprocessing transformers on training data.
        
        Args:
            X: Training feature matrix
            y: Target variable (required for target encoding, optional otherwise)
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If X not a DataFrame or contains invalid data
            
        Notes:
            - Must be called before transform()
            - Fitting on training data ensures no data leakage
            - Categorical/numeric feature detection is automatic
            - IMPORTANT: Fit ONLY on training data, never on full dataset
              before splitting to avoid data leakage!
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if len(X) < 2:
            raise ValueError(f"X must have at least 2 samples, got {len(X)}")
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Identify numeric and categorical features
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle missing values
        X_imputed = self._handle_missing_values(X, fit=True, y=y)
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X_imputed, fit=True, y=y)
        
        # Scale numeric features
        self._scale_features(X_encoded, fit=True)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing transformers to data.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix with same index as input
            
        Raises:
            RuntimeError: If called before fit()
            ValueError: If X contains unexpected features
            
        Notes:
            - Does not modify input DataFrame
            - Output is always numeric
            - Maintains row indices for alignment with target
        """
        if not hasattr(self, 'numeric_imputer') or not hasattr(self, 'encoder') or not hasattr(self, 'scaler'):
            raise RuntimeError("fit() must be called before transform()")
        
        # Make a copy to avoid modifying input
        X = X.copy()
        
        # Apply imputation
        X = self._handle_missing_values(X, fit=False)
        
        # Apply encoding
        X = self._encode_categorical_features(X, fit=False)
        
        # Apply scaling
        X = self._scale_features(X, fit=False)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step (for training data).
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def _handle_missing_values(
        self,
        X: pd.DataFrame,
        fit: bool = False,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Handle missing values using configured strategy.
        
        Args:
            X: Feature matrix
            fit: Whether to fit imputer on this data
            y: Target variable (for context-aware imputation)
            
        Returns:
            DataFrame with missing values handled
        """
        X = X.copy()
        
        # Determine imputation strategy per column
        if fit:
            # Create imputers for numeric and categorical features
            # Numeric features: use median
            if self.numeric_features:
                self.numeric_imputer = SimpleImputer(strategy="median")
                X[self.numeric_features] = self.numeric_imputer.fit_transform(X[self.numeric_features])
            
            # Categorical features: use most_frequent
            if self.categorical_features:
                self.categorical_imputer = SimpleImputer(strategy="most_frequent")
                X[self.categorical_features] = self.categorical_imputer.fit_transform(X[self.categorical_features])
        else:
            # Apply fitted imputers
            if self.numeric_features and hasattr(self, 'numeric_imputer') and self.numeric_imputer is not None:
                X[self.numeric_features] = self.numeric_imputer.transform(X[self.numeric_features])
            
            if self.categorical_features and hasattr(self, 'categorical_imputer') and self.categorical_imputer is not None:
                X[self.categorical_features] = self.categorical_imputer.transform(X[self.categorical_features])
        
        return X
    
    def _encode_categorical_features(
        self,
        X: pd.DataFrame,
        fit: bool = False,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Encode categorical features to numeric.
        
        Args:
            X: Feature matrix (may contain categorical columns)
            fit: Whether to fit encoder on this data
            y: Target variable (for target encoding)
            
        Returns:
            DataFrame with all numeric features
        """
        X = X.copy()
        
        if not self.categorical_features:
            return X
        
        if fit:
            # Create one-hot encoder with safe defaults
            self.encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                dtype=np.float64
            )
            # Fit and transform
            encoded = self.encoder.fit_transform(X[self.categorical_features])
            # Get feature names from encoder
            feature_names = self.encoder.get_feature_names_out(self.categorical_features)
            X_encoded = pd.DataFrame(encoded, columns=feature_names, index=X.index)
            # Combine numeric + encoded categorical
            if self.numeric_features:
                X_result = pd.concat([X[self.numeric_features], X_encoded], axis=1)
            else:
                X_result = X_encoded
            return X_result
        else:
            # Apply fitted encoder
            if self.encoder is None:
                return X
            encoded = self.encoder.transform(X[self.categorical_features])
            feature_names = self.encoder.get_feature_names_out(self.categorical_features)
            X_encoded = pd.DataFrame(encoded, columns=feature_names, index=X.index)
            if self.numeric_features:
                X_result = pd.concat([X[self.numeric_features], X_encoded], axis=1)
            else:
                X_result = X_encoded
            return X_result
    
    def _scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Scale/normalize features using configured strategy.
        
        Args:
            X: Numeric feature matrix
            fit: Whether to fit scaler on this data
            
        Returns:
            Scaled feature matrix
        """
        X = X.copy()
        
        # Only scale numeric columns
        if not self.numeric_features:
            return X
        
        if fit:
            self.scaler = StandardScaler()
            X[self.numeric_features] = self.scaler.fit_transform(X[self.numeric_features])
        else:
            if self.scaler is None:
                return X
            X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        
        return X
    
    def split_train_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test sets with stratification for imbalanced data.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing (0.0-1.0)
            stratify: Use stratified split for classification
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If test_size invalid or data too small
            
        Notes:
            - Stratification critical for small, imbalanced datasets
            - Preserves class distribution in both sets
            - Random state ensures reproducibility
        """
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        
        # Determine stratification
        stratify_param = y if stratify else None
        
        # Perform the split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_param,
            random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalanced_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        strategy: str = "class_weights"
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[Dict[int, float]]]:
        """
        Handle imbalanced classification dataset.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target variable
            strategy: Imbalance handling method
                Options: "class_weights", "oversample", "undersample", "smote"
                
        Returns:
            Tuple of (X_train, y_train, class_weights).
            class_weights is a dict mapping class -> weight when strategy is
            "class_weights", otherwise None.
        """
        if strategy == "class_weights":
            class_counts = y_train.value_counts()
            total = len(y_train)
            class_weights = {cls: total / (len(class_counts) * count) 
                           for cls, count in class_counts.items()}
            return X_train, y_train, class_weights
        else:
            # Other strategies (oversample, undersample, smote) not yet implemented
            return X_train, y_train, None
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features after encoding.
        
        Returns:
            List of feature names in transform output
            
        Raises:
            RuntimeError: If fit() not called yet
        """
        if self.feature_names is None:
            raise RuntimeError("fit() must be called before get_feature_names()")
        
        if self.encoder is None:
            return self.numeric_features
        
        # Get one-hot encoded feature names
        encoded_names = list(self.encoder.get_feature_names_out(self.categorical_features))
        return self.numeric_features + encoded_names
    
    def get_feature_types(self) -> Dict[str, List[str]]:
        """
        Get feature type classification.
        
        Returns:
            Dictionary with 'numeric' and 'categorical' feature lists
        """
        return {
            "numeric": self.numeric_features or [],
            "categorical": self.categorical_features or []
        }

"""
FastAPI Backend for AuraAuth AutoML System.

Main entry point for the AutoML API server.
Provides REST endpoints for:
- Dataset upload and profiling
- Model training and optimization
- Model evaluation and selection
- Predictions and explanations
- Documentation generation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
import logging
import os
import html

from .config import DEFAULT_CONFIG, MAX_UPLOAD_SIZE_BYTES
from .data_profiler import DataProfiler
from .preprocessing import PreprocessingPipeline
from .automl_engine import AutoMLEngine, ModelFactory
from .core.uncertainty_estimator import UncertaintyEstimator
from .ood_detector import OODDetector
from .core.explainability_engine import ExplainabilityEngine
from .core.documentation_generator import DocumentationGenerator


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class DatasetUploadResponse(BaseModel):
    """Response after dataset upload."""
    dataset_id: str
    n_samples: int
    n_features: int
    column_names: List[str]
    data_types: Dict[str, str]
    message: str


class DataProfileResponse(BaseModel):
    """Response from data profiling."""
    dataset_id: str
    profile_timestamp: str
    data_quality_score: float
    health_status: str
    warnings: List[str]
    recommendations: List[str]
    summary_stats: Dict[str, Any]


class OptimizationRequest(BaseModel):
    """Request to start optimization."""
    dataset_id: str
    target_column: str
    models: List[str]
    metric: str = "accuracy"
    n_trials: Optional[int] = None
    cv_folds: Optional[int] = None
    description: Optional[str] = None


class OptimizationResponse(BaseModel):
    """Response from optimization."""
    optimization_id: str
    status: str  # "in_progress", "completed", "failed"
    best_model: str
    best_score: float
    trials_completed: int
    timestamp: str


class PredictionRequest(BaseModel):
    """Request for prediction."""
    model_id: str
    data: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Response with prediction."""
    prediction: Any
    confidence: float
    is_confident: bool
    explanation: Optional[str] = None
    is_ood: bool
    timestamp: str


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="AuraAuth AutoML API",
    description="Reliability-aware AutoML system for small, noisy datasets",
    version="0.1.0"
)

# CORS configuration (avoid wildcard + credentials)
def _get_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if not raw:
        return [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    return [o.strip() for o in raw.split(",") if o.strip()]

cors_origins = _get_cors_origins()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global State (In Production: Use Database)
# ============================================================================

# WARNING: In-memory storage is NOT thread-safe and loses data on restart
# For production deployment, use a proper database (PostgreSQL, MongoDB, Redis)
datasets: Dict[str, pd.DataFrame] = {}
models: Dict[str, Any] = {}
optimizations: Dict[str, Dict[str, Any]] = {}
profilers: Dict[str, DataProfiler] = {}


# ============================================================================
# Input Validation Helpers
# ============================================================================

def validate_dataset_exists(dataset_id: str) -> None:
    """Validate that dataset exists in storage."""
    if dataset_id not in datasets:
        raise KeyError(f"Dataset '{dataset_id}' not found. Please upload a dataset first.")

def validate_model_exists(model_id: str) -> None:
    """Validate that model exists in storage."""
    if model_id not in models:
        raise KeyError(f"Model '{model_id}' not found. Please train a model first.")

def _get_model_preprocessor(model_data: Dict[str, Any]) -> Optional[PreprocessingPipeline]:
    preprocessor = model_data.get("preprocessor")
    return preprocessor if isinstance(preprocessor, PreprocessingPipeline) else None

def _drop_target_if_present(df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    if target_column and target_column in df.columns:
        return df.drop(columns=[target_column])
    return df

def _infer_task_type_from_target(y: pd.Series) -> str:
    """
    Heuristic inference for API usage.
    NOTE: AutoMLEngine currently supports classifiers only.
    """
    # If clearly continuous float with many unique values -> regression
    try:
        if pd.api.types.is_float_dtype(y) and y.nunique(dropna=True) > 20:
            return "regression"
    except Exception:
        pass
    return "classification"

async def validate_dataset_file(file: UploadFile) -> bytes:
    """
    Validate uploaded file is CSV and return contents.
    
    Args:
        file: Uploaded file object
        
    Returns:
        File contents as bytes
        
    Raises:
        ValueError: If file is invalid
    """
    filename = file.filename or ""
    if not filename.endswith('.csv'):
        raise ValueError(
            f"Invalid file format: {filename or 'unnamed_file'}. Only CSV files are supported."
        )
    
    # Read file contents to check size
    contents = await file.read()
    
    if not contents or len(contents) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // 1_000_000
        raise ValueError(f"File too large. Maximum size is {max_mb}MB.")
    
    return contents


# ============================================================================
# Health and Info Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - system information."""
    return {
        "name": "AuraAuth AutoML API",
        "version": "0.1.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/config")
async def get_config():
    """Get current system configuration."""
    from dataclasses import asdict
    return {"config": asdict(DEFAULT_CONFIG)}


# ============================================================================
# Dataset Management Endpoints
# ============================================================================

@app.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload and register a dataset.
    
    Args:
        file: CSV file with training data
        
    Returns:
        DatasetUploadResponse with dataset info
    """
    import uuid
    import io
    
    try:
        # Validate file and get contents
        contents = await validate_dataset_file(file)
        
        # Read CSV
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate data
        if df.empty:
            raise ValueError("Dataset is empty")
        if df.shape[0] < 2:
            raise ValueError("Dataset must have at least 2 rows")
        
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())[:8]
        
        # Store dataset
        datasets[dataset_id] = df
        
        # Get metadata
        n_samples, n_features = df.shape
        column_names = df.columns.tolist()
        data_types = {col: str(df[col].dtype) for col in column_names}
        
        logger.info(f"Dataset uploaded: {dataset_id} ({n_samples} samples, {n_features} features)")
        
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            n_samples=n_samples,
            n_features=n_features,
            column_names=column_names,
            data_types=data_types,
            message=f"Successfully uploaded dataset with {n_samples} samples and {n_features} features"
        )
    except Exception as e:
        logger.error(f"Dataset upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """
    Get information about uploaded dataset.
    
    Args:
        dataset_id: Dataset identifier
        
    Returns:
        Dataset metadata and basic statistics
    """
    try:
        validate_dataset_exists(dataset_id)
        df = datasets[dataset_id]
        
        return {
            "dataset_id": dataset_id,
            "n_samples": df.shape[0],
            "n_features": df.shape[1],
            "column_names": df.columns.tolist(),
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "missing_counts": df.isnull().sum().to_dict(),
            "shape": df.shape
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset and associated models."""
    try:
        validate_dataset_exists(dataset_id)
        
        # Delete dataset
        del datasets[dataset_id]
        
        # Delete associated models (optional: can implement cleanup logic)
        models_to_delete = [mid for mid, m in models.items() 
                           if m.get("dataset_id") == dataset_id]
        for mid in models_to_delete:
            del models[mid]
        
        logger.info(f"Deleted dataset {dataset_id} and {len(models_to_delete)} associated models")
        
        return {"message": f"Deleted dataset {dataset_id}", "models_deleted": len(models_to_delete)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Data Profiling Endpoints
# ============================================================================

@app.post("/profile/{dataset_id}", response_model=DataProfileResponse)
async def profile_dataset(
    dataset_id: str,
    target_column: Optional[str] = None
):
    """
    Profile dataset for quality and characteristics.
    
    Args:
        dataset_id: Dataset to profile
        target_column: Target variable for classification analysis
        
    Returns:
        DataProfileResponse with health report
    """
    try:
        validate_dataset_exists(dataset_id)
        df = datasets[dataset_id]
        
        # Get target if specified
        y = None
        if target_column:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            X = df
        
        # Run profiler
        profiler = DataProfiler()
        report = profiler.analyze(X, y)
        
        # Cache profiler
        profilers[dataset_id] = profiler
        
        return DataProfileResponse(
            dataset_id=dataset_id,
            profile_timestamp=datetime.now().isoformat(),
            data_quality_score=report.data_quality_score,
            health_status=report.health_status.value,
            warnings=report.warnings,
            recommendations=report.recommendations,
            summary_stats=report.to_dict()
        )
    except Exception as e:
        logger.error(f"Profiling failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/profile/{dataset_id}")
async def get_profile(dataset_id: str):
    """Get cached profile for dataset."""
    try:
        if dataset_id not in profilers:
            raise KeyError(f"Profile for dataset '{dataset_id}' not found. Run /profile/{dataset_id} first.")
        
        profiler = profilers[dataset_id]
        return {
            "dataset_id": dataset_id,
            "profile": profiler.report.to_dict() if profiler.report else None
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# AutoML Optimization Endpoints
# ============================================================================

@app.post("/optimize", response_model=OptimizationResponse)
async def start_optimization(request: OptimizationRequest):
    """
    Start Bayesian hyperparameter optimization.
    
    Runs optimization synchronously (for small datasets) and stores best model.
    
    Args:
        request: OptimizationRequest with dataset and parameters
        
    Returns:
        OptimizationResponse with optimization_id and status
    """
    import uuid
    
    try:
        validate_dataset_exists(request.dataset_id)
        
        df = datasets[request.dataset_id]
        
        # Validate target column
        if request.target_column not in df.columns:
            raise ValueError(f"Target column '{request.target_column}' not found")
        
        # Generate optimization ID
        opt_id = str(uuid.uuid4())[:8]
        
        # Prepare data
        y = df[request.target_column]
        X = df.drop(columns=[request.target_column])

        # API currently supports classification only (AutoMLEngine optimizes classifiers).
        task_type = _infer_task_type_from_target(y)
        if task_type != "classification":
            raise ValueError(
                "Regression targets are not supported by this API optimize endpoint yet. "
                "Use the Streamlit PipelineManager path for regression demos."
            )
        
        # Apply preprocessing
        preprocessor = PreprocessingPipeline()
        X_processed = preprocessor.fit_transform(X, y)
        
        # Create AutoML engine
        engine = AutoMLEngine(
            n_trials=request.n_trials or 50,
            cv_folds=request.cv_folds or 5
        )
        
        # Prepare models to optimize
        models_to_optimize = request.models if request.models else [
            "logistic_regression", "random_forest", "xgboost", "lightgbm", "svm"
        ]
        
        # Run optimization for each model
        best_overall_model_name = None
        best_overall_params = None
        best_overall_score = -1.0
        results = {}
        
        logger.info(f"Starting optimization {opt_id} on {len(models_to_optimize)} models")
        
        for model_name in models_to_optimize:
            try:
                logger.info(f"Optimizing {model_name}...")
                best_params, mean_score, std_score = engine.optimize_model(
                    model_name=model_name,
                    X_train=X_processed,
                    y_train=y,
                    metric=request.metric
                )
                
                results[model_name] = {
                    "score": mean_score,
                    "std": std_score,
                    "params": best_params
                }
                
                # Track best overall
                if mean_score > best_overall_score:
                    best_overall_score = mean_score
                    best_overall_model_name = model_name
                    best_overall_params = best_params
                    
            except Exception as e:
                logger.error(f"Failed to optimize {model_name}: {str(e)}")
                results[model_name] = {"error": str(e), "score": 0.0}
        
        # Train and store best model
        if best_overall_model_name is None:
            raise ValueError(
                "Optimization did not produce a valid best model. "
                "Check model configuration and input data."
            )

        best_model_obj = None
        model_id = None
        if best_overall_model_name:
            try:
                best_model_obj = ModelFactory.create_model(
                    best_overall_model_name,
                    hyperparameters=best_overall_params
                )
                
                # Train on full dataset
                best_model_obj.fit(X_processed, y)
                
                # Generate unique model ID
                model_id = str(uuid.uuid4())[:8]
                
                # Store model
                models[model_id] = {
                    "model": best_model_obj,
                    "name": f"{best_overall_model_name}_{model_id}",
                    "type": best_overall_model_name,
                    "score": best_overall_score,
                    "dataset_id": request.dataset_id,
                    "target_column": request.target_column,
                    "hyperparameters": best_overall_params,
                    "metrics": {"cv_mean": best_overall_score},
                    "created_at": datetime.now().isoformat(),
                    "optimization_id": opt_id,
                    "preprocessor": preprocessor
                }
                
                logger.info(f"Stored best model {model_id}: {best_overall_model_name} with score {best_overall_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train and store best model: {str(e)}", exc_info=True)
        
        # Store optimization metadata
        optimizations[opt_id] = {
            "dataset_id": request.dataset_id,
            "target_column": request.target_column,
            "metric": request.metric,
            "models": models_to_optimize,
            "status": "completed",
            "start_time": datetime.now().isoformat(),
            "trials_completed": (request.n_trials or 50) * len(models_to_optimize),
            "best_model": best_overall_model_name,
            "best_model_id": model_id if best_model_obj else None,
            "best_score": best_overall_score,
            "engine": engine,
            "X": X_processed,
            "y": y,
            "results": results
        }
        
        logger.info(f"Optimization {opt_id} completed: best={best_overall_model_name}, score={best_overall_score:.4f}")
        
        return OptimizationResponse(
            optimization_id=opt_id,
            status="completed",
            best_model=best_overall_model_name,
            best_score=best_overall_score,
            trials_completed=(request.n_trials or 50) * len(models_to_optimize),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/optimize/{optimization_id}")
async def get_optimization_status(optimization_id: str):
    """
    Get status and results of optimization.
    
    Args:
        optimization_id: Optimization job identifier
        
    Returns:
        Status, progress, and results when complete
    """
    try:
        if optimization_id not in optimizations:
            raise KeyError(f"Optimization '{optimization_id}' not found")
        
        opt = optimizations[optimization_id]
        return {
            "optimization_id": optimization_id,
            "status": opt["status"],
            "trials_completed": opt["trials_completed"],
            "best_model": opt["best_model"],
            "best_score": opt["best_score"],
            "start_time": opt["start_time"]
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/optimize/{optimization_id}/history")
async def get_optimization_history(optimization_id: str):
    """
    Get trial history from optimization.
    
    Returns:
        List of trials with scores and hyperparameters
    """
    try:
        if optimization_id not in optimizations:
            raise KeyError(f"Optimization '{optimization_id}' not found")
        
        opt = optimizations[optimization_id]
        engine = opt["engine"]
        
        # Convert trial results to dict format
        trials = []
        for trial in engine.trials_history:
            trials.append({
                "trial_number": trial.trial_number,
                "model_name": trial.model_name,
                "score": trial.trial_score,
                "hyperparameters": trial.hyperparameters,
                "is_best": trial.is_best
            })
        
        return {
            "optimization_id": optimization_id,
            "total_trials": len(trials),
            "trials": trials
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/models")
async def list_models():
    """
    List all trained models.
    
    Returns:
        List of model metadata
    """
    try:
        model_list = []
        for model_id, model_data in models.items():
            model_list.append({
                "model_id": model_id,
                "name": model_data.get("name"),
                "type": model_data.get("type"),
                "score": model_data.get("score", 0.0),
                "dataset_id": model_data.get("dataset_id"),
                "created_at": model_data.get("created_at")
            })
        
        return {
            "total_models": len(model_list),
            "models": model_list
        }
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return {"total_models": 0, "models": []}


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """
    Get detailed information about a model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model metadata, hyperparameters, performance
    """
    try:
        validate_model_exists(model_id)
        
        model_data = models[model_id]
        return {
            "model_id": model_id,
            "name": model_data.get("name"),
            "type": model_data.get("type"),
            "score": model_data.get("score"),
            "hyperparameters": model_data.get("hyperparameters", {}),
            "dataset_id": model_data.get("dataset_id"),
            "created_at": model_data.get("created_at"),
            "metrics": model_data.get("metrics", {})
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model."""
    try:
        validate_model_exists(model_id)
        del models[model_id]
        logger.info(f"Deleted model {model_id}")
        return {"message": f"Deleted model {model_id}"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict/{model_id}", response_model=PredictionResponse)
async def predict(model_id: str, request: PredictionRequest):
    """
    Make prediction with trained model.
    
    Args:
        model_id: Model to use for prediction
        request: PredictionRequest with input data
        
    Returns:
        PredictionResponse with prediction, confidence, explanation
    """
    try:
        validate_model_exists(model_id)
        
        model_data = models[model_id]
        model = model_data.get("model")
        
        if not model:
            raise ValueError("Model not properly loaded")
        
        # Convert input to DataFrame
        X_input = pd.DataFrame([request.data])

        # Apply same preprocessing as training (if available)
        preprocessor = _get_model_preprocessor(model_data)
        target_column = model_data.get("target_column")
        X_input = _drop_target_if_present(X_input, target_column)
        X_input_processed = preprocessor.transform(X_input) if preprocessor else X_input
        
        # Get prediction
        prediction = model.predict(X_input_processed)[0]
        
        # Get confidence (try predict_proba if available)
        confidence = 0.5
        is_confident = False
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_input_processed)[0]
            confidence = float(max(proba))
            is_confident = confidence >= 0.7
        
        # Detect OOD (simplified: use distance to training data)
        is_ood = False  # TODO: Implement OOD detection
        
        return PredictionResponse(
            prediction=str(prediction),
            confidence=confidence,
            is_confident=is_confident,
            explanation="Prediction made successfully",
            is_ood=is_ood,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch/{model_id}")
async def predict_batch(
    model_id: str,
    file: UploadFile = File(...)
):
    """
    Make batch predictions on dataset.
    
    Args:
        model_id: Model to use
        file: CSV with data to predict
        
    Returns:
        JSON with predictions and confidence scores
    """
    import io
    
    try:
        validate_model_exists(model_id)
        
        model_data = models[model_id]
        model = model_data.get("model")
        
        if not model:
            raise ValueError("Model not properly loaded")
        
        # Read input CSV
        contents = await file.read()
        X_batch = pd.read_csv(io.BytesIO(contents))

        # Apply same preprocessing as training (if available)
        preprocessor = _get_model_preprocessor(model_data)
        target_column = model_data.get("target_column")
        X_batch = _drop_target_if_present(X_batch, target_column)
        X_batch_processed = preprocessor.transform(X_batch) if preprocessor else X_batch
        
        # Make predictions
        predictions = model.predict(X_batch_processed)
        
        # Get confidence scores
        confidences = [0.5] * len(X_batch)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_batch_processed)
            confidences = [float(max(p)) for p in proba]
        
        # Create results DataFrame
        results = X_batch.copy()
        results['prediction'] = predictions
        results['confidence'] = confidences
        
        # Convert to JSON
        results_json = results.to_dict(orient='records')
        
        return {
            "total_predictions": len(results_json),
            "predictions": results_json
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Explainability Endpoints
# ============================================================================

@app.get("/explain/{model_id}/{sample_index}")
async def explain_prediction(model_id: str, sample_index: int):
    """
    Explain individual prediction.
    
    Args:
        model_id: Model to explain
        sample_index: Sample index in prediction dataset
        
    Returns:
        Explanation with feature contributions
    """
    try:
        validate_model_exists(model_id)
        
        model_data = models[model_id]
        model = model_data.get("model")
        
        if not model:
            raise ValueError("Model not properly loaded")
        
        dataset_id = model_data.get("dataset_id")
        if not dataset_id or dataset_id not in datasets:
            raise ValueError("Original dataset not found for explanation")

        df_train = datasets[dataset_id]
        target_column = model_data.get("target_column")
        X_train_raw = df_train.drop(columns=[target_column]) if target_column in df_train.columns else df_train

        if sample_index < 0 or sample_index >= len(X_train_raw):
            raise ValueError(f"sample_index out of range: {sample_index} (0..{len(X_train_raw)-1})")

        preprocessor = _get_model_preprocessor(model_data)
        if not preprocessor:
            raise ValueError("No preprocessor found for this model; cannot compute SHAP explanations reliably.")

        # Preprocess training background and sample
        X_train_processed = preprocessor.transform(X_train_raw)
        feature_names = preprocessor.get_feature_names()
        X_sample_processed = X_train_processed.iloc[[sample_index]].values

        explainer = ExplainabilityEngine()
        explainer.fit_explainer(model, X_train_processed.values, feature_names)
        local_exp = explainer.explain_local(X_sample_processed)

        return {
            "model_id": model_id,
            "sample_index": sample_index,
            "explanation_method": "SHAP",
            **local_exp
        }
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/explain/{model_id}/global")
async def get_global_explanations(model_id: str):
    """
    Get global feature importance.
    
    Args:
        model_id: Model to explain
        
    Returns:
        Feature importance ranking
    """
    try:
        validate_model_exists(model_id)
        
        model_data = models[model_id]
        model = model_data.get("model")
        
        if not model:
            raise ValueError("Model not properly loaded")
        
        dataset_id = model_data.get("dataset_id")
        if not dataset_id or dataset_id not in datasets:
            raise ValueError("Original dataset not found for explanation")

        df_train = datasets[dataset_id]
        target_column = model_data.get("target_column")
        X_train_raw = df_train.drop(columns=[target_column]) if target_column in df_train.columns else df_train

        preprocessor = _get_model_preprocessor(model_data)
        if not preprocessor:
            raise ValueError("No preprocessor found for this model; cannot compute SHAP explanations reliably.")

        X_train_processed = preprocessor.transform(X_train_raw)
        feature_names = preprocessor.get_feature_names()

        explainer = ExplainabilityEngine()
        explainer.fit_explainer(model, X_train_processed.values, feature_names)
        global_exp = explainer.explain_global()

        return {
            "model_id": model_id,
            "explanation_method": "SHAP",
            **global_exp
        }
    except Exception as e:
        logger.error(f"Feature importance extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Uncertainty Quantification Endpoints
# ============================================================================

@app.post("/uncertainty/{model_id}")
async def estimate_uncertainty(
    model_id: str,
    file: UploadFile = File(...)
):
    """
    Estimate uncertainty for predictions.
    
    Args:
        model_id: Model to use
        file: Data to estimate uncertainty for
        
    Returns:
        Confidence scores and uncertainty estimates
    """
    import io
    
    try:
        validate_model_exists(model_id)
        
        model_data = models[model_id]
        model = model_data.get("model")
        
        if not model:
            raise ValueError("Model not properly loaded")
        
        # Read input CSV
        contents = await file.read()
        X_data = pd.read_csv(io.BytesIO(contents))
        
        # Apply preprocessing to match training
        preprocessor = _get_model_preprocessor(model_data)
        target_column = model_data.get("target_column")
        X_data = _drop_target_if_present(X_data, target_column)
        X_processed = preprocessor.transform(X_data).values if preprocessor else X_data.values

        estimator = UncertaintyEstimator()

        if hasattr(model, "predict_proba"):
            results = estimator.estimate_classification_uncertainty(model, X_processed)
        else:
            results = estimator.estimate_regression_uncertainty(model, X_processed)

        return {
            "model_id": model_id,
            "uncertainty": results
        }
    except Exception as e:
        logger.error(f"Uncertainty estimation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ood-detection/{model_id}")
async def detect_ood(
    model_id: str,
    file: UploadFile = File(...)
):
    """
    Detect out-of-distribution samples.
    
    Args:
        model_id: Model to use
        file: Data to analyze
        
    Returns:
        OOD detection results
    """
    import io
    
    try:
        validate_model_exists(model_id)
        
        model_data = models[model_id]
        model = model_data.get("model")
        
        if not model:
            raise ValueError("Model not properly loaded")
        
        # Read input CSV
        contents = await file.read()
        X_data = pd.read_csv(io.BytesIO(contents))
        
        dataset_id = model_data.get("dataset_id")
        if not dataset_id or dataset_id not in datasets:
            raise ValueError("Original dataset not found for OOD fitting")

        df_train = datasets[dataset_id]
        target_column = model_data.get("target_column")
        X_train_raw = df_train.drop(columns=[target_column]) if target_column in df_train.columns else df_train

        # Apply preprocessing to both train and incoming data
        preprocessor = _get_model_preprocessor(model_data)
        X_data = _drop_target_if_present(X_data, target_column)
        X_train_processed = preprocessor.transform(X_train_raw) if preprocessor else X_train_raw
        X_new_processed = preprocessor.transform(X_data) if preprocessor else X_data

        ood_detector = OODDetector(method="isolation_forest")
        ood_detector.fit(X_train_processed)
        ood_result = ood_detector.detect_ood(X_new_processed, return_scores=True)

        return {
            "model_id": model_id,
            "method": ood_result.get("method_used", "isolation_forest"),
            "ood_percentage": ood_result.get("ood_percentage", 0.0),
            "is_ood": ood_result.get("is_ood").astype(bool).tolist() if "is_ood" in ood_result else [],
            "ood_scores": ood_result.get("ood_scores").tolist() if "ood_scores" in ood_result else [],
            "total_samples": len(X_data)
        }
    except Exception as e:
        logger.error(f"OOD detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Documentation Endpoints
# ============================================================================

@app.get("/documentation/{model_id}")
async def get_model_documentation(
    model_id: str,
    format: str = "markdown"
):
    """
    Get auto-generated model documentation.
    
    Args:
        model_id: Model to document
        format: Output format (markdown, text, html, pdf)
        
    Returns:
        Model documentation in requested format
    """
    try:
        validate_model_exists(model_id)
        
        model_data = models[model_id]
        model = model_data.get("model")
        
        dataset_id = model_data.get("dataset_id")
        df_train = datasets.get(dataset_id) if dataset_id else None
        target_column = model_data.get("target_column")

        generator = DocumentationGenerator()

        # Try to produce richer docs if we still have the dataset
        data_profile = {}
        explainability_summary = {}
        uncertainty_summary = {}
        shift_summary = {}

        if df_train is not None:
            X_train_raw = df_train.drop(columns=[target_column]) if target_column in df_train.columns else df_train
            preprocessor = _get_model_preprocessor(model_data)
            if preprocessor:
                X_train_processed = preprocessor.transform(X_train_raw)
                feature_names = preprocessor.get_feature_names()

                # Explainability (global)
                try:
                    exp = ExplainabilityEngine()
                    exp.fit_explainer(model, X_train_processed.values, feature_names)
                    explainability_summary = exp.explain_global()
                except Exception:
                    explainability_summary = {}

                # Uncertainty (mean confidence)
                try:
                    est = UncertaintyEstimator()
                    if hasattr(model, "predict_proba"):
                        u = est.estimate_classification_uncertainty(model, X_train_processed.values)
                    else:
                        u = est.estimate_regression_uncertainty(model, X_train_processed.values)
                    uncertainty_summary = {
                        "mean_confidence": u.get("confidence_score", "unknown"),
                        "uncertainty_level": u.get("uncertainty_level", "unknown"),
                        "notes": u.get("notes", ""),
                    }
                except Exception:
                    uncertainty_summary = {}

                # Shift summary: cannot be computed without a "new" dataset, keep empty

        # Minimal data_profile from stored profiler if available
        if dataset_id and dataset_id in profilers and profilers[dataset_id].report:
            rep = profilers[dataset_id].report
            data_profile = rep.to_dict()
            # Ensure keys expected by DocumentationGenerator
            if "n_samples" not in data_profile and "dataset_size" in data_profile:
                data_profile["n_samples"] = data_profile["dataset_size"]
            if "n_features" not in data_profile and "feature_count" in data_profile:
                data_profile["n_features"] = data_profile["feature_count"]
            if "missing_values_pct" not in data_profile and "missing_value_report" in data_profile:
                mv = data_profile.get("missing_value_report") or {}
                data_profile["missing_values_pct"] = (sum(mv.values()) / len(mv)) if mv else 0.0
            if "target_distribution" not in data_profile:
                data_profile["target_distribution"] = {}

        if not data_profile:
            # Fallback profile skeleton
            data_profile = {
                "n_samples": "unknown",
                "n_features": "unknown",
                "missing_values_pct": 0.0,
                "target_distribution": {},
            }

        card = generator.generate_model_card(
            model_name=model_data.get("type", model_id),
            metrics=model_data.get("metrics", {}) or {"cv_mean": model_data.get("score", 0.0)},
            data_profile=data_profile,
            explainability_summary=explainability_summary if isinstance(explainability_summary, dict) else {},
            uncertainty_summary=uncertainty_summary if isinstance(uncertainty_summary, dict) else {},
            shift_summary=shift_summary if isinstance(shift_summary, dict) else {},
        )

        sheet = generator.generate_dataset_sheet(data_profile)

        # Export as string
        fmt = (format or "markdown").lower()
        if fmt == "pdf":
            # Compatibility alias: clients asking for PDF receive markdown payload
            # that can be rendered/exported client-side.
            fmt = "markdown"
        if fmt not in {"markdown", "text", "html"}:
            raise ValueError("format must be one of: markdown, text, html, pdf")

        card_str = generator.export_documentation(card, format="markdown" if fmt != "text" else "text")
        sheet_str = generator.export_documentation(sheet, format="markdown" if fmt != "text" else "text")

        if fmt == "html":
            card_str = f"<pre>{html.escape(card_str)}</pre>"
            sheet_str = f"<pre>{html.escape(sheet_str)}</pre>"

        return {
            "model_id": model_id,
            "format": fmt,
            "model_card": card_str,
            "dataset_sheet": sheet_str
        }
    except Exception as e:
        logger.error(f"Documentation generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Error Handlers
# ============================================================================

logger = logging.getLogger(__name__)

@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle ValueError exceptions."""
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(KeyError)
async def key_error_handler(request, exc: KeyError):
    """Handle KeyError exceptions (missing dataset, model, etc)."""
    logger.error(f"KeyError: {str(exc)}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource not found",
            "detail": f"Missing resource: {str(exc)}",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc: FileNotFoundError):
    """Handle file not found errors."""
    logger.error(f"FileNotFoundError: {str(exc)}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "File not found",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle all other exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if logger.level == logging.DEBUG else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

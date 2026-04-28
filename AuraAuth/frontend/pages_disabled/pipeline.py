"""Model training pipeline page."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_model(task_type: str, model_name: str, *, fast_mode: bool):
    rf_trees = 80 if fast_mode else 250
    if task_type == "classification":
        if model_name == "logistic_regression":
            return LogisticRegression(max_iter=120 if fast_mode else 200)
        return RandomForestClassifier(
            n_estimators=rf_trees,
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "linear_regression":
        return LinearRegression()
    return RandomForestRegressor(
        n_estimators=rf_trees,
        random_state=42,
        n_jobs=-1,
    )


def _score(task_type: str, y_true, y_pred) -> tuple[float, str]:
    if task_type == "classification":
        return float(accuracy_score(y_true, y_pred)), "accuracy"
    return float(r2_score(y_true, y_pred)), "r2"


def _prepare_target_for_task(y: pd.Series, task_type: str) -> tuple[pd.Series | None, str | None]:
    """Validate and normalize target labels for the selected task."""
    if y.isna().any():
        return None, "Target column contains missing values. Please clean target values first."

    if task_type == "classification":
        if pd.api.types.is_numeric_dtype(y):
            y_numeric = pd.to_numeric(y, errors="coerce")
            if y_numeric.isna().any():
                return None, "Classification target contains invalid numeric values."

            # Allow integer-like numeric labels (e.g., 0.0, 1.0), reject continuous labels.
            is_integer_like = (y_numeric % 1 == 0).all()
            if not is_integer_like:
                return (
                    None,
                    "Selected task is classification, but target looks continuous. "
                    "Use regression for continuous targets.",
                )
            y_prepared = y_numeric.astype(int)
        else:
            y_prepared = y.astype(str)

        if y_prepared.nunique() < 2:
            return None, "Classification target needs at least two classes."
        return y_prepared, None

    y_numeric = pd.to_numeric(y, errors="coerce")
    if y_numeric.isna().any():
        return None, "Selected task is regression, but target contains non-numeric values."
    return y_numeric.astype(float), None


def page_run_pipeline() -> None:
    """Train baseline models locally and persist results in session state."""
    st.title("Run Pipeline")

    df = st.session_state.get("dataset")
    target = st.session_state.get("target_column")
    task_type = st.session_state.get("task_type")

    if df is None or not target or not task_type:
        st.warning("Please upload a dataset and select target/task first.")
        return

    candidate_models = [
        "logistic_regression" if task_type == "classification" else "linear_regression",
        "random_forest",
    ]
    selected = st.multiselect("Candidate models", options=candidate_models, default=candidate_models)

    test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    fast_mode = st.checkbox(
        "Fast mode (recommended for large datasets)",
        value=True,
        help="Uses fewer trees and optional row sampling to finish faster.",
    )

    max_rows = st.number_input(
        "Max training rows (fast mode only)",
        min_value=500,
        max_value=200000,
        value=10000,
        step=500,
        disabled=not fast_mode,
    )

    if st.button("Run local pipeline", use_container_width=True):
        if not selected:
            st.error("Select at least one model.")
            return

        working_df = df
        if fast_mode and len(df) > int(max_rows):
            working_df = df.sample(n=int(max_rows), random_state=42)
            st.info(f"Fast mode: sampled {len(working_df)} rows from {len(df)} total rows.")

        X = working_df.drop(columns=[target])
        y, target_error = _prepare_target_for_task(working_df[target], task_type)
        if target_error:
            st.error(target_error)
            return

        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop",
        )

        stratify = y if task_type == "classification" and y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        leaderboard = []
        fitted_pipelines = {}

        model_errors: list[str] = []
        for model_name in selected:
            estimator = _build_model(task_type, model_name, fast_mode=fast_mode)
            pipe = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
            try:
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                score_value, score_name = _score(task_type, y_test, preds)

                extra_metric = None
                if task_type == "classification":
                    extra_metric = float(f1_score(y_test, preds, average="weighted"))
                else:
                    extra_metric = float(mean_absolute_error(y_test, preds))

                leaderboard.append(
                    {
                        "model": model_name,
                        score_name: round(score_value, 4),
                        "f1_weighted" if task_type == "classification" else "mae": round(extra_metric, 4),
                    }
                )
                fitted_pipelines[model_name] = pipe
            except Exception as exc:
                model_errors.append(f"{model_name}: {exc}")

        if not leaderboard:
            st.error("No model could be trained with the current task/target settings.")
            if model_errors:
                st.write("Model errors:")
                for err in model_errors:
                    st.write(f"- {err}")
            return

        if model_errors:
            st.warning("Some models failed and were skipped.")

        leaderboard_df = pd.DataFrame(leaderboard).sort_values(by=score_name, ascending=False)
        best_row = leaderboard_df.iloc[0]
        best_model_name = str(best_row["model"])

        st.session_state.pipeline_results = {
            "task_type": task_type,
            "metric": score_name,
            "leaderboard": leaderboard_df,
            "best_model": best_model_name,
            "best_score": float(best_row[score_name]),
            "model_objects": fitted_pipelines,
            "feature_columns": list(X.columns),
        }
        st.session_state.pipeline_executed = True

        st.success(f"Pipeline complete. Best model: {best_model_name} ({score_name}={best_row[score_name]:.4f})")
        st.dataframe(leaderboard_df, use_container_width=True)

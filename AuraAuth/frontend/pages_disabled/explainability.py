"""Simple explainability page."""

import streamlit as st


def page_explainability() -> None:
    """Show available explainability metadata for the best model."""
    st.title("Explainability")

    results = st.session_state.get("pipeline_results")
    if not results:
        st.warning("Run the pipeline first to inspect explainability.")
        return

    model_name = results.get("best_model")
    model_objects = results.get("model_objects", {})
    pipeline = model_objects.get(model_name)
    if pipeline is None:
        st.warning("No fitted model found in session.")
        return

    model = pipeline.named_steps.get("model")
    st.subheader(f"Model: {model_name}")

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        st.bar_chart(values)
        st.caption("Feature importance values from tree-based model.")
    elif hasattr(model, "coef_"):
        coeffs = model.coef_
        if hasattr(coeffs, "ndim") and coeffs.ndim > 1:
            coeffs = coeffs[0]
        st.bar_chart(coeffs)
        st.caption("Coefficient magnitudes from linear model.")
    else:
        st.info("This model type does not expose built-in feature importance.")

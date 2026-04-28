"""
Documentation Generator Module for AuraAuth AutoML System

Automatically generates human-readable model cards and dataset sheets following best
practices from model card frameworks and dataset sheets literature.

Auto-generated documentation improves transparency, accountability, and responsible AI
deployment by providing stakeholders with comprehensive, structured information about
model behavior, limitations, and intended use cases.

Author: AuraAuth Development Team
"""

from typing import Dict, Any
from datetime import datetime


class DocumentationGenerator:
    """
    Generates structured model cards and dataset sheets for ML models.

    This class produces human-readable documentation suitable for academic evaluation,
    industry reporting, and web app display. Documentation follows model card and
    dataset sheet best practices, emphasizing transparency and honest assessment of
    limitations.

    The generator takes structured metadata (metrics, profiles, summaries) and produces
    clean, readable documentation in multiple formats (text, markdown).

    Attributes:
        None (stateless utility class)

    Example:
        >>> from backend.core.documentation_generator import DocumentationGenerator
        >>> import numpy as np
        >>>
        >>> generator = DocumentationGenerator()
        >>>
        >>> # Prepare inputs
        >>> metrics = {"accuracy": 0.92, "precision": 0.89, "recall": 0.85}
        >>> data_profile = {
        ...     "n_samples": 1000,
        ...     "n_features": 15,
        ...     "missing_values_pct": 2.5,
        ...     "target_distribution": {"class_0": 0.6, "class_1": 0.4}
        ... }
        >>> explainability_summary = {
        ...     "feature_importance": [
        ...         {"feature": "age", "importance": 0.35},
        ...         {"feature": "income", "importance": 0.28}
        ...     ]
        ... }
        >>> uncertainty_summary = {"mean_confidence": 0.85, "uncertainty_level": "LOW"}
        >>> shift_summary = {"shift_score": 0.15, "shift_level": "LOW"}
        >>>
        >>> # Generate model card
        >>> model_card = generator.generate_model_card(
        ...     model_name="RandomForestClassifier",
        ...     metrics=metrics,
        ...     data_profile=data_profile,
        ...     explainability_summary=explainability_summary,
        ...     uncertainty_summary=uncertainty_summary,
        ...     shift_summary=shift_summary
        ... )
        >>>
        >>> # Export to markdown
        >>> markdown_doc = generator.export_documentation(model_card, format="markdown")
        >>> print(markdown_doc)
    """

    def __init__(self) -> None:
        """Initialize the DocumentationGenerator (stateless)."""
        pass

    def generate_model_card(
        self,
        model_name: str,
        metrics: Dict[str, float],
        data_profile: Dict[str, Any],
        explainability_summary: Dict[str, Any],
        uncertainty_summary: Dict[str, Any],
        shift_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive model card documenting model behavior and deployment context.

        A model card is a structured document that communicates model characteristics,
        intended use, performance, and limitations to stakeholders. This implementation
        follows best practices from the model card framework literature, emphasizing
        transparency and responsible AI.

        Args:
            model_name (str): Name or type of the model (e.g., "RandomForestClassifier").
            metrics (Dict[str, float]): Performance metrics (accuracy, precision, recall, F1, AUC, etc.).
                Keys should be metric names, values should be floats in [0, 1].
            data_profile (Dict[str, Any]): Dataset profile including:
                - "n_samples" (int): Number of training samples
                - "n_features" (int): Number of features
                - "missing_values_pct" (float): Percentage of missing values
                - "target_distribution" (Dict): Class distribution
                - Optional: "feature_types", "data_quality_score"
            explainability_summary (Dict[str, Any]): Feature importance summary from ExplainabilityEngine:
                - "feature_importance" (List[Dict]): List of features and importance scores
                - Optional: "notes"
            uncertainty_summary (Dict[str, Any]): Uncertainty metrics from UncertaintyEstimator:
                - "mean_confidence" (float): Mean confidence [0, 1]
                - "uncertainty_level" (str): "LOW", "MEDIUM", "HIGH"
                - Optional: "notes"
            shift_summary (Dict[str, Any]): Distribution shift metrics from DistributionShiftDetector:
                - "shift_score" (float): Shift magnitude [0, 1]
                - "shift_level" (str): "LOW", "MEDIUM", "HIGH"
                - Optional: "notes"

        Returns:
            Dict[str, Any]: Structured model card containing:
                - "title" (str): Model card title
                - "timestamp" (str): Generation timestamp
                - "overview" (str): Model overview and purpose
                - "intended_use" (str): Intended use cases and constraints
                - "performance" (str): Performance summary with metrics
                - "feature_importance" (str): Top features and their impact
                - "reliability" (str): Uncertainty and reliability assessment
                - "distribution_shift" (str): Distribution shift warnings
                - "limitations" (str): Model limitations and failure modes
                - "ethical_considerations" (str): Ethical and fairness considerations
                - "recommendations" (str): Recommendations for deployment and monitoring

        Raises:
            ValueError: If required keys missing from input dictionaries.
            TypeError: If inputs are wrong type.

        Notes:
            - All inputs should be validated upstream before calling this method.
            - This method produces documentation structure; use export_documentation()
              to format for display.
            - Missing optional sections are handled gracefully with defaults.

        Example:
            >>> generator = DocumentationGenerator()
            >>> card = generator.generate_model_card(
            ...     model_name="XGBClassifier",
            ...     metrics={"accuracy": 0.95, "auc": 0.92},
            ...     data_profile={
            ...         "n_samples": 1500,
            ...         "n_features": 20,
            ...         "missing_values_pct": 1.2,
            ...         "target_distribution": {"negative": 0.7, "positive": 0.3}
            ...     },
            ...     explainability_summary={
            ...         "feature_importance": [
            ...             {"feature": "feature_1", "importance": 0.42},
            ...             {"feature": "feature_2", "importance": 0.28}
            ...         ]
            ...     },
            ...     uncertainty_summary={"mean_confidence": 0.88, "uncertainty_level": "LOW"},
            ...     shift_summary={"shift_score": 0.12, "shift_level": "LOW"}
            ... )
        """
        # Validate required inputs
        self._validate_model_card_inputs(
            model_name, metrics, data_profile, explainability_summary,
            uncertainty_summary, shift_summary
        )

        # Generate sections
        overview = self._generate_overview(model_name, data_profile)
        intended_use = self._generate_intended_use(model_name)
        performance = self._generate_performance_section(metrics)
        feature_importance = self._generate_feature_importance_section(explainability_summary)
        reliability = self._generate_reliability_section(uncertainty_summary)
        distribution_shift = self._generate_shift_section(shift_summary)
        limitations = self._generate_limitations_section(metrics, data_profile)
        ethical_considerations = self._generate_ethical_considerations()
        recommendations = self._generate_recommendations(
            uncertainty_summary, shift_summary, data_profile
        )

        return {
            "title": f"Model Card: {model_name}",
            "timestamp": datetime.now().isoformat(),
            "overview": overview,
            "intended_use": intended_use,
            "performance": performance,
            "feature_importance": feature_importance,
            "reliability": reliability,
            "distribution_shift": distribution_shift,
            "limitations": limitations,
            "ethical_considerations": ethical_considerations,
            "recommendations": recommendations
        }

    def generate_dataset_sheet(
        self,
        data_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a dataset sheet documenting training data characteristics and quality.

        A dataset sheet provides structured information about the data used to train
        the model, including size, composition, quality metrics, and known issues.
        This follows best practices from the dataset sheets literature.

        Args:
            data_profile (Dict[str, Any]): Comprehensive data profile including:
                - "n_samples" (int): Number of samples
                - "n_features" (int): Number of features
                - "missing_values_pct" (float): Percentage of missing values
                - "target_distribution" (Dict): Class/value distribution
                - Optional: "feature_types", "data_quality_score", "imbalance_ratio",
                  "duplicate_rows_pct", "outlier_ratio"

        Returns:
            Dict[str, Any]: Structured dataset sheet containing:
                - "title" (str): "Dataset Sheet"
                - "timestamp" (str): Generation timestamp
                - "description" (str): Dataset overview
                - "composition" (str): Size and feature information
                - "distribution" (str): Target distribution and class balance
                - "data_quality" (str): Data quality assessment and issues
                - "warnings" (str): Known issues and limitations
                - "preprocessing_notes" (str): Suggested data preprocessing steps

        Raises:
            ValueError: If required keys missing from data_profile.
            TypeError: If inputs are wrong type.

        Example:
            >>> generator = DocumentationGenerator()
            >>> sheet = generator.generate_dataset_sheet({
            ...     "n_samples": 2000,
            ...     "n_features": 25,
            ...     "missing_values_pct": 3.5,
            ...     "target_distribution": {"class_0": 0.65, "class_1": 0.35},
            ...     "data_quality_score": 0.82,
            ...     "imbalance_ratio": 1.86
            ... })
        """
        self._validate_dataset_sheet_input(data_profile)

        # Generate sections
        description = self._generate_dataset_description(data_profile)
        composition = self._generate_composition_section(data_profile)
        distribution = self._generate_distribution_section(data_profile)
        data_quality = self._generate_data_quality_section(data_profile)
        warnings = self._generate_data_warnings(data_profile)
        preprocessing = self._generate_preprocessing_notes(data_profile)

        return {
            "title": "Dataset Sheet",
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "composition": composition,
            "distribution": distribution,
            "data_quality": data_quality,
            "warnings": warnings,
            "preprocessing_notes": preprocessing
        }

    def export_documentation(
        self,
        doc: Dict[str, Any],
        format: str = "markdown"
    ) -> str:
        """
        Export documentation dictionary to formatted string representation.

        Supports multiple output formats suitable for different use cases:
        - "markdown": For web display, GitHub, Jupyter notebooks
        - "text": For plain text reports, terminal display

        Args:
            doc (Dict[str, Any]): Documentation dictionary (from generate_model_card
                or generate_dataset_sheet).
            format (str): Output format. Options: "markdown", "text". Default: "markdown".

        Returns:
            str: Formatted documentation string, ready for display or export.

        Raises:
            ValueError: If format not recognized or doc missing required keys.
            TypeError: If doc is not a dictionary.

        Notes:
            - This method does NOT write to files; it returns formatted string.
            - For file export, use external write operation on the returned string.
            - Markdown format includes proper heading levels and formatting.
            - Text format uses ASCII formatting for terminal display.

        Example:
            >>> generator = DocumentationGenerator()
            >>> card = generator.generate_model_card(...)
            >>>
            >>> # Export as markdown
            >>> markdown_str = generator.export_documentation(card, format="markdown")
            >>> print(markdown_str)
            >>>
            >>> # Export as plain text
            >>> text_str = generator.export_documentation(card, format="text")
            >>>
            >>> # Save to file (external operation)
            >>> with open("model_card.md", "w") as f:
            ...     f.write(markdown_str)
        """
        if not isinstance(doc, dict):
            raise TypeError(f"doc must be dictionary, received {type(doc).__name__}")

        if format not in ["markdown", "text"]:
            raise ValueError(
                f"format must be 'markdown' or 'text', received '{format}'"
            )

        if "title" not in doc or "timestamp" not in doc:
            raise ValueError(
                "doc missing required keys: 'title', 'timestamp'"
            )

        if format == "markdown":
            return self._export_as_markdown(doc)
        else:
            return self._export_as_text(doc)

    # ==================== Private Helper Methods ====================

    def _validate_model_card_inputs(
        self,
        model_name: str,
        metrics: Dict,
        data_profile: Dict,
        explainability_summary: Dict,
        uncertainty_summary: Dict,
        shift_summary: Dict
    ) -> None:
        """Validate inputs to generate_model_card."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty string")

        if not isinstance(metrics, dict):
            raise TypeError("metrics must be dictionary")

        if not isinstance(data_profile, dict) or "n_samples" not in data_profile:
            raise ValueError("data_profile must contain 'n_samples'")

        if not isinstance(explainability_summary, dict):
            raise TypeError("explainability_summary must be dictionary")

        if not isinstance(uncertainty_summary, dict):
            raise TypeError("uncertainty_summary must be dictionary")

        if not isinstance(shift_summary, dict):
            raise TypeError("shift_summary must be dictionary")

    def _validate_dataset_sheet_input(self, data_profile: Dict) -> None:
        """Validate input to generate_dataset_sheet."""
        if not isinstance(data_profile, dict):
            raise TypeError("data_profile must be dictionary")

        required_keys = ["n_samples", "n_features", "missing_values_pct", "target_distribution"]
        missing = [k for k in required_keys if k not in data_profile]
        if missing:
            raise ValueError(
                f"data_profile missing required keys: {', '.join(missing)}"
            )

    def _generate_overview(self, model_name: str, data_profile: Dict) -> str:
        """Generate model overview section."""
        n_samples = data_profile.get("n_samples", "unknown")
        n_features = data_profile.get("n_features", "unknown")

        overview = (
            f"This model is a {model_name} trained on a dataset with {n_samples} samples "
            f"and {n_features} features. The model is optimized for small-to-medium structured "
            f"datasets typical of enterprise and research environments."
        )

        return overview

    def _generate_intended_use(self, model_name: str) -> str:
        """Generate intended use section."""
        return (
            "This model is intended for binary or multiclass classification tasks on tabular data. "
            "It is suitable for:\n"
            "- Decision support systems\n"
            "- Exploratory analysis and prototyping\n"
            "- Academic research and evaluation\n"
            "\n"
            "This model is NOT suitable for:\n"
            "- Critical safety-sensitive applications (e.g., medical diagnosis without human review)\n"
            "- Real-time systems with strict latency requirements\n"
            "- Datasets significantly different from training distribution"
        )

    def _generate_performance_section(self, metrics: Dict[str, float]) -> str:
        """Generate performance metrics section."""
        if not metrics:
            return "No performance metrics provided."

        lines = ["Model performance on training/validation data:"]
        for metric_name, metric_value in sorted(metrics.items()):
            if isinstance(metric_value, float):
                lines.append(f"- {metric_name.title()}: {metric_value:.4f}")
            else:
                lines.append(f"- {metric_name.title()}: {metric_value}")

        return "\n".join(lines)

    def _generate_feature_importance_section(self, explainability_summary: Dict) -> str:
        """Generate feature importance section."""
        if "feature_importance" not in explainability_summary:
            return "Feature importance data not available."

        features = explainability_summary.get("feature_importance", [])
        if not features:
            return "No feature importance scores computed."

        lines = ["Top features by importance:"]
        for i, item in enumerate(features[:5], 1):
            feature_name = item.get("feature", "unknown")
            importance = item.get("importance", 0)
            lines.append(f"{i}. {feature_name}: {importance:.3f}")

        return "\n".join(lines)

    def _generate_reliability_section(self, uncertainty_summary: Dict) -> str:
        """Generate reliability and uncertainty section."""
        confidence = uncertainty_summary.get("mean_confidence", "unknown")
        level = uncertainty_summary.get("uncertainty_level", "unknown")

        reliability = (
            f"Mean prediction confidence: {confidence if isinstance(confidence, str) else f'{confidence:.3f}'}\n"
            f"Uncertainty level: {level}\n"
            f"\n"
            f"Interpretation: "
        )

        if level == "LOW":
            reliability += (
                "Model predictions are reliable with high confidence. "
                "Predictions can be used directly with minimal caution."
            )
        elif level == "MEDIUM":
            reliability += (
                "Model predictions show moderate uncertainty. "
                "Recommend human review for critical decisions."
            )
        else:
            reliability += (
                "Model predictions are uncertain. "
                "Use only as a preliminary signal; human judgment is essential."
            )

        return reliability

    def _generate_shift_section(self, shift_summary: Dict) -> str:
        """Generate distribution shift section."""
        shift_score = shift_summary.get("shift_score", "unknown")
        shift_level = shift_summary.get("shift_level", "unknown")

        shift_text = (
            f"Distribution shift score: {shift_score if isinstance(shift_score, str) else f'{shift_score:.3f}'}\n"
            f"Shift level: {shift_level}\n"
            f"\n"
            f"Warning: "
        )

        if shift_level == "LOW":
            shift_text += (
                "No significant distribution shift detected. "
                "Model should perform reliably on new data similar to training."
            )
        elif shift_level == "MEDIUM":
            shift_text += (
                "Moderate distribution shift detected. "
                "Monitor model performance on new data and consider retraining if needed."
            )
        else:
            shift_text += (
                "Significant distribution shift detected. "
                "Model may not generalize well to new data. Retraining is recommended."
            )

        return shift_text

    def _generate_limitations_section(self, metrics: Dict, data_profile: Dict) -> str:
        """Generate limitations section."""
        limitations = [
            "This model has the following known limitations:",
            "",
            "1. Data Size: Model trained on limited data (500-5000 samples typical). "
            "Performance may degrade on larger or more complex distributions.",
            "",
            "2. Feature Scope: Model only uses available tabular features. "
            "Important contextual information may not be captured.",
            "",
            "3. Temporal Generalization: Model reflects patterns in training data. "
            "Performance may degrade if underlying distributions shift over time.",
            "",
            "4. Edge Cases: Model may perform poorly on rare or unusual samples "
            "not well-represented in training data.",
            "",
            "5. Interpretability-Performance Trade-off: Some models prioritize interpretability "
            "over maximum predictive accuracy."
        ]

        return "\n".join(limitations)

    def _generate_ethical_considerations(self) -> str:
        """Generate ethical considerations section."""
        ethics = [
            "Ethical and fairness considerations:",
            "",
            "1. Bias & Fairness: Model may perpetuate biases present in training data. "
            "Evaluate performance across demographic groups.",
            "",
            "2. Transparency: Use explainability features to provide rationale for predictions. "
            "This is especially important for high-impact decisions.",
            "",
            "3. Human Judgment: Model is a decision support tool, not a replacement for human judgment. "
            "Critical decisions should involve human review.",
            "",
            "4. Accountability: Organization deploying this model is responsible for outcomes. "
            "Maintain audit trails and monitoring."
        ]

        return "\n".join(ethics)

    def _generate_recommendations(
        self,
        uncertainty_summary: Dict,
        shift_summary: Dict,
        data_profile: Dict
    ) -> str:
        """Generate recommendations section."""
        recommendations = ["Recommended practices for deployment:"]

        # Confidence-based recommendations
        uncertainty_level = uncertainty_summary.get("uncertainty_level", "MEDIUM")
        if uncertainty_level == "HIGH":
            recommendations.append(
                "- Use confidence scores to flag uncertain predictions for human review"
            )
        else:
            recommendations.append(
                "- Monitor confidence scores on new data; alert if they decline"
            )

        # Shift-based recommendations
        shift_level = shift_summary.get("shift_level", "MEDIUM")
        if shift_level in ["MEDIUM", "HIGH"]:
            recommendations.append(
                "- Implement distribution shift monitoring; retrain if shift is detected"
            )
        else:
            recommendations.append(
                "- Continue periodic retraining to maintain performance over time"
            )

        # Data quality recommendations
        missing_pct = data_profile.get("missing_values_pct", 0)
        if missing_pct > 5:
            recommendations.append(
                f"- Current data has {missing_pct}% missing values; improve data quality processes"
            )

        recommendations.extend([
            "- Use explainability features to understand predictions",
            "- Evaluate model performance on new data regularly",
            "- Document any changes to features, preprocessing, or model type",
            "- Maintain this model card and dataset sheet up-to-date"
        ])

        return "\n".join(recommendations)

    def _generate_dataset_description(self, data_profile: Dict) -> str:
        """Generate dataset description."""
        n_samples = data_profile.get("n_samples", "unknown")
        n_features = data_profile.get("n_features", "unknown")

        return (
            f"This dataset contains {n_samples} samples with {n_features} features. "
            "It represents a curated collection of structured tabular data suitable for "
            "machine learning model training and evaluation. The data is assumed to have "
            "undergone basic preprocessing and validation before use."
        )

    def _generate_composition_section(self, data_profile: Dict) -> str:
        """Generate dataset composition section."""
        n_samples = data_profile.get("n_samples", "unknown")
        n_features = data_profile.get("n_features", "unknown")
        missing_pct = data_profile.get("missing_values_pct", "unknown")

        composition = [
            f"- Total samples: {n_samples}",
            f"- Total features: {n_features}",
            f"- Missing values: {missing_pct}%" if isinstance(missing_pct, (int, float)) else f"- Missing values: {missing_pct}"
        ]

        return "\n".join(composition)

    def _generate_distribution_section(self, data_profile: Dict) -> str:
        """Generate target distribution section."""
        distribution = data_profile.get("target_distribution", {})

        if not distribution:
            return "Target distribution information not available."

        lines = ["Target distribution:"]
        for class_label, proportion in sorted(distribution.items()):
            if isinstance(proportion, float):
                lines.append(f"- {class_label}: {proportion:.1%}")
            else:
                lines.append(f"- {class_label}: {proportion}")

        imbalance_ratio = data_profile.get("imbalance_ratio", None)
        if imbalance_ratio:
            lines.append(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")

        return "\n".join(lines)

    def _generate_data_quality_section(self, data_profile: Dict) -> str:
        """Generate data quality assessment."""
        quality_score = data_profile.get("data_quality_score", None)

        quality = []

        if quality_score:
            quality.append(f"Overall data quality score: {quality_score:.2f}/1.0")

        missing_pct = data_profile.get("missing_values_pct", 0)
        if missing_pct > 10:
            quality.append(f"⚠️  High missing values: {missing_pct}% missing")
        elif missing_pct > 5:
            quality.append(f"⚠️  Moderate missing values: {missing_pct}% missing")
        elif missing_pct > 0:
            quality.append(f"✓ Low missing values: {missing_pct}% missing")
        else:
            quality.append("✓ No missing values detected")

        outlier_ratio = data_profile.get("outlier_ratio", None)
        if outlier_ratio and outlier_ratio > 0.05:
            quality.append(f"⚠️  Outliers detected: {outlier_ratio:.1%}")

        if not quality:
            quality.append("Data quality assessment: No detailed metrics available")

        return "\n".join(quality)

    def _generate_data_warnings(self, data_profile: Dict) -> str:
        """Generate data quality warnings."""
        warnings = []

        missing_pct = data_profile.get("missing_values_pct", 0)
        if missing_pct > 10:
            warnings.append("- High missing values may require careful imputation strategy")

        imbalance = data_profile.get("imbalance_ratio", None)
        if imbalance and imbalance > 3:
            warnings.append("- Severe class imbalance detected; consider resampling techniques")

        duplicates = data_profile.get("duplicate_rows_pct", 0)
        if duplicates > 5:
            warnings.append("- Multiple duplicate rows; verify data integrity")

        if not warnings:
            warnings.append("✓ No major data quality issues detected")

        return "\n".join(warnings)

    def _generate_preprocessing_notes(self, data_profile: Dict) -> str:
        """Generate preprocessing recommendations."""
        notes = ["Recommended preprocessing steps:"]

        missing_pct = data_profile.get("missing_values_pct", 0)
        if missing_pct > 0:
            notes.append("1. Handle missing values (imputation or removal)")

        notes.extend([
            "2. Feature scaling (standardization or normalization)",
            "3. Feature engineering (interactions, polynomial features if needed)",
            "4. Outlier detection and treatment",
            "5. Categorical encoding (one-hot, label encoding)"
        ])

        imbalance = data_profile.get("imbalance_ratio", None)
        if imbalance and imbalance > 2:
            notes.append("6. Address class imbalance (SMOTE, class weights, stratified sampling)")

        return "\n".join(notes)

    def _export_as_markdown(self, doc: Dict[str, Any]) -> str:
        """Export documentation as markdown."""
        lines = [
            f"# {doc.get('title', 'Documentation')}",
            "",
            f"**Generated:** {doc.get('timestamp', 'unknown')}",
            "",
        ]

        # Add sections in order
        sections = [
            ("Overview", "overview"),
            ("Intended Use", "intended_use"),
            ("Performance", "performance"),
            ("Feature Importance", "feature_importance"),
            ("Reliability & Uncertainty", "reliability"),
            ("Distribution Shift", "distribution_shift"),
            ("Limitations", "limitations"),
            ("Ethical Considerations", "ethical_considerations"),
            ("Recommendations", "recommendations"),
            ("Dataset Description", "description"),
            ("Composition", "composition"),
            ("Target Distribution", "distribution"),
            ("Data Quality", "data_quality"),
            ("Known Issues", "warnings"),
            ("Preprocessing Notes", "preprocessing_notes"),
        ]

        for section_title, section_key in sections:
            if section_key in doc:
                lines.append(f"## {section_title}")
                lines.append("")
                lines.append(doc[section_key])
                lines.append("")

        return "\n".join(lines)

    def _export_as_text(self, doc: Dict[str, Any]) -> str:
        """Export documentation as plain text."""
        lines = [
            "=" * 80,
            doc.get("title", "DOCUMENTATION"),
            "=" * 80,
            "",
            f"Generated: {doc.get('timestamp', 'unknown')}",
            "",
        ]

        sections = [
            ("OVERVIEW", "overview"),
            ("INTENDED USE", "intended_use"),
            ("PERFORMANCE", "performance"),
            ("FEATURE IMPORTANCE", "feature_importance"),
            ("RELIABILITY & UNCERTAINTY", "reliability"),
            ("DISTRIBUTION SHIFT", "distribution_shift"),
            ("LIMITATIONS", "limitations"),
            ("ETHICAL CONSIDERATIONS", "ethical_considerations"),
            ("RECOMMENDATIONS", "recommendations"),
            ("DATASET DESCRIPTION", "description"),
            ("COMPOSITION", "composition"),
            ("TARGET DISTRIBUTION", "distribution"),
            ("DATA QUALITY", "data_quality"),
            ("KNOWN ISSUES", "warnings"),
            ("PREPROCESSING NOTES", "preprocessing_notes"),
        ]

        for section_title, section_key in sections:
            if section_key in doc:
                lines.append("-" * 80)
                lines.append(section_title)
                lines.append("-" * 80)
                lines.append(doc[section_key])
                lines.append("")

        return "\n".join(lines)

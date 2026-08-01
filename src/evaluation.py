"""Reusable evaluation utilities for the Phi-3 job-skill project.

This module extracts the reusable evaluation logic from
``notebooks/04-model-evaluation.ipynb``.

It supports:

- cleaning generated labels;
- normalizing labels;
- calculating Accuracy, Precision, Recall, and F1;
- comparing baseline and fine-tuned predictions;
- generating classification reports;
- building a confusion matrix;
- saving row-level results and summary metrics;
- creating evaluation visualizations.

The module can be imported by notebooks or used directly from the command line
when prediction files have already been generated.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_GROUND_TRUTH_COLUMN = "Ground Truth"
DEFAULT_BASELINE_COLUMN = "Baseline Prediction"
DEFAULT_FINETUNED_COLUMN = "Fine-tuned Prediction"


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file with clear error reporting."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation file was not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)

    if suffix in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(file_path)
        except Exception:
            # Some Kaggle files use an XLS extension but contain CSV text.
            return pd.read_csv(file_path)

    raise ValueError(
        f"Unsupported evaluation format '{suffix}'. Use CSV, XLS, or XLSX."
    )


def validate_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str = "evaluation data",
) -> None:
    """Confirm that all required columns are available."""
    required_columns = list(required_columns)
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def clean_prediction(text: object) -> str:
    """Clean generated model output and retain one label.

    This matches the prediction-cleaning logic used in the evaluation notebook.
    """
    text = str(text).strip()

    text = re.sub(
        r"^(skill category|skill_name|skill name|answer|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return "[empty]"

    result = lines[0]
    result = result.strip(" \t\n\r\"'`.,;:")

    return result if result else "[empty]"


def normalize_label(text: object) -> str:
    """Normalize labels before exact-match evaluation.

    The normalization follows the notebook's intent while preventing repeated
    spaces after commas.
    """
    normalized = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace(".", "")
    normalized = re.sub(r"\s*,\s*", ", ", normalized)

    return normalized


def normalize_labels(values: Sequence[object]) -> list[str]:
    """Normalize a sequence of labels."""
    return [normalize_label(value) for value in values]


def validate_prediction_lengths(
    ground_truths: Sequence[object],
    baseline_predictions: Sequence[object] | None,
    finetuned_predictions: Sequence[object],
) -> None:
    """Ensure all prediction lists have matching lengths."""
    expected_length = len(ground_truths)

    if len(finetuned_predictions) != expected_length:
        raise ValueError(
            "Fine-tuned prediction count does not match ground truths: "
            f"{len(finetuned_predictions)} vs {expected_length}."
        )

    if (
        baseline_predictions is not None
        and len(baseline_predictions) != expected_length
    ):
        raise ValueError(
            "Baseline prediction count does not match ground truths: "
            f"{len(baseline_predictions)} vs {expected_length}."
        )


def calculate_metrics(
    ground_truths: Sequence[object],
    predictions: Sequence[object],
    average: str = "weighted",
) -> dict[str, float | int]:
    """Calculate classification metrics using normalized exact-match labels."""
    if len(ground_truths) != len(predictions):
        raise ValueError(
            "Prediction count must match ground-truth count before evaluation."
        )

    ground_truth_norm = normalize_labels(ground_truths)
    prediction_norm = normalize_labels(predictions)

    return {
        "num_samples": len(ground_truth_norm),
        "accuracy": float(
            accuracy_score(ground_truth_norm, prediction_norm)
        ),
        "precision": float(
            precision_score(
                ground_truth_norm,
                prediction_norm,
                average=average,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                ground_truth_norm,
                prediction_norm,
                average=average,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                ground_truth_norm,
                prediction_norm,
                average=average,
                zero_division=0,
            )
        ),
    }


def compare_models(
    ground_truths: Sequence[object],
    baseline_predictions: Sequence[object] | None,
    finetuned_predictions: Sequence[object],
    average: str = "weighted",
) -> pd.DataFrame:
    """Create a baseline-versus-fine-tuned metrics table."""
    validate_prediction_lengths(
        ground_truths,
        baseline_predictions,
        finetuned_predictions,
    )

    metric_rows: list[dict[str, object]] = []

    if baseline_predictions is not None:
        baseline_metrics = calculate_metrics(
            ground_truths,
            baseline_predictions,
            average=average,
        )
        metric_rows.append(
            {
                "Model": "Baseline",
                "Accuracy": baseline_metrics["accuracy"],
                "Precision": baseline_metrics["precision"],
                "Recall": baseline_metrics["recall"],
                "F1": baseline_metrics["f1"],
                "Samples": baseline_metrics["num_samples"],
            }
        )

    finetuned_metrics = calculate_metrics(
        ground_truths,
        finetuned_predictions,
        average=average,
    )
    metric_rows.append(
        {
            "Model": "Fine-tuned",
            "Accuracy": finetuned_metrics["accuracy"],
            "Precision": finetuned_metrics["precision"],
            "Recall": finetuned_metrics["recall"],
            "F1": finetuned_metrics["f1"],
            "Samples": finetuned_metrics["num_samples"],
        }
    )

    return pd.DataFrame(metric_rows)


def build_results_table(
    source_df: pd.DataFrame,
    ground_truths: Sequence[object],
    finetuned_predictions: Sequence[object],
    baseline_predictions: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Create row-level prediction results and correctness indicators."""
    validate_prediction_lengths(
        ground_truths,
        baseline_predictions,
        finetuned_predictions,
    )

    if len(source_df) != len(ground_truths):
        raise ValueError(
            "Source DataFrame row count must match the number of labels."
        )

    results_df = source_df.copy().reset_index(drop=True)

    results_df[DEFAULT_GROUND_TRUTH_COLUMN] = list(ground_truths)

    if baseline_predictions is not None:
        results_df[DEFAULT_BASELINE_COLUMN] = list(baseline_predictions)
        results_df["Baseline Correct"] = [
            normalize_label(truth) == normalize_label(prediction)
            for truth, prediction in zip(
                ground_truths,
                baseline_predictions,
            )
        ]

    results_df[DEFAULT_FINETUNED_COLUMN] = list(finetuned_predictions)
    results_df["Fine-tuned Correct"] = [
        normalize_label(truth) == normalize_label(prediction)
        for truth, prediction in zip(
            ground_truths,
            finetuned_predictions,
        )
    ]

    return results_df


def get_classification_report(
    ground_truths: Sequence[object],
    predictions: Sequence[object],
    output_dict: bool = False,
) -> str | dict:
    """Return a scikit-learn classification report."""
    if len(ground_truths) != len(predictions):
        raise ValueError(
            "Prediction count must match ground-truth count."
        )

    return classification_report(
        normalize_labels(ground_truths),
        normalize_labels(predictions),
        zero_division=0,
        output_dict=output_dict,
    )


def get_confusion_matrix(
    ground_truths: Sequence[object],
    predictions: Sequence[object],
    labels: Sequence[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Build a confusion matrix using normalized labels."""
    if len(ground_truths) != len(predictions):
        raise ValueError(
            "Prediction count must match ground-truth count."
        )

    ground_truth_norm = normalize_labels(ground_truths)
    prediction_norm = normalize_labels(predictions)

    if labels is None:
        # Match the notebook: define classes from the observed ground truths.
        labels = sorted(set(ground_truth_norm))
    else:
        labels = [normalize_label(label) for label in labels]

    matrix = confusion_matrix(
        ground_truth_norm,
        prediction_norm,
        labels=list(labels),
    )

    return list(labels), matrix


def plot_confusion_matrix(
    ground_truths: Sequence[object],
    predictions: Sequence[object],
    output_path: str | Path | None = None,
    title: str = "Fine-tuned Confusion Matrix",
) -> plt.Figure:
    """Plot and optionally save the fine-tuned confusion matrix."""
    labels, matrix = get_confusion_matrix(
        ground_truths,
        predictions,
    )

    figure = plt.figure(figsize=(12, 10))
    plt.imshow(matrix)
    plt.colorbar()

    positions = np.arange(len(labels))

    plt.xticks(
        positions,
        labels,
        rotation=90,
    )
    plt.yticks(
        positions,
        labels,
    )

    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

    return figure


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Baseline vs Fine-tuned Performance",
) -> plt.Figure:
    """Plot grouped bars for Accuracy, Precision, Recall, and F1."""
    validate_columns(
        metrics_df,
        ["Model", "Accuracy", "Precision", "Recall", "F1"],
        "metrics DataFrame",
    )

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    figure = plt.figure(figsize=(8, 5))
    x_positions = np.arange(len(metric_names))

    if len(metrics_df) == 1:
        row = metrics_df.iloc[0]
        plt.bar(
            x_positions,
            [row[name] for name in metric_names],
            width=0.6,
            label=str(row["Model"]),
        )
    else:
        width = 0.35

        baseline_row = metrics_df.iloc[0]
        finetuned_row = metrics_df.iloc[1]

        plt.bar(
            x_positions - width / 2,
            [baseline_row[name] for name in metric_names],
            width,
            label=str(baseline_row["Model"]),
        )
        plt.bar(
            x_positions + width / 2,
            [finetuned_row[name] for name in metric_names],
            width,
            label=str(finetuned_row["Model"]),
        )

    plt.xticks(x_positions, metric_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

    return figure


def save_evaluation_outputs(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: str | Path,
    ground_truths: Sequence[object] | None = None,
    baseline_predictions: Sequence[object] | None = None,
    finetuned_predictions: Sequence[object] | None = None,
    save_plots: bool = True,
) -> dict[str, Path]:
    """Save evaluation tables, reports, metrics, and optional plots."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {
        "evaluation_results": (
            output_directory / "evaluation_results.csv"
        ),
        "metrics_csv": output_directory / "metrics.csv",
        "metrics_json": output_directory / "metrics.json",
    }

    results_df.to_csv(
        output_paths["evaluation_results"],
        index=False,
        encoding="utf-8",
    )

    metrics_df.to_csv(
        output_paths["metrics_csv"],
        index=False,
        encoding="utf-8",
    )

    with output_paths["metrics_json"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_df.to_dict(orient="records"),
            file,
            indent=2,
        )

    if (
        ground_truths is not None
        and finetuned_predictions is not None
    ):
        finetuned_report_path = (
            output_directory
            / "finetuned_classification_report.txt"
        )

        finetuned_report_path.write_text(
            str(
                get_classification_report(
                    ground_truths,
                    finetuned_predictions,
                )
            ),
            encoding="utf-8",
        )
        output_paths[
            "finetuned_classification_report"
        ] = finetuned_report_path

        if baseline_predictions is not None:
            baseline_report_path = (
                output_directory
                / "baseline_classification_report.txt"
            )
            baseline_report_path.write_text(
                str(
                    get_classification_report(
                        ground_truths,
                        baseline_predictions,
                    )
                ),
                encoding="utf-8",
            )
            output_paths[
                "baseline_classification_report"
            ] = baseline_report_path

        if save_plots:
            confusion_matrix_path = (
                output_directory
                / "finetuned_confusion_matrix.png"
            )
            comparison_plot_path = (
                output_directory
                / "model_comparison.png"
            )

            confusion_figure = plot_confusion_matrix(
                ground_truths,
                finetuned_predictions,
                output_path=confusion_matrix_path,
            )
            plt.close(confusion_figure)

            comparison_figure = plot_model_comparison(
                metrics_df,
                output_path=comparison_plot_path,
            )
            plt.close(comparison_figure)

            output_paths[
                "finetuned_confusion_matrix"
            ] = confusion_matrix_path
            output_paths[
                "model_comparison"
            ] = comparison_plot_path

    return output_paths


def evaluate_prediction_dataframe(
    df: pd.DataFrame,
    ground_truth_column: str = DEFAULT_GROUND_TRUTH_COLUMN,
    baseline_column: str | None = DEFAULT_BASELINE_COLUMN,
    finetuned_column: str = DEFAULT_FINETUNED_COLUMN,
    output_dir: str | Path | None = None,
    save_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a DataFrame containing labels and model predictions."""
    required_columns = [
        ground_truth_column,
        finetuned_column,
    ]

    if baseline_column is not None:
        required_columns.append(baseline_column)

    validate_columns(
        df,
        required_columns,
        "prediction DataFrame",
    )

    working_df = df.copy().reset_index(drop=True)

    ground_truths = working_df[ground_truth_column].tolist()
    finetuned_predictions = (
        working_df[finetuned_column]
        .apply(clean_prediction)
        .tolist()
    )

    baseline_predictions = None
    if baseline_column is not None:
        baseline_predictions = (
            working_df[baseline_column]
            .apply(clean_prediction)
            .tolist()
        )

    source_columns = [
        column
        for column in working_df.columns
        if column not in {
            ground_truth_column,
            baseline_column,
            finetuned_column,
            "Baseline Correct",
            "Fine-tuned Correct",
        }
    ]

    results_df = build_results_table(
        source_df=working_df[source_columns],
        ground_truths=ground_truths,
        finetuned_predictions=finetuned_predictions,
        baseline_predictions=baseline_predictions,
    )

    metrics_df = compare_models(
        ground_truths=ground_truths,
        baseline_predictions=baseline_predictions,
        finetuned_predictions=finetuned_predictions,
    )

    if output_dir is not None:
        save_evaluation_outputs(
            results_df=results_df,
            metrics_df=metrics_df,
            output_dir=output_dir,
            ground_truths=ground_truths,
            baseline_predictions=baseline_predictions,
            finetuned_predictions=finetuned_predictions,
            save_plots=save_plots,
        )

    return results_df, metrics_df


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline and fine-tuned job-skill predictions."
        )
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help=(
            "CSV/XLS file containing ground-truth and prediction columns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory for evaluation tables and plots.",
    )
    parser.add_argument(
        "--ground-truth-column",
        default=DEFAULT_GROUND_TRUTH_COLUMN,
        help="Ground-truth label column.",
    )
    parser.add_argument(
        "--baseline-column",
        default=DEFAULT_BASELINE_COLUMN,
        help=(
            "Baseline prediction column. Use 'none' when the file only "
            "contains fine-tuned predictions."
        ),
    )
    parser.add_argument(
        "--finetuned-column",
        default=DEFAULT_FINETUNED_COLUMN,
        help="Fine-tuned prediction column.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Calculate metrics without saving chart images.",
    )

    return parser.parse_args()


def main() -> None:
    """Evaluate saved predictions from the command line."""
    args = parse_args()

    predictions_df = read_table(args.predictions)

    baseline_column = args.baseline_column
    if baseline_column.lower() in {"none", "null", ""}:
        baseline_column = None

    results_df, metrics_df = evaluate_prediction_dataframe(
        df=predictions_df,
        ground_truth_column=args.ground_truth_column,
        baseline_column=baseline_column,
        finetuned_column=args.finetuned_column,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )

    print("=" * 65)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 65)
    print(metrics_df.to_string(index=False))
    print(f"\nEvaluated rows: {len(results_df):,}")
    print(f"Outputs saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

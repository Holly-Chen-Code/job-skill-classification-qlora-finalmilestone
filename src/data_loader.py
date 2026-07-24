"""Data loading and preprocessing utilities for inference."""

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["title", "description", "skill_name"]


def build_user_prompt(title: str, description: str) -> str:
    """Build the same instruction format used during model training/evaluation."""
    return (
        "Classify the following job posting into one functional skill category.\n\n"
        f"Job Title:\n{str(title).strip()}\n\n"
        f"Job Description:\n{str(description).strip()}\n\n"
        "Return only the category name."
    )


def load_test_data(file_path: str | Path) -> pd.DataFrame:
    """Load, validate, clean, and format the processed test dataset."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Test data file was not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif suffix in {".xls", ".xlsx"}:
        # Some Kaggle files have an .xls extension but contain CSV text.
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported data format '{suffix}'. Use CSV, XLS, or XLSX."
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["skill_name"]).copy()

    for column in ["title", "description"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["skill_name"] = df["skill_name"].astype(str).str.strip()
    df = df[(df["title"] != "") | (df["description"] != "")].copy()

    df["instruction"] = df.apply(
        lambda row: build_user_prompt(row["title"], row["description"]),
        axis=1,
    )

    columns = [
        column
        for column in ["job_id", "title", "description", "instruction", "skill_name"]
        if column in df.columns
    ]

    return (
        df[columns]
        .drop_duplicates(subset=["instruction", "skill_name"])
        .reset_index(drop=True)
    )

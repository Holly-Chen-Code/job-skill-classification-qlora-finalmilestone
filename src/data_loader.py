"""Data loading and preprocessing utilities for inference."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["title", "description", "skill_name"]


def build_user_prompt(title: str, description: str) -> str:
    """Create the instruction prompt used for model inference."""
    return (
        "Classify the following job posting into one functional skill category.\n\n"
        f"Job Title:\n{title.strip()}\n\n"
        f"Job Description:\n{description.strip()}\n\n"
        "Return only the category name."
    )


def read_dataset(file_path: Path) -> pd.DataFrame:
    """Read a CSV or Excel dataset."""
    suffix = file_path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)

    if suffix in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(file_path)
        except Exception:
            # Supports files with an Excel extension that contain CSV text.
            return pd.read_csv(file_path)

    raise ValueError(
        f"Unsupported file format: {suffix}. "
        "Please use CSV, TXT, XLS, or XLSX."
    )


def load_test_data(file_path: str | Path) -> pd.DataFrame:
    """Load, validate, and prepare the test dataset."""
    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Test data file was not found: {path}")

    df = read_dataset(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["skill_name"]).copy()

    for column in REQUIRED_COLUMNS:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Keep rows containing at least a title or description.
    df = df[
        (df["title"] != "")
        | (df["description"] != "")
    ].copy()

    df["instruction"] = df.apply(
        lambda row: build_user_prompt(
            row["title"],
            row["description"],
        ),
        axis=1,
    )

    output_columns = [
        column
        for column in [
            "job_id",
            "title",
            "description",
            "instruction",
            "skill_name",
        ]
        if column in df.columns
    ]

    df = df[output_columns]

    df = df.drop_duplicates(
        subset=["instruction", "skill_name"]
    )

    return df.reset_index(drop=True)

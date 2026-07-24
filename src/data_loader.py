"""Data loading and preprocessing utilities for inference."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["title", "description", "skill_name"]


def build_user_prompt(title: str, description: str) -> str:
    """Create the prompt used for model inference."""
    return (
        "Classify the following job posting into one functional skill category.\n\n"
        f"Job Title:\n{title.strip()}\n\n"
        f"Job Description:\n{description.strip()}\n\n"
        "Return only the category name."
    )


def load_test_data(file_path: str | Path) -> pd.DataFrame:
    """Load and prepare the processed test dataset."""
    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Test data file was not found: {path}")

    if path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            df = pd.read_excel(path)
        except Exception:
            # Handles files saved as CSV text with an Excel extension.
            df = pd.read_csv(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(
            "Unsupported data format. Please use CSV, TXT, XLS, or XLSX."
        )

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["skill_name"]).copy()

    for column in ["title", "description", "skill_name"]:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df = df[
        (df["title"] != "") |
        (df["description"] != "")
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

    return (
        df[output_columns]
        .drop_duplicates(
            subset=["instruction", "skill_name"]
        )
        .reset_index(drop=True)
    )

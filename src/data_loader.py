"""Data loading and preprocessing utilities for inference."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["title", "description", "skill_name"]
SUPPORTED_DATA_EXTENSIONS = {".csv", ".txt", ".xls", ".xlsx"}


def build_user_prompt(title: str, description: str) -> str:
    """Build the instruction format used during model training and evaluation."""
    return (
        "Classify the following job posting into one functional skill category.\n\n"
        f"Job Title:\n{str(title).strip()}\n\n"
        f"Job Description:\n{str(description).strip()}\n\n"
        "Return only the category name."
    )


def read_data_file(file_path: Path) -> pd.DataFrame:
    """Read a supported CSV or Excel data file."""
    suffix = file_path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)

    if suffix in {".xls", ".xlsx"}:
        # Some Kaggle files use an Excel extension but contain CSV text.
        try:
            return pd.read_excel(file_path)
        except Exception:
            return pd.read_csv(file_path)

    raise ValueError(
        f"Unsupported data format '{suffix}'. "
        "Use CSV, TXT, XLS, XLSX, or ZIP."
    )


def read_zip_file(zip_path: Path) -> pd.DataFrame:
    """Extract a ZIP archive and read the first supported dataset inside it."""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"The file is not a valid ZIP archive: {zip_path}")

    with tempfile.TemporaryDirectory() as temp_directory:
        extraction_path = Path(temp_directory)

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extraction_path)

        candidate_files = sorted(
            file
            for file in extraction_path.rglob("*")
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
            and not file.name.startswith("__MACOSX")
        )

        if not candidate_files:
            raise FileNotFoundError(
                "No CSV, TXT, XLS, or XLSX file was found inside "
                f"the ZIP archive: {zip_path}"
            )

        # Prefer a file whose name begins with "test".
        test_files = [
            file
            for file in candidate_files
            if file.stem.lower().startswith("test")
        ]

        selected_file = test_files[0] if test_files else candidate_files[0]

        print(f"Reading extracted test data from: {selected_file.name}")
        return read_data_file(selected_file)


def load_test_data(file_path: str | Path) -> pd.DataFrame:
    """Load, validate, clean, and format the processed test dataset."""
    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Test data file was not found: {path}")

    if path.suffix.lower() == ".zip":
        df = read_zip_file(path)
    else:
        df = read_data_file(path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["skill_name"]).copy()

    for column in ["title", "description"]:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["skill_name"] = (
        df["skill_name"]
        .astype(str)
        .str.strip()
    )

    # Keep records containing at least a title or description.
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

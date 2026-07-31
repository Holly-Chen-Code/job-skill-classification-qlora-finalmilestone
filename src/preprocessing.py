"""Reusable data-preprocessing utilities for the Phi-3 job-skill project.

This module extracts the reusable data-cleaning and dataset-building logic from:

- notebooks/01_data_preprocessing.ipynb
- notebooks/02_build_training_dataset.ipynb

It can be imported by the notebooks or executed directly from the project root.

Example
-------
python src/preprocessing.py \
    --postings postings.csv \
    --job-skills job_skills.csv \
    --skills skills.csv \
    --output-dir processed_data
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


POSTING_COLUMNS = [
    "job_id",
    "company_name",
    "title",
    "description",
    "location",
    "formatted_work_type",
    "formatted_experience_level",
]

TEXT_COLUMNS = [
    "company_name",
    "title",
    "description",
    "location",
    "formatted_work_type",
    "formatted_experience_level",
]

OPTIONAL_COLUMNS = [
    "company_name",
    "location",
    "formatted_work_type",
    "formatted_experience_level",
]


def read_csv_file(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and raise a clear error when it is unavailable."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file was not found: {file_path}")

    try:
        return pd.read_csv(file_path)
    except Exception as error:
        raise RuntimeError(f"Could not read CSV file: {file_path}") from error


def validate_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Confirm that a DataFrame contains all required columns."""
    required_columns = list(required_columns)
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def clean_text(value: object) -> str | float:
    """Remove HTML markup, control characters, and repeated whitespace.

    Parameters
    ----------
    value:
        Original text value.

    Returns
    -------
    str or np.nan
        Cleaned text, or ``np.nan`` when the input is missing or empty.
    """
    if pd.isna(value):
        return np.nan

    text = html.unescape(str(value))

    # Replace common structural HTML tags with spaces.
    text = re.sub(
        r"<\s*(br|p|div|li)\s*/?\s*>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove remaining HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize non-breaking spaces, line breaks, tabs, and control characters.
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else np.nan


def preprocess_postings(
    postings: pd.DataFrame,
    min_description_words: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the raw LinkedIn job-postings dataset.

    The processing steps match the original preprocessing notebook:

    1. Select the project columns.
    2. Remove rows missing title or description.
    3. Clean text fields.
    4. Remove duplicated title-description pairs.
    5. Calculate description length.
    6. Remove descriptions shorter than the selected threshold.
    7. Fill missing optional fields with ``"Not specified"``.
    8. Create a cleaning report.

    Parameters
    ----------
    postings:
        Raw postings DataFrame.
    min_description_words:
        Minimum number of words required in a job description.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The cleaned postings and a cleaning-summary report.
    """
    validate_columns(postings, POSTING_COLUMNS, "postings")

    if min_description_words < 0:
        raise ValueError("min_description_words must be zero or greater.")

    initial_rows = len(postings)
    clean_df = postings[POSTING_COLUMNS].copy()

    # Remove rows without the fields required for modeling.
    clean_df = clean_df.dropna(subset=["title", "description"]).copy()

    # Clean all text fields.
    for column in TEXT_COLUMNS:
        clean_df[column] = clean_df[column].apply(clean_text)

    # Cleaning can turn whitespace-only values into missing values.
    clean_df = clean_df.dropna(subset=["title", "description"]).copy()

    # Keep one copy of each job posting.
    clean_df = clean_df.drop_duplicates(
        subset=["title", "description"],
        keep="first",
    ).copy()

    # Add length features used for quality checks.
    clean_df["description_character_count"] = clean_df["description"].str.len()
    clean_df["description_word_count"] = (
        clean_df["description"].str.split().str.len()
    )

    # Remove very short descriptions.
    clean_df = clean_df[
        clean_df["description_word_count"] >= min_description_words
    ].copy()

    # Preserve rows with missing nonessential fields.
    for column in OPTIONAL_COLUMNS:
        clean_df[column] = clean_df[column].fillna("Not specified")

    clean_df = clean_df.reset_index(drop=True)

    final_rows = len(clean_df)
    removed_rows = initial_rows - final_rows
    retention_rate = (
        final_rows / initial_rows * 100 if initial_rows > 0 else 0.0
    )

    cleaning_report = pd.DataFrame(
        {
            "metric": [
                "original_rows",
                "final_rows",
                "removed_rows",
                "retention_rate_percentage",
                "minimum_description_words",
                "remaining_duplicate_jobs",
                "missing_titles",
                "missing_descriptions",
            ],
            "value": [
                initial_rows,
                final_rows,
                removed_rows,
                round(retention_rate, 2),
                min_description_words,
                clean_df.duplicated(
                    subset=["title", "description"]
                ).sum(),
                clean_df["title"].isna().sum(),
                clean_df["description"].isna().sum(),
            ],
        }
    )

    return clean_df, cleaning_report


def build_training_dataset(
    cleaned_postings: pd.DataFrame,
    job_skills: pd.DataFrame,
    skill_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Attach functional-skill labels to cleaned job postings.

    Multiple skills associated with the same job are deduplicated, sorted, and
    joined into one comma-separated label string, matching the original
    training-dataset notebook.
    """
    validate_columns(cleaned_postings, ["job_id"], "cleaned_postings")
    validate_columns(job_skills, ["job_id", "skill_abr"], "job_skills")
    validate_columns(
        skill_mapping,
        ["skill_abr", "skill_name"],
        "skill_mapping",
    )

    mapped_skills = job_skills.merge(
        skill_mapping[["skill_abr", "skill_name"]],
        on="skill_abr",
        how="left",
    )

    # The notebook checked missing mappings. Dropping them prevents NaN values
    # from entering sorted sets during label aggregation.
    mapped_skills = mapped_skills.dropna(subset=["skill_name"]).copy()

    grouped_skills = (
        mapped_skills.groupby("job_id")["skill_name"]
        .apply(
            lambda values: ", ".join(
                sorted(
                    {
                        str(value).strip()
                        for value in values
                        if str(value).strip()
                    }
                )
            )
        )
        .reset_index()
    )

    training_df = cleaned_postings.merge(
        grouped_skills,
        on="job_id",
        how="left",
    )

    return training_df


def split_labeled_dataset(
    training_df: pd.DataFrame,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible train, validation, and test datasets.

    The default split matches the notebook: 80% training, 10% validation,
    and 10% testing.
    """
    validate_columns(training_df, ["skill_name"], "training_df")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    labeled_df = training_df.dropna(subset=["skill_name"]).copy()
    labeled_df["skill_name"] = (
        labeled_df["skill_name"].astype(str).str.strip()
    )
    labeled_df = labeled_df[labeled_df["skill_name"] != ""].copy()

    labeled_df = labeled_df.sample(
        frac=1,
        random_state=random_seed,
    ).reset_index(drop=True)

    total_rows = len(labeled_df)
    train_end = int(total_rows * train_ratio)
    validation_end = int(
        total_rows * (train_ratio + validation_ratio)
    )

    train_df = labeled_df.iloc[:train_end].copy()
    validation_df = labeled_df.iloc[train_end:validation_end].copy()
    test_df = labeled_df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def save_preprocessing_outputs(
    cleaned_postings: pd.DataFrame,
    cleaning_report: pd.DataFrame,
    training_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save all generated preprocessing files."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "cleaned_postings": output_directory / "IE7374 clean_postings.csv",
        "cleaning_report": output_directory / "IE7374 cleaning_report.csv",
        "training_dataset": output_directory / "training_dataset.csv",
        "train": output_directory / "train.csv",
        "validation": output_directory / "validation.csv",
        "test": output_directory / "test.csv",
    }

    cleaned_postings.to_csv(
        output_paths["cleaned_postings"],
        index=False,
        encoding="utf-8",
    )
    cleaning_report.to_csv(
        output_paths["cleaning_report"],
        index=False,
        encoding="utf-8",
    )
    training_df.to_csv(
        output_paths["training_dataset"],
        index=False,
        encoding="utf-8",
    )
    train_df.to_csv(
        output_paths["train"],
        index=False,
        encoding="utf-8",
    )
    validation_df.to_csv(
        output_paths["validation"],
        index=False,
        encoding="utf-8",
    )
    test_df.to_csv(
        output_paths["test"],
        index=False,
        encoding="utf-8",
    )

    return output_paths


def run_preprocessing_pipeline(
    postings_path: str | Path,
    job_skills_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path = "processed_data",
    min_description_words: int = 50,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    random_seed: int = 42,
) -> dict[str, object]:
    """Run the complete preprocessing and dataset-building pipeline."""
    postings = read_csv_file(postings_path)
    job_skills = read_csv_file(job_skills_path)
    skill_mapping = read_csv_file(skills_path)

    cleaned_postings, cleaning_report = preprocess_postings(
        postings,
        min_description_words=min_description_words,
    )

    training_df = build_training_dataset(
        cleaned_postings,
        job_skills,
        skill_mapping,
    )

    train_df, validation_df, test_df = split_labeled_dataset(
        training_df,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )

    output_paths = save_preprocessing_outputs(
        cleaned_postings=cleaned_postings,
        cleaning_report=cleaning_report,
        training_df=training_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        output_dir=output_dir,
    )

    labeled_count = int(training_df["skill_name"].notna().sum())
    total_count = len(training_df)

    summary = {
        "original_postings": len(postings),
        "cleaned_postings": len(cleaned_postings),
        "training_dataset_rows": total_count,
        "labeled_rows": labeled_count,
        "label_coverage": (
            labeled_count / total_count if total_count > 0 else 0.0
        ),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "output_paths": output_paths,
    }

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Clean LinkedIn job postings, attach skill labels, and create "
            "train/validation/test files."
        )
    )
    parser.add_argument(
        "--postings",
        default="postings.csv",
        help="Path to postings.csv.",
    )
    parser.add_argument(
        "--job-skills",
        default="job_skills.csv",
        help="Path to job_skills.csv.",
    )
    parser.add_argument(
        "--skills",
        default="skills.csv",
        help="Path to skills.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_data",
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--min-description-words",
        type=int,
        default=50,
        help="Minimum accepted description length in words.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when shuffling labeled records.",
    )
    return parser.parse_args()


def main() -> None:
    """Run preprocessing from the command line."""
    args = parse_args()

    summary = run_preprocessing_pipeline(
        postings_path=args.postings,
        job_skills_path=args.job_skills,
        skills_path=args.skills,
        output_dir=args.output_dir,
        min_description_words=args.min_description_words,
        random_seed=args.random_seed,
    )

    print("=" * 60)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Original postings:       {summary['original_postings']:,}")
    print(f"Cleaned postings:        {summary['cleaned_postings']:,}")
    print(f"Training dataset rows:   {summary['training_dataset_rows']:,}")
    print(f"Labeled rows:            {summary['labeled_rows']:,}")
    print(f"Label coverage:          {summary['label_coverage']:.2%}")
    print(f"Train rows:              {summary['train_rows']:,}")
    print(f"Validation rows:         {summary['validation_rows']:,}")
    print(f"Test rows:               {summary['test_rows']:,}")
    print("\nGenerated files:")

    for name, path in summary["output_paths"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

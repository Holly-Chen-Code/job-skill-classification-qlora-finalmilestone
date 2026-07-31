"""QLoRA training utilities for the Phi-3 job-skill classification project.

This module reorganizes the reusable training logic from
``notebooks/03_phi3_qlora_training.ipynb``.

Important
---------
Full QLoRA fine-tuning is computationally expensive and requires a CUDA GPU.
To avoid accidentally starting a long training run, this script only validates
the data and configuration by default. Add ``--train`` to begin fine-tuning.

Examples
--------
Validate the setup without training:

    python src/training.py \
        --train-data processed_data/train.csv \
        --validation-data processed_data/validation.csv

Start training on Kaggle or another CUDA environment:

    python src/training.py \
        --train-data /kaggle/input/.../train.csv \
        --validation-data /kaggle/input/.../validation.csv \
        --output-dir /kaggle/working/phi3_skill_lora \
        --train
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


REQUIRED_COLUMNS = ["title", "description", "skill_name"]


@dataclass
class TrainingConfig:
    """Configuration matching the final QLoRA training notebook."""

    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    seed: int = 42

    max_length: int = 512
    max_description_tokens: int = 400

    train_sample_size: int | None = 1000
    validation_sample_size: int | None = 200

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optimizer: str = "paged_adamw_8bit"

    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2

    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    target_modules: tuple[str, ...] = (
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    )


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Excel data, including CSV text saved with an XLS suffix."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Training data file was not found: {file_path}")

    try:
        return pd.read_csv(file_path)
    except Exception as csv_error:
        try:
            return pd.read_excel(file_path)
        except Exception as excel_error:
            raise RuntimeError(
                f"Could not read {file_path} as CSV or Excel.\n"
                f"CSV error: {csv_error}\n"
                f"Excel error: {excel_error}"
            ) from excel_error


def validate_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate required fields and normalize text columns."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    clean_df = df.copy()

    for column in REQUIRED_COLUMNS:
        clean_df[column] = clean_df[column].fillna("").astype(str).str.strip()

    clean_df = clean_df[
        (clean_df["title"] != "")
        | (clean_df["description"] != "")
    ].copy()

    clean_df = clean_df[clean_df["skill_name"] != ""].copy()
    clean_df = clean_df.reset_index(drop=True)

    if clean_df.empty:
        raise ValueError(f"No usable rows remained in {name} after validation.")

    return clean_df


def sample_dataframe(
    df: pd.DataFrame,
    sample_size: int | None,
    seed: int,
) -> pd.DataFrame:
    """Return a reproducible sample or the complete DataFrame."""
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)

    if sample_size <= 0:
        raise ValueError("Sample size must be positive or None.")

    return df.sample(
        n=sample_size,
        random_state=seed,
    ).reset_index(drop=True)


def build_user_prompt(title: object, description: object) -> str:
    """Build the same instruction format used in training and inference."""
    return (
        "Classify the following job posting into one functional skill category.\n\n"
        f"Job Title:\n{str(title).strip()}\n\n"
        f"Job Description:\n{str(description).strip()}\n\n"
        "Return only the category name."
    )


class SkillExtractionDataset(Dataset):
    """Tokenized supervised fine-tuning dataset with prompt-label masking."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 512,
        max_description_tokens: int = 400,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_description_tokens = max_description_tokens

    def __len__(self) -> int:
        return len(self.df)

    def truncate_description(self, description: object) -> str:
        """Truncate descriptions before prompt construction."""
        description_ids = self.tokenizer(
            str(description),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_description_tokens,
        )["input_ids"]

        return self.tokenizer.decode(
            description_ids,
            skip_special_tokens=True,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]

        title = str(row["title"])
        description = self.truncate_description(row["description"])
        skill_name = str(row["skill_name"])

        user_prompt = build_user_prompt(title, description)

        prompt_messages = [
            {"role": "user", "content": user_prompt},
        ]
        full_messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": skill_name},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        prompt_length = min(
            len(prompt_encoded["input_ids"]),
            self.max_length,
        )

        # Train only on the assistant answer, not on the user prompt or padding.
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def ensure_cuda_available() -> None:
    """Require a CUDA GPU before loading the quantized model."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. Full QLoRA fine-tuning should be run "
            "in Kaggle or another GPU environment."
        )


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and Transformers random seeds."""
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_tokenizer(model_name: str):
    """Load and configure the Phi-3 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def build_quantization_config() -> BitsAndBytesConfig:
    """Create the 4-bit NF4 configuration used by QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(config: TrainingConfig) -> LoraConfig:
    """Create LoRA adapters for Phi-3 fused projection modules."""
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )


def load_qlora_model(config: TrainingConfig):
    """Load Phi-3 in 4-bit and attach trainable LoRA adapters."""
    ensure_cuda_available()

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=build_quantization_config(),
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(config))

    return model


def build_training_arguments(
    config: TrainingConfig,
    output_dir: str | Path,
) -> TrainingArguments:
    """Create version-compatible Hugging Face training arguments."""
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_total_limit": config.save_total_limit,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "optim": config.optimizer,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": config.gradient_checkpointing,
        "seed": config.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in signature:
        training_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in signature:
        training_kwargs["evaluation_strategy"] = "steps"

    if "save_strategy" in signature:
        training_kwargs["save_strategy"] = "steps"

    return TrainingArguments(**training_kwargs)


def prepare_training_data(
    train_path: str | Path,
    validation_path: str | Path,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and optionally sample training data."""
    train_df = validate_dataframe(
        read_table(train_path),
        "training dataset",
    )
    validation_df = validate_dataframe(
        read_table(validation_path),
        "validation dataset",
    )

    train_df = sample_dataframe(
        train_df,
        config.train_sample_size,
        config.seed,
    )
    validation_df = sample_dataframe(
        validation_df,
        config.validation_sample_size,
        config.seed,
    )

    return train_df, validation_df


def save_training_metadata(
    config: TrainingConfig,
    output_dir: str | Path,
    train_rows: int,
    validation_rows: int,
) -> Path:
    """Save the exact experiment configuration for reproducibility."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "training_config": asdict(config),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "pytorch_version": torch.__version__,
    }

    metadata_path = output_path / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata_path


def train_qlora(
    train_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Run the complete Phi-3 QLoRA fine-tuning workflow."""
    set_reproducible_seed(config.seed)

    train_df, validation_df = prepare_training_data(
        train_path,
        validation_path,
        config,
    )

    tokenizer = load_tokenizer(config.model_name)

    train_dataset = SkillExtractionDataset(
        train_df,
        tokenizer,
        max_length=config.max_length,
        max_description_tokens=config.max_description_tokens,
    )
    validation_dataset = SkillExtractionDataset(
        validation_df,
        tokenizer,
        max_length=config.max_length,
        max_description_tokens=config.max_description_tokens,
    )

    model = load_qlora_model(config)
    model.print_trainable_parameters()

    training_args = build_training_arguments(config, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    train_result = trainer.train()
    evaluation_result = trainer.evaluate()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    metadata_path = save_training_metadata(
        config,
        output_path,
        train_rows=len(train_df),
        validation_rows=len(validation_df),
    )

    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", evaluation_result)
    trainer.save_state()

    return {
        "output_dir": output_path,
        "metadata_path": metadata_path,
        "train_metrics": train_result.metrics,
        "evaluation_metrics": evaluation_result,
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
    }


def parse_optional_sample_size(value: str) -> int | None:
    """Parse an integer sample size or the word 'none'."""
    if value.lower() in {"none", "null", "full", "all"}:
        return None

    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            "Sample size must be positive or one of: none, full, all."
        )
    return parsed_value


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Prepare or run Phi-3 QLoRA fine-tuning."
    )

    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to train.csv or train.xls.",
    )
    parser.add_argument(
        "--validation-data",
        required=True,
        help="Path to validation.csv or validation.xls.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phi3_skill_lora",
        help="Directory for the trained LoRA adapter and training metadata.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=parse_optional_sample_size,
        default=1000,
        help="Training rows to use; enter 'none' for the full dataset.",
    )
    parser.add_argument(
        "--validation-sample-size",
        type=parse_optional_sample_size,
        default=200,
        help="Validation rows to use; enter 'none' for the full dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help=(
            "Actually start the long GPU training run. Without this flag, "
            "the script only validates data and prints the configuration."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Validate the setup or start QLoRA fine-tuning."""
    args = parse_args()

    config = TrainingConfig(
        train_sample_size=args.train_sample_size,
        validation_sample_size=args.validation_sample_size,
        num_train_epochs=args.epochs,
    )

    train_df, validation_df = prepare_training_data(
        args.train_data,
        args.validation_data,
        config,
    )

    print("=" * 65)
    print("PHI-3 QLORA TRAINING CONFIGURATION")
    print("=" * 65)
    print(f"Base model:              {config.model_name}")
    print(f"Training rows:           {len(train_df):,}")
    print(f"Validation rows:         {len(validation_df):,}")
    print(f"Maximum sequence length: {config.max_length}")
    print(f"LoRA rank:               {config.lora_rank}")
    print(f"Target modules:          {', '.join(config.target_modules)}")
    print(f"Epochs:                  {config.num_train_epochs}")
    print(f"Learning rate:           {config.learning_rate}")
    print(f"Output directory:        {args.output_dir}")

    if not args.train:
        print("\nValidation completed successfully.")
        print(
            "Training was not started. Add --train when running in a CUDA "
            "GPU environment to reproduce the fine-tuning experiment."
        )
        return

    result = train_qlora(
        train_path=args.train_data,
        validation_path=args.validation_data,
        output_dir=args.output_dir,
        config=config,
    )

    print("\nTraining completed.")
    print(f"Adapter saved to: {result['output_dir']}")
    print(f"Metadata saved to: {result['metadata_path']}")
    print(f"Train metrics: {result['train_metrics']}")
    print(f"Evaluation metrics: {result['evaluation_metrics']}")


if __name__ == "__main__":
    main()

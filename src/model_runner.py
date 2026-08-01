"""Run Phi-3 with a LoRA adapter and save prediction samples.

Run from the project root:

    python src/model_runner.py --config configs/model_config.yaml

Paths and inference settings are stored in:

    configs/model_config.yaml

A CUDA GPU is required because the model uses 4-bit quantization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_loader import load_test_data  # noqa: E402
from utils.helpers import clean_prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read optional command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phi-3 LoRA inference."
    )

    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--test-data",
        default=None,
        help="Optional test-data path overriding the YAML setting.",
    )

    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter path overriding the YAML setting.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of prediction samples.",
    )

    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    """Resolve relative paths from the project root."""
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate the nested YAML configuration."""
    path = resolve_path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    required_sections = [
        "model",
        "data",
        "generation",
        "evaluation",
        "output",
    ]

    missing_sections = [
        section
        for section in required_sections
        if not isinstance(config.get(section), dict)
    ]

    if missing_sections:
        raise ValueError(
            f"Missing required configuration sections: {missing_sections}"
        )

    required_values = {
        "model.base_model_id": config["model"].get("base_model_id"),
        "model.adapter_path": config["model"].get("adapter_path"),
        "data.test_path": config["data"].get("test_path"),
        "output.directory": config["output"].get("directory"),
    }

    missing_values = [
        key
        for key, value in required_values.items()
        if not value
    ]

    if missing_values:
        raise ValueError(
            f"Missing required configuration values: {missing_values}"
        )

    return config


def build_quantization_config() -> BitsAndBytesConfig:
    """Create the 4-bit quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(
    base_model_id: str,
    adapter_path: Path,
):
    """Load the tokenizer, base model, and LoRA adapter."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. "
            "Enable a GPU accelerator before running this script."
        )

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter was not found: {adapter_path}"
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading tokenizer from: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print(f"Loading base model: {base_model_id}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=build_quantization_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager",
    )

    print(f"Loading LoRA adapter from: {adapter_path}")

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
    )

    model.eval()

    print("Fine-tuned model loaded successfully.")

    return model, tokenizer


def predict_skills(
    model,
    tokenizer,
    instructions: list[str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
) -> list[str]:
    """Generate skill-category predictions."""
    predictions: list[str] = []

    for start_index in tqdm(
        range(0, len(instructions), batch_size),
        desc="Generating predictions",
    ):
        batch = instructions[
            start_index : start_index + batch_size
        ]

        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": instruction,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for instruction in batch
        ]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

        encoded = {
            key: value.to(model.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = encoded["input_ids"].shape[1]
        generated_tokens = generated[:, prompt_length:]

        decoded_outputs = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        predictions.extend(
            clean_prediction(output)
            for output in decoded_outputs
        )

    return predictions


def main() -> None:
    """Run the complete inference pipeline."""
    args = parse_args()
    config = load_config(args.config)

    model_config = config["model"]
    data_config = config["data"]
    generation_config = config["generation"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]

    test_data_path = resolve_path(
        args.test_data or data_config["test_path"]
    )

    adapter_path = resolve_path(
        args.adapter_path or model_config["adapter_path"]
    )

    output_path = resolve_path(
        str(
            Path(output_config["directory"])
            / "prediction_samples.csv"
        )
    )

    sample_size = (
        args.sample_size
        if args.sample_size is not None
        else int(evaluation_config.get("sample_size", 10))
    )

    random_seed = int(
        evaluation_config.get("random_seed", 42)
    )

    print(f"Loading test data from: {test_data_path}")

    test_df = load_test_data(test_data_path)

    if test_df.empty:
        raise ValueError(
            "No valid test samples remained after preprocessing."
        )

    sample_size = min(sample_size, len(test_df))

    sample_df = test_df.sample(
        n=sample_size,
        random_state=random_seed,
    ).reset_index(drop=True)

    model, tokenizer = load_model_and_tokenizer(
        base_model_id=model_config["base_model_id"],
        adapter_path=adapter_path,
    )

    predictions = predict_skills(
        model=model,
        tokenizer=tokenizer,
        instructions=sample_df["instruction"].tolist(),
        batch_size=int(
            generation_config.get("batch_size", 2)
        ),
        max_input_length=int(
            generation_config.get("max_input_length", 512)
        ),
        max_new_tokens=int(
            generation_config.get("max_new_tokens", 12)
        ),
    )

    results = sample_df.copy()
    results["prediction"] = predictions

    results["correct"] = (
        results["skill_name"]
        .str.strip()
        .str.lower()
        ==
        results["prediction"]
        .str.strip()
        .str.lower()
    )

    sample_accuracy = results["correct"].mean()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # Shorten long titles only for terminal display.
    # The CSV file still contains the complete titles.
    display_results = results.copy()

    display_results["title"] = (
        display_results["title"]
        .astype(str)
        .str.slice(0, 60)
    )

    print(
        f"\nGenerated {len(results)} representative samples."
    )

    print("\nPrediction Results")
    print("-" * 110)

    print(
        display_results[
            [
                "title",
                "skill_name",
                "prediction",
                "correct",
            ]
        ].to_string(index=False)
    )

    print("-" * 110)
    print(f"Sample Accuracy: {sample_accuracy:.2%}")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error

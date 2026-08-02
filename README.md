# Phi-3 QLoRA Job Skill Classification

## Project Overview

This project presents a reproducible inference pipeline for job skill classification using Microsoft's Phi-3 Mini 4K Instruct model fine-tuned with QLoRA (Quantized Low-Rank Adaptation).

The model predicts functional skill categories from LinkedIn job postings. The repository provides an end-to-end workflow for loading the fine-tuned model, running inference, generating predictions, and evaluating classification performance.

---

## Project Objectives

This project aims to:

- Build a reproducible inference pipeline for job skill classification.
- Load a fine-tuned Phi-3 Mini QLoRA model.
- Generate skill predictions from unseen job postings.
- Evaluate prediction performance using standard classification metrics.
- Save representative outputs for reproducibility.
---

## Repository Structure

```
.
├── configs/
│   └── model_config.yaml
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_build_training_dataset.ipynb
│   ├── 03_phi3_qlora_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── outputs/
│   ├── preprocessing_summary.png
│   ├── prediction_samples.csv
│   └── evaluation_results.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── model_runner.py
│   └── evaluation.py
│
├── utils/
│   └── helpers.py
│
├── README.md
└── requirements.txt
```

---

## Dataset

The project uses the **LinkedIn Job Postings Dataset**, including:

- postings.csv
- job_skills.csv
- skills.csv

The preprocessing pipeline:

- Removes missing records
- Cleans HTML tags and special characters
- Removes duplicate postings
- Filters short job descriptions
- Maps skill abbreviations to functional skill names
- Splits the dataset into training, validation, and testing sets

---

## Methodology

### Base Model

- Microsoft Phi-3 Mini 4K Instruct

### Fine-tuning Method

- QLoRA
- 4-bit NF4 Quantization
- PEFT (Parameter Efficient Fine Tuning)

### LoRA Configuration

- Rank (r): 16
- Alpha: 32
- Dropout: 0.05

Target modules:

- qkv_proj
- o_proj
- gate_up_proj
- down_proj

---

## Installation

```bash
git clone <repository-url>

cd project

pip install -r requirements.txt
```

---

## Data Preprocessing

Generate cleaned datasets:

```bash
python src/preprocessing.py
```

Outputs:

- cleaned_postings.csv
- training_dataset.csv
- train.csv
- validation.csv
- test.csv

---

## Model Training

The complete QLoRA training workflow is implemented in

```
src/training.py
```

Because QLoRA fine-tuning requires significant GPU resources, training is **not executed by default**.

To validate the configuration:

```bash
python src/training.py \
    --train-data train.csv \
    --validation-data validation.csv
```

To reproduce the complete fine-tuning experiment on a CUDA-enabled GPU:

```bash
python src/training.py \
    --train-data train.csv \
    --validation-data validation.csv \
    --train
```

---

## Inference

Run inference using the fine-tuned Phi-3 model:

```bash
python src/model_runner.py \
    --config configs/model_config.yaml
```

The script automatically:

- Loads the tokenizer
- Loads the Phi-3 base model
- Loads the trained LoRA adapter
- Generates predictions
- Saves prediction samples

---

## Evaluation

Evaluate generated predictions:

```bash
python src/evaluation.py \
    --predictions outputs/evaluation_results.csv
```

Metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

## Results

The fine-tuned model demonstrates improved instruction-following capability compared with the original pretrained Phi-3 model.

The repository includes:

- prediction samples
- evaluation reports
- confusion matrices
- comparison plots

---

## Example Output

Example prediction:

| Job Title | Ground Truth | Prediction |
|-----------|--------------|------------|
| Data Analyst | Data Analysis | Data Analysis |

---

## Reproducibility

To reproduce this project:

1. Run preprocessing
2. Generate train/validation/test datasets
3. Fine-tune Phi-3 with QLoRA (optional)
4. Run inference
5. Evaluate predictions

The repository contains all scripts, configuration files, and sample outputs required to reproduce the workflow.

---

## Future Work

Potential future improvements include:

- Larger instruction datasets
- Multi-label skill prediction
- Better prompt engineering
- Additional LLM backbone comparisons
- Domain-specific instruction tuning

---

## Authors

- Holly Chen
- Team Members

---

### 5. Miscellaneous
If you elect to test-run this on Kaggle.com with their GPU usage, please remeber to set the GPU to "GPU T4 *2", in our testing, this is the only GPU compitable. The other GPU "GPU P1000" results in hung-kernal and does not produce resutls.

Another poitners is to rember change the testing dataset file path, if you run into problem. Our hosted file is open to the public, however, in some enviroment, such as Kaggle's virtual one, it prevents you from loading. In that case, please download the dataset lcoally and intergret into the system, thank you.

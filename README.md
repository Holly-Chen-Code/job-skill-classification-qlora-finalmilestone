# Phi-3 QLoRA Job Skill Classification

## Project Overview

This project presents an end-to-end generative AI pipeline for job skill classification using Microsoft's Phi-3 Mini 4K Instruct model fine-tuned with QLoRA (Quantized Low-Rank Adaptation).

The repository includes the complete workflow for data preprocessing, dataset construction, QLoRA fine-tuning, inference, and evaluation. To support reproducibility, a lightweight training configuration is provided to demonstrate the complete pipeline, while the final prediction examples and evaluation results are generated using the fully trained model developed during the project.

The model predicts functional skill categories from LinkedIn job postings and demonstrates how parameter-efficient fine-tuning can be applied to a real-world job classification task.

---

## Project Objectives

This project aims to:

- Build a reproducible inference pipeline for job skill classification.
- Load a fine-tuned Phi-3 Mini QLoRA model.
- Generate skill predictions from unseen job postings.
- Evaluate prediction performance using standard classification metrics.
- Save representative outputs for reproducibility.
---

## Authors

Group 20
- Holly Chen
- Michael Dong

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

The notebooks document the model development process completed throughout the course, while the modular implementation used for the final pipeline is located in the `src/` directory.

---

## Dataset

This project uses two datasets at different stages of the workflow.

### 1. Raw Dataset (Pipeline Demonstration)

The end-to-end preprocessing pipeline starts from the original **LinkedIn Job Postings Dataset**, which includes:

- `postings.csv` – Job posting information and descriptions https://www.kaggle.com/datasets/hollychen12345/postings-csv
- `job_skills.csv` – Relationships between job postings and required skills https://www.kaggle.com/datasets/hollychen12345/job-skills-csv
- `skills.csv` – Skill definitions and metadata https://www.kaggle.com/datasets/hollychen12345/skills-csv

The preprocessing pipeline cleans the raw data, maps skill labels, removes invalid records, and generates the processed training, validation, and test datasets.

### 2. Final Inference Resources

For the final inference and evaluation, the repository uses:

- A **fully trained Phi-3 QLoRA LoRA adapter** https://www.kaggle.com/datasets/hollychen12345/phi3-skill-lora-adapter
- The **held-out test dataset (`test.xls`)** specified in `configs/model_config.yaml` https://www.kaggle.com/datasets/hollychen12345/test-xls

The lightweight training example included in this repository uses a subset of the processed training data to demonstrate and verify the complete QLoRA training pipeline. This configuration allows the entire workflow to be reproduced efficiently within the computational constraints of the course environment.

The final project model was trained using the complete processed training dataset, which requires substantially more GPU time and computational resources than the demonstration workflow. Therefore, the inference step loads the fully trained LoRA adapter together with the held-out test dataset (`test.xls`) generated during the original project. This configuration reproduces the final experimental results reported in the technical report and presentation while preserving a practical, reproducible end-to-end pipeline.

---

# Methodology

The project fine-tunes **Microsoft Phi-3 Mini 4K Instruct** for job skill classification using **QLoRA**, a parameter-efficient fine-tuning method that enables efficient training with limited GPU resources.

### Model Configuration

- **Base Model:** Microsoft Phi-3 Mini 4K Instruct
- **Fine-tuning:** QLoRA (PEFT)
- **Quantization:** 4-bit NF4
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.05

### Target Modules

- `qkv_proj`
- `o_proj`
- `gate_up_proj`
- `down_proj`

The complete training configuration is provided in `configs/model_config.yaml`, and the implementation is available in `src/training.py`.

---

# Running the Project

The recommended way to reproduce this project is to execute the provided Kaggle notebook sequentially. Each stage calls the reusable modules implemented in the `src/` directory.

---

## 1. Setup

Clone the repository, install the required dependencies, and verify that a CUDA-enabled GPU is available.

```python
%cd /kaggle/working

# Clone once; pull updates when the repository already exists.
!if [ -d "job-skill-classification-qlora-finalmilestone/.git" ]; then \
    git -C job-skill-classification-qlora-finalmilestone pull; \
else \
    git clone https://github.com/Holly-Chen-Code/job-skill-classification-qlora-finalmilestone.git; \
fi

%cd /kaggle/working/job-skill-classification-qlora-finalmilestone

!pip install -q -r requirements.txt

import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### Expected Output

```
CUDA available: True
GPU: Tesla T4
```

---

## 2. Preprocess the Raw Data

Run the preprocessing pipeline using the original **LinkedIn Job Postings Dataset**.

The preprocessing pipeline:

- Loads `postings.csv`, `job_skills.csv`, and `skills.csv`
- Cleans HTML tags and special characters
- Removes duplicate and invalid records
- Maps skill labels to standardized functional skill categories
- Generates the processed training, validation, and test datasets

```python
!python src/preprocessing.py \
    --postings ".../postings.csv" \
    --job-skills ".../job_skills.csv" \
    --skills ".../skills.csv"
    --output-dir "/kaggle/working/processed_data"
```

### Expected Output

```
PREPROCESSING PIPELINE COMPLETE

Original postings: ...
Cleaned postings: ...
Training dataset rows: ...
Train rows: ...
Validation rows: ...
Test rows: ...
```

Generated files:

```
processed_data/
├── IE7374_clean_postings.csv
├── IE7374_cleaning_report.csv
├── training_dataset.csv
├── train.csv
├── validation.csv
└── test.csv
```

---

## 3. Test the Training Pipeline

Run a lightweight QLoRA training example using a subset of the processed training data.

The demonstration uses the **same preprocessing pipeline, model architecture, QLoRA configuration, and training procedure** as the final project model. Only the training dataset size is reduced so that the complete workflow can be reproduced efficiently within the computational limits of the course environment.

For the complete experiment, simply increase the sample sizes or train on the full processed dataset.

```python
!python src/training.py \
    --train-data "/kaggle/working/processed_data/train.csv" \
    --validation-data "/kaggle/working/processed_data/validation.csv" \
    --train-sample-size 50 \
    --validation-sample-size 20 \
    --epochs 1 \
    --output-dir "/kaggle/working/phi3_skill_lora_test" \
    --train
```

### Expected Output

```
PHI-3 QLORA TRAINING CONFIGURATION

Base model:
microsoft/Phi-3-mini-4k-instruct

Training completed.

Adapter saved to:
phi3_skill_lora_test/

Training metrics:
...

Evaluation metrics:
...
```

---

## 4. Run Inference

Run inference using the configuration defined in `configs/model_config.yaml`.

The lightweight training demonstration in **Step 3** verifies that the complete QLoRA training pipeline executes successfully.

Training the final model on the complete processed training dataset requires substantially more GPU time and computational resources than the demonstration workflow. Therefore, the inference step loads the **fully trained LoRA adapter（Phi-3 QLoRA LoRA adapter）** together with the **held-out test dataset (`test.xls`)** generated during the original project, ensuring that the reported prediction examples and evaluation metrics are consistent with the final experimental results presented in the technical report and presentation.

```python
!python src/model_runner.py --config configs/model_config.yaml
```

### Expected Output

```
Generating predictions...

Generated 20 representative samples.

Prediction Results
...

Prediction samples saved to:

outputs/prediction_samples.csv
```

---

## 5. Evaluate Predictions

Evaluate the generated predictions using standard classification metrics.

The evaluation reports:

- Accuracy
- Precision
- Recall
- F1-score

```python
!python src/evaluation.py \
    --predictions outputs/prediction_samples.csv \
    --ground-truth-column skill_name \
    --finetuned-column prediction \
    --baseline-column none \
    --output-dir outputs/evaluation
```

### Expected Output

```
MODEL EVALUATION COMPLETE

Accuracy
Precision
Recall
F1-score
```

---

## 6. Review Results

Review the generated outputs.

```python
import pandas as pd
from IPython.display import Image, display

print("Prediction samples")
display(pd.read_csv("outputs/prediction_samples.csv").head())

print("Evaluation metrics")
display(pd.read_csv("outputs/evaluation/metrics.csv"))

display(Image(filename="outputs/evaluation/finetuned_confusion_matrix.png"))
```

The workflow automatically generates:

```
outputs/
├── preprocessing_summary.png
├── prediction_samples.csv
└── evaluation_results.png
```

These outputs summarize:

- Data preprocessing statistics
- Representative prediction examples
- Final evaluation metrics


---

### Notes

### GPU Requirements

This project is designed to run on a CUDA-enabled GPU.

During development and testing, the pipeline was successfully executed on a **Kaggle Tesla T4 GPU**. Training and inference on lower-memory GPUs (such as the Tesla P100 in our testing environment) may fail due to insufficient GPU memory.

If GPU memory issues occur, we recommend using a **Tesla T4 (or equivalent/higher-memory GPU)**.


### Dataset Paths

The default dataset paths in `configs/model_config.yaml` point to the public Kaggle datasets used during development.

If you download the datasets locally or use a different Kaggle environment, simply update the dataset paths in `configs/model_config.yaml` before running the notebook.

Another poitners is to rember change the testing dataset file path, if you run into problem. Our hosted file is open to the public, however, in some enviroment, such as Kaggle's virtual one, it prevents you from loading. In that case, please download the dataset lcoally and intergret into the system, thank you.


### Reproducibility

The lightweight QLoRA training example included in this repository is intended to demonstrate the complete training pipeline within the computational limits of the course environment.

The final inference and evaluation results are reproduced using the fully trained LoRA adapter developed during the original project.

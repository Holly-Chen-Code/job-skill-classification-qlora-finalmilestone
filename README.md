# Job Skill Classification using Phi-3 + QLoRA

## Project Overview

This project fine-tunes Microsoft's **Phi-3 Mini** model using **QLoRA** to classify the functional skill category of job postings.

Given a job title and description, the model predicts the corresponding **skill_name** and generates representative prediction samples.

---

## Repository Structure

```
<img width="1021" height="1090" alt="image" src="https://github.com/user-attachments/assets/ca4919a1-a52f-4270-ae9f-b5e231364407" />
```

---

## Running the inference pipeline

1. Add the following Kaggle datasets as **Input**:

   - `test.xls`
   - `phi3_skill_lora_adapter`

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run in a CUDA GPU environment (e.g. Kaggle):

```bash
python src/model_runner.py
```

The script generates **10 representative predictions** and saves them to:

```
outputs/prediction_samples.csv
```

---

## Model

- Base Model: Microsoft Phi-3 Mini 4K Instruct
- Fine-tuning Method: QLoRA (PEFT)
- Quantization: 4-bit NF4

---

## Example Output

The inference pipeline prints:

- Job title
- Ground truth skill category
- Predicted skill category
- Prediction correctness
- Sample accuracy

Example:

```
Prediction Results
----------------------------------------------------
Title                    Ground Truth     Prediction
----------------------------------------------------
Registered Nurse         Health Care      Health Care
Teacher                  Education        Other
...
----------------------------------------------------
Sample Accuracy: 40.00%
```

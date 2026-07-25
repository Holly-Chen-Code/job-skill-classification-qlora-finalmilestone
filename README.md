## Running the Inference Pipeline

### 1. Clone the repository

```bash
git clone https://github.com/Holly-Chen-Code/job-skill-classification-qlora-milestone-4.git
cd job-skill-classification-qlora-milestone-4
```

### 2. Add the following public Kaggle datasets as Input

- **test.xls**
  https://www.kaggle.com/datasets/hollychen12345/test-xls

- **phi3_skill_lora_adapter**
  https://www.kaggle.com/datasets/hollychen12345/phi3-skill-lora-adapter

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run inference

```bash
python src/model_runner.py
```

The script generates 10 representative predictions and saves them to:

```
outputs/prediction_samples.csv
```

### 5. Miscellaneous
```
If you elect to setup a Kaggle notebook to take advantage of their GPU useage, please remember to set the GPU to "GPT T4 *2", this is the only compitable GPU in our experienment. The other GPU "GPU P1000" results in a hung-kernal and does not produce results.

Another pointer is to recheck testing dataset direcotry in the model_runner; in some cases, virtual envieroment has a different path for the file.
```

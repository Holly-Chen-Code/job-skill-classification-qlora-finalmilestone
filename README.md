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
If you elect to test-run this on Kaggle.com with their GPU usage, please remeber to set the GPU to "GPU T4 *2", in our testing, this is the only GPU compitable. The other GPU "GPU P1000" results in hung-kernal and does not produce resutls.

Another poitners is to rember change the testing dataset file path, if you run into problem. Our hosted file is open to the public, however, in some enviroment, such as Kaggle's virtual one, it prevents you from loading. In that case, please download the dataset lcoally and intergret into the system, thank you.

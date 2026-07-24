# job-skill-classification-qlora-milestone-4
<img width="1021" height="1090" alt="image" src="https://github.com/user-attachments/assets/ca4919a1-a52f-4270-ae9f-b5e231364407" />




# Running the inference pipeline

1. Place the processed test dataset at `data/processed/test.csv`.
2. Place the saved LoRA adapter at `models/phi3_skill_lora_adapter_v2/`.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run in a CUDA GPU environment such as Kaggle:

```bash
python src/model_runner.py
```

The script generates 10 representative predictions and saves them to
`outputs/prediction_samples.csv`.

For Kaggle paths, either edit `configs/model_config.yaml` or run:

```bash
python src/model_runner.py \
  --test-data /kaggle/input/.../test.csv \
  --adapter-path /kaggle/input/.../phi3_skill_lora_adapter_v2
```

# Generative Representational Learning of Foundation Models for Recommendation

This is the github repository for paper *Generative Representational Learning of Foundation Models for Recommendation*.

## Requirements

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Install GradCache**
```bash
cd training/GradCache
pip install -e .
cd ../..
```

3. **Environment Setup**

This project requires modifications to specific packages for training and inference environments.

**For Training Environment:**
- Replace `transformers/models/llama/modeling_llama.py` with `env/Training Environment/transformers/modeling_llama.py`
- Replace `peft/tuners/lora/model.py` with `env/Training Environment/peft/model.py`
- Apply similar replacements for other files in the Training Environment folder

**For Inference Environment:**
- Replace files similarly using the `env/Inference Environment/` versions

## Start

Here is a simplified command to run the program. For more detailed parameter settings, please refer to the corresponding configuration file.

### Training

```bash
python scripts/run_script.py \
    --base_model_path /path/to/base/model \
    --train_data_path /path/to/train/data \
    --output_dir /path/to/output
```

### Evaluation

**For Generative Tasks:**
```bash
python scripts/run_inference_eval_gen.py \
    --base_model_path /path/to/base/model \
    --peft_path /path/to/trained/adapter \
    --test_data_path /path/to/test/data
```

**For Embedding Tasks:**
```bash
python scripts/run_inference_eval_emb.py \
    --base_model_path /path/to/base/model \
    --peft_path /path/to/trained/adapter \
    --test_data_path /path/to/test/data
```

### Model Merging

```bash
python merge/merge_moe.py
```
### Dataset

The Dataset is available in `https://huggingface.co/datasets/Anonqwq/RecFound`.

### Model

The TMoLE model checkpoint is available in `https://huggingface.co/Anonqwq/RecFound-7B`. You can load the module on Mistral-7B-v0.3-Instruct after getting ready for the environment.

## Implementation Details

- **Backbone LLM**  
  Mistral-7B-Instruct

- **Hardware**  
  8 × H100 GPUs

- **Training Configuration**
  - Epochs: **3**
  - Dataset: **RecFound**
  - Batch size:
    - Embedding tasks: **2048**
    - Generative tasks: **1024**
  - Optimizer: **AdamW**
  - Learning rate: **2e-5**

---

### Task-wise Mixture of Low-rank Experts (TMoLE)

- LoRA applied to Q/K/V/Output projections
- Rank: **16**
- Alpha: **64**
- Dropout: **0.1**
- Experts per projection: **6**
  - Embedding experts (**E**): 2
  - Generative experts (**G**): 2
  - Shared experts (**S**): 2
- Task embedding dimension: **512**

---

### Step-wise Convergence-oriented Scheduler (S2Sched)

- Warmup ratio (**Ϛ**): **10%**
- History window (**L**): **64** steps
- Temperature (**τ**): **10**
- Validation sampling:
  - **128** instances per task
  - Compute normalized validation loss every step

## Acknowledgments

Built upon:
- [GritLM](https://github.com/ContextualAI/gritlm)
- [PEFT](https://github.com/huggingface/peft)
- [Transformers](https://github.com/huggingface/transformers)

## License

MIT License

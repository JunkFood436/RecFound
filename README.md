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

The Dataset is available in `https://drive.google.com/file/d/10gg8cjqpDku8NBU9zXeS9-LvxuK3CQDP/view?usp=sharing`.

## Acknowledgments

Built upon:
- [GritLM](https://github.com/ContextualAI/gritlm)
- [PEFT](https://github.com/huggingface/peft)
- [Transformers](https://github.com/huggingface/transformers)

## License

MIT License

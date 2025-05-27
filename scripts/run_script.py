import sys
import subprocess
import argparse
import os

def parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_url', default="./ckpts/RecFound-3epoch-tau=1-mistral")
    parser.add_argument('--init_method', default=None)
    parser.add_argument('--root', default="../")
    parser.add_argument('--model', default="Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()
    return args

# Parse arguments
args = parse()

# Target py file
TARGET_PY_FILE = f'{args.root}/training/run.py'
os.system(f'cd {args.root}')

# per_device_eval_batch_size can only be set to 1
# 当前版本的代码在训练mode是embedding的时候，可能会有loss_weight时间轴没对齐的问题，可能需要修改grad_cache中传入loss_weight参数的方法

command = [
    "torchrun", 
    "--nproc_per_node", "8",
    "--master_port", "29505",
    # "-m", "training.run",
    TARGET_PY_FILE,
    "--output_dir", args.train_url,
    "--model_name_or_path", args.model,
    "--tokenizer_name", args.model,
    "--train_data", f"{args.root}/data/origin_data",
    "--eval_data", f"{args.root}/data/eval_data",
    "--learning_rate", "2e-5",
    "--lr_scheduler_type", "linear",
    "--num_train_epochs", "1",
    "--warmup_ratio", "0.03",
    "--per_device_train_batch_size", "16",
    "--gradient_accumulation_steps", "64",
    "--per_device_eval_batch_size", "1", 
    "--per_device_generative_bs", "512", 
    "--coba_batch_rate", "0.25",
    "--dataloader_drop_last",
    "--normalized",
    "--temperature", "0.02",
    "--train_group_size", "2",
    "--negatives_cross_device",
    "--query_max_len", "256",
    "--passage_max_len", "2048",
    "--mode", "unified",
    "--report_to", "tensorboard",
    "--logging_dir", args.train_url,
    "--logging_strategy", "steps",
    "--logging_steps", "1",
    "--bf16",
    "--pooling_method", "mean",
    "--use_unique_indices",
    "--loss_gen_factor", "1",
    "--loss_gen_type", "mixed",
    "--attn", "bbcc",
    "--attn_implementation", "sdpa",
    # "--no_gen_gas",
    "--save_strategy", "steps",
    "--save_steps", "30",
    "--gradient_checkpointing",
    "--repeat_times", "3",
    "--lora",
    "--ddp_find_unused_parameters", "False",
    "--weighted_loss_mode", "coba",
    "--coba_warmup_steps", "32",
    "--coba_history_length", "64",
    "--coba_tau", "1",
    "--coba_update_interval", "1",
    "--coba_sample_valid_num", "128",
]

subprocess.run(command)

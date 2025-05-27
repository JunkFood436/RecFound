#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and evaluation pipeline")
    
    parser.add_argument("--base_model_path", type=str, 
                        default="Mistral-7B-Instruct-v0.3",
                        help="Path to base model")
    parser.add_argument("--peft_path", type=str, 
                        default="ckpts/CobaData-3epoch-tau=1-mistral",
                        help="Path to PEFT model")
    parser.add_argument("--test_data_path", type=str, 
                        default="Embedding_dataset_test.jsonl",
                        help="Path to test dataset")
    parser.add_argument("--item_data_path", type=str, 
                        default="item_embedding_test.json",
                        help="Path to item embedding dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as peft_path)")
    
    parser.add_argument("--sample_rate", type=int, default=10,
                        help="Sample rate for data (take every Nth sample)")
    
    parser.add_argument("--num_neg_samples", type=int, default=19,
                        help="Number of negative samples for evaluation")
    
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference step and only run evaluation")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation step and only run inference")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()

def run_command(cmd, verbose=False):
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    
    if verbose:
        print(f"Output: {result.stdout}")
    
    return result

def main():
    args = parse_args()
    
    script_dir = Path(__file__).parent
    inference_script = script_dir / "inference_emb.py"
    eval_script = script_dir / "eval_emb.py"
    
    if not inference_script.exists():
        print(f"Error: {inference_script} not found")
        sys.exit(1)
    if not eval_script.exists():
        print(f"Error: {eval_script} not found")
        sys.exit(1)
    
    output_dir = args.output_dir if args.output_dir else args.peft_path
    
    if not args.skip_inference:
        print("=" * 50)
        print("Step 1: Running Inference...")
        print("=" * 50)
        
        inference_cmd = [
            sys.executable, str(inference_script),
            "--base_model_path", args.base_model_path,
            "--peft_path", args.peft_path,
            "--test_data_path", args.test_data_path,
            "--item_data_path", args.item_data_path,
            "--output_dir", output_dir,
            "--sample_rate", str(args.sample_rate),
            "--seed", str(args.seed)
        ]
        
        run_command(inference_cmd, args.verbose)
        print("Inference completed successfully!")
    else:
        print("Skipping inference step...")
    
    if not args.skip_evaluation:
        print("=" * 50)
        print("Step 2: Running Evaluation...")
        print("=" * 50)
        
        query_pos_path = os.path.join(output_dir, "Emb_QP.jsonl")
        neg_samples_path = os.path.join(output_dir, "Emb_Ne.json")
        
        if not os.path.exists(query_pos_path):
            print(f"Error: {query_pos_path} not found. Please run inference first.")
            sys.exit(1)
        if not os.path.exists(neg_samples_path):
            print(f"Error: {neg_samples_path} not found. Please run inference first.")
            sys.exit(1)
        
        eval_cmd = [
            sys.executable, str(eval_script),
            "--query_pos_path", query_pos_path,
            "--neg_samples_path", neg_samples_path,
            "--num_neg_samples", str(args.num_neg_samples),
            "--seed", str(args.seed)
        ]
        
        run_command(eval_cmd, args.verbose)
        print("Evaluation completed successfully!")
    else:
        print("Skipping evaluation step...")
    
    print("=" * 50)
    print("Pipeline completed successfully!")
    print(f"Results saved in: {output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run complete inference and evaluation pipeline")
    
    parser.add_argument("--base_model_path", type=str, 
                        default="Mistral-7B-Instruct-v0.3",
                        help="Path to base model")
    parser.add_argument("--peft_path", type=str, 
                        default="ckpts/RecFound-3epoch-tau=0.5-mistral",
                        help="Path to PEFT model")
    parser.add_argument("--test_data_path", type=str, 
                        default="Generative_dataset_zero_shot_test_reducued.jsonl",
                        help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as peft_path)")
    parser.add_argument("--instruction_format", type=str, default="llama", 
                        help="Instruction format for generative tasks")
    parser.add_argument("--tasks", type=str, nargs='+', default=None,
                        help="Specific tasks to evaluate (default: all tasks)")
    parser.add_argument("--label_file", type=str, default=None,
                        help="Path to label file (for ranking tasks)")
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
        return False
    
    if verbose:
        print(f"Output: {result.stdout}")
    else:
        print(result.stdout)
    
    return True

def run_inference(args, script_dir, output_dir):
    print("=" * 50)
    print("Step 1: Running Inference...")
    print("=" * 50)
    
    inference_script = script_dir / "inference.py"
    if not inference_script.exists():
        print(f"Error: {inference_script} not found")
        return False
    
    output_path = os.path.join(output_dir, "inference_result.jsonl")
    
    inference_cmd = [
        sys.executable, str(inference_script),
        "--base_model_path", args.base_model_path,
        "--peft_path", args.peft_path,
        "--test_data_path", args.test_data_path,
        "--output_path", output_path,
        "--instruction_format", args.instruction_format,
        "--seed", str(args.seed)
    ]
    
    if not run_command(inference_cmd, args.verbose):
        return False
    
    print("Inference completed successfully!")
    return True

def run_evaluation(args, script_dir, output_dir):
    print("=" * 50)
    print("Step 2: Running Evaluation...")
    print("=" * 50)
    
    eval_script = script_dir / "eval.py"
    if not eval_script.exists():
        print(f"Error: {eval_script} not found")
        return False
    
    result_file = os.path.join(output_dir, "inference_result.jsonl")
    if not os.path.exists(result_file):
        print(f"Error: {result_file} not found. Please run inference first.")
        return False
    
    default_tasks = [
        "Attribute_Value_Extraction",
        "Answerability_Prediction", 
        "Product_Matching",
        "Sequential_Recommendation",
        "Sentiment_Analysis",
        "Product_Relation_Prediction",
        "Answer_Generation",
        "Query_Rewriting",
        "item_profile",
        "user_profile",
    ]
    
    tasks = args.tasks if args.tasks else default_tasks
    
    for task in tasks:
        print("="*30)
        print(f"Evaluating Task: {task}")
        print("="*30)
        
        eval_cmd = [
            sys.executable, str(eval_script),
            "--result_file", result_file,
            "--task", task
        ]
        
        if args.label_file:
            eval_cmd.extend(["--label_file", args.label_file])
        
        if not run_command(eval_cmd, args.verbose):
            print(f"Warning: Failed to evaluate task {task}")
            continue
    
    print("Evaluation completed successfully!")
    return True

def main():
    args = parse_args()
    
    script_dir = Path(__file__).parent
    
    output_dir = args.output_dir if args.output_dir else args.peft_path
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GRITLM GENERATIVE INFERENCE AND EVALUATION PIPELINE")
    print("=" * 60)
    print(f"PEFT Path: {args.peft_path}")
    print(f"Test Data: {args.test_data_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Instruction Format: {args.instruction_format}")
    print("=" * 60)
    
    success = True
    
    if not args.skip_inference:
        success &= run_inference(args, script_dir, output_dir)
    else:
        print("Skipping inference step...")
    
    if not args.skip_evaluation and success:
        success &= run_evaluation(args, script_dir, output_dir)
    elif args.skip_evaluation:
        print("Skipping evaluation step...")
    
    if success:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Results saved in: {output_dir}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("PIPELINE FAILED!")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()

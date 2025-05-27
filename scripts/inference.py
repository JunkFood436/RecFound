import os
import time
from gritlm import GritLM
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
import json
import re
from tqdm import tqdm
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run generative inference")
    parser.add_argument("--base_model_path", type=str, 
                        default="Mistral-7B-Instruct-v0.3",
                        help="Path to base model")
    parser.add_argument("--peft_path", type=str, 
                        default="ckpts/RecFound-3epoch-tau=0.5-mistral",
                        help="Path to PEFT model")
    parser.add_argument("--test_data_path", type=str, 
                        default="Generative_dataset_zero_shot_test_reducued.jsonl",
                        help="Path to test dataset")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output file path (default: peft_path/inference_result.jsonl)")
    parser.add_argument("--instruction_format", type=str, default="llama", 
                        help="Instruction format to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    output_path = args.output_path if args.output_path else f'{args.peft_path}/inference_result.jsonl'
    
    def llama_instruction_prompt(user_input):
        return f'<|start_header_id|>user<|end_header_id|>\n{user_input}<|start_header_id|>assistant<|end_header_id|>\n'
    
    def mistral_instruction_prompt(user_input):
        return f'<s>[INST]\nPlease understand the instructions given below and give your answer. You should avoid outputting phrases like "Based on the input ..." or similar sentences.\n{user_input}[/INST]</s>\n'
    
    def grit_instruction_prompt(user_input):
        return f'<s><|user|>\nInstruction:\n{user_input}\n<|assistant|>\n'
    
    # Select instruction format
    if args.instruction_format == "llama":
        instruction_prompt = llama_instruction_prompt
        extract_start = "<|start_header_id|>assistant<|end_header_id|>\n"
    elif args.instruction_format == "mistral":
        instruction_prompt = mistral_instruction_prompt
        extract_start = " [/INST] </s>\n"
    else:
        instruction_prompt = grit_instruction_prompt
        extract_start = "<|assistant|>\n"

    model = GritLM(args.base_model_path, torch_dtype="auto", mode="unified", attn="bbcc")
    model.model = PeftModel.from_pretrained(model.model, args.peft_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.task_embedding.load_state_dict(torch.load(f'{args.peft_path}/embedding_weights.pth', map_location=device))

    def load_jsonl(filename):
        data = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                d = json.loads(line)
                data.append(d)
        return data

    def task_convert(task):
        task_dict = {"Emb_user":0, "Emb_query":1, "Emb_item":2, "Query_Rewriting":3, "Sequential_Recommendation":4, "Product_Relation_Prediction":5,
                    "Sentiment_Analysis":6, "Attribute_Value_Extraction":7, "Product_Matching":8, "user_profile":9, "item_profile":10, 
                    "Answerability_Prediction":11, "Answer_Generation":12}
        if task == None:
            return 13
        if task in task_dict:
            return task_dict[task]
        else:
            raise("Unexpected Task Type.")

    data = load_jsonl(args.test_data_path)
    results = []
        
    for d in tqdm(data, desc="Processing data"):
        max_new_token = 1 if d['task'] in ['Sequential_Recommendation', 'Product_Relation_Prediction', 'Product_Matching', 'Sentiment_Analysis', 'Answerability_Prediction'] else \
                        256 if d['task'] in ['Query_Rewriting', 'Attribute_Value_Extraction'] else \
                        2048
        
        max_length= 1024 if d['task'] in ['Query_Rewriting', 'Product_Relation_Prediction', 'Product_Matching', 'item_profile'] else \
                    2048 if d['task'] in ['Sentiment_Analysis', 'user_profile'] else \
                    4096 if d['task'] in ['Answerability_Prediction', 'Answer_Generation', 'Attribute_Value_Extraction'] else \
                    4400

        encoded = model.tokenizer(instruction_prompt(d["text"][0]), return_tensors="pt", max_length=max_length, truncation=True)
        encoded = encoded.to(model.device)
        task_type = model.task_embedding(torch.tensor(task_convert(d['task'])).reshape(-1,1))
        gen = model.generate(encoded['input_ids'], max_new_tokens=max_new_token, attention_mask=encoded['attention_mask'], pad_token_id=128001, task_types=task_type)
        decoded = model.tokenizer.batch_decode(gen)
        decoded = decoded[0]
        
        # Clean up end tokens
        end_tokens = ['<|eot_id|>', '</s>', '<|end_of_text|>']
        for token in end_tokens:
            while decoded.endswith(token):
                decoded = decoded[:-len(token)]
            
        def extract_dialogue(text):
            start_index = text.find(extract_start) + len(extract_start)
            return text[start_index:].strip()
        
        output = extract_dialogue(decoded)
        result = {"output": output, "answer": d["text"][1], "source": d["source"], "task": d["task"]}
        results.append(result)

    with open(output_path, 'w', encoding='utf-8') as file:
        for item in results:
            file.write(json.dumps(item) + '\n')

    print(f"Inference completed. Results saved to {output_path}")
    torch.cuda.empty_cache()
    del model
    return output_path

if __name__ == "__main__":
    main()

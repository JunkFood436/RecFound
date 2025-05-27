from gritlm import GritLM
from transformers import AutoTokenizer, AutoModel, set_seed
from peft import PeftModel
import json
from tqdm import tqdm
import torch
from itertools import chain
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inference embeddings using GritLM model")
    parser.add_argument("--base_model_path", type=str, default="Mistral-7B-Instruct-v0.3",
                        help="Path to base model")
    parser.add_argument("--peft_path", type=str, default="ckpts/CobaData-3epoch-tau=1-mistral",
                        help="Path to PEFT model")
    parser.add_argument("--test_data_path", type=str, default="./Embedding_dataset_test.jsonl",
                        help="Path to test dataset")
    parser.add_argument("--item_data_path", type=str, default="./item_embedding_test.json",
                        help="Path to item embedding dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as peft_path)")
    parser.add_argument("--sample_rate", type=int, default=10,
                        help="Sample rate for data (default: 10, means take every 10th sample)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    output_dir = args.output_dir if args.output_dir else args.peft_path
    
    model = GritLM(args.base_model_path, torch_dtype="auto", mode="unified", attn="bbcc")
    model.model = PeftModel.from_pretrained(model.model, args.peft_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.task_embedding.load_state_dict(torch.load(f'{args.peft_path}/embedding_weights.pth', map_location=device))

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

    def grit_instruction(instruction):
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

    def base_instruction(instruction):
        return instruction + '\n' if instruction else ""

    def load_data(file1, file2):
        data = []
        with open(file1, 'r', encoding='utf-8') as file:
            for line in file:
                d = json.loads(line)
                if d['task'] == 'Emb_item':
                    continue
                data.append(d)
        with open(file2, 'r', encoding='utf-8') as file:
            d = json.load(file)
            data = list(chain(d, data))
        return data[::args.sample_rate]

    def encode_single(text, model, instruction,task_types):
        precision = 6
        task_types = [[task_types] for _ in range(len(text))]
        res = model.encode(text, instruction=instruction, task_types=task_types)
        res = np.array(res,dtype=np.float64)
        res = np.round(res, decimals=precision)
        return res.tolist()

    data = load_data(args.test_data_path, args.item_data_path)
    query_pos_list = []
    neg_dict = {}

    queries = [d['query'][1] for d in data]
    instructions = [base_instruction(d['query'][0]) for d in data]
    poss = [d['pos'][0][1] for d in data]
    negs = [d['neg'][0][1] for d in data]
    tasks = [task_convert(d['task']) for d in data]
    categories = [d['source'] for d in data]


    with torch.no_grad():
        n_reps = []
        i = 0
        for n in tqdm(negs, total=len(negs), desc="Encoding Negative Samples"):
            n_rep = encode_single(n, model, grit_instruction(""), task_types=tasks[i])
            category = categories[i]
            if tasks[i] not in neg_dict:
                neg_dict[tasks[i]] = {}
            if category not in neg_dict[tasks[i]]:
                neg_dict[tasks[i]][category] = []
            neg_dict[tasks[i]][category].append(n_rep)
            i = i + 1
        
        
        with open(os.path.join(output_dir, "Emb_Ne.json"), 'w', encoding='utf-8') as f:
            json.dump(neg_dict, f)
        
        i = 0
        for q, inst, p in tqdm(zip(queries, instructions,poss), total=len(queries), desc="Encoding Queries"):
            q_rep = encode_single(q, model, inst, task_types=tasks[i])
            p_rep = encode_single(p, model, grit_instruction(""), task_types=tasks[i])
            query_pos_list.append({
                'query': q_rep,
                'pos': p_rep,
                'category': categories[i],
                'task': data[i]['task']
            })
            i = i + 1

        with open(os.path.join(output_dir, "Emb_QP.jsonl"), 'w', encoding='utf-8') as f:
            for item in query_pos_list:
                f.write(json.dumps(item) + '\n')
    
    print(f"Inference completed. Results saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    main()



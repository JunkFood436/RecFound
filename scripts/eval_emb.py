import json
import random
import numpy as np
from tqdm import tqdm
from transformers import set_seed
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate embedding results")
    parser.add_argument("--query_pos_path", type=str, required=True,
                        help="Path to query-positive results file")
    parser.add_argument("--neg_samples_path", type=str, required=True,
                        help="Path to negative samples file")
    parser.add_argument("--num_neg_samples", type=int, default=19,
                        help="Number of negative samples to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def task_convert(task):
    task_dict = {"Emb_user":'0', "Emb_query":'1', "Emb_item":'2'}
    if task in task_dict:
        return task_dict[task]
    else:
        raise("Unexpected Task Type.")

def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def compute_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_ndcg(similarities, labels):
    sorted_indices = np.argsort(similarities)[::-1]
    
    dcg = 0.0
    for i, idx in enumerate(sorted_indices):
        dcg += labels[idx] / np.log2(i + 2)
    
    ideal_sorted_indices = np.argsort(labels)[::-1]
    idcg = 0.0
    for i, idx in enumerate(ideal_sorted_indices):
        idcg += labels[idx] / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

def calculate_average_ndcg(query_emb, positive_embeddings, negative_embeddings):
    total_ndcg = 0

    sample_pool = np.vstack((positive_embeddings, negative_embeddings))
    labels = np.concatenate([np.ones(positive_embeddings.shape[0]), np.zeros(negative_embeddings.shape[0])])
    similarities = [compute_cosine_similarity(query_emb, sample) for sample in sample_pool]
    ndcg = compute_ndcg(similarities, labels)
    total_ndcg += ndcg
    
    average_ndcg = total_ndcg
    return average_ndcg

def evaluate_results(query_pos_results, neg_samples_by_category):
    skipped = 0
    score = {"Emb_item":[],"Emb_query":[],"Emb_user":[]}

    for item in tqdm(query_pos_results, desc="Evaluating results"):
        category = item['category']
        task = task_convert(item['task'])
        query_emb = np.array(item['query'])
        pos_emb = np.array(item['pos'])
        if len(neg_samples_by_category[task][category]) < num_neg_samples:
            skipped += 1
            continue
        negative_samples = random.sample(neg_samples_by_category[task][category], num_neg_samples)
        negative_embeddings = [np.array(neg) for neg in negative_samples]

        if task != '2':
            sample_pool = [pos_emb] + negative_embeddings
            if np.isnan(sample_pool).any() or np.isnan(query_emb).any():
                skipped += 1
                continue
            similarities = [compute_cosine_similarity(query_emb, sample) for sample in sample_pool]
            sorted_indices = np.argsort(similarities)[::-1]  
            rank_of_pos = np.where(sorted_indices == 0)[0][0] + 1
            score[item['task']].append(1.0/rank_of_pos)
        else:
            average_ndcg = calculate_average_ndcg(query_emb, pos_emb, negative_embeddings[0][:5])
            score[item['task']].append(average_ndcg)

    return score, skipped

def main():
    args = parse_args()
    set_seed(args.seed)
    
    global num_neg_samples
    num_neg_samples = args.num_neg_samples

    query_pos_results = load_jsonl(args.query_pos_path)
    neg_samples_by_category = load_json(args.neg_samples_path)
    score, skipped = evaluate_results(query_pos_results, neg_samples_by_category)

    for task in ['Emb_item','Emb_query','Emb_user']:
        if score[task]:
            print(f"Rating For task {task}: {np.mean(score[task]):.4f}")
        else:
            print(f"Rating For task {task}: No valid samples")
    print(f"Skipped: {skipped}")
    
    return score

if __name__ == "__main__":
    main()
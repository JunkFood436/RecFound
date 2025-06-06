import os
import math
import torch
from packaging import version
import importlib
import random
from collections import defaultdict

# Modified
TASK2ID = {"Emb_user":0, "Emb_query":1, "Emb_item":2, "Query_Rewriting":3, "Sequential_Recommendation":4, "Product_Relation_Prediction":5,
                    "Sentiment_Analysis":6, "Attribute_Value_Extraction":7, "Product_Matching":8, "user_profile":9, "item_profile":10, 
                    "Answerability_Prediction":11, "Answer_Generation":12}
ID2TASK = {0:"Emb_user", 1:"Emb_query", 2:"Emb_item", 3:"Query_Rewriting", 4:"Sequential_Recommendation", 5:"Product_Relation_Prediction",
                    6:"Sentiment_Analysis", 7:"Attribute_Value_Extraction", 8:"Product_Matching", 9:"user_profile", 10:"item_profile", 
                    11:"Answerability_Prediction", 12:"Answer_Generation"}
EMBTASKLIST = [0, 1, 2]
GENTASKLIST = list(range(3, 13))

# This is used for comparing with baseline
TASK_NUMS = [9095, 4090, 8837, 579, 1985, 2040, 1994, 2012, 400, 1239, 1499, 1972, 2001]
sampled_ids = set()


def is_flash_attn_2_available():

    # Let's add an extra check to see if cuda is available

    if not torch.cuda.is_available():
        return False

    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    elif torch.version.hip:
        # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.0.4")
    else:
        return False


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)


def wait_for_everyone():
    torch.distributed.barrier()


def generate_task_id(data_paths):
    data_prefixes = list(data_paths[1:-1].split(","))
    print("data paths: ")
    print(data_prefixes)

    for i, prefix in enumerate(data_prefixes):
        task_name = prefix.split("/")[-1]
        TASK2ID[task_name] = i
        ID2TASK[i] = task_name

def allocate_samples(weights, total_samples):
    """Allocate sample counts based on weights, ensuring total equals target samples."""
    fractions = [w * total_samples for w in weights]
    integer_parts = list(map(int, fractions))
    remainders = [frac - int_part for frac, int_part in zip(fractions, integer_parts)]
    current_total = sum(integer_parts)
    remaining = total_samples - current_total
    
    indexed_remainders = sorted(enumerate(remainders), key=lambda x: (-x[1], x[0]))
    for i in range(remaining):
        idx = indexed_remainders[i][0]
        integer_parts[idx] += 1
    return integer_parts

def dynamic_allocation(weights, total_samples, available_counts):
    """Dynamically allocate samples considering available sample count limitations for each task."""
    allocated = allocate_samples(weights, total_samples)
    
    adjusted = []
    total_alloc = 0
    for i in range(len(allocated)):
        alloc = allocated[i]
        avail = available_counts[i]
        if alloc > avail:
            adjusted.append(avail)
            total_alloc += avail
        else:
            adjusted.append(alloc)
            total_alloc += alloc
    
    remaining = total_samples - total_alloc
    
    if remaining > 0:
        valid_indices = [i for i in range(len(adjusted)) if adjusted[i] < available_counts[i]]
        if valid_indices:
            valid_weights = [weights[i] for i in valid_indices]
            sum_weights = sum(valid_weights)
            if sum_weights > 0:
                normalized_weights = [w/sum_weights for w in valid_weights]
                additional_alloc = allocate_samples(normalized_weights, remaining)
                
                for idx, add in zip(valid_indices, additional_alloc):
                    max_possible = available_counts[idx] - adjusted[idx]
                    actual_add = min(add, max_possible)
                    adjusted[idx] += actual_add
                    remaining -= actual_add
                    if remaining <= 0:
                        break
    
    return adjusted

def sample_batch(input_dict, per_task_weight, coba_batch_rate):
    """Sample input data based on task weights and batch rate, returning sampled batch data."""
    emb_task = input_dict.get('emb_task', [])
    gen_task = input_dict.get('gen_task', [])
    
    emb_num_samples = int(len(emb_task) * coba_batch_rate)
    gen_num_samples = int(len(gen_task) * coba_batch_rate)
    
    device = input_dict['query']['input_ids'].device if 'query' in input_dict else None
    if not device and 'generative' in input_dict:
        device = input_dict['generative']['input_ids'].device

    selected_emb_indices = []
    if emb_num_samples > 0 and emb_task:
        emb_weights = per_task_weight[:len(EMBTASKLIST)].tolist()
        sum_emb = sum(emb_weights)
        normalized_weights = [w/sum_emb for w in emb_weights]

        emb_unmarked_indices = defaultdict(list)
        available_unmarked = []
        for t in EMBTASKLIST:
            task_indices = [idx for idx, task in enumerate(emb_task) if task == t]
            unmarked = [idx for idx in task_indices if input_dict['id'][idx] not in sampled_ids]
            emb_unmarked_indices[t] = unmarked
            available_unmarked.append(len(unmarked))
        
        alloc_unmarked = dynamic_allocation(normalized_weights, emb_num_samples, available_unmarked)
        sum_unmarked = sum(alloc_unmarked)
        remaining_emb = emb_num_samples - sum_unmarked

        alloc_marked = [0] * len(EMBTASKLIST)
        if remaining_emb > 0:
            emb_marked_indices = defaultdict(list)
            available_marked = []
            for t in EMBTASKLIST:
                task_indices = [idx for idx, task in enumerate(emb_task) if task == t]
                marked = [idx for idx in task_indices if input_dict['id'][idx] in sampled_ids]
                emb_marked_indices[t] = marked
                available_marked.append(len(marked))
            
            alloc_marked = dynamic_allocation(normalized_weights, remaining_emb, available_marked)
            
        allocated = [alloc_unmarked[i] + alloc_marked[i] for i in range(len(alloc_unmarked))]
        print(f'Emb Task Samples:{allocated}')

        for i, task in enumerate(EMBTASKLIST):
            needed_unmarked = alloc_unmarked[i]
            available_unmarked = emb_unmarked_indices[task]
            if needed_unmarked > 0:
                selected = random.sample(available_unmarked, needed_unmarked)
                for idx in selected:
                    sampled_ids.add(input_dict['id'][idx])
                selected_emb_indices.extend(selected)

            if remaining_emb > 0:
                needed_marked = alloc_marked[i]
                available_marked = emb_marked_indices.get(task, [])
                if needed_marked > 0 and available_marked:
                    selected = random.sample(available_marked, needed_marked)
                    selected_emb_indices.extend(selected)
        
        random.shuffle(selected_emb_indices)

    selected_gen_indices = []
    if gen_num_samples > 0 and gen_task:
        gen_weights = per_task_weight[len(EMBTASKLIST):].tolist()
        sum_gen = sum(gen_weights)
        normalized_weights_gen = [w/sum_gen for w in gen_weights]

        gen_unmarked_indices = defaultdict(list)
        available_unmarked_gen = []
        for t in GENTASKLIST:
            task_indices = [idx for idx, task in enumerate(gen_task) if task == t]
            unmarked = [idx for idx in task_indices if input_dict['id'][idx] not in sampled_ids]
            gen_unmarked_indices[t] = unmarked
            available_unmarked_gen.append(len(unmarked))
        
        alloc_unmarked_gen = dynamic_allocation(normalized_weights_gen, gen_num_samples, available_unmarked_gen)
        sum_unmarked_gen = sum(alloc_unmarked_gen)
        remaining_gen = gen_num_samples - sum_unmarked_gen

        alloc_marked_gen = [0] * len(GENTASKLIST)
        if remaining_gen > 0:
            gen_marked_indices = defaultdict(list)
            available_marked_gen = []
            for t in GENTASKLIST:
                task_indices = [idx for idx, task in enumerate(gen_task) if task == t]
                marked = [idx for idx in task_indices if input_dict['id'][idx] in sampled_ids]
                gen_marked_indices[t] = marked
                available_marked_gen.append(len(marked))
            
            alloc_marked_gen = dynamic_allocation(normalized_weights_gen, remaining_gen, available_marked_gen)

        allocated_gen = [alloc_unmarked_gen[i] + alloc_marked_gen[i] for i in range(len(alloc_unmarked_gen))]
        print(f'Gen Task Samples:{allocated_gen}')

        for i, task in enumerate(GENTASKLIST):
            needed_unmarked = alloc_unmarked_gen[i]
            available_unmarked = gen_unmarked_indices[task]
            if needed_unmarked > 0:
                selected = random.sample(available_unmarked, needed_unmarked)
                for idx in selected:
                    sampled_ids.add(input_dict['id'][idx])
                selected_gen_indices.extend(selected)
            
            if remaining_gen > 0:
                needed_marked = alloc_marked_gen[i]
                available_marked = gen_marked_indices.get(task, [])
                if needed_marked > 0 and available_marked:
                    selected = random.sample(available_marked, needed_marked)
                    selected_gen_indices.extend(selected)
        
        random.shuffle(selected_gen_indices)

    new_input = {}
    if selected_emb_indices:
        emb_idx_tensor = torch.tensor(selected_emb_indices, device=device)
        new_input['query'] = {
            'input_ids': input_dict['query']['input_ids'].index_select(0, emb_idx_tensor),
            'attention_mask': input_dict['query']['attention_mask'].index_select(0, emb_idx_tensor),
            'instruction_lens': input_dict['query']['instruction_lens'].index_select(0, emb_idx_tensor),
        }
        passage_indices = []
        for i in selected_emb_indices:
            passage_indices.extend([2*i, 2*i+1])
        passage_idx_tensor = torch.tensor(passage_indices, device=device)
        new_input['passage'] = {
            'input_ids': input_dict['passage']['input_ids'].index_select(0, passage_idx_tensor),
            'attention_mask': input_dict['passage']['attention_mask'].index_select(0, passage_idx_tensor)
        }
        new_input['emb_task'] = [emb_task[i] for i in selected_emb_indices]

    if selected_gen_indices:
        gen_idx_tensor = torch.tensor(selected_gen_indices, device=device)
        new_input['generative'] = {
            'input_ids': input_dict['generative']['input_ids'].index_select(0, gen_idx_tensor),
            'attention_mask': input_dict['generative']['attention_mask'].index_select(0, gen_idx_tensor),
            'labels': input_dict['generative']['labels'].index_select(0, gen_idx_tensor),
        }
        new_input['gen_task'] = [gen_task[i] for i in selected_gen_indices]

    del input_dict, selected_emb_indices, selected_gen_indices
    torch.cuda.empty_cache()
    
    return new_input

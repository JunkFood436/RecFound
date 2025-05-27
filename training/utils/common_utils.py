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


def _goes_first(is_main):
    if is_main is False:
        wait_for_everyone()
    yield
    if is_main is True:
        wait_for_everyone()


def get_model_params_num(model):
    """
    Get params number of the model
    Args:
        model: model(required)
    Returns:
        the number of parameters of model
    """
    num = 0
    for _, param in model.named_parameters():
        num += param.nelement()
    return num


def unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    try:
        return type(obj)(generator)
    except TypeError:
        # Some objects may not be able to instantiate from a generator directly
        return type(obj)(*list(generator))


def get_computation_speed(batch_size_per_device, seq_len, step_time):

    return batch_size_per_device * seq_len / (step_time + 1e-12)


def human_readable_flops(num):
    for unit in [
        "",
        "KFLOPS",
        "MFLOPS",
        "GFLOPS",
        "TFLOPS",
        "PFLOPS",
        "EFLOPS",
        "ZFLOPS",
    ]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")


def get_tflops_new(args, batch_size, seq_len, step_time):
    sl = seq_len
    L = args.num_hidden_layers
    h = args.hidden_size
    V = args.vocab_size
    flops = 96 * batch_size * sl * L * h * h * (1 + sl / (6 * h) + V / (16 * L * h)) / step_time
    return human_readable_flops(flops)


def get_tflops_megatron(total_model_param, hidden_size, num_hidden_layers, batch_size_per_device, seq_len, step_time):

    ff = total_model_param * 6
    attn = seq_len * hidden_size * num_hidden_layers * 60
    flops = batch_size_per_device * seq_len * (ff + attn) / step_time
    return human_readable_flops(flops)


def generate_task_id(data_paths):
    data_prefixes = list(data_paths[1:-1].split(","))
    print("data paths: ")
    print(data_prefixes)

    for i, prefix in enumerate(data_prefixes):
        task_name = prefix.split("/")[-1]
        TASK2ID[task_name] = i
        ID2TASK[i] = task_name

def allocate_samples(weights, total_samples):
    # 保持原有实现不变
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
    """动态调整分配逻辑"""
    # 初始分配
    allocated = allocate_samples(weights, total_samples)
    
    # 第一轮调整（处理样本不足的任务）
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
    
    # 剩余名额再分配
    if remaining > 0:
        valid_indices = [i for i in range(len(adjusted)) if adjusted[i] < available_counts[i]]
        if valid_indices:
            # 重新计算有效权重
            valid_weights = [weights[i] for i in valid_indices]
            sum_weights = sum(valid_weights)
            if sum_weights > 0:
                normalized_weights = [w/sum_weights for w in valid_weights]
                additional_alloc = allocate_samples(normalized_weights, remaining)
                
                # 应用补充分配
                for idx, add in zip(valid_indices, additional_alloc):
                    max_possible = available_counts[idx] - adjusted[idx]
                    actual_add = min(add, max_possible)
                    adjusted[idx] += actual_add
                    remaining -= actual_add
                    if remaining <= 0:
                        break
    
    return adjusted

def sample_batch(input_dict, per_task_weight, coba_batch_rate):
    emb_task = input_dict.get('emb_task', [])
    gen_task = input_dict.get('gen_task', [])
    
    emb_num_samples = int(len(emb_task) * coba_batch_rate)
    gen_num_samples = int(len(gen_task) * coba_batch_rate)
    
    device = input_dict['query']['input_ids'].device if 'query' in input_dict else None
    if not device and 'generative' in input_dict:
        device = input_dict['generative']['input_ids'].device

    selected_emb_indices = []
    if emb_num_samples > 0 and emb_task:
        # 分两阶段分配：未采样样本优先
        emb_weights = per_task_weight[:len(EMBTASKLIST)].tolist()
        sum_emb = sum(emb_weights)
        normalized_weights = [w/sum_emb for w in emb_weights]

        # 阶段1：从未采样的样本中分配
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

        # 阶段2：从已采样的样本中分配（如果有剩余）
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

        # 合并分配结果
        allocated = [alloc_unmarked[i] + alloc_marked[i] for i in range(len(alloc_unmarked))]
        print(f'Emb Task Samples:{allocated}')

        # 收集样本索引
        for i, task in enumerate(EMBTASKLIST):
            # 处理未采样的样本
            needed_unmarked = alloc_unmarked[i]
            available_unmarked = emb_unmarked_indices[task]
            if needed_unmarked > 0:
                selected = random.sample(available_unmarked, needed_unmarked)
                # 记录采样ID
                for idx in selected:
                    sampled_ids.add(input_dict['id'][idx])
                selected_emb_indices.extend(selected)
            
            # 处理已采样的样本
            if remaining_emb > 0:
                needed_marked = alloc_marked[i]
                available_marked = emb_marked_indices.get(task, [])
                if needed_marked > 0 and available_marked:
                    selected = random.sample(available_marked, needed_marked)
                    selected_emb_indices.extend(selected)
        
        random.shuffle(selected_emb_indices)

    selected_gen_indices = []
    if gen_num_samples > 0 and gen_task:
        # 类似分阶段处理Generative任务
        gen_weights = per_task_weight[len(EMBTASKLIST):].tolist()
        sum_gen = sum(gen_weights)
        normalized_weights_gen = [w/sum_gen for w in gen_weights]

        # 阶段1：从未采样的样本中分配
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

        # 阶段2：从已采样的样本中分配
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

        # 收集样本索引
        for i, task in enumerate(GENTASKLIST):
            # 处理未采样的样本
            needed_unmarked = alloc_unmarked_gen[i]
            available_unmarked = gen_unmarked_indices[task]
            if needed_unmarked > 0:
                selected = random.sample(available_unmarked, needed_unmarked)
                for idx in selected:
                    sampled_ids.add(input_dict['id'][idx])
                selected_gen_indices.extend(selected)
            
            # 处理已采样的样本
            if remaining_gen > 0:
                needed_marked = alloc_marked_gen[i]
                available_marked = gen_marked_indices.get(task, [])
                if needed_marked > 0 and available_marked:
                    selected = random.sample(available_marked, needed_marked)
                    selected_gen_indices.extend(selected)
        
        random.shuffle(selected_gen_indices)

    new_input = {}
    # 处理Embedding相关数据
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

    # 处理Generation相关数据
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

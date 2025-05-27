# This is the python script for merging models into a moe-structure model. (Not changing the original model structure)
import os
from safetensors.torch import load_file, save_file
import torch
import shutil
import numpy as np
from typing import Union
import sparsify


# Args 




# Please set the first model as the initialized LoRA model when using the TIES method
models_path = [    
]

# Sparsify method: 'magnitude', 'random', 'magnitude_outliers', 'rank_magnitude_sampling', 'consensus_ties', 'consensus_ta'
# Set to '' if not using sparsification
sparsify_method = 'magnitude'

merge_method = 'ties'

t = 0.6 # Only used when merge_method is 'slerp'

density = 0.8 # Only used when setting sparsify_method

output_path = f'results/{merge_method}/final(10,20),sp0.8'
os.makedirs(output_path, exist_ok=True)




def copy(src,dst):
    shutil.copy(f'{src}/adapter_config.json', dst)
    shutil.copy(f'{src}/special_tokens_map.json', dst)
    shutil.copy(f'{src}/tokenizer_config.json', dst)
    shutil.copy(f'{src}/tokenizer.json', dst)
    
def lerp(
    t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return (1 - t) * v0 + t * v1
    
def slerp(
    t: Union[float, np.ndarray],
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles
    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return maybe_torch(res, is_torch)

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return maybe_torch(res, is_torch)

def maybe_torch(v: np.ndarray, is_torch: bool):
    if is_torch:
        return torch.from_numpy(v)
    return v

def normalize(v: np.ndarray, eps: float):
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v
    
def main():
    if merge_method == 'linear':
        # Average weight for each model
        n = len(models_path)
        weight = torch.full((n,), 1/n, dtype=torch.bfloat16)
        # weight = torch.tensor([0.3,0.3,0.4], dtype=torch.bfloat16)
        final_data = None
        final_embed = None
        
        for i in range(n):
            data = load_file(f'{models_path[i]}/adapter_model.safetensors')
            embed = torch.load(f'{models_path[i]}/embedding_weights.pth', map_location='cuda:0')
            
            # 对当前模型进行sparsify
            if sparsify_method:
                for key in data.keys():
                    data[key] = sparsify.sparsify(data[key], density, sparsify_method)
                embed['weight'] = sparsify.sparsify(embed['weight'], density, sparsify_method)
            
            # 加权累加
            if i == 0:  # 第一个模型初始化
                final_data = {k: v * weight[i] for k, v in data.items()}
                final_embed = {'weight': embed['weight'] * weight[i]}
            else:       # 后续模型累加
                for key in data.keys():
                    final_data[key] += data[key] * weight[i]
                final_embed['weight'] += embed['weight'] * weight[i]
            
        copy(models_path[0],output_path)
        save_file(final_data, f'{output_path}/adapter_model.safetensors')
        torch.save(final_embed, f'{output_path}/embedding_weights.pth')

    elif merge_method == 'slerp':
        # Only support merging two models
        assert len(models_path) == 2
        state_dict1 = load_file(f'{models_path[0]}/adapter_model.safetensors')
        state_dict2 = load_file(f'{models_path[1]}/adapter_model.safetensors')
        interpolated_state_dict = {}

        for key in state_dict1:
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]

            assert (tensor1.shape == tensor2.shape)
            interpolated_tensor = slerp(t, tensor1, tensor2)
            interpolated_state_dict[key] = interpolated_tensor
        
        # Embedding weights
        embed1 = torch.load(f'{models_path[0]}/embedding_weights.pth', map_location='cuda:0')
        embed2 = torch.load(f'{models_path[1]}/embedding_weights.pth', map_location='cuda:0')
        embed1['weight'] = slerp(t, embed1['weight'], embed2['weight'])
        
        copy(models_path[0],output_path)
        save_file(interpolated_state_dict, f'{output_path}/adapter_model.safetensors')
        torch.save(embed1, f'{output_path}/embedding_weights.pth')
        
    elif merge_method == 'ties':
        # The more models there are, the better the result will be
        base_model = load_file(f'{models_path[0]}/adapter_model.safetensors')
        vector_model = {}
        final_model = {}
        for key, tensor in base_model.items():
            vector_model[key] = torch.zeros_like(tensor) # All zero initialization
            final_model[key] = torch.zeros_like(tensor)
        
        for i in range(1,len(models_path)):
            model = load_file(f'{models_path[i]}/adapter_model.safetensors')
            for key, tensor in model.items():
                vector_model[key] += tensor - base_model[key]
        
        sign_mask = {}
        count_mask = {}
        for key, tensor in vector_model.items():
            sign_mask[key] = torch.sign(tensor)
            count_mask[key] = torch.zeros_like(tensor)
        
        for i in range(1,len(models_path)):
            model = load_file(f'{models_path[i]}/adapter_model.safetensors')
            for key, tensor in model.items():
                tensor = tensor - base_model[key]
                mask = (torch.sign(tensor) != sign_mask[key])
                count_mask[key] += torch.ones_like(mask).int() - mask.int()
                tensor[mask] = 0
                if sparsify_method:
                    tensor = sparsify.sparsify(tensor, density, sparsify_method)
                final_model[key] += tensor
        
        for key, tensor in final_model.items():
            final_model[key] = tensor / (count_mask[key] + 1e-9) + base_model[key]
            
        # Embedding weights
        base_embed = torch.load(f'{models_path[0]}/embedding_weights.pth', map_location='cuda:0')
        vector_embed = torch.zeros_like(base_embed['weight'])
        final_embed = torch.zeros_like(base_embed['weight'])
        
        for i in range(1,len(models_path)):
            embed = torch.load(f'{models_path[i]}/embedding_weights.pth', map_location='cuda:0')
            vector_embed += embed['weight'] - base_embed['weight']
        
        sign_mask = torch.sign(vector_embed)
        count_mask = torch.zeros_like(sign_mask)
        
        for i in range(1,len(models_path)):
            embed = torch.load(f'{models_path[i]}/embedding_weights.pth', map_location='cuda:0')
            embed['weight'] = embed['weight'] - base_embed['weight']
            mask = (torch.sign(embed['weight']) != sign_mask)
            count_mask += torch.ones_like(mask).int() - mask.int()
            embed['weight'][mask] = 0
            if sparsify_method:
                embed['weight'] = sparsify.sparsify(embed['weight'], density, sparsify_method)
            final_embed += embed['weight']
            
        base_embed['weight'] = final_embed / (count_mask + 1e-9) + base_embed['weight']
        
        copy(models_path[1],output_path)
        save_file(final_model, f'{output_path}/adapter_model.safetensors')
        torch.save(base_embed, f'{output_path}/embedding_weights.pth')
    
if __name__ == '__main__':
    main()
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    



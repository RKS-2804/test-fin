import os
import shutil
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict

# Merge method enum
class MergeMethod:
    WEIGHTED_AVERAGE = "weighted_average"
    TASK_ARITHMETIC = "task_arithmetic"
    TIES = "ties"
    DARE = "dare"

def lora_merge(weights, lora_name_list, output_name, gpu_id, directly_load_safetensors=0,
               merge_method=MergeMethod.WEIGHTED_AVERAGE, merge_params=None):
    """
    Merge multiple LoRA adapters with specified weights using the selected merge method.
    
    Args:
        weights (list): List of weights for each adapter.
        lora_name_list (list): List of paths to LoRA adapters.
        output_name (str): Path to save the merged adapter.
        gpu_id (int): GPU ID to use for merging.
        directly_load_safetensors (int): Whether to directly load safetensors (faster).
        merge_method (str): Method to use for merging (from MergeMethod enum).
        merge_params (dict, optional): Additional parameters for the merge method.
    
    Returns:
        dict: The merged state dict if directly_load_safetensors is True.
    """
    # Default merge parameters if none provided
    if merge_params is None:
        merge_params = {}
    
    # Load state dicts based on the loading method
    if not directly_load_safetensors:
        # Load full models
        models = []
        state_dicts = []
        for lora_name in lora_name_list:
            model = AutoModelForCausalLM.from_pretrained(lora_name).to(f"cuda:{gpu_id}")
            models.append(model)
            state_dicts.append(get_peft_model_state_dict(model))
        
        # Apply the selected merge method
        if merge_method == MergeMethod.WEIGHTED_AVERAGE:
            final_state_dict = weighted_average_merge(state_dicts, weights)
        elif merge_method == MergeMethod.TASK_ARITHMETIC:
            final_state_dict = task_arithmetic_merge(state_dicts, weights, **merge_params)
        elif merge_method == MergeMethod.TIES:
            final_state_dict = ties_merge(state_dicts, weights, **merge_params)
        elif merge_method == MergeMethod.DARE:
            final_state_dict = dare_merge(state_dicts, weights, **merge_params)
        else:
            raise ValueError(f"Unknown merge method: {merge_method}")
        
        # Save the merged model
        set_peft_model_state_dict(models[0], final_state_dict)
        if os.path.exists(output_name):
            shutil.rmtree(output_name)
        models[0].save_pretrained(output_name)
        
        # Clean up
        for model in models[1:]:
            del model
    else:
        # Load only state dicts (faster)
        state_dicts = []
        for lora_name in lora_name_list:
            state_dict = load_file(os.path.join(lora_name, "adapter_model.safetensors"), device="cpu")
            state_dicts.append(state_dict)
        
        # Apply the selected merge method
        if merge_method == MergeMethod.WEIGHTED_AVERAGE:
            final_state_dict = weighted_average_merge(state_dicts, weights)
        elif merge_method == MergeMethod.TASK_ARITHMETIC:
            final_state_dict = task_arithmetic_merge(state_dicts, weights, **merge_params)
        elif merge_method == MergeMethod.TIES:
            final_state_dict = ties_merge(state_dicts, weights, **merge_params)
        elif merge_method == MergeMethod.DARE:
            final_state_dict = dare_merge(state_dicts, weights, **merge_params)
        else:
            raise ValueError(f"Unknown merge method: {merge_method}")
        
        # Save the merged state dict
        if not os.path.exists(output_name):
            os.mkdir(output_name)
        save_file(final_state_dict, os.path.join(output_name, "adapter_model.safetensors"))
        
        return final_state_dict

def weighted_average_merge(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """
    Merge state dictionaries using weighted average.
    
    Args:
        state_dicts: List of state dictionaries to merge
        weights: List of weights for each state dictionary
    
    Returns:
        Merged state dictionary
    """
    final_state_dict = {}
    for i, state_dict in enumerate(state_dicts):
        if i == 0:
            for key in state_dict.keys():
                final_state_dict[key] = weights[i] * state_dict[key]
        else:
            for key in state_dict.keys():
                assert key in final_state_dict.keys(), f"Key {key} not found in final state dict"
                final_state_dict[key] += weights[i] * state_dict[key]
    
    return final_state_dict

def task_arithmetic_merge(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float],
                          base_model_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    Merge state dictionaries using task arithmetic.
    
    Args:
        state_dicts: List of state dictionaries to merge
        weights: List of weights for each state dictionary
        base_model_idx: Index of the base model in state_dicts
    
    Returns:
        Merged state dictionary
    """
    # Extract the base model state dict
    base_state_dict = state_dicts[base_model_idx]
    
    # Build task vectors (differences between each model and the base model)
    task_vectors = []
    for i, state_dict in enumerate(state_dicts):
        if i == base_model_idx:
            continue
        
        task_vector = {}
        for key in base_state_dict.keys():
            if key in state_dict:
                task_vector[key] = state_dict[key] - base_state_dict[key]
        
        task_vectors.append((task_vector, weights[i]))
    
    # Apply weighted task vectors to the base model
    final_state_dict = {k: v.clone() for k, v in base_state_dict.items()}
    
    for task_vector, weight in task_vectors:
        for key in final_state_dict.keys():
            if key in task_vector:
                final_state_dict[key] += weight * task_vector[key]
    
    return final_state_dict

def ties_merge(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float],
               threshold: float = 0.01) -> Dict[str, torch.Tensor]:
    """
    Merge state dictionaries using TIES merging method.
    
    Args:
        state_dicts: List of state dictionaries to merge
        weights: List of weights for each state dictionary
        threshold: Threshold for minimal changes
    
    Returns:
        Merged state dictionary
    """
    # Step 1: Reset minimal changes
    processed_dicts = []
    for state_dict in state_dicts:
        processed_dict = {k: v.clone() for k, v in state_dict.items()}
        for key, param in processed_dict.items():
            minimal_change_mask = torch.abs(param) < threshold
            param[minimal_change_mask] = 0.0
        processed_dicts.append(processed_dict)
    
    # Step 2: Resolve sign conflicts
    final_state_dict = {}
    for key in processed_dicts[0].keys():
        # Stack parameters from all models for this key
        stacked_params = torch.stack([state_dict[key] for state_dict in processed_dicts])
        
        # Calculate sign consensus
        sign_consensus = torch.sign(torch.mean(stacked_params, dim=0))
        
        # Apply sign consensus to absolute values
        resolved_params = torch.mean(torch.abs(stacked_params), dim=0) * sign_consensus
        
        final_state_dict[key] = resolved_params
    
    # Step 3: Merge aligned parameters
    for key in final_state_dict.keys():
        stacked_params = torch.stack([state_dict[key] for state_dict in processed_dicts])
        alignment_mask = torch.prod(torch.sign(stacked_params), dim=0) > 0
        final_state_dict[key] = final_state_dict[key] * alignment_mask.float()
    
    return final_state_dict

def dare_merge(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float],
               threshold: float = 0.01, amplification_factor: float = 2.0,
               base_model_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    Merge state dictionaries using DARE merging method.
    
    Args:
        state_dicts: List of state dictionaries to merge
        weights: List of weights for each state dictionary
        threshold: Threshold for small differences
        amplification_factor: Amplification factor for large differences
        base_model_idx: Index of the base model in state_dicts
    
    Returns:
        Merged state dictionary
    """
    # Extract the base model state dict
    base_state_dict = state_dicts[base_model_idx]
    
    # Create a copy of the base state dict for the result
    final_state_dict = {k: v.clone() for k, v in base_state_dict.items()}
    
    # Process each fine-tuned model
    for i, state_dict in enumerate(state_dicts):
        if i == base_model_idx:
            continue
        
        # Apply DARE method
        for key in final_state_dict.keys():
            if key in state_dict:
                # Calculate difference between fine-tuned and base model
                difference = state_dict[key] - base_state_dict[key]
                
                # Apply threshold and amplification
                small_differences_mask = torch.abs(difference) < threshold
                difference[small_differences_mask] = 0.0
                difference[~small_differences_mask] *= amplification_factor * weights[i]
                
                # Update the final state dict
                final_state_dict[key] += difference
    
    return final_state_dict
# Example usage:
# Basic weighted average merge (original method)
# lora_merge([0.3, 0.6, 0.8], ["./initial_experts/lima", "./initial_experts/cot", "./initial_experts/science"],
#           "./new", 0, directly_load_safetensors=1)

# Task arithmetic merge
# lora_merge([0.3, 0.6, 0.8], ["./initial_experts/lima", "./initial_experts/cot", "./initial_experts/science"],
#           "./new", 0, directly_load_safetensors=1, merge_method=MergeMethod.TASK_ARITHMETIC,
#           merge_params={"base_model_idx": 0})

# TIES merge
# lora_merge([0.3, 0.6, 0.8], ["./initial_experts/lima", "./initial_experts/cot", "./initial_experts/science"],
#           "./new", 0, directly_load_safetensors=1, merge_method=MergeMethod.TIES,
#           merge_params={"threshold": 0.01})

# DARE merge
# lora_merge([0.3, 0.6, 0.8], ["./initial_experts/lima", "./initial_experts/cot", "./initial_experts/science"],
#           "./new", 0, directly_load_safetensors=1, merge_method=MergeMethod.DARE,
#           merge_params={"threshold": 0.01, "amplification_factor": 2.0, "base_model_idx": 0})
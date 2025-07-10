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
        
        # Copy adapter_config.json from the first source model to ensure model_type is preserved
        source_config_path = os.path.join(lora_name_list[0], "adapter_config.json")
        if os.path.exists(source_config_path):
            shutil.copy(source_config_path, os.path.join(output_name, "adapter_config.json"))
        
        return final_state_dict

def weighted_average_merge(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """
    Merge state dictionaries using weighted average, handling tensor size mismatches.
    
    Args:
        state_dicts: List of state dictionaries to merge
        weights: List of weights for each state dictionary
    
    Returns:
        Merged state dictionary
    """
    final_state_dict = {}
    incompatible_keys = set()
    reference_shapes = {}
    
    # First pass: Initialize final_state_dict with the first state_dict and record tensor shapes
    for key in state_dicts[0].keys():
        final_state_dict[key] = weights[0] * state_dicts[0][key]
        reference_shapes[key] = state_dicts[0][key].shape
    
    # Second pass: Add weighted tensors, handling dimension mismatches
    for i, state_dict in enumerate(state_dicts):
        if i == 0:
            continue  # Already processed
            
        for key in state_dict.keys():
            if key not in final_state_dict:
                print(f"Warning: Key {key} found in state_dict {i} but not in the first state_dict. Skipping.")
                continue
                
            # Check if tensor shapes are compatible
            if state_dict[key].shape != reference_shapes[key]:
                # If this is the first time we've seen this incompatible key, log it
                if key not in incompatible_keys:
                    print(f"Warning: Shape mismatch for key {key}. "
                          f"Reference shape: {reference_shapes[key]}, "
                          f"State dict {i} shape: {state_dict[key].shape}. "
                          f"Skipping this tensor for merge.")
                    incompatible_keys.add(key)
                continue
                
            # Add weighted tensor if shapes match
            final_state_dict[key] += weights[i] * state_dict[key]
    
    # Report summary of incompatible keys
    if incompatible_keys:
        print(f"Completed merge with {len(incompatible_keys)} incompatible keys skipped: {incompatible_keys}")
    
    return final_state_dict

def task_arithmetic_merge(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float],
                          base_model_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    Merge state dictionaries using task arithmetic, handling tensor size mismatches.
    
    Args:
        state_dicts: List of state dictionaries to merge
        weights: List of weights for each state dictionary
        base_model_idx: Index of the base model in state_dicts
    
    Returns:
        Merged state dictionary
    """
    # Extract the base model state dict
    base_state_dict = state_dicts[base_model_idx]
    incompatible_keys = set()
    
    # Build task vectors (differences between each model and the base model)
    task_vectors = []
    for i, state_dict in enumerate(state_dicts):
        if i == base_model_idx:
            continue
        
        task_vector = {}
        for key in base_state_dict.keys():
            if key in state_dict:
                # Check if tensor shapes are compatible
                if state_dict[key].shape != base_state_dict[key].shape:
                    if key not in incompatible_keys:
                        print(f"Warning: Shape mismatch for key {key} in model {i}. "
                              f"Base shape: {base_state_dict[key].shape}, "
                              f"Model {i} shape: {state_dict[key].shape}. "
                              f"Skipping this tensor for task vector calculation.")
                        incompatible_keys.add(key)
                    continue
                task_vector[key] = state_dict[key] - base_state_dict[key]
        
        task_vectors.append((task_vector, weights[i]))
    
    # Apply weighted task vectors to the base model
    final_state_dict = {k: v.clone() for k, v in base_state_dict.items()}
    
    for task_vector, weight in task_vectors:
        for key in final_state_dict.keys():
            if key in task_vector:
                final_state_dict[key] += weight * task_vector[key]
    
    # Report summary of incompatible keys
    if incompatible_keys:
        print(f"Completed task arithmetic merge with {len(incompatible_keys)} incompatible keys skipped: {incompatible_keys}")
    
    return final_state_dict

def ties_merge(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float],
               threshold: float = 0.01) -> Dict[str, torch.Tensor]:
    """
    Merge state dictionaries using TIES merging method, handling tensor size mismatches.
    
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
    
    # Identify common keys with compatible shapes
    reference_shapes = {k: v.shape for k, v in processed_dicts[0].items()}
    compatible_keys = set(reference_shapes.keys())
    
    for i, state_dict in enumerate(processed_dicts[1:], 1):
        for key in list(compatible_keys):
            if key not in state_dict:
                compatible_keys.remove(key)
                print(f"Warning: Key {key} not found in state_dict {i}. Removing from compatible keys.")
            elif state_dict[key].shape != reference_shapes[key]:
                compatible_keys.remove(key)
                print(f"Warning: Shape mismatch for key {key}. "
                      f"Reference shape: {reference_shapes[key]}, "
                      f"State dict {i} shape: {state_dict[key].shape}. "
                      f"Removing from compatible keys.")
    
    incompatible_keys = set(reference_shapes.keys()) - compatible_keys
    if incompatible_keys:
        print(f"Found {len(incompatible_keys)} incompatible keys that will be skipped in TIES merge.")
    
    # Step 2: Resolve sign conflicts (only for compatible keys)
    final_state_dict = {}
    for key in compatible_keys:
        # Stack parameters from all models for this key
        stacked_params = torch.stack([state_dict[key] for state_dict in processed_dicts])
        
        # Calculate sign consensus
        sign_consensus = torch.sign(torch.mean(stacked_params, dim=0))
        
        # Apply sign consensus to absolute values
        resolved_params = torch.mean(torch.abs(stacked_params), dim=0) * sign_consensus
        
        final_state_dict[key] = resolved_params
    
    # Step 3: Merge aligned parameters
    for key in compatible_keys:
        stacked_params = torch.stack([state_dict[key] for state_dict in processed_dicts])
        alignment_mask = torch.prod(torch.sign(stacked_params), dim=0) > 0
        final_state_dict[key] = final_state_dict[key] * alignment_mask.float()
    
    # For incompatible keys, use values from the first state dict
    for key in incompatible_keys:
        final_state_dict[key] = processed_dicts[0][key]
    
    # Report summary
    if incompatible_keys:
        print(f"Completed TIES merge with {len(incompatible_keys)} incompatible keys using first state dict values: {incompatible_keys}")
    
    return final_state_dict

def dare_merge(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float],
               threshold: float = 0.01, amplification_factor: float = 2.0,
               base_model_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    Merge state dictionaries using DARE merging method, handling tensor size mismatches.
    
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
    
    # Track incompatible keys
    incompatible_keys = set()
    
    # Process each fine-tuned model
    for i, state_dict in enumerate(state_dicts):
        if i == base_model_idx:
            continue
        
        # Apply DARE method
        for key in final_state_dict.keys():
            if key in state_dict:
                # Check if tensor shapes are compatible
                if state_dict[key].shape != base_state_dict[key].shape:
                    if key not in incompatible_keys:
                        print(f"Warning: Shape mismatch for key {key} in model {i}. "
                              f"Base shape: {base_state_dict[key].shape}, "
                              f"Model {i} shape: {state_dict[key].shape}. "
                              f"Skipping this tensor for DARE merge.")
                        incompatible_keys.add(key)
                    continue
                
                # Calculate difference between fine-tuned and base model
                difference = state_dict[key] - base_state_dict[key]
                
                # Apply threshold and amplification
                small_differences_mask = torch.abs(difference) < threshold
                difference[small_differences_mask] = 0.0
                difference[~small_differences_mask] *= amplification_factor * weights[i]
                
                # Update the final state dict
                final_state_dict[key] += difference
    
    # Report summary of incompatible keys
    if incompatible_keys:
        print(f"Completed DARE merge with {len(incompatible_keys)} incompatible keys skipped: {incompatible_keys}")
    
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
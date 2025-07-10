#!/bin/bash

# Set the domain list
domain_list="math"

# Set the name for the search
name="math_bwr"

# Set the evaluation type
eval_type="exact_match"

# Set the dataset
dataset="gsm8k"

# Set the base model
base_model="google/gemma-7b-it"

# Set the project name for wandb
project_name="bwr"

# Set the GPUs to use
gpus="0"

# Run the search
python search.py \
    -n $name \
    -e $eval_type \
    -d $dataset \
    -g $gpus \
    -b $base_model \
    --project_name_wb $project_name \
    --populate_initial_experts 1 \
    --initial_experts_num 20 \
    --step_length 1 \
    --step_length_factor 0.95 \
    --minimum_step_length 0.1 \
    --restart_stray_candidates 1 \
    --restart_patience 0.5 \
    --merge_method weighted_average \
    --merge_params "{}" \
    -p 10 \
    -m 200

# Example usage with different merge methods:
#
# Task Arithmetic merge:
# --merge_method task_arithmetic \
# --merge_params "{\"base_model_idx\": 0}" \
#
# TIES merge:
# --merge_method ties \
# --merge_params "{\"threshold\": 0.01}" \
#
# DARE merge:
# --merge_method dare \
# --merge_params "{\"threshold\": 0.01, \"amplification_factor\": 2.0, \"base_model_idx\": 0}" \
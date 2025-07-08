#!/bin/bash

# Set the name for the search
name="hindi_math_bwr"

# Set the evaluation type - we'll need to use exact_match for math problems
eval_type="exact_match"

# Set the dataset - we'll need to create this dataset file
dataset="mgsm_hindi"

# Set the base model - OpenHathi is based on Mistral-7B
base_model="mistralai/Mistral-7B-v0.1"

# Set the project name for wandb
project_name="bwr_hindi_math"

# Set the GPUs to use - adjust based on your available GPUs
gpus="0,1,2,3,4"

# Run the search
python search.py \
    -n $name \
    -e $eval_type \
    -d $dataset \
    -g $gpus \
    -b $base_model \
    --project_name_wb $project_name \
    --populate_initial_experts 0 \
    --step_length 1 \
    --step_length_factor 0.95 \
    --minimum_step_length 0.1 \
    --restart_stray_candidates 1 \
    --restart_patience 0.5 \
    --merge_method task_arithmetic \
    --merge_params "{\"base_model_idx\": 0}" \
    -p 10 \
    -m 200

# Notes:
# 1. You'll need to download and prepare the following models as initial experts:
#    - sarvamai/OpenHathi-7B-Hi-v0.1-Base (Hindi LLM)
#    - WizardMath-7B-V1.1 (Math LLM)
#    - Abel-7B-002 (Math LLM)
#
# 2. You'll need to create the MGSM Hindi dataset file in data/eval/mgsm_hindi.json
#
# 3. We're using task_arithmetic as the merge method with the Hindi model as the base,
#    which should help preserve Hindi language capabilities while adding math skills
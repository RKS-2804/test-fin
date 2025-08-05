#!/bin/bash

# Set the name for the search
name="japan_finance_fraud_detection"

# Set the evaluation type for fraud detection
eval_type="fraud_detection"

# Set the dataset
dataset="japan_finance_fraude"

# Set the base model - select one of the Japanese financial models
base_model="pfnet/nekomata-7b-pfn-qfin"

# Set the project name for wandb
project_name="japan_finance"

# Set the GPUs to use
gpus="0"

# Run the search
python jaya_search.py \
    -n $name \
    -e $eval_type \
    -d $dataset \
    -g $gpus \
    -b $base_model \
    --project_name_wb $project_name \
    --populate_initial_experts 1 \
    --initial_experts_num 10 \
    --step_length 1 \
    --step_length_factor 0.95 \
    --minimum_step_length 0.1 \
    --restart_stray_candidates 1 \
    --restart_patience 0.5 \
    --merge_method weighted_average \
    --merge_params "{}" \
    --fitness_function "combined" \
    -p 10 \
    -m 50

# Note: fitness_function options:
# - "roc_auc" - optimize only for ROC-AUC
# - "mcc" - optimize only for Matthews Correlation Coefficient
# - "combined" - optimize for weighted combination (0.2*acc + 0.5*roc_auc + 0.3*mcc)
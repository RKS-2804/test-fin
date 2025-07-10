# #!/bin/bash

# # Set the name for the search
# name="japanese_math_bwr"

# # Set the evaluation type - we'll need to use exact_match for math problems
# eval_type="exact_match"

# # Set the dataset - we'll need to create this dataset file
# dataset="mgsm_japanese"

# # Set the base model - all source models are based on Mistral-7B
# base_model="mistralai/Mistral-7B-v0.1"

# # Set the project name for wandb
# project_name="bwr_japanese_math"

# # Set GPU visibility
# export CUDA_VISIBLE_DEVICES=0
# gpus="0"
# # gpus="0"  # keep this for CUDA_VISIBLE_DEVICES
# gpus_list=(0)


# # Run the search
# python search.py \
#     -n $name \
#     -e $eval_type \
#     -d $dataset \
#     -g ${gpus_list[@]} \
#     -b $base_model \
#     --project_name_wb $project_name \
#     --populate_initial_experts 0 \
#     --step_length 1 \
#     --step_length_factor 0.95 \
#     --minimum_step_length 0.1 \
#     --restart_stray_candidates 1 \
#     --restart_patience 0.5 \
#     --merge_method task_arithmetic \
#     --merge_params "{\"base_model_idx\": 0}" \
#     -p 10 \
#     -m 200

# # Notes:
# # 1. You'll need to download and prepare the following models as initial experts:
# #    - shisa-gamma-7b-v1 (Japanese LLM)
# #    - WizardMath-7B-V1.1
# #    - Abel-7B-002
# #
# # 2. You'll need to create the MGSM Japanese dataset file in data/eval/mgsm_japanese.json
# #
# # 3. We're using task_arithmetic as the merge method with the Japanese model as the base,
# #    which should help preserve Japanese language capabilities while adding math skills

#!/bin/bash

name="hindi_math_bwr"
eval_type="exact_match"
dataset="mgsm_hindi"
base_model="mistralai/Mistral-7B-v0.1"
project_name="bwr_hindi_math"

# Correct way to set GPU list
gpus="0"
gpu_array=(0)  # So Python sees it as a list

# Ensure CUDA sees the right device
export CUDA_VISIBLE_DEVICES=$gpus

# Run search with correctly passed GPU list
python search.py \
    -n $name \
    -e $eval_type \
    -d $dataset \
    -g ${gpu_array[@]} \
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

# Japanese Math LLM using BWR_model

This guide explains how to use the BWR_model framework to create a Japanese Math LLM by merging a Japanese language model with math-specialized models.

## Overview

This project aims to create a model that can solve math problems in Japanese by merging:
- A Japanese language model: shisa-gamma-7b-v1
- Math-specialized models: WizardMath-7B-V1.1 and Abel-7B-002

All these models are based on Mistral-7B-v0.1, making them compatible for merging.

## Setup Instructions

### 1. Prepare the Initial Expert Models

Run the provided script to download and prepare the initial expert models:

```bash
cd BWR_model/initial_experts
python japanese_math_init.py
cd ..
```

This script will:
- Download the three models (shisa-gamma-7b-v1, WizardMath-7B-V1.1, and Abel-7B-002)
- Convert them to LoRA adapters if needed
- Place them in the initial_experts directory

### 2. Prepare the Dataset

We use the MGSM (Multilingual GSM8k) dataset for evaluation. The Japanese test set consists of 250 samples.

For training/search, we use samples from the GSM8k test set (IDs 250-1318) translated to Japanese.

A template file is provided at `data/eval/mgsm_japanese.json`. Replace this with your actual dataset following the same format.

### 3. Run the BWR Algorithm

Execute the search script to run the BWR algorithm:

```bash
bash search_japanese_math.sh
```

This script is configured to:
- Use the task_arithmetic merge method with the Japanese model as the base
- This helps preserve Japanese language capabilities while adding math skills
- Evaluate models on the Japanese math problems

### 4. Customize the Search

You can modify `search_japanese_math.sh` to adjust parameters:

- Change the merge method:
  ```bash
  # Task Arithmetic (default in the script)
  --merge_method task_arithmetic --merge_params "{\"base_model_idx\": 0}"
  
  # TIES merge
  --merge_method ties --merge_params "{\"threshold\": 0.01}"
  
  # DARE merge
  --merge_method dare --merge_params "{\"threshold\": 0.01, \"amplification_factor\": 2.0, \"base_model_idx\": 0}"
  ```

- Adjust other parameters:
  ```bash
  --step_length 0.5  # Smaller steps for more conservative updates
  -p 5               # Patience for early stopping
  -m 100             # Maximum iterations
  ```

## Expected Results

After running the search, you'll find:
- The best model in `search/japanese_math_bwr_*/best/`
- Performance metrics in `search/japanese_math_bwr_*/utility_scratchpad.json`
- Detailed logs in `search/japanese_math_bwr_*/log.txt`

The best model should be able to solve math problems in Japanese by combining:
1. Japanese language understanding from shisa-gamma-7b-v1
2. Mathematical reasoning capabilities from WizardMath-7B-V1.1 and Abel-7B-002

## References

This approach is based on the methodology described in research on evolving Japanese Math LLMs, where evolutionary model merging is applied to combine language-specific and task-specific models.
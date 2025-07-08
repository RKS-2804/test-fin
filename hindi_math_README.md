# Hindi Math LLM using BWR_model

This guide explains how to use the BWR_model framework to create a Hindi Math LLM by merging a Hindi language model with math-specialized models.

## Overview

This project aims to create a model that can solve math problems in Hindi by merging:
- A Hindi language model: sarvamai/OpenHathi-7B-Hi-v0.1-Base
- Math-specialized models: WizardMath-7B-V1.1 and Abel-7B-002

All these models are based on Mistral-7B-v0.1, making them compatible for merging.

## Setup Instructions

### 1. Prepare the Initial Expert Models

Run the provided script to download and prepare the initial expert models:

```bash
cd BWR_model/initial_experts
python hindi_math_init.py
cd ..
```

This script will:
- Download the three models (OpenHathi-7B-Hi-v0.1-Base, WizardMath-7B-V1.1, and Abel-7B-002)
- Convert them to LoRA adapters if needed
- Place them in the initial_experts directory

### 2. Prepare the Dataset

We use a Hindi version of math problems for evaluation. A template file is provided at `data/eval/mgsm_hindi.json`. Replace this with your actual dataset following the same format.

For the dataset, you can:
1. Use existing Hindi math datasets if available
2. Translate GSM8k problems to Hindi (similar to how MGSM was created)
3. Create your own Hindi math problems dataset

### 3. Run the BWR Algorithm

Execute the search script to run the BWR algorithm:

```bash
bash search_hindi_math.sh
```

This script is configured to:
- Use the task_arithmetic merge method with the Hindi model as the base
- This helps preserve Hindi language capabilities while adding math skills
- Evaluate models on the Hindi math problems

### 4. Customize the Search

You can modify `search_hindi_math.sh` to adjust parameters:

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
- The best model in `search/hindi_math_bwr_*/best/`
- Performance metrics in `search/hindi_math_bwr_*/utility_scratchpad.json`
- Detailed logs in `search/hindi_math_bwr_*/log.txt`

The best model should be able to solve math problems in Hindi by combining:
1. Hindi language understanding from OpenHathi-7B-Hi-v0.1-Base
2. Mathematical reasoning capabilities from WizardMath-7B-V1.1 and Abel-7B-002

## Evaluation

To evaluate your model:
1. Prepare a test set of Hindi math problems
2. Run the model on these problems
3. Compare the model's answers with the correct answers
4. Calculate accuracy and other relevant metrics

## Tips for Better Results

1. **Base Model Selection**: Make sure the Hindi model (OpenHathi) is set as the base model in the task_arithmetic merge method to preserve Hindi language capabilities.

2. **Dataset Quality**: The quality of your Hindi math dataset will significantly impact the results. Ensure the problems are well-formulated and the answers are correct.

3. **Hyperparameter Tuning**: Experiment with different merge methods and parameters to find the optimal configuration for your specific use case.

4. **Multiple Runs**: Consider running the BWR algorithm multiple times with different random seeds to get more robust results.
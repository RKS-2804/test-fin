# BWR_model

Repository for "BWR_model: Collaborative Search to Adapt LLM Experts via Best-Worst-Random Algorithm".

## Quick Start

#### Initialization

Create a conda environment for BWR_model.
```
conda env create -f bwr.yml
conda activate bwr
```

Log into huggingface (for gemma access).
```
huggingface-cli login
```

Download initial experts.
```
cd initial_experts
python expert_init.py
cd ..
```

#### Execute your first BWR_model search.

Let's run BWR_model on the [NLGraph](https://arxiv.org/abs/2305.10037) dataset focusing on LLM graph reasoning. `search_nlgraph.sh` provides the starter script for this.

Before running, how many GPUs do you have (and what are the GPU ids?). Change `-g` in line 23 of `search_nlgraph.sh`: by default five GPUs with ids `0,1,2,3,4`, but you can change to `0`, `0,1`, `0,2,4,5,6` or any combination you'd like.

Run it!
```
bash search_nlgraph.sh
```

You might be prompted to log into WandB and it is highly recommended. There will be a directory in `search/` that starts with `nlgraph_...`, all the logs, models, and results will be stored there. You can check out `search/nlgraph_.../log.txt` to see current progress.

[My GPUs are not so powerful and it runs kind of slow?] change `--populate_initial_experts` from 1 to 0, change `--initial_experts_num` from 20 to 10, add `--step_length 0.5`, `-p 5`, `-m 20` together for a toned-down version.

End of search: the best-found model will be in `search/nlgraph_.../global_best` and performance metrics will be in `search/nlgraph_.../utility_scratchpad.json`. The json file presents the change in utility values in the search process: for global best, global worst, and individual models. The `log.txt` will contain in the end several overall metrics. Watch for `starting_best_validation_utility`, the best initial expert's utility value on the dev set; `starting_best_single_test_accuracy`, this initial expert's utility on the test set; `ending_best_validation_utility`, the ending global best's utility on the dev set; `ending_best_single_test_accuracy`, the ending global best's utility on the test set.

## BWR Algorithm

The BWR (Best-Worst-Random) algorithm is a population-based, parameter-free algorithm for model merging. It works as follows:

1. Initialize a population of candidate models
2. Evaluate each candidate model and identify the best and worst models
3. For each iteration:
   - For each candidate model:
     - Generate random variables r1, r2, r3, r4 ~ Uniform(0, 1)
     - Randomly choose T from {1, 2}
     - Randomly select another candidate model
     - If r4 > 0.5:
       - Update the candidate using the formula:
         - New = Current + r1 * (Best - T * Random) - r2 * (Worst - Random)
     - Otherwise:
       - Reset the candidate with a random model: New = Random * r3
   - Evaluate all candidates
   - Update the best and worst models
4. Return the best model found

The BWR algorithm is parameter-free, requiring only the population size and maximum number of iterations. It balances exploration and exploitation through its dual update mechanism.

## Model Merging Methods

BWR_model supports multiple model merging methods that can be specified using the `--merge_method` parameter:

### Weighted Average (Default)

The default merging method that performs a weighted average of model parameters. This is the original method used in the BWR algorithm.

```
--merge_method weighted_average
```

### Task Arithmetic

This method treats the difference between models as task vectors and applies them to the base model. It's useful when you want to preserve the base model's knowledge while adding specific capabilities from other models.

```
--merge_method task_arithmetic --merge_params "{\"base_model_idx\": 0}"
```

Parameters:
- `base_model_idx`: Index of the base model in the list of models to merge (default: 0)

### TIES Merging

TIES (Task-Informed Expert Selection) merging resolves sign conflicts between parameters and merges only aligned parameters. This method is effective when models have conflicting parameter changes.

```
--merge_method ties --merge_params "{\"threshold\": 0.01}"
```

Parameters:
- `threshold`: Threshold for minimal changes (default: 0.01)

### DARE Merging

DARE (Difference-Amplified REcombination) merging amplifies significant parameter differences while eliminating small differences. This helps focus on the most important changes between models.

```
--merge_method dare --merge_params "{\"threshold\": 0.01, \"amplification_factor\": 2.0, \"base_model_idx\": 0}"
```

Parameters:
- `threshold`: Threshold for small differences (default: 0.01)
- `amplification_factor`: Factor to amplify large differences (default: 2.0)
- `base_model_idx`: Index of the base model (default: 0)

## Other Objectives

For Objective 1: single task, follow `search_nlgraph.sh` and change `-e` and `-d` to combinations of `(multiple_choice, mmlu)`, `(multiple_choice, mmlu_pro)`, `(multiple_choice, knowledge_crosswords)`, `(multiple_choice, hellaswag)`, `(exact_match, gsm8k)`, `(exact_match, nlgraph)`, `(multiple_choice, truthfulqa)`, `(external_api, realtoxicityprompts)`, `(AbstainQA, mmlu)`. Make sure to change `-n` to include the dataset name in the search directory name. For RealToxicityPrompts you will be prompted to setup Perspective API through google cloud.

For Objective 2: multi-task domains, follow `search_legal.sh` and change `-d` to `legal`, `medical`, `science`, and `culture`. Make sure to change `-n` as well.

For Objective 3: reward models, follow `search_concise.sh` and change `-e` to `rm_default`, `rm_verbose`, and `rm_concise`. Make sure to change `-n` as well.

For Objective 4: human interests, follow `search_phd_application.sh` and change line 8 `domain_list` into the `human_<name>.json` in `data/eval/`. Make sure to change `-n` as well. You will be prompted to setup Vertex AI API to access Gemini APIs through google cloud.

Adding your data: just follow the format of existing data files in `data/eval/` and then change `-n` and use a respective `-e`. For example, for multiple-choice, follow the format of `data/eval/hellaswag.json` and use `-e` as `multiple_choice`; for answer match, follow the format of `data/eval/nlgraph.json` and use `-e` as `exact_match`; for optimizing reward model scores (objective 3 or 4), follow `data/eval/rm.json` and use `-e` as either `rm_default` for a local reward model or `human` for gemini-as-a-judge.

Adding your models: look at `initial_experts/`: it is essentially a folder of 10 LoRA adapters of Gemma-7B. Create your own folder of models with the same architecture like `initial_experts` and change add the argument `-i <path_to_folder>` (see search.py). If the models are adapters/only have 1 shard, you don't need to change anything else. If they are full models/multiple shards, change `--fast_merge` from `1` to `0`.

Adding your evaluation: well that will be a bit more workload. Go to `evaluate.py`: there's a `evaluate()` for valiation set and `evaluate_test()` for test set (intentionally kept separate), essentially you specify path to a model and these functions give you a scalar score. In both of them there are `if eval_type == ...` clauses: name your evaluation type, open a new if clause `if eval_type == <your_eval>`, implement how you load the data and get a scalar score. You need to do it for both `evaluate()` and `evaluate_test()`. `batch_generate()` and `batch_generate_chat_template()` provide two helper functions to generate text from a model and please use them for model output. You could refer to `eval_type == multiple_choice` or `eval_type == exact_match` as examples.

## Changing Hyperparameters and Settings

Do `python search.py -h` to see a list of all possible hyperparameters and settings. Additionally look at the comments for hyperparameters in `search.py`. We already included the default settings in the four `search_<name>.sh` starter scripts, but feel free to play around different settings.

## Citation

If BWR_model is helpful to you:

```
@article{bwr_model2025,
  title={BWR_model: Collaborative Search to Adapt LLM Experts via Best-Worst-Random Algorithm},
  author={Ravi},
  year={2025}
}

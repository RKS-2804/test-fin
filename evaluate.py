import json
import math
import time
import torch
import datetime
import numpy as np
import sys
from tqdm import tqdm
from datasets import load_dataset
from tenacity import retry, wait_random_exponential, stop_after_attempt
from googleapiclient import discovery
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
import random
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print GPU information at startup
available_gpus = torch.cuda.device_count()
gpu_ids = list(range(available_gpus))
print(f"[INFO] Available GPUs: {available_gpus} (IDs: {gpu_ids})")

# Global variables
ICL_PROMPT = None
model = None
tokenizer = None
PERSPECTIVE_API_KEY = None  # Provide your own perspective API key through google cloud
perspective_already_warned = False

# Try to initialize the Perspective API client
try:
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
except:
    if not perspective_already_warned:
        warnings.warn("Ignore this if not running RealToxicityPrompts evaluation: provide your own perspective API key through google cloud.")
        perspective_already_warned = True

# Domain dataset mappings
multitask_domain_dataset_dict = {
    "legal": ["hearsay", "citation_prediction_classification"],
    "medical": ["medqa", "medmcqa"],
    "science": ["scifact", "stem"],
    "culture": ["normad_country", "normad_value"],
}

# Global variable for one_or_two option
ONLY_ONE_OR_TWO = None

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def lora_weight_visualize(path):
    """
    Visualize LoRA weights for debugging and analysis.
    
    Args:
        path (str): Path to the LoRA weights file.
        
    Returns:
        list: Coordinates for visualization.
    """
    state_dict_this = load_file(path, device="cpu")
    return [
        state_dict_this["base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"][0][0].item(), 
        state_dict_this["base_model.model.model.layers.10.self_attn.k_proj.lora_B.weight"][0][0].item()
    ]

def update_only_one_or_two(value):
    """
    Update the ONLY_ONE_OR_TWO global variable.
    
    Args:
        value: The value to set ONLY_ONE_OR_TWO to.
    """
    global ONLY_ONE_OR_TWO
    ONLY_ONE_OR_TWO = value

def current_time_string():
    """
    Get the current time as a formatted string.
    
    Returns:
        str: Formatted current time.
    """
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

def get_effective_reliability(correct_flags, abstain_flags):
    """
    Calculate effective reliability for AbstainQA.
    
    Args:
        correct_flags (list): List of correct flags.
        abstain_flags (list): List of abstain flags.
        
    Returns:
        float: Effective reliability score.
    """
    answered_correct = 0
    answered_incorrect = 0

    for i in range(len(correct_flags)):
        if abstain_flags[i] == 0:
            if correct_flags[i] == 1:
                answered_correct += 1
            else:
                answered_incorrect += 1

    return (answered_correct - answered_incorrect) / len(correct_flags)

@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=20))
def get_toxicity_score(text):
    """
    Get toxicity score from Perspective API.
    
    Args:
        text (str): Text to analyze.
        
    Returns:
        float: Toxicity score.
    """
    text = text.split("\n")[0].strip()
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

def multiple_choice_prompt(instance_dict, dataset):
    """
    Create a prompt for multiple choice questions.
    
    Args:
        instance_dict (dict): Instance dictionary.
        dataset (str): Dataset name.
        
    Returns:
        str: Formatted prompt.
    """
    prompt = "Question: " + instance_dict["question"] + "\n"

    if dataset == "knowledge_crosswords":
        prompt = prompt
    elif dataset == "hellaswag":
        prompt = "Please choose an option that best completes the sentence.\n" + prompt
    else:
        prompt = "Please choose an option that best answers the question.\n" + prompt

    for key in instance_dict["choices"].keys():
        prompt += (key + ": " + instance_dict["choices"][key] + "\n")

    prompt += "The answer is"

    # The format of Knowledge Crosswords is more complex and necessitates an in-context example
    if dataset == "knowledge_crosswords":
        prompt = ICL_PROMPT + "\n" + prompt

    return prompt

def multiple_choice_answer_parsing(instance_dict, output_text):
    """
    Parse the answer from the model output for multiple choice questions.
    
    Args:
        instance_dict (dict): Instance dictionary.
        output_text (str): Model output text.
        
    Returns:
        str: Parsed answer.
    """
    # Directly answer
    for key in instance_dict["choices"].keys():
        if key in output_text[:5]:
            return key
    # "The answer is ."
    for key in instance_dict["choices"].keys():
        if key in output_text[-5:]:
            return key
    # Answer text exact match
    for key in instance_dict["choices"].keys():
        if instance_dict["choices"][key].lower() in output_text.lower():
            return key
    return "Z"  # So that it is absolutely incorrect

def batch_generate_chat_template(model, tokenizer, prompts, gpu_id, batch_size=10, max_new_tokens=512):
    """
    Generate responses using chat template.
    
    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        prompts (list): List of prompts.
        gpu_id (int): GPU ID to use.
        batch_size (int): Batch size.
        max_new_tokens (int): Maximum number of new tokens to generate.
        
    Returns:
        list: Generated outputs.
    """
    outputs = []
    # Batch_size argument is useless here, sequential generation is necessary
    for prompt in tqdm(prompts):
        chat = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output = model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens, do_sample=False)
        outputs.append(tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True).strip())
    return outputs

def batch_generate(model, tokenizer, prompts, gpu_id, batch_size=10, max_new_tokens=10):
    """
    Generate responses in batches.
    
    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        prompts (list): List of prompts.
        gpu_id (int): GPU ID to use.
        batch_size (int): Batch size.
        max_new_tokens (int): Maximum number of new tokens to generate.
        
    Returns:
        list: Generated outputs.
    """
    num_batches = math.ceil(len(prompts) / batch_size)
    outputs = []
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        input_ids = tokenizer(batch_prompts, return_tensors="pt", padding=True).input_ids.to(f"cuda:{gpu_id}")
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        for j in range(len(output)):
            outputs.append(tokenizer.decode(output[j][len(input_ids[j]):], skip_special_tokens=True).strip())
        
        del input_ids, output
        torch.cuda.empty_cache()
    
    return outputs

def validate_gpu_id(gpu_id):
    """
    Validate that the requested GPU ID is available on the system.
    
    Args:
        gpu_id (int): The GPU ID to validate
        
    Returns:
        int: A valid GPU ID (either the requested one or GPU 0 as fallback)
        
    Raises:
        RuntimeError: If no GPUs are available on the system
    """
    available_gpu_count = torch.cuda.device_count()
    available_gpu_ids = list(range(available_gpu_count))
    
    if available_gpu_count == 0:
        raise RuntimeError("No GPUs available on this system. Cannot proceed with evaluation.")
    
    if gpu_id >= available_gpu_count:
        logger.warning(f"Invalid GPU ID: {gpu_id}. Only {available_gpu_count} GPUs available (IDs: {available_gpu_ids})")
        logger.warning(f"Falling back to GPU 0")
        return 0
    
    return gpu_id

def evaluate(model_path, eval_type, dataset, gpu_id, base_model="google/gemma-7b-it", save_dev_flag=False, only_one_or_two=None, skip_flag=False):
    """
    Evaluate a model on a dataset.
    
    Args:
        model_path (str): Path to the model.
        eval_type (str): Type of evaluation.
        dataset (str): Dataset to evaluate on.
        gpu_id (int): GPU ID to use.
        base_model (str): Base model to use.
        save_dev_flag (bool): Whether to save dev set results.
        only_one_or_two (str): Whether to only evaluate on one or two datasets.
        skip_flag (bool): Whether to skip evaluation.
        
    Returns:
        float: Evaluation score.
    """
    if skip_flag:
        return None

    global model
    global tokenizer
    only_one_or_two = ONLY_ONE_OR_TWO
    
    try:
        # Validate GPU ID before attempting to use it
        valid_gpu_id = validate_gpu_id(gpu_id)
        if valid_gpu_id != gpu_id:
            logger.info(f"Using GPU {valid_gpu_id} instead of requested GPU {gpu_id}")
            gpu_id = valid_gpu_id
            
        logger.info(f"Loading model on GPU {gpu_id}")
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
        
        # Try to load the adapter with dimension mismatch handling
        try:
            logger.info(f"Loading adapter from {model_path}")
            model.load_adapter(model_path)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.warning(f"Dimension mismatch detected when loading adapter: {str(e)}")
                logger.warning(f"Attempting to load adapter with dimension mismatch handling...")
                
                # Load adapter state dict manually
                from safetensors.torch import load_file
                import os
                
                adapter_path = os.path.join(model_path, "adapter_model.safetensors")
                if os.path.exists(adapter_path):
                    # Load adapter state dict
                    adapter_state_dict = load_file(adapter_path)
                    
                    # Get model state dict for reference
                    model_state_dict = model.state_dict()
                    
                    # Create a filtered state dict with only compatible tensors
                    filtered_state_dict = {}
                    skipped_keys = []
                    
                    for key, tensor in adapter_state_dict.items():
                        if key in model_state_dict:
                            if model_state_dict[key].shape == tensor.shape:
                                filtered_state_dict[key] = tensor
                            else:
                                skipped_keys.append(key)
                                logger.warning(f"Skipping incompatible tensor: {key}, "
                                              f"adapter shape: {tensor.shape}, "
                                              f"model shape: {model_state_dict[key].shape}")
                    
                    # Load the filtered state dict
                    logger.info(f"Loading filtered adapter state dict with {len(filtered_state_dict)} compatible tensors")
                    logger.info(f"Skipped {len(skipped_keys)} incompatible tensors")
                    
                    # Use strict=False to ignore missing keys
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
                    logger.info(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                else:
                    logger.error(f"Adapter file not found at {adapter_path}")
                    raise FileNotFoundError(f"Adapter file not found at {adapter_path}")
            else:
                # Re-raise if it's not a size mismatch error
                raise
        
        model.to(f"cuda:{gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error(f"CUDA out of memory on GPU {gpu_id}. Try using a different GPU or reducing batch size.")
            sys.exit(1)
        elif "invalid device ordinal" in str(e).lower():
            logger.error(f"Invalid device ordinal. GPU {gpu_id} does not exist. Available GPUs: {list(range(torch.cuda.device_count()))}")
            sys.exit(1)
        else:
            logger.error(f"Error loading model: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        try:
            del model
            del tokenizer
            logger.info(f"Attempting to load model directly from path on GPU {gpu_id}")
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
            model.to(f"cuda:{gpu_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e2:
            logger.error(f"Second attempt to load model failed: {str(e2)}")
            raise
    tokenizer.pad_token = tokenizer.eos_token

    # Task 1: single task, multiple choice questions
    if eval_type == "multiple_choice":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"]
        golds = []
        preds = []
        global ICL_PROMPT
        try:
            # In case an ICL prompt is provided for datasets such as Knowledge Crosswords
            ICL_PROMPT = json.load(open("data/eval/" + dataset + ".json"))["icl_prompt"]
        except:
            pass

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)
        
        # Hard-defined batch_size for multiple choice questions, reduce if OOM
        BATCH_SIZE = 10

        # Change max_new_tokens to larger values for intermediate reasoning
        outputs = batch_generate(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE, max_new_tokens=10)

        for question, output in zip(eval_data, outputs):
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))

        if save_dev_flag:
            with open(model_path + "/golds_dev.json", "w") as f:
                json.dump(golds, f)
            with open(model_path + "/preds_dev.json", "w") as f:
                json.dump(preds, f)
        
        # Utility function value is the accuracy score of the model on the multiple choice questions
        return accuracy_score(golds, preds)

    # Task 1: single task, exact match questions
    elif eval_type == "exact_match":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"]
        scores = []  # Could be 0/1 binary, could be continuous scores

        prompts = []
        for question in eval_data:
            prompts.append(question["question"])

        # Hard-defined batch_size for exact match questions, reduce if OOM
        BATCH_SIZE = 10
        if dataset == "nlgraph" or dataset == "nlgraph_mini":
            BATCH_SIZE = 5
        
        MAX_NEW_TOKENS = 10
        # Math reasoning datasets require more tokens for reasoning
        if dataset == "gsm8k":
            MAX_NEW_TOKENS = 200

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE)

        # Retain only the last 5 tokens for number disambiguation
        if dataset == "gsm8k":
            outputs = [" ".join(output.split(" ")[-5:]) for output in outputs]

        # Exact match evaluation
        for question, output in zip(eval_data, outputs):
            if question["answer"] in output:
                scores.append(1)
                time.sleep(0.2)
            else:
                scores.append(0)

        if save_dev_flag:
            with open(model_path + "/scores_dev.json", "w") as f:
                json.dump(scores, f)

        # Utility function value is the accuracy score of the model on the exact match questions
        return sum(scores) / len(scores)

    # Task 2: multi-task domains
    elif eval_type == "multitask":  # medical, legal, science, culture
        per_dataset_scores = []
        eval_datasets = multitask_domain_dataset_dict[dataset][:2]
        for eval_dataset in eval_datasets:
            if eval_dataset in ["nlgraph", "gsm8k", "xstreet_ar", "xstreet_es"]:
                per_dataset_scores.append(evaluate(model_path, "exact_match", eval_dataset, gpu_id, save_dev_flag=True))
            else:
                per_dataset_scores.append(evaluate(model_path, "multiple_choice", eval_dataset, gpu_id, save_dev_flag=True))
        assert len(per_dataset_scores) == 2
        if sum(per_dataset_scores) == 0:
            per_dataset_scores = [0.01, 0.01]  # Dummy scores
        harmonic_mean = 2 * per_dataset_scores[0] * per_dataset_scores[1] / (per_dataset_scores[0] + per_dataset_scores[1])
        if only_one_or_two == "one":
            return per_dataset_scores[0]
        elif only_one_or_two == "two":
            return per_dataset_scores[1]
        # Utility function value is the harmonic mean of the two scores on two datasets
        return harmonic_mean

    # Default case
    return 0.0

def evaluate_test(model_path, eval_type, dataset, gpu_id, base_model="google/gemma-7b-it", only_one_or_two=None):
    """
    Evaluate a model on the test set.
    
    Args:
        model_path (str): Path to the model.
        eval_type (str): Type of evaluation.
        dataset (str): Dataset to evaluate on.
        gpu_id (int): GPU ID to use.
        base_model (str): Base model to use.
        only_one_or_two (str): Whether to only evaluate on one or two datasets.
        
    Returns:
        float: Evaluation score.
    """
    global model
    global tokenizer

    only_one_or_two = ONLY_ONE_OR_TWO

    try:
        # Validate GPU ID before attempting to use it
        valid_gpu_id = validate_gpu_id(gpu_id)
        if valid_gpu_id != gpu_id:
            logger.info(f"Using GPU {valid_gpu_id} instead of requested GPU {gpu_id}")
            gpu_id = valid_gpu_id
            
        logger.info(f"Loading model for test evaluation on GPU {gpu_id}")
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
        
        # Try to load the adapter with dimension mismatch handling
        try:
            logger.info(f"Loading adapter from {model_path} for test evaluation")
            model.load_adapter(model_path)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.warning(f"Dimension mismatch detected when loading adapter for test: {str(e)}")
                logger.warning(f"Attempting to load adapter with dimension mismatch handling...")
                
                # Load adapter state dict manually
                from safetensors.torch import load_file
                import os
                
                adapter_path = os.path.join(model_path, "adapter_model.safetensors")
                if os.path.exists(adapter_path):
                    # Load adapter state dict
                    adapter_state_dict = load_file(adapter_path)
                    
                    # Get model state dict for reference
                    model_state_dict = model.state_dict()
                    
                    # Create a filtered state dict with only compatible tensors
                    filtered_state_dict = {}
                    skipped_keys = []
                    
                    for key, tensor in adapter_state_dict.items():
                        if key in model_state_dict:
                            if model_state_dict[key].shape == tensor.shape:
                                filtered_state_dict[key] = tensor
                            else:
                                skipped_keys.append(key)
                                logger.warning(f"Skipping incompatible tensor: {key}, "
                                              f"adapter shape: {tensor.shape}, "
                                              f"model shape: {model_state_dict[key].shape}")
                    
                    # Load the filtered state dict
                    logger.info(f"Loading filtered adapter state dict with {len(filtered_state_dict)} compatible tensors")
                    logger.info(f"Skipped {len(skipped_keys)} incompatible tensors")
                    
                    # Use strict=False to ignore missing keys
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
                    logger.info(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                else:
                    logger.error(f"Adapter file not found at {adapter_path}")
                    raise FileNotFoundError(f"Adapter file not found at {adapter_path}")
            else:
                # Re-raise if it's not a size mismatch error
                raise
        
        model.to(f"cuda:{gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error(f"CUDA out of memory on GPU {gpu_id}. Try using a different GPU or reducing batch size.")
            sys.exit(1)
        elif "invalid device ordinal" in str(e).lower():
            logger.error(f"Invalid device ordinal. GPU {gpu_id} does not exist. Available GPUs: {list(range(torch.cuda.device_count()))}")
            sys.exit(1)
        else:
            logger.error(f"Error loading model for test evaluation: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Error loading model for test evaluation: {str(e)}")
        try:
            del model
            del tokenizer
            logger.info(f"Attempting to load model directly from path on GPU {gpu_id}")
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
            model.to(f"cuda:{gpu_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e2:
            logger.error(f"Second attempt to load model failed: {str(e2)}")
            raise
    tokenizer.pad_token = tokenizer.eos_token

    # Task 1: single task, multiple choice questions
    if eval_type == "multiple_choice":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        golds = []
        preds = []
        global ICL_PROMPT
        try:
            ICL_PROMPT = json.load(open("data/eval/" + dataset + ".json"))["icl_prompt"]
        except:
            pass

        # Hard-defined batch_size for multiple choice questions, reduce if OOM
        BATCH_SIZE = 10

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)
        
        outputs = batch_generate(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE, max_new_tokens=10)

        for question, output in zip(eval_data, outputs):
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))

        # Save golds and preds for later ensemble
        with open(model_path + "/golds.json", "w") as f:
            json.dump(golds, f)
        with open(model_path + "/preds.json", "w") as f:
            json.dump(preds, f)

        return accuracy_score(golds, preds)

    # Task 1: single task, exact match questions
    elif eval_type == "exact_match":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        scores = []  # Could be 0/1 binary, could be continuous scores

        prompts = []
        for question in eval_data:
            prompts.append(question["question"])

        # Hard-defined batch_size for exact match questions, reduce if OOM
        BATCH_SIZE = 10
        if dataset == "nlgraph" or dataset == "nlgraph_mini":
            BATCH_SIZE = 5

        MAX_NEW_TOKENS = 10
        # Math reasoning datasets require more tokens for reasoning
        if dataset == "gsm8k":
            MAX_NEW_TOKENS = 200

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE)

        if dataset == "gsm8k":
            outputs = [" ".join(output.split(" ")[-10:]) for output in outputs]

        for question, output in zip(eval_data, outputs):
            if question["answer"] in output:
                scores.append(1)
            else:
                scores.append(0)

        with open(model_path + "/scores.json", "w") as f:
            json.dump(scores, f)

        return sum(scores) / len(scores)

    # Task 2: multi-task domains
    elif eval_type == "multitask":
        per_dataset_scores = []
        eval_datasets = multitask_domain_dataset_dict[dataset]
        for eval_dataset in eval_datasets:
            # Default multi-task evaluation sets are all MC
            per_dataset_scores.append(evaluate_test(model_path, "multiple_choice", eval_dataset, gpu_id))
        assert len(per_dataset_scores) == 2
        if sum(per_dataset_scores) == 0:
            per_dataset_scores = [0.01, 0.01]  # Dummy scores
        harmonic_mean = 2 * per_dataset_scores[0] * per_dataset_scores[1] / (per_dataset_scores[0] + per_dataset_scores[1])
        if only_one_or_two == "one":
            return per_dataset_scores[0]
        elif only_one_or_two == "two":
            return per_dataset_scores[1]
        return harmonic_mean

    # Default case
    return 0.0
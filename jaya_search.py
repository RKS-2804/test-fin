import os
import math
import json
import torch
import shutil
import socket
import argparse
import random
import logging
import datetime
import wandb
import sys
from overall_metrics import overall_metrics
from merge import lora_merge, MergeMethod
from evaluate import evaluate, evaluate_test, update_only_one_or_two, lora_weight_visualize
from multiprocessing import Pool
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForLanguageModeling


from peft import LoraConfig
from safetensors.torch import load_file, save_file
import torch

# Create a compatibility class if needed
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Print GPU information at startup
available_gpus = torch.cuda.device_count()
gpu_ids = list(range(available_gpus))
print(f"[INFO] Available GPUs: {available_gpus} (IDs: {gpu_ids})")

def log_with_flush(message, level=logging.INFO):
    """Log a message and flush the log handler."""
    logging.log(level, message)
    logging.getLogger().handlers[0].flush()

def current_time_string():
    """Get the current time as a formatted string."""
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

def assign_gpu(num_gpus, process_idx, total_processes):
    """Assign a GPU to a process."""
    process_per_gpu = math.ceil(total_processes / num_gpus)
    gpu_idx = math.floor(process_idx / process_per_gpu)
    return gpu_idx

# Initialize a directory in search/ for the JAYA model search
def initialize_search_records(search_pass_name, candidate_paths, eval_type, dataset, gpus, base_model, fast_merge, fitness_function="accuracy"):
    """
    Initialize the search records directory structure and initial models.
    
    Args:
        search_pass_name (str): Name of the search pass.
        candidate_paths (list): List of paths to initial candidate models.
        eval_type (str): Type of evaluation.
        dataset (str): Dataset to use for evaluation.
        gpus (list): List of GPU IDs to use.
        base_model (str): Base model to use.
        fast_merge (int): Whether to use fast merge.
        fitness_function (str): Fitness function to use (accuracy, roc_auc, mcc, or combined).
    """
    # Create directory structure
    for i in range(len(candidate_paths)):
        os.mkdir(os.path.join("search", search_pass_name, "candidate_"+str(i)))
    
    os.mkdir(os.path.join("search", search_pass_name, "best"))  # weights directly in this folder
    os.mkdir(os.path.join("search", search_pass_name, "worst"))  # weights directly in this folder
    
    # Initialize utility scratchpad with fitness function information
    utility_scratchpad = {
        "best": None, 
        "worst": None, 
        "history": [],
        "fitness_function": fitness_function
    }
    
    for i in range(len(candidate_paths)):
        utility_scratchpad[f"candidate_{i}"] = None
        utility_scratchpad[f"candidate_{i}_history"] = []
    
    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
        json.dump(utility_scratchpad, f, indent=4)

    # Initialize candidate models
    for i in range(len(candidate_paths)):
        shutil.copytree(candidate_paths[i], os.path.join("search", search_pass_name, "candidate_"+str(i)), dirs_exist_ok=True)
    
    # Evaluate the utility of starting candidates
    eval_args = []
    for i in range(len(candidate_paths)):
        eval_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                          eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                          base_model, True, None, False, fitness_function))
    
    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
    pool.close()
    pool.join()

    # Update utility scratchpad with initial results
    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
        utility_scratchpad = json.load(f)
    
    utility_scratchpad["best"] = max(results)
    utility_scratchpad["worst"] = min(results)
    utility_scratchpad["history"].append(utility_scratchpad["best"])

    for i in range(len(candidate_paths)):
        utility_scratchpad[f"candidate_{i}"] = results[i]
        utility_scratchpad[f"candidate_{i}_history"].append(results[i])

    # Logging at iteration=0
    wandb_log = {
        "best": utility_scratchpad["best"],
        "worst": utility_scratchpad["worst"],
    }
    for i in range(len(candidate_paths)):
        wandb_log["candidate_" + str(i)] = utility_scratchpad["candidate_" + str(i)]
    
    wandb.log(wandb_log)
    
    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
        json.dump(utility_scratchpad, f, indent=4)
    
    # Initialize best model checkpoint
    best_idx = results.index(max(results))
    shutil.copytree(os.path.join("search", search_pass_name, "candidate_"+str(best_idx)),
                   os.path.join("search", search_pass_name, "best"),
                   dirs_exist_ok=True)

    # Initialize worst model checkpoint
    worst_idx = results.index(min(results))
    shutil.copytree(os.path.join("search", search_pass_name, "candidate_"+str(worst_idx)),
                   os.path.join("search", search_pass_name, "worst"),
                   dirs_exist_ok=True)

# JAYA algorithm implementation for candidate update
def jaya_candidate_update(i, gpu_id, search_pass_name, fast_merge, step_length, restart_flag, 
                         exploration_prob=0.1, merge_method=MergeMethod.WEIGHTED_AVERAGE, merge_params=None):
    """
    Update a candidate model using the JAYA algorithm with greedy selection.
    
    JAYA Algorithm Process:
    1. Apply formula: X_new = X_old + r1 * (X_best - |X_old|) - r2 * (X_worst - |X_old|)
    2. Evaluate both X_old and X_new
    3. GREEDY SELECTION: Keep X_new only if f(X_new) > f(X_old), otherwise keep X_old
    
    Args:
        i (int): Candidate index.
        gpu_id (int): GPU ID to use.
        search_pass_name (str): Name of the search pass.
        fast_merge (int): Whether to use fast merge.
        step_length (float): Step length for update.
        restart_flag (bool): Whether to restart the candidate.
        exploration_prob (float): Probability of pure exploration.
        merge_method: Method for merging models.
        merge_params: Parameters for merge method.
    """
    # Get paths
    candidate_path = os.path.join("search", search_pass_name, "candidate_"+str(i))
    
    # Handle restart if needed
    if restart_flag:
        # Reset to a random model from other candidates
        all_candidates = []
        for j in range(len(os.listdir(os.path.join("search", search_pass_name)))):
            if os.path.isdir(os.path.join("search", search_pass_name, "candidate_"+str(j))) and j != i:
                all_candidates.append(j)
        
        if all_candidates:
            random_idx = random.choice(all_candidates)
            random_path = os.path.join("search", search_pass_name, "candidate_"+str(random_idx))
            
            # Create a new candidate path for the restart
            output_path = candidate_path + "_new"
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            
            # Check if the original candidate exists
            if os.path.exists(candidate_path):
                # Copy the random candidate to the new path for comparison
                shutil.copytree(random_path, output_path)
                log_with_flush(f"JAYA: Candidate_{i} restarted with a copy of candidate_{random_idx}")
            else:
                # If original doesn't exist, create it directly
                shutil.copytree(random_path, candidate_path)
                log_with_flush(f"JAYA: Candidate_{i} initialized with a copy of candidate_{random_idx}")
        return

    # Generate random variables for JAYA algorithm
    r1 = random.uniform(0, 1)  # Random factor for moving towards best
    r2 = random.uniform(0, 1)  # Random factor for moving away from worst
    
    # Get paths for best and worst models
    best_model_path = os.path.join("search", search_pass_name, "best")
    worst_model_path = os.path.join("search", search_pass_name, "worst")
    
    # Create temporary directory for intermediate results
    temp_path = os.path.join("search", search_pass_name, "temp_"+str(i))
    os.makedirs(temp_path, exist_ok=True)
    
    # Check for exploration vs exploitation
    if random.uniform(0, 1) < exploration_prob:
        # === Pure Exploration: Generate random solution ===
        log_with_flush(f"JAYA: Exploration mode for candidate_{i}")
        
        # Create a new candidate path for exploration
        output_path = candidate_path + "_new"
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Load a reference model to get shapes and bounds
        reference_path = best_model_path if os.path.exists(best_model_path) else candidate_path
        sd = load_file(
            os.path.join(reference_path, "adapter_model.safetensors"),
            device="cpu"
        )
        new_sd = {}
        for name, tensor in sd.items():
            max_abs = float(tensor.abs().max())
            L, U = -max_abs, +max_abs
            # Sample uniformly in [L, U]
            new_sd[name] = torch.empty_like(tensor).uniform_(L, U)
        
        # Save the random adapter
        save_file(new_sd, os.path.join(output_path, "adapter_model.safetensors"))
        
        # Copy adapter configuration
        config_src = os.path.join(reference_path, "adapter_config.json")
        config_dst = os.path.join(output_path, "adapter_config.json")
        if os.path.exists(config_src):
            shutil.copy(config_src, config_dst)
        
    else:
        # === JAYA Algorithm: X_new = X_old + r1 * (X_best - |X_old|) - r2 * (X_worst - |X_old|) ===
        log_with_flush(f"JAYA: Optimization mode for candidate_{i} (r1={r1:.3f}, r2={r2:.3f})")
        
        # Step 1: Create |X_old| (absolute value of current candidate)
        abs_candidate_path = os.path.join(temp_path, "abs_candidate")
        os.makedirs(abs_candidate_path, exist_ok=True)
        
        # Load current candidate and create absolute value version
        candidate_sd = load_file(
            os.path.join(candidate_path, "adapter_model.safetensors"),
            device="cpu"
        )
        abs_candidate_sd = {}
        for name, tensor in candidate_sd.items():
            abs_candidate_sd[name] = torch.abs(tensor)  # Take absolute value
        
        # Save absolute value candidate
        save_file(abs_candidate_sd, os.path.join(abs_candidate_path, "adapter_model.safetensors"))
        
        # Copy adapter configuration for absolute candidate
        config_src = os.path.join(candidate_path, "adapter_config.json")
        config_dst = os.path.join(abs_candidate_path, "adapter_config.json")
        if os.path.exists(config_src):
            shutil.copy(config_src, config_dst)
        
        # Step 2: Compute r1 * (X_best - |X_old|)
        lora_merge(
            weights=[r1, -r1],  # r1 * X_best - r1 * |X_old|
            lora_name_list=[best_model_path, abs_candidate_path],
            output_name=os.path.join(temp_path, "towards_best"),
            gpu_id=gpu_id,
            directly_load_safetensors=fast_merge,
            merge_method=merge_method,
            merge_params=merge_params
        )
        
        # Step 3: Compute r2 * (X_worst - |X_old|)
        lora_merge(
            weights=[r2, -r2],  # r2 * X_worst - r2 * |X_old|
            lora_name_list=[worst_model_path, abs_candidate_path],
            output_name=os.path.join(temp_path, "away_from_worst"),
            gpu_id=gpu_id,
            directly_load_safetensors=fast_merge,
            merge_method=merge_method,
            merge_params=merge_params
        )
        
        # Step 4: Final JAYA update: X_new = X_old + step_length * [r1*(X_best - |X_old|) - r2*(X_worst - |X_old|)]
        output_path = candidate_path + "_new"
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        lora_merge(
            weights=[1, step_length, -step_length],  # X_old + step_length * towards_best - step_length * away_from_worst
            lora_name_list=[
                candidate_path,
                os.path.join(temp_path, "towards_best"),
                os.path.join(temp_path, "away_from_worst")
            ],
            output_name=output_path,
            gpu_id=gpu_id,
            directly_load_safetensors=fast_merge,
            merge_method=merge_method,
            merge_params=merge_params
        )
        
        log_with_flush(f"JAYA: Updated candidate_{i} using JAYA formula")
    
    # Clean up temporary directory
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of this JAYA model search, also directory name in search/")
    argParser.add_argument("-e", "--eval_type", help="evaluation types")  # multiple_choice, exact_match, multitask, rm_default, rm_verbose, rm_concise, human, fraud_detection
    argParser.add_argument("-d", "--dataset", help="dataset as the search objective/evaluation")  # file names in data/eval, be mindful of using the right --eval_type
    argParser.add_argument("-g", "--gpus", help="available gpu ids in a string")  # such as 0,1,2,3,4
    argParser.add_argument("--num_cpu_when_merging", default=1, help="number of cpu cores when merging")
    argParser.add_argument("--step_length", default=1, help="step length for JAYA updates")
    argParser.add_argument("-p", "--patience", default=10, help="patience of the search")
    argParser.add_argument("-m", "--max_iteration", default=200, help="max iteration of the search")
    argParser.add_argument("-i", "--initial_expert_directory", default="./initial_experts", help="initial expert directory")
    argParser.add_argument("-b", "--base_model", default="google/gemma-7b-it", help="base model of the lora experts")
    argParser.add_argument("--starting_test_set_eval", default=0, help="starting test set evaluation")  # 0, 1
    argParser.add_argument("--fast_merge", default=1, help="whether to use fast merge by only loading the safetensor file")
    argParser.add_argument("--project_name_wb", default="jaya", help="wandb project name")
    argParser.add_argument("--populate_initial_experts", default=0, help="whether to populate initial experts")  # 0, 1
    argParser.add_argument("--initial_experts_num", default=None, help="number of initial experts to populate, when populate flag is 1")
    argParser.add_argument("--step_length_factor", default=0.95, help="step length *= step_length_factor every iteration")
    argParser.add_argument("--minimum_step_length", default=0.1, help="minimum step length")
    argParser.add_argument("--restart_stray_candidates", default=1, help="whether to restart stray candidates")  # 0, 1
    argParser.add_argument("--restart_patience", default=0.5, help="restart patience * patience = when to restart candidates")
    argParser.add_argument("--clean_up_on_end", default=1, help="whether to clean up on end")  # 0, 1
    argParser.add_argument("--only_one_or_two", default=None, help="whether to only optimize with dataset 1 or 2 in multitask")
    argParser.add_argument("--to_visualize", default=False, help="whether to visualize the search process")  # 0, 1
    argParser.add_argument("--correctness_emergence", default=False, help="whether to track correctness changes wrt iteration")  # 0, 1
    argParser.add_argument("--dropK", default=0, help="dropout-K, 0-1")
    argParser.add_argument("--dropN", default=0, help="dropout-N, 0-1")
    argParser.add_argument("--exploration_prob", default=0.1, help="probability of pure exploration in JAYA (0-1)")
    argParser.add_argument("--merge_method", default="weighted_average",
                          help="method to use for merging models: weighted_average, task_arithmetic, ties, dare")
    argParser.add_argument("--merge_params", default="{}",
                          help="JSON string of parameters for the merge method, e.g., '{\"threshold\": 0.01}'")
    argParser.add_argument("--fitness_function", default="accuracy", 
                          help="Fitness function to optimize: accuracy, roc_auc, mcc, or combined")

    args = argParser.parse_args()
    search_pass_name = args.name
    eval_type = args.eval_type
    dataset = args.dataset
    gpus = args.gpus
    num_cpu_when_merging = int(args.num_cpu_when_merging)
    patience = int(args.patience)
    step_length = float(args.step_length)
    max_iteration = int(args.max_iteration)
    initial_expert_directory = args.initial_expert_directory
    base_model = args.base_model
    starting_test_set_eval = int(args.starting_test_set_eval)
    fast_merge = int(args.fast_merge)
    project_name_wb = args.project_name_wb
    populate_initial_experts = int(args.populate_initial_experts)
    try:
        initial_experts_num = int(args.initial_experts_num)
    except:
        initial_experts_num = None
    step_length_factor = float(args.step_length_factor)
    minimum_step_length = float(args.minimum_step_length)
    restart_stray_candidates = int(args.restart_stray_candidates)
    restart_patience = float(args.restart_patience)
    clean_up_on_end = int(args.clean_up_on_end)
    only_one_or_two = args.only_one_or_two
    update_only_one_or_two(only_one_or_two)
    to_visualize_flag = args.to_visualize
    correctness_emergence = args.correctness_emergence
    dropK = float(args.dropK)
    dropN = float(args.dropN)
    exploration_prob = float(args.exploration_prob)
    merge_method = args.merge_method
    fitness_function = args.fitness_function
    try:
        merge_params = json.loads(args.merge_params)
    except json.JSONDecodeError:
        log_with_flush("Error parsing merge_params JSON. Using empty dict.")
        merge_params = {}

    search_pass_name += ("_" + socket.gethostname())
    args.name = search_pass_name

    perplexity_extrinsic_test_dict = {
        "legal": ["hearsay", "citation_prediction_classification"],
        "medical": ["medqa", "medmcqa"],
        "science": ["scifact", "stem"],
        "culture": ["normad_country", "normad_value"]
    }
    
    # Create search directory
    if os.path.exists(os.path.join("search", search_pass_name)):
        search_pass_name += current_time_string().replace(" ", "_")
    os.mkdir(os.path.join("search", search_pass_name))

    # Write args to file
    with open(os.path.join("search", args.name, "args.txt"), "w") as f:
        f.write(str(args))

    run = wandb.init(name=search_pass_name, project=project_name_wb)
    run.config.update(args)
    torch.multiprocessing.set_start_method('spawn')
    random.seed(42)
    
    # Configure logging to write to a file
    logging.basicConfig(filename=os.path.join("search", search_pass_name, "log.txt"), level=logging.DEBUG)
    log_with_flush("=== JAYA ALGORITHM FOR MODEL OPTIMIZATION ===")
    log_with_flush(f"Fitness function: {fitness_function}")
    log_with_flush(f"Exploration probability: {exploration_prob}")
    log_with_flush(f"Step length: {step_length}")
    log_with_flush(f"Step length factor: {step_length_factor}")

    # Parse GPU IDs and validate against available GPUs
    available_gpu_count = torch.cuda.device_count()
    available_gpu_ids = list(range(available_gpu_count))
    log_with_flush(f"Available GPUs: {available_gpu_count} (IDs: {available_gpu_ids})")
    
    # Parse requested GPUs
    try:
        if isinstance(gpus, list):
            requested_gpus = [int(gpu) for gpu in gpus]
        else:
            requested_gpus = [int(gpu) for gpu in gpus.split(",") if gpu.strip()]
        log_with_flush(f"Requested GPUs: {requested_gpus}")
    except ValueError as e:
        log_with_flush(f"ERROR: Invalid GPU specification: {gpus}. Must be comma-separated integers.")
        log_with_flush(f"Defaulting to GPU 0 if available.")
        requested_gpus = [0]
    
    # Filter to only use available GPUs
    valid_gpus = [gpu for gpu in requested_gpus if gpu < available_gpu_count]
    invalid_gpus = [gpu for gpu in requested_gpus if gpu >= available_gpu_count]
    
    if invalid_gpus:
        log_with_flush(f"WARNING: Requested GPU IDs {invalid_gpus} are not available and will be ignored.")
    
    if not valid_gpus:
        log_with_flush("WARNING: No valid GPUs specified. Defaulting to GPU 0 if available.")
        valid_gpus = [0] if available_gpu_count > 0 else []
        if not valid_gpus:
            log_with_flush("ERROR: No GPUs available on this system!")
            sys.exit(1)
    
    gpus = valid_gpus
    log_with_flush(f"Using GPUs: {gpus}")
    candidate_paths = []
    for candidate_path in os.listdir(initial_expert_directory):
        if os.path.isdir(os.path.join(initial_expert_directory, candidate_path)):
            candidate_paths.append(os.path.join(initial_expert_directory, candidate_path))
    candidate_paths = sorted(candidate_paths)

    # Populate initial experts
    if populate_initial_experts and initial_experts_num and len(candidate_paths) < initial_experts_num:
        log_with_flush("JAYA: Populating initial experts...")
        log_with_flush("previously " + str(len(candidate_paths)) + " experts")
        log_with_flush("now " + str(initial_experts_num))
        log_with_flush("adding " + str(initial_experts_num - len(candidate_paths)) + " experts")

        os.mkdir(os.path.join("search", search_pass_name, "tmp"))
        candidates_now = len(candidate_paths)
        for i in range(initial_experts_num - candidates_now):
            parent_1 = random.choice(candidate_paths)
            parent_2 = random.choice(candidate_paths)
            while parent_1 == parent_2:
                parent_2 = random.choice(candidate_paths)
            child_path = os.path.join("search", search_pass_name, "tmp", "child_"+str(i))
            w_1 = random.random() * 2  # half interpolation, half extrapolation
            w_2 = 1 - w_1
            shutil.copytree(parent_1, child_path)
            lora_merge([w_1, w_2], [parent_1, parent_2], child_path, gpus[0], fast_merge, merge_method, merge_params)
            candidate_paths.append(child_path)

    if correctness_emergence:
        correctness_emergence_dict = {}
        for i in range(len(candidate_paths)):
            correctness_emergence_dict[i] = []

    if to_visualize_flag:
        candidate_trajectory = {}
        for i in range(len(candidate_paths)):
            candidate_trajectory[i] = []

    log_with_flush("JAYA: Initializing search... "+current_time_string())
    initialize_search_records(search_pass_name, candidate_paths, eval_type, dataset, gpus, base_model, fast_merge, fitness_function)
    log_with_flush("JAYA: Search initialized")
    for i in range(len(candidate_paths)):
        log_with_flush("expert " + str(i) + ": " + candidate_paths[i])

    if os.path.exists(os.path.join("search", search_pass_name, "tmp")):
        shutil.rmtree(os.path.join("search", search_pass_name, "tmp"))

    # Test set evaluation
    if starting_test_set_eval:
        eval_test_args = []
        for i in range(len(candidate_paths)):
            eval_test_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                                  eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                                  base_model, fitness_function))

        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
        pool.close()
        pool.join()

        log_with_flush("JAYA: Test set results:")
        for i in range(len(candidate_paths)):
            log_with_flush("candidate_"+str(i)+": "+str(results[i]))

    log_with_flush("JAYA: Starting search... "+current_time_string())
    log_with_flush("=== JAYA ALGORITHM PROCESS ===")
    log_with_flush("1. Generate new candidates: X_new = X_old + r1*(X_best - |X_old|) - r2*(X_worst - |X_old|)")
    log_with_flush("2. Evaluate both old and new candidates")
    log_with_flush("3. GREEDY SELECTION: Keep new candidate only if better than old")
    log_with_flush("4. Update best/worst solutions globally")
    log_with_flush("=" * 50)

    # Main JAYA search iteration
    iter_count = 0
    while iter_count < max_iteration:
        iter_count += 1
        log_with_flush("--------------------------")
        log_with_flush(f"JAYA ITERATION {iter_count}! "+current_time_string())
        log_with_flush("JAYA: Updating candidates...")

        # Patience and ending condition
        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
            utility_scratchpad = json.load(f)
        best_score = utility_scratchpad["best"]
        score_history = utility_scratchpad["history"]
        if len(score_history) > patience:
            score_history = score_history[-patience:]
            # If score_history hasn't changed
            if max(score_history) == min(score_history):
                log_with_flush("JAYA: Patience reached!")
                break

        if to_visualize_flag:
            for i in range(len(candidate_paths)):
                lora_weight_path = os.path.join("search", search_pass_name, "candidate_"+str(i), "adapter_model.safetensors")
                coords = lora_weight_visualize(lora_weight_path)
                candidate_trajectory[i].append(coords)
            with open(os.path.join("search", search_pass_name, "candidate_trajectory.json"), "w") as f:
                json.dump(candidate_trajectory, f, indent=4)
        
        if correctness_emergence:
            for i in range(len(candidate_paths)):
                model_path = os.path.join("search", search_pass_name, "candidate_"+str(i))
                golds = json.load(open(os.path.join(model_path, "golds_dev.json"), "r"))
                preds = json.load(open(os.path.join(model_path, "preds_dev.json"), "r"))
                correctness = []
                assert len(golds) == len(preds)
                for j in range(len(golds)):
                    if golds[j] == preds[j]:
                        correctness.append(1)
                    else:
                        correctness.append(0)
                correctness_emergence_dict[i].append(correctness)
            
            with open(os.path.join("search", search_pass_name, "correctness_emergence.json"), "w") as f:
                json.dump(correctness_emergence_dict, f, indent=4)
        
        # Update each candidate using JAYA algorithm
        update_args = []
        for i in range(len(candidate_paths)):
            if restart_stray_candidates:
                candidate_history = utility_scratchpad["candidate_"+str(i)+"_history"]
                candidate_best_score = max(candidate_history)
                first_time_best_idx = candidate_history.index(candidate_best_score)
                if len(candidate_history) - first_time_best_idx >= restart_patience * patience:
                    restart_flag = True
                    log_with_flush(f"JAYA: candidate_{i} restarted!")
                else:
                    restart_flag = False
            else:
                restart_flag = False

            update_args.append((i, gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                               search_pass_name, fast_merge, step_length, restart_flag,
                               exploration_prob, merge_method, merge_params))

        pool = Pool(processes=num_cpu_when_merging)
        results = pool.starmap(jaya_candidate_update, update_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
        pool.close()
        pool.join()
        log_with_flush("JAYA: All candidates updated! "+current_time_string())

        # Evaluate each candidate and update utility_scratchpad and weights
        log_with_flush("JAYA: Evaluating candidates...")

        if random.random() < dropK:  # iteration drop
            log_with_flush("JAYA: Dropped iteration!")
            global_skip_flag = True
        else:
            global_skip_flag = False

        eval_args = []
        for i in range(len(candidate_paths)):
            if random.random() < dropN:  # candidate drop
                local_skip_flag = True
            else:
                local_skip_flag = False

            if not correctness_emergence:
                eval_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                                 eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                                 base_model, False, None, global_skip_flag or local_skip_flag, fitness_function))
            else:
                eval_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                                 eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                                 base_model, True, None, global_skip_flag or local_skip_flag, fitness_function))
        
        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
        pool.close()
        pool.join()

        # Now evaluate the new candidate versions
        log_with_flush("JAYA: Evaluating new candidate versions...")
        eval_new_args = []
        new_candidate_indices = []  # Track which candidates have new versions
        
        for i in range(len(candidate_paths)):
            new_candidate_path = os.path.join("search", search_pass_name, "candidate_"+str(i)+"_new")
            if os.path.exists(new_candidate_path):
                eval_new_args.append((new_candidate_path,
                                    eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                                    base_model, False, None, global_skip_flag or local_skip_flag, fitness_function))
                new_candidate_indices.append(i)
        
        # Only evaluate if there are new candidates
        new_results = []
        if eval_new_args:
            pool = Pool(processes=min(len(gpus), len(eval_new_args)))
            new_results = pool.starmap(evaluate, eval_new_args, chunksize=math.ceil(len(eval_new_args)/len(gpus)))
            pool.close()
            pool.join()
        
        with open("search/"+search_pass_name+"/utility_scratchpad.json", "r") as f:
            utility_scratchpad = json.load(f)

        # If skipped, pull performance from last step
        for i in range(len(candidate_paths)):
            if results[i] is None:
                results[i] = utility_scratchpad["candidate_"+str(i)]
                assert results[i] == utility_scratchpad["candidate_"+str(i)+"_history"][-1]
        
        # JAYA GREEDY SELECTION: Accept new solution only if it's better
        log_with_flush("=== JAYA GREEDY SELECTION: COMPARING OLD AND NEW VERSIONS ===")
        
        # Create a mapping from candidate index to its result index
        new_result_map = {}
        for idx, candidate_idx in enumerate(new_candidate_indices):
            new_result_map[candidate_idx] = idx
        
        for i in range(len(candidate_paths)):
            new_candidate_path = os.path.join("search", search_pass_name, "candidate_"+str(i)+"_new")
            old_candidate_path = os.path.join("search", search_pass_name, "candidate_"+str(i))
            
            if os.path.exists(new_candidate_path) and i in new_result_map:
                new_score = new_results[new_result_map[i]]
                old_score = results[i]
                
                if new_score is None:
                    # If evaluation was skipped, assume no improvement
                    log_with_flush(f"JAYA: Candidate_{i} evaluation skipped, keeping old version")
                    shutil.rmtree(new_candidate_path)
                    utility_scratchpad["candidate_" + str(i)] = old_score
                    utility_scratchpad["candidate_" + str(i) + "_history"].append(old_score)
                elif new_score > old_score:
                    # GREEDY SELECTION: New version is better, accept it (replace old with new)
                    log_with_flush(f"JAYA GREEDY: Candidate_{i} improved: {old_score} -> {new_score} ✓ ACCEPTED")
                    shutil.rmtree(old_candidate_path)
                    os.rename(new_candidate_path, old_candidate_path)
                    utility_scratchpad["candidate_" + str(i)] = new_score
                    utility_scratchpad["candidate_" + str(i) + "_history"].append(new_score)
                    results[i] = new_score  # Update results for best/worst tracking
                else:
                    # GREEDY SELECTION: Old version is better or equal, reject new candidate
                    log_with_flush(f"JAYA GREEDY: Candidate_{i} did not improve: {old_score} vs {new_score} ✗ REJECTED")
                    shutil.rmtree(new_candidate_path)
                    utility_scratchpad["candidate_" + str(i)] = old_score
                    utility_scratchpad["candidate_" + str(i) + "_history"].append(old_score)
            else:
                # No new version exists (possibly due to restart or other reasons)
                utility_scratchpad["candidate_" + str(i)] = results[i]
                utility_scratchpad["candidate_" + str(i) + "_history"].append(results[i])
        
        # Best model update
        if max(results) > utility_scratchpad["best"]:
            utility_scratchpad["best"] = max(results)
            utility_scratchpad["history"].append(max(results))
            log_with_flush(f"JAYA: New best model found: {utility_scratchpad['best']}")
            for i in range(len(candidate_paths)):
                if results[i] == utility_scratchpad["best"]:
                    shutil.copytree(os.path.join("search", search_pass_name, "candidate_"+str(i)),
                                   os.path.join("search", search_pass_name, "best"),
                                   dirs_exist_ok=True)
                    break
        else:
            utility_scratchpad["history"].append(utility_scratchpad["best"])

        # Worst model update
        if min(results) < utility_scratchpad["worst"]:
            utility_scratchpad["worst"] = min(results)
            for i in range(len(candidate_paths)):
                if results[i] == utility_scratchpad["worst"]:
                    shutil.copytree(os.path.join("search", search_pass_name, "candidate_"+str(i)),
                                   os.path.join("search", search_pass_name, "worst"),
                                   dirs_exist_ok=True)

        wandb_log = {
            "best": utility_scratchpad["best"],
            "worst": utility_scratchpad["worst"],
        }
        for i in range(len(candidate_paths)):
            wandb_log["candidate_" + str(i)] = utility_scratchpad["candidate_" + str(i)]
        
        wandb.log(wandb_log)
        
        with open("search/"+search_pass_name+"/utility_scratchpad.json", "w") as f:
            json.dump(utility_scratchpad, f, indent=4)
        
        log_with_flush("JAYA: All candidates evaluated! "+current_time_string())
        log_with_flush("--------------------------")

        # Step length update (adaptive step size)
        step_length = max(step_length * step_length_factor, minimum_step_length)
        log_with_flush(f"JAYA: Updated step length to {step_length}")

    log_with_flush("JAYA: Ending search and starting test set evaluation... "+current_time_string())

    # Which candidate is the best?
    with open("search/"+search_pass_name+"/utility_scratchpad.json", "r") as f:
        utility_scratchpad = json.load(f)
    best_score = utility_scratchpad["best"]
    for i in range(len(candidate_paths)):
        if utility_scratchpad["candidate_" + str(i)] == best_score:
            best_candidate = i
    log_with_flush(f"JAYA: Best candidate: {best_candidate}")

    # Dev set evaluation for all candidates
    eval_args = []
    for i in range(len(candidate_paths)):
        eval_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                         eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                         base_model, True, None, False, fitness_function))

    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
    pool.close()
    pool.join()

    # Test set evaluation
    eval_test_args = []
    for i in range(len(candidate_paths)):
        eval_test_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                              eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                              base_model, fitness_function))

    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
    pool.close()
    pool.join()

    log_with_flush("JAYA: Test set results:")
    for i in range(len(candidate_paths)):
        log_with_flush("candidate_"+str(i)+": "+str(results[i]))

    final_metrics = overall_metrics(search_pass_name, eval_type)

    if eval_type == "AbstainQA":
        best_candidate_idx = final_metrics["ending_best_candidate_on_validation"]
        final_metrics["ending_best_single_test_accuracy"] = results[best_candidate_idx]
    
    if eval_type == "perplexity" or eval_type == "multitask":
        dataset_1_name = perplexity_extrinsic_test_dict[dataset][0]
        eval_test_args = []
        for i in range(len(candidate_paths)):
            eval_test_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                                  "multiple_choice", dataset_1_name,
                                  gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                                  base_model, fitness_function))
        
        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_test_" + dataset_1_name] = results[final_metrics["ending_best_candidate_on_validation"]]

        dataset_2_name = perplexity_extrinsic_test_dict[dataset][1]
        eval_test_args = []
        for i in range(len(candidate_paths)):
            eval_test_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                                  "multiple_choice", dataset_2_name,
                                  gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                                  base_model, fitness_function))
        
        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_test_" + dataset_2_name] = results[final_metrics["ending_best_candidate_on_validation"]]

    if eval_type == "multitask":
        dataset_1_name = perplexity_extrinsic_test_dict[dataset][0]
        eval_args = []
        for i in range(len(candidate_paths)):
            eval_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                              "multiple_choice", dataset_1_name,
                              gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                              base_model, True, None, False, fitness_function))
        
        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_dev_" + dataset_1_name] = results[final_metrics["ending_best_candidate_on_validation"]]

        dataset_2_name = perplexity_extrinsic_test_dict[dataset][1]
        eval_args = []
        for i in range(len(candidate_paths)):
            eval_args.append((os.path.join("search", search_pass_name, "candidate_"+str(i)),
                              "multiple_choice", dataset_2_name,
                              gpus[assign_gpu(len(gpus), i, len(candidate_paths))],
                              base_model, True, None, False, fitness_function))

        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(candidate_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_dev_" + dataset_2_name] = results[final_metrics["ending_best_candidate_on_validation"]]

    wandb.log(final_metrics)
    log_with_flush("JAYA: Final metrics for test: "+str(final_metrics))

    # Ensemble for dev set
    try:
        for i in range(len(candidate_paths)):
            os.remove(os.path.join("search", search_pass_name, "candidate_"+str(i), "golds.json"))
            os.remove(os.path.join("search", search_pass_name, "candidate_"+str(i), "preds.json"))

            os.rename(os.path.join("search", search_pass_name, "candidate_"+str(i), "golds_dev.json"),
                      os.path.join("search", search_pass_name, "candidate_"+str(i), "golds.json"))
            os.rename(os.path.join("search", search_pass_name, "candidate_"+str(i), "preds_dev.json"),
                      os.path.join("search", search_pass_name, "candidate_"+str(i), "preds.json"))
            
            # Also rename probs file if it exists (for ROC-AUC calculation)
            try:
                os.remove(os.path.join("search", search_pass_name, "candidate_"+str(i), "probs.json"))
                os.rename(os.path.join("search", search_pass_name, "candidate_"+str(i), "probs_dev.json"),
                          os.path.join("search", search_pass_name, "candidate_"+str(i), "probs.json"))
            except:
                pass
    except:
        for i in range(len(candidate_paths)):
            os.remove(os.path.join("search", search_pass_name, "candidate_"+str(i), "scores.json"))
            os.rename(os.path.join("search", search_pass_name, "candidate_"+str(i), "scores_dev.json"),
                      os.path.join("search", search_pass_name, "candidate_"+str(i), "scores.json"))

    final_metrics = overall_metrics(search_pass_name, eval_type)
    dev_final_metrics = {
        "starting_top-k_ensemble_dev_accuracy": final_metrics["starting_top-k_ensemble_test_accuracy"],
        "ending_top-k_ensemble_dev_accuracy": final_metrics["ending_top-k_ensemble_test_accuracy"],
        "starting_top-k_ensemble_dev_roc_auc": final_metrics.get("starting_top-k_ensemble_test_roc_auc", 0.5),
        "ending_top-k_ensemble_dev_roc_auc": final_metrics.get("ending_top-k_ensemble_test_roc_auc", 0.5),
        "starting_top-k_ensemble_dev_mcc": final_metrics.get("starting_top-k_ensemble_test_mcc", 0.0),
        "ending_top-k_ensemble_dev_mcc": final_metrics.get("ending_top-k_ensemble_test_mcc", 0.0)
    }
    wandb.log(dev_final_metrics)
    log_with_flush("JAYA: Final ensemble metrics for dev: "+str(dev_final_metrics))

    if clean_up_on_end:
        shutil.rmtree(os.path.join("search", search_pass_name, "worst"))
        for i in range(len(candidate_paths)):
            try:
                shutil.rmtree(os.path.join("search", search_pass_name, "temp_"+str(i)))
            except:
                pass

    log_with_flush("JAYA: The end of search... "+current_time_string())
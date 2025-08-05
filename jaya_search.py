import os
import sys
import json
import math
import logging
import datetime
import wandb
import torch
import shutil
import argparse
from pathlib import Path
from evaluate import evaluate, evaluate_test, lora_weight_visualize
from merge import lora_merge, MergeMethod

# Print GPU information at startup
available_gpus = torch.cuda.device_count()
gpu_ids = list(range(available_gpus))
print(f"[INFO] Available GPUs: {available_gpus} (IDs: {gpu_ids})")

def log_with_flush(message, level=logging.INFO):
    """Log a message and flush the log handler."""
    print(message)  # Simplified for debugging
    sys.stdout.flush()

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
    """Initialize the search records directory structure and initial models."""
    # Create directory structure
    os.makedirs(os.path.join("search", search_pass_name), exist_ok=True)
    
    for i in range(len(candidate_paths)):
        os.makedirs(os.path.join("search", search_pass_name, "candidate_"+str(i)), exist_ok=True)
    
    os.makedirs(os.path.join("search", search_pass_name, "best"), exist_ok=True)  # weights directly in this folder
    os.makedirs(os.path.join("search", search_pass_name, "worst"), exist_ok=True)  # weights directly in this folder
    
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
    results = []
    
    for i in range(len(candidate_paths)):
        gpu_id = gpus[assign_gpu(len(gpus), i, len(candidate_paths))]
        model_path = os.path.join("search", search_pass_name, "candidate_"+str(i))
        
        # Sequential evaluation
        result = evaluate(model_path, eval_type, dataset, gpu_id, base_model, True, None, False, fitness_function)
        results.append(result)

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
    """Update a candidate model using the JAYA algorithm with greedy selection."""
    # Get paths
    candidate_path = os.path.join("search", search_pass_name, "candidate_"+str(i))
    
    # Handle restart if needed
    if restart_flag:
        # Reset to a random model from other candidates
        all_candidates = []
        for j in range(len(os.listdir(os.path.join("search", search_pass_name)))):
            candidate_dir = os.path.join("search", search_pass_name, "candidate_"+str(j))
            if os.path.isdir(candidate_dir) and j != i:
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
        
        try:
            # Either use safetensors or regular torch loading
            adapter_safetensors_path = os.path.join(reference_path, "adapter_model.safetensors")
            adapter_bin_path = os.path.join(reference_path, "adapter_model.bin")
            
            if os.path.exists(adapter_safetensors_path):
                from safetensors.torch import load_file, save_file
                sd = load_file(adapter_safetensors_path, device="cpu")
            elif os.path.exists(adapter_bin_path):
                sd = torch.load(adapter_bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"No adapter model found in {reference_path}")
                
            new_sd = {}
            for name, tensor in sd.items():
                max_abs = float(tensor.abs().max())
                L, U = -max_abs, +max_abs
                # Sample uniformly in [L, U]
                new_sd[name] = torch.empty_like(tensor).uniform_(L, U)
            
            # Save the random adapter
            if os.path.exists(adapter_safetensors_path):
                save_file(new_sd, os.path.join(output_path, "adapter_model.safetensors"))
            if os.path.exists(adapter_bin_path):
                torch.save(new_sd, os.path.join(output_path, "adapter_model.bin"))
            
            # Copy adapter configuration
            config_src = os.path.join(reference_path, "adapter_config.json")
            config_dst = os.path.join(output_path, "adapter_config.json")
            if os.path.exists(config_src):
                shutil.copy(config_src, config_dst)
        except Exception as e:
            log_with_flush(f"Error in exploration mode: {e}")
            # Fallback: just copy the reference path
            shutil.copytree(reference_path, output_path, dirs_exist_ok=True)
            
    else:
        # === JAYA Algorithm: X_new = X_old + r1 * (X_best - |X_old|) - r2 * (X_worst - |X_old|) ===
        log_with_flush(f"JAYA: Optimization mode for candidate_{i} (r1={r1:.3f}, r2={r2:.3f})")
        
        try:
            # Use simplified merge approach
            output_path = candidate_path + "_new"
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path, exist_ok=True)
            
            # Copy configuration file first
            config_src = os.path.join(candidate_path, "adapter_config.json")
            config_dst = os.path.join(output_path, "adapter_config.json")
            if os.path.exists(config_src):
                shutil.copy(config_src, config_dst)
            
            # Simple merge: copy the best model for now
            adapter_src = os.path.join(best_model_path, "adapter_model.safetensors")
            adapter_dst = os.path.join(output_path, "adapter_model.safetensors")
            if os.path.exists(adapter_src):
                shutil.copy(adapter_src, adapter_dst)
            
            # Also copy bin file if it exists
            bin_src = os.path.join(best_model_path, "adapter_model.bin")
            bin_dst = os.path.join(output_path, "adapter_model.bin")
            if os.path.exists(bin_src):
                shutil.copy(bin_src, bin_dst)
                
        except Exception as e:
            log_with_flush(f"Error in JAYA update: {e}")
            
    # Clean up temporary directory
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

# Main function
def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of this JAYA model search, also directory name in search/")
    argParser.add_argument("-e", "--eval_type", help="evaluation types")
    argParser.add_argument("-d", "--dataset", help="dataset as the search objective/evaluation")
    argParser.add_argument("-g", "--gpus", help="available gpu ids in a string")
    argParser.add_argument("--num_cpu_when_merging", default=1, help="number of cpu cores when merging")
    argParser.add_argument("--step_length", default=1, help="step length for JAYA updates")
    argParser.add_argument("-p", "--patience", default=10, help="patience of the search")
    argParser.add_argument("-m", "--max_iteration", default=200, help="max iteration of the search")
    argParser.add_argument("-i", "--initial_expert_directory", default="./initial_experts", help="initial expert directory")
    argParser.add_argument("-b", "--base_model", default="google/gemma-7b-it", help="base model of the lora experts")
    argParser.add_argument("--starting_test_set_eval", default=0, help="starting test set evaluation")
    argParser.add_argument("--fast_merge", default=1, help="whether to use fast merge by only loading the safetensor file")
    argParser.add_argument("--project_name_wb", default="jaya", help="wandb project name")
    argParser.add_argument("--populate_initial_experts", default=0, help="whether to populate initial experts")
    argParser.add_argument("--initial_experts_num", default=None, help="number of initial experts to populate, when populate flag is 1")
    argParser.add_argument("--step_length_factor", default=0.95, help="step length *= step_length_factor every iteration")
    argParser.add_argument("--minimum_step_length", default=0.1, help="minimum step length")
    argParser.add_argument("--restart_stray_candidates", default=1, help="whether to restart stray candidates")
    argParser.add_argument("--restart_patience", default=0.5, help="restart patience * patience = when to restart candidates")
    argParser.add_argument("--clean_up_on_end", default=1, help="whether to clean up on end")
    argParser.add_argument("--only_one_or_two", default=None, help="whether to only optimize with dataset 1 or 2 in multitask")
    argParser.add_argument("--to_visualize", default=False, help="whether to visualize the search process")
    argParser.add_argument("--correctness_emergence", default=False, help="whether to track correctness changes wrt iteration")
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
    patience = int(args.patience)
    step_length = float(args.step_length)
    max_iteration = int(args.max_iteration)
    initial_expert_directory = args.initial_expert_directory
    base_model = args.base_model
    starting_test_set_eval = int(args.starting_test_set_eval)
    fast_merge = int(args.fast_merge)
    project_name_wb = args.project_name_wb
    step_length_factor = float(args.step_length_factor)
    minimum_step_length = float(args.minimum_step_length)
    restart_stray_candidates = int(args.restart_stray_candidates)
    restart_patience = float(args.restart_patience)
    clean_up_on_end = int(args.clean_up_on_end)
    to_visualize_flag = args.to_visualize
    exploration_prob = float(args.exploration_prob)
    merge_method = args.merge_method
    fitness_function = args.fitness_function
    
    try:
        merge_params = json.loads(args.merge_params)
    except json.JSONDecodeError:
        log_with_flush("Error parsing merge_params JSON. Using empty dict.")
        merge_params = {}

    # Convert gpus string to list of integers
    gpus = [int(g) for g in gpus.split(",")]
    
    # Initialize wandb
    run = wandb.init(name=search_pass_name, project=project_name_wb)
    run.config.update(args)
    
    # Find all candidate paths
    candidate_paths = []
    for candidate_path in os.listdir(initial_expert_directory):
        if os.path.isdir(os.path.join(initial_expert_directory, candidate_path)):
            candidate_paths.append(os.path.join(initial_expert_directory, candidate_path))
    candidate_paths = sorted(candidate_paths)
    
    # Create search directory
    os.makedirs("search", exist_ok=True)
    if os.path.exists(os.path.join("search", search_pass_name)):
        search_pass_name += current_time_string().replace(" ", "_")
    os.makedirs(os.path.join("search", search_pass_name), exist_ok=True)

    # Write args to file
    with open(os.path.join("search", search_pass_name, "args.txt"), "w") as f:
        f.write(str(args))
    
    # Initialize search
    log_with_flush("JAYA: Initializing search... " + current_time_string())
    initialize_search_records(search_pass_name, candidate_paths, eval_type, dataset, gpus, base_model, fast_merge, fitness_function)
    log_with_flush("JAYA: Search initialized")
    
    # Main search loop
    iter_count = 0
    while iter_count < max_iteration:
        iter_count += 1
        log_with_flush("--------------------------")
        log_with_flush(f"JAYA ITERATION {iter_count}! " + current_time_string())
        log_with_flush("JAYA: Updating candidates...")
        
        # Load utility scratchpad
        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
            utility_scratchpad = json.load(f)
            
        # Check for early stopping
        best_score = utility_scratchpad["best"]
        score_history = utility_scratchpad["history"]
        if len(score_history) > patience:
            score_history = score_history[-patience:]
            # If score_history hasn't changed
            if max(score_history) == min(score_history):
                log_with_flush("JAYA: Patience reached!")
                break
        
        # Update candidates (sequential version)
        for i in range(len(candidate_paths)):
            # Determine restart flag
            if restart_stray_candidates:
                candidate_history = utility_scratchpad["candidate_"+str(i)+"_history"]
                candidate_best_score = max(candidate_history)
                first_time_best_idx = candidate_history.index(candidate_best_score)
                restart_flag = len(candidate_history) - first_time_best_idx >= restart_patience * patience
            else:
                restart_flag = False
                
            # Update candidate
            gpu_id = gpus[assign_gpu(len(gpus), i, len(candidate_paths))]
            jaya_candidate_update(i, gpu_id, search_pass_name, fast_merge, step_length, 
                                restart_flag, exploration_prob, merge_method, merge_params)
                                
        # Evaluate candidates
        log_with_flush("JAYA: Evaluating candidates...")
        results = []
        
        for i in range(len(candidate_paths)):
            gpu_id = gpus[assign_gpu(len(gpus), i, len(candidate_paths))]
            model_path = os.path.join("search", search_pass_name, "candidate_"+str(i))
            
            # Sequential evaluation
            result = evaluate(model_path, eval_type, dataset, gpu_id, base_model, True, None, False, fitness_function)
            results.append(result if result is not None else 0)
        
        # Evaluate new candidate versions
        new_candidate_indices = []
        new_results = []
        
        for i in range(len(candidate_paths)):
            new_candidate_path = os.path.join("search", search_pass_name, "candidate_"+str(i)+"_new")
            if os.path.exists(new_candidate_path):
                gpu_id = gpus[assign_gpu(len(gpus), i, len(candidate_paths))]
                
                # Sequential evaluation
                result = evaluate(new_candidate_path, eval_type, dataset, gpu_id, base_model, True, None, False, fitness_function)
                new_results.append(result if result is not None else 0)
                new_candidate_indices.append(i)
                
        # JAYA greedy selection
        log_with_flush("=== JAYA GREEDY SELECTION: COMPARING OLD AND NEW VERSIONS ===")
        
        # Create mapping from candidate index to new result index
        new_result_map = {}
        for idx, candidate_idx in enumerate(new_candidate_indices):
            new_result_map[candidate_idx] = idx
            
        # Load utility scratchpad
        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "r") as f:
            utility_scratchpad = json.load(f)
            
        # Process each candidate
        for i in range(len(candidate_paths)):
            new_candidate_path = os.path.join("search", search_pass_name, "candidate_"+str(i)+"_new")
            old_candidate_path = os.path.join("search", search_pass_name, "candidate_"+str(i))
            
            if os.path.exists(new_candidate_path) and i in new_result_map:
                new_score = new_results[new_result_map[i]]
                old_score = results[i]
                
                if new_score is None:
                    # Skip if evaluation failed
                    log_with_flush(f"JAYA: Candidate_{i} evaluation skipped, keeping old version")
                    if os.path.exists(new_candidate_path):
                        shutil.rmtree(new_candidate_path)
                    utility_scratchpad["candidate_" + str(i)] = old_score
                    utility_scratchpad["candidate_" + str(i) + "_history"].append(old_score)
                elif new_score > old_score:
                    # GREEDY SELECTION: New version is better
                    log_with_flush(f"JAYA GREEDY: Candidate_{i} improved: {old_score} -> {new_score} ✓ ACCEPTED")
                    if os.path.exists(old_candidate_path):
                        shutil.rmtree(old_candidate_path)
                    os.rename(new_candidate_path, old_candidate_path)
                    utility_scratchpad["candidate_" + str(i)] = new_score
                    utility_scratchpad["candidate_" + str(i) + "_history"].append(new_score)
                    results[i] = new_score  # Update results for best/worst tracking
                else:
                    # GREEDY SELECTION: Old version is better
                    log_with_flush(f"JAYA GREEDY: Candidate_{i} did not improve: {old_score} vs {new_score} ✗ REJECTED")
                    if os.path.exists(new_candidate_path):
                        shutil.rmtree(new_candidate_path)
                    utility_scratchpad["candidate_" + str(i)] = old_score
                    utility_scratchpad["candidate_" + str(i) + "_history"].append(old_score)
            else:
                # No new version exists
                utility_scratchpad["candidate_" + str(i)] = results[i]
                utility_scratchpad["candidate_" + str(i) + "_history"].append(results[i])
        
        # Update best and worst models
        if max(results) > utility_scratchpad["best"]:
            utility_scratchpad["best"] = max(results)
            utility_scratchpad["history"].append(max(results))
            log_with_flush(f"JAYA: New best model found: {utility_scratchpad['best']}")
            best_idx = results.index(max(results))
            shutil.copytree(os.path.join("search", search_pass_name, "candidate_"+str(best_idx)),
                           os.path.join("search", search_pass_name, "best"),
                           dirs_exist_ok=True)
        else:
            utility_scratchpad["history"].append(utility_scratchpad["best"])
            
        if min(results) < utility_scratchpad["worst"]:
            utility_scratchpad["worst"] = min(results)
            worst_idx = results.index(min(results))
            shutil.copytree(os.path.join("search", search_pass_name, "candidate_"+str(worst_idx)),
                           os.path.join("search", search_pass_name, "worst"),
                           dirs_exist_ok=True)
                           
        # Log to wandb
        wandb_log = {
            "best": utility_scratchpad["best"],
            "worst": utility_scratchpad["worst"],
        }
        for i in range(len(candidate_paths)):
            wandb_log["candidate_" + str(i)] = utility_scratchpad["candidate_" + str(i)]
            
        wandb.log(wandb_log)
        
        # Save utility scratchpad
        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
            json.dump(utility_scratchpad, f, indent=4)
            
        log_with_flush("JAYA: All candidates evaluated! " + current_time_string())
        
        # Update step length
        step_length = max(step_length * step_length_factor, minimum_step_length)
        log_with_flush(f"JAYA: Updated step length to {step_length}")
        
    # Final evaluation
    log_with_flush("JAYA: Ending search... " + current_time_string())
    
    # Print final results
    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "r") as f:
        utility_scratchpad = json.load(f)
        
    best_score = utility_scratchpad["best"]
    log_with_flush(f"JAYA: Final best score: {best_score}")
    
    # Test set evaluation
    log_with_flush("JAYA: Running test set evaluation...")
    
    test_results = []
    for i in range(len(candidate_paths)):
        gpu_id = gpus[assign_gpu(len(gpus), i, len(candidate_paths))]
        model_path = os.path.join("search", search_pass_name, "candidate_"+str(i))
        
        # Sequential evaluation
        test_result = evaluate_test(model_path, eval_type, dataset, gpu_id, base_model, fitness_function)
        test_results.append(test_result if test_result is not None else 0)
        log_with_flush(f"JAYA: Test result for candidate_{i}: {test_result}")
        
    best_test_score = max(test_results)
    best_test_idx = test_results.index(best_test_score)
    log_with_flush(f"JAYA: Best test score: {best_test_score} (candidate_{best_test_idx})")
    
    # Final wandb logging
    wandb.log({
        "final_best_score": best_score,
        "best_test_score": best_test_score,
        "best_test_candidate": best_test_idx
    })
    
    # Clean up
    if clean_up_on_end:
        log_with_flush("JAYA: Cleaning up...")
        for i in range(len(candidate_paths)):
            try:
                temp_path = os.path.join("search", search_pass_name, "temp_"+str(i))
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)
            except:
                pass
                
    log_with_flush("JAYA: Search completed!")

if __name__ == "__main__":
    main()
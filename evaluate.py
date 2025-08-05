import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
from sklearn.decomposition import PCA
import re

# Global variable for multitask optimization
ONLY_ONE_OR_TWO = None

def update_only_one_or_two(only_one_or_two):
    """Update the global variable for multitask optimization."""
    global ONLY_ONE_OR_TWO
    ONLY_ONE_OR_TWO = only_one_or_two

def extract_json_from_text(text):
    """Extract JSON object from generated text."""
    # Find anything that looks like a JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def evaluate(model_path, eval_type, dataset, gpu_id, base_model, save_preds=False, suffix=None, 
             skip_flag=False, fitness_function="accuracy"):
    """
    Evaluate a model and calculate metrics including ROC-AUC and MCC.
    """
    if skip_flag:
        return None
    
    # Load the dataset
    if eval_type == "multiple_choice" or eval_type == "AbstainQA" or eval_type == "fraud_detection":
        # Load data for classification tasks
        try:
            # Load a single JSON file with both dev and test data
            with open(f"data/eval/{dataset}.json", "r") as f:
                data = json.load(f)
            
            # Extract data based on suffix
            split_name = "test" if suffix and "_test" in suffix else "dev"
            split_data = data[split_name]
            
            # Extract texts and labels from your custom format
            texts = [item["prompt"] for item in split_data]
            
            # Handle the "lable" field (with typo)
            labels = []
            for item in split_data:
                if isinstance(item["lable"], str):
                    labels.append(1 if item["lable"].lower() in ["1", "true", "yes"] else 0)
                else:
                    labels.append(1 if item["lable"] else 0)
            
            # Keep the options handling the same (if needed)
            options = data.get("options", None)
        except Exception as e:
            print(f"Error loading dataset: data/eval/{dataset}.json")
            print(f"Error details: {e}")
            return 0
        
        # Load model and tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map={"": gpu_id}
            )
            model.load_adapter(model_path)
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return 0
        
        # Make predictions
        preds = []
        probabilities = []  # For ROC-AUC calculation
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                if options is not None:
                    # Multiple choice prediction (keep original implementation)
                    scores = []
                    for option in options[i]:
                        prompt = text + "\n" + option
                        inputs = tokenizer(prompt, return_tensors="pt").to(gpu_id)
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                        
                        token_id = outputs.sequences[0, -1].item()
                        score = outputs.scores[0][0, token_id].item()
                        scores.append(score)
                    
                    pred_idx = np.argmax(scores)
                    preds.append(pred_idx)
                    
                    if len(options[i]) == 2:
                        softmax_scores = torch.softmax(torch.tensor(scores), dim=0)
                        probabilities.append(softmax_scores[1].item())
                else:
                    # MODIFIED: Generate complete JSON response
                    inputs = tokenizer(text, return_tensors="pt").to(gpu_id)
                    
                    # Generate a longer response that includes the JSON
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,  # Increased to capture full JSON response
                        temperature=0.0,     # Deterministic output
                        do_sample=False
                    )
                    
                    # Decode the generated text
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract JSON from generated text
                    result_json = extract_json_from_text(generated_text)
                    
                    if result_json and "prediction" in result_json and "prob" in result_json:
                        # Extract prediction (0 or 1)
                        preds.append(result_json["prediction"])
                        
                        # Extract probability directly from the model output
                        probabilities.append(float(result_json["prob"]))
                    else:
                        # Fallback if JSON extraction fails
                        print(f"Warning: Failed to extract valid JSON from output for example {i}")
                        preds.append(0)  # Default prediction
                        probabilities.append(0.5)  # Default probability
        
        # Save predictions if needed
        if save_preds:
            preds_file = os.path.join(model_path, f"preds{suffix or ''}.json")
            golds_file = os.path.join(model_path, f"golds{suffix or ''}.json")
            
            with open(preds_file, "w") as f:
                json.dump(preds, f)
            with open(golds_file, "w") as f:
                json.dump(labels, f)
            
            # For fraud detection, also save probabilities
            if eval_type == "fraud_detection":
                probs_file = os.path.join(model_path, f"probs{suffix or ''}.json")
                with open(probs_file, "w") as f:
                    json.dump(probabilities, f)
        
        # Calculate metrics
        if eval_type == "fraud_detection":
            # Convert labels and predictions to numeric for binary classification
            numeric_labels = [1 if l in [1, True, "1", "True"] else 0 for l in labels]
            numeric_preds = [1 if p in [1, True, "1", "True"] else 0 for p in preds]
            
            # Calculate metrics
            acc = accuracy_score(numeric_labels, numeric_preds)
            roc_auc = roc_auc_score(numeric_labels, probabilities)
            mcc = matthews_corrcoef(numeric_labels, numeric_preds)
            
            # Return metric based on fitness function
            if fitness_function == "accuracy":
                return acc
            elif fitness_function == "roc_auc":
                return roc_auc
            elif fitness_function == "mcc":
                return mcc
            elif fitness_function == "combined":
                # Weighted combination, emphasizing ROC-AUC more
                return 0.2 * acc + 0.5 * roc_auc + 0.3 * mcc
            else:
                return roc_auc  # Default to ROC-AUC for fraud detection
        else:
            # For other classification tasks, return accuracy
            return accuracy_score(labels, preds)
    
    # Other evaluation types remain the same
    elif eval_type == "exact_match" or eval_type == "rm_default" or eval_type == "rm_verbose" or eval_type == "rm_concise":
        # Add implementation for other evaluation types
        return 0.0
    
    elif eval_type == "perplexity" or eval_type == "multitask":
        # Add implementation for perplexity and multitask evaluation
        return 0.0
    
    else:
        print(f"Unknown evaluation type: {eval_type}")
        return 0.0

def evaluate_test(model_path, eval_type, dataset, gpu_id, base_model, fitness_function="accuracy"):
    """
    Evaluate a model on the test set.
    """
    return evaluate(model_path, eval_type, dataset, gpu_id, base_model, True, "_test", False, fitness_function)

# lora_weight_visualize function remains unchanged
def lora_weight_visualize(lora_weight_path, num_components=2):
    """
    Visualize LoRA weights using PCA.
    """
    # Implementation unchanged
    weights = torch.load(lora_weight_path, map_location="cpu")
    
    flattened_weights = []
    for name, weight in weights.items():
        if isinstance(weight, torch.Tensor):
            flattened_weights.append(weight.reshape(-1))
    
    if not flattened_weights:
        return (0, 0)
    
    concatenated_weights = torch.cat(flattened_weights).numpy()
    
    pca = PCA(n_components=num_components)
    coords = pca.fit_transform([concatenated_weights])[0]
    
    return tuple(coords)
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

def overall_metrics(search_pass_name, eval_type):
    """
    Calculate overall metrics for a search pass, now including ROC-AUC and MCC.
    
    Args:
        search_pass_name (str): Name of the search pass.
        eval_type (str): Type of evaluation.
        
    Returns:
        dict: Dictionary of overall metrics.
    """
    # Load utility scratchpad
    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
        utility_scratchpad = json.load(f)
    
    # Get number of candidates
    num_candidates = 0
    for key in utility_scratchpad.keys():
        if "candidate_" in key and not "_history" in key:
            num_candidates += 1
    
    # Get starting and ending best validation utility
    starting_best_validation_utility = utility_scratchpad["history"][0]
    ending_best_validation_utility = utility_scratchpad["best"]
    
    # Find the best candidate on validation
    ending_best_candidate_on_validation = -1
    for i in range(num_candidates):
        if utility_scratchpad[f"candidate_{i}"] == ending_best_validation_utility:
            ending_best_candidate_on_validation = i
            break
    
    # Calculate ensemble metrics for test set
    if eval_type == "multiple_choice" or eval_type == "AbstainQA" or eval_type == "fraud_detection":
        # Load gold and pred files for each candidate
        golds_list = []
        preds_list = []
        probs_list = []  # For ROC-AUC calculation
        
        # Starting ensemble
        for i in range(num_candidates):
            try:
                with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "golds.json")) as f:
                    golds = json.load(f)
                with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "preds.json")) as f:
                    preds = json.load(f)
                # Load probabilities if available
                try:
                    with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "probs.json")) as f:
                        probs = json.load(f)
                    probs_list.append(probs)
                except:
                    probs_list.append(None)
                    
                golds_list.append(golds)
                preds_list.append(preds)
            except:
                pass
        
        # Ensure all golds are the same
        for i in range(1, len(golds_list)):
            assert golds_list[i] == golds_list[0]
        
        # Calculate ensemble predictions
        ensemble_preds = []
        ensemble_probs = []  # For ROC-AUC calculation
        
        for j in range(len(golds_list[0])):
            votes = {}
            prob_sum = 0.0
            prob_count = 0
            
            for i in range(len(preds_list)):
                if preds_list[i][j] not in votes:
                    votes[preds_list[i][j]] = 0
                votes[preds_list[i][j]] += 1
                
                # Accumulate probabilities if available
                if probs_list[i] is not None:
                    prob_sum += probs_list[i][j]
                    prob_count += 1
            
            # Find the most voted prediction
            max_vote = 0
            max_pred = None
            for pred, vote in votes.items():
                if vote > max_vote:
                    max_vote = vote
                    max_pred = pred
            
            ensemble_preds.append(max_pred)
            # Calculate average probability
            if prob_count > 0:
                ensemble_probs.append(prob_sum / prob_count)
            else:
                ensemble_probs.append(0.5)  # Default if no probabilities available
        
        starting_ensemble_test_accuracy = accuracy_score(golds_list[0], ensemble_preds)
        
        # Calculate ROC-AUC if we have binary classification
        starting_ensemble_test_roc_auc = 0.5
        starting_ensemble_test_mcc = 0.0
        
        try:
            # For binary classification with 0/1 or False/True
            if len(set(golds_list[0])) == 2:
                # Convert to numeric if needed
                numeric_golds = [1 if g in [1, True, "1", "True"] else 0 for g in golds_list[0]]
                numeric_preds = [1 if p in [1, True, "1", "True"] else 0 for p in ensemble_preds]
                
                # Calculate ROC-AUC
                starting_ensemble_test_roc_auc = roc_auc_score(numeric_golds, ensemble_probs)
                
                # Calculate MCC
                starting_ensemble_test_mcc = matthews_corrcoef(numeric_golds, numeric_preds)
        except:
            pass
        
        # Ending ensemble (same process as above but for current models)
        golds_list = []
        preds_list = []
        probs_list = []
        
        for i in range(num_candidates):
            try:
                with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "golds.json")) as f:
                    golds = json.load(f)
                with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "preds.json")) as f:
                    preds = json.load(f)
                try:
                    with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "probs.json")) as f:
                        probs = json.load(f)
                    probs_list.append(probs)
                except:
                    probs_list.append(None)
                    
                golds_list.append(golds)
                preds_list.append(preds)
            except:
                pass
        
        # Ensure all golds are the same
        for i in range(1, len(golds_list)):
            assert golds_list[i] == golds_list[0]
        
        # Calculate ensemble predictions and probabilities
        ensemble_preds = []
        ensemble_probs = []
        
        for j in range(len(golds_list[0])):
            votes = {}
            prob_sum = 0.0
            prob_count = 0
            
            for i in range(len(preds_list)):
                if preds_list[i][j] not in votes:
                    votes[preds_list[i][j]] = 0
                votes[preds_list[i][j]] += 1
                
                if probs_list[i] is not None:
                    prob_sum += probs_list[i][j]
                    prob_count += 1
            
            # Find the most voted prediction
            max_vote = 0
            max_pred = None
            for pred, vote in votes.items():
                if vote > max_vote:
                    max_vote = vote
                    max_pred = pred
            
            ensemble_preds.append(max_pred)
            if prob_count > 0:
                ensemble_probs.append(prob_sum / prob_count)
            else:
                ensemble_probs.append(0.5)
        
        ending_ensemble_test_accuracy = accuracy_score(golds_list[0], ensemble_preds)
        
        # Calculate ROC-AUC and MCC for ending ensemble
        ending_ensemble_test_roc_auc = 0.5
        ending_ensemble_test_mcc = 0.0
        
        try:
            if len(set(golds_list[0])) == 2:
                numeric_golds = [1 if g in [1, True, "1", "True"] else 0 for g in golds_list[0]]
                numeric_preds = [1 if p in [1, True, "1", "True"] else 0 for p in ensemble_preds]
                
                ending_ensemble_test_roc_auc = roc_auc_score(numeric_golds, ensemble_probs)
                ending_ensemble_test_mcc = matthews_corrcoef(numeric_golds, numeric_preds)
        except:
            pass
        
        # Get the best single test metrics
        try:
            with open(os.path.join("search", search_pass_name, "candidate_"+str(ending_best_candidate_on_validation), "golds.json")) as f:
                golds = json.load(f)
            with open(os.path.join("search", search_pass_name, "candidate_"+str(ending_best_candidate_on_validation), "preds.json")) as f:
                preds = json.load(f)
                
            ending_best_single_test_accuracy = accuracy_score(golds, preds)
            
            # Calculate ROC-AUC and MCC for best single model
            ending_best_single_test_roc_auc = 0.5
            ending_best_single_test_mcc = 0.0
            
            try:
                if len(set(golds)) == 2:
                    numeric_golds = [1 if g in [1, True, "1", "True"] else 0 for g in golds]
                    numeric_preds = [1 if p in [1, True, "1", "True"] else 0 for p in preds]
                    
                    # Try to load probabilities for ROC-AUC
                    try:
                        with open(os.path.join("search", search_pass_name, "candidate_"+str(ending_best_candidate_on_validation), "probs.json")) as f:
                            probs = json.load(f)
                        ending_best_single_test_roc_auc = roc_auc_score(numeric_golds, probs)
                    except:
                        pass
                    
                    ending_best_single_test_mcc = matthews_corrcoef(numeric_golds, numeric_preds)
            except:
                pass
                
        except:
            ending_best_single_test_accuracy = 0
            ending_best_single_test_roc_auc = 0.5
            ending_best_single_test_mcc = 0.0
        
        # Get the starting best single test metrics
        starting_best_single_test_accuracy = 0
        starting_best_single_test_roc_auc = 0.5
        starting_best_single_test_mcc = 0.0
        
        for i in range(num_candidates):
            try:
                with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "golds.json")) as f:
                    golds = json.load(f)
                with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "preds.json")) as f:
                    preds = json.load(f)
                    
                acc = accuracy_score(golds, preds)
                
                # Calculate ROC-AUC and MCC if possible
                roc_auc = 0.5
                mcc = 0.0
                
                try:
                    if len(set(golds)) == 2:
                        numeric_golds = [1 if g in [1, True, "1", "True"] else 0 for g in golds]
                        numeric_preds = [1 if p in [1, True, "1", "True"] else 0 for p in preds]
                        
                        # Try to load probabilities for ROC-AUC
                        try:
                            with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "probs.json")) as f:
                                probs = json.load(f)
                            roc_auc = roc_auc_score(numeric_golds, probs)
                        except:
                            pass
                        
                        mcc = matthews_corrcoef(numeric_golds, numeric_preds)
                except:
                    pass
                
                # Update starting best metrics if better
                if acc > starting_best_single_test_accuracy:
                    starting_best_single_test_accuracy = acc
                    starting_best_single_test_roc_auc = roc_auc
                    starting_best_single_test_mcc = mcc
            except:
                pass
    
    elif eval_type == "exact_match" or eval_type == "external_api" or eval_type == "rm_default" or eval_type == "rm_concise" or eval_type == "rm_verbose" or eval_type == "human":
        # Load scores files for each candidate
        scores_list = []
        
        # Starting ensemble
        for i in range(num_candidates):
            try:
                with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "scores.json")) as f:
                    scores = json.load(f)
                scores_list.append(scores)
            except:
                pass
        
        # Calculate ensemble scores
        ensemble_scores = []
        for j in range(len(scores_list[0])):
            ensemble_scores.append(np.mean([scores_list[i][j] for i in range(len(scores_list))]))
        
        starting_ensemble_test_accuracy = np.mean(ensemble_scores)
        
        # Ending ensemble
        scores_list = []
        
        for i in range(num_candidates):
            try:
                with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "scores.json")) as f:
                    scores = json.load(f)
                scores_list.append(scores)
            except:
                pass
        
        # Calculate ensemble scores
        ensemble_scores = []
        for j in range(len(scores_list[0])):
            ensemble_scores.append(np.mean([scores_list[i][j] for i in range(len(scores_list))]))
        
        ending_ensemble_test_accuracy = np.mean(ensemble_scores)
        
        # Get the best single test accuracy
        try:
            with open(os.path.join("search", search_pass_name, "candidate_"+str(ending_best_candidate_on_validation), "scores.json")) as f:
                scores = json.load(f)
            ending_best_single_test_accuracy = np.mean(scores)
        except:
            ending_best_single_test_accuracy = 0
        
        # Get the starting best single test accuracy
        starting_best_single_test_accuracy = 0
        for i in range(num_candidates):
            try:
                with open(os.path.join("search", search_pass_name, "candidate_"+str(i), "scores.json")) as f:
                    scores = json.load(f)
                acc = np.mean(scores)
                if acc > starting_best_single_test_accuracy:
                    starting_best_single_test_accuracy = acc
            except:
                pass
        
        # Default values for ROC-AUC and MCC
        starting_ensemble_test_roc_auc = 0.5
        ending_ensemble_test_roc_auc = 0.5
        starting_best_single_test_roc_auc = 0.5
        ending_best_single_test_roc_auc = 0.5
        
        starting_ensemble_test_mcc = 0.0
        ending_ensemble_test_mcc = 0.0
        starting_best_single_test_mcc = 0.0
        ending_best_single_test_mcc = 0.0
    
    else:
        # For other evaluation types, set default values
        starting_ensemble_test_accuracy = 0
        ending_ensemble_test_accuracy = 0
        ending_best_single_test_accuracy = 0
        starting_best_single_test_accuracy = 0
        
        # Default values for ROC-AUC and MCC
        starting_ensemble_test_roc_auc = 0.5
        ending_ensemble_test_roc_auc = 0.5
        starting_best_single_test_roc_auc = 0.5
        ending_best_single_test_roc_auc = 0.5
        
        starting_ensemble_test_mcc = 0.0
        ending_ensemble_test_mcc = 0.0
        starting_best_single_test_mcc = 0.0
        ending_best_single_test_mcc = 0.0
    
    # Return overall metrics including new ROC-AUC and MCC metrics
    return {
        "starting_best_validation_utility": starting_best_validation_utility,
        "ending_best_validation_utility": ending_best_validation_utility,
        "starting_best_single_test_accuracy": starting_best_single_test_accuracy,
        "ending_best_single_test_accuracy": ending_best_single_test_accuracy,
        "starting_top-k_ensemble_test_accuracy": starting_ensemble_test_accuracy,
        "ending_top-k_ensemble_test_accuracy": ending_ensemble_test_accuracy,
        "starting_top-k_ensemble_test_roc_auc": starting_ensemble_test_roc_auc,
        "ending_top-k_ensemble_test_roc_auc": ending_ensemble_test_roc_auc,
        "starting_best_single_test_roc_auc": starting_best_single_test_roc_auc,
        "ending_best_single_test_roc_auc": ending_best_single_test_roc_auc,
        "starting_top-k_ensemble_test_mcc": starting_ensemble_test_mcc,
        "ending_top-k_ensemble_test_mcc": ending_ensemble_test_mcc,
        "starting_best_single_test_mcc": starting_best_single_test_mcc,
        "ending_best_single_test_mcc": ending_best_single_test_mcc,
        "ending_best_candidate_on_validation": ending_best_candidate_on_validation
    }
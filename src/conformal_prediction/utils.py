"""
Utility functions for conformal risk control.

This module provides FNR computation and statistical utilities for conformal prediction.

Architecture:
- Threshold application logic lives in ThreeHopPipeline.apply_thresholds_to_scores()
- This ensures calibration and inference use the SAME prediction logic
- VectorOptimizer uses the model's static method during threshold optimization
- VectorCRC uses model.predict_with_thresholds() during inference

Key functions:
- compute_fnr_metrics: Unified metric computation (used by calibration and validation)
- binomial_upper_bound: Statistical confidence bound for CRC

All threshold comparisons use >= (not >) per conformal prediction theory.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from scipy.stats import beta
import logging


def calculate_false_negative_rate(scores: List[np.ndarray], 
                                true_labels: List[List[int]], 
                                threshold: float) -> float:
    """
    Calculate false negative rate for given threshold.
    
    Args:
        scores: List of score arrays
        true_labels: List of ground truth label lists
        threshold: Threshold value for prediction
        
    Returns:
        False negative rate
    """
    overlap_ratios = []
    
    for i in range(len(scores)):
        scores_i = np.array(scores[i]) if not isinstance(scores[i], np.ndarray) else scores[i]
        threshold_np = np.array(threshold) if not isinstance(threshold, np.ndarray) else threshold
        
        predictions = np.where(scores_i >= threshold_np)[0]
        gt_labels = set(true_labels[i])
        
        intersection = len(set(predictions) & gt_labels)
        
        if len(gt_labels) > 0:
            overlap_ratio = 1 - intersection / len(gt_labels)
        else:
            overlap_ratio = 0  # avoid division by zero
            
        overlap_ratios.append(overlap_ratio)
    
    return np.mean(overlap_ratios)


def lamhat_threshold(threshold: float, 
                    cal_scores: List[np.ndarray], 
                    true_labels: List[List[int]], 
                    alpha: float) -> float:
    """
    Calculate threshold function for conformal risk control.
    
    Args:
        threshold: Threshold value
        cal_scores: Calibration scores
        true_labels: True labels
        alpha: Target false negative rate
        
    Returns:
        Threshold function value
    """
    n = len(cal_scores)
    fnr = calculate_false_negative_rate(cal_scores, true_labels, threshold)
    return fnr - ((n + 1) / n * alpha - 1 / (n + 1))


def get_alpha_bounds(n: int) -> Tuple[float, float]:
    """
    Calculate alpha bounds for conformal risk control.
    
    Args:
        n: Number of calibration samples
        
    Returns:
        Tuple of (alpha_lowerbound, alpha_upperbound)
    """
    alpha_lowerbound = n / ((n + 1) ** 2)
    alpha_upperbound = n * (n + 2) / ((n + 1) ** 2)
    return alpha_lowerbound, alpha_upperbound


def binomial_upper_bound(m: int, n: int, delta: float) -> float:
    """
    One-sided Clopper–Pearson upper confidence bound.
    
    Args:
        m: Number of successes
        n: Number of trials
        delta: Confidence level
        
    Returns:
        Upper confidence bound
    """
    if m == n:
        return 1.0
    return beta.ppf(1 - delta, m + 1, n - m)


def validate_model_name(model_name: str) -> None:
    """
    Validate that the model name is supported.
    
    Args:
        model_name: Name of the model
        
    Raises:
        ValueError: If model name is not supported
    """
    supported_models = {
        "ultra",
        "dbexecmodel", "multihoppredictor", "threehoppipeline",
        "twounionpipeline", "twointersectprojectpipeline"
    }
    
    if model_name not in supported_models:
        raise ValueError(f"Model {model_name} not supported. Supported models: {supported_models}")


def validate_calibration_data(cal_scores: Union[np.ndarray, List[Dict]], true_labels: List[Dict]) -> None:
    """
    Validate calibration data format and content.
    
    Args:
        cal_scores: Calibration scores (either old format numpy array or new path-aware list of dicts)
        true_labels: True labels list
        
    Raises:
        ValueError: If data format is invalid
    """
    if isinstance(cal_scores, list):
        # New path-aware format
        if len(cal_scores) == 0:
            raise ValueError("Calibration scores cannot be empty")
        
        if len(cal_scores) != len(true_labels):
            raise ValueError("Number of calibration scores must match number of true labels")
        
        # Validate structure of first entry
        if len(cal_scores) > 0:
            sample = cal_scores[0]
            if not isinstance(sample, dict):
                raise ValueError("Path-aware calibration data must be list of dicts")

            # Support both 3p (hop1/hop2/hop3) and 2u (branch1/branch2) vector formats.
            if "hop1" in sample:
                required_keys = ["hop1", "hop2", "hop3"]
            elif "branch1" in sample:
                required_keys = ["branch1", "branch2"]
            else:
                raise ValueError(f"Unrecognized calibration dict keys: {list(sample.keys())}")

            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Path-aware calibration data missing key: {key}")
            
    elif isinstance(cal_scores, np.ndarray):
        # Old format (for backward compatibility)
        if len(cal_scores) == 0:
            raise ValueError("Calibration scores cannot be empty")
        
        if len(cal_scores) != len(true_labels):
            raise ValueError("Number of calibration scores must match number of true labels")
        
        if cal_scores.ndim != 3:
            raise ValueError("Calibration scores must be 3-dimensional (n, k, d)")
        
        n, k, d = cal_scores.shape
        if k != 3:
            raise ValueError("Expected 3-hop calibration data (k=3)")
        
        logging.info(f"Validated calibration data: shape={cal_scores.shape}")
    
    else:
        raise ValueError(f"Calibration scores must be numpy array or list of dicts, got {type(cal_scores)}")


def compute_fnr_metrics(preds: List[Union[List[int], Set[int]]], 
                       ground_truth: List[Union[List[int], Set[int]]]) -> tuple[float, float, float]:
    """
    Unified FNR computation for Conformal Risk Control.
    Ensures every query in the input contributes to the risk average to maintain 
    the validity of the (n/n+1) inflation factor.
    """
    n_total = len(preds)
    if n_total == 0:
        return 0.0, 0.0, 0.0

    fnrs = np.zeros(n_total)
    precisions = np.zeros(n_total)
    f1s = np.zeros(n_total)

    for i in range(n_total):
        # 1. Standardize inputs to sets
        gt_labels = set(int(x) for x in ground_truth[i]) if ground_truth[i] else set()
        pred_labels = set(int(x) for x in preds[i]) if preds[i] else set()

        # 2. Handle queries with no ground truth
        if len(gt_labels) == 0:
            # If there is nothing to find, FNR is 0 (you missed nothing).
            # However, precision is tricky. Usually, if pred_labels is empty, 
            # precision is 1.0 (no false positives). If not empty, 0.0.
            fnrs[i] = 0.0
            precisions[i] = 1.0 if len(pred_labels) == 0 else 0.0
            f1s[i] = precisions[i] # Simple mapping for the no-GT case
            continue

        # 3. Calculate Recall and FNR
        # If pred_labels is empty (abstention), recall is 0, FNR is 1.0.
        hits = len(pred_labels & gt_labels)
        recall = hits / len(gt_labels)
        fnrs[i] = 1.0 - recall

        # 4. Calculate Precision
        if len(pred_labels) == 0:
            precisions[i] = 0.0 
        else:
            precisions[i] = hits / len(pred_labels)

        # 5. Calculate F1
        if (precisions[i] + recall) > 0:
            f1s[i] = (2 * precisions[i] * recall) / (precisions[i] + recall)
        else:
            f1s[i] = 0.0

    # Return macro-averages (mean of per-query metrics)
    # This ensures the denominator is ALWAYS n_total queries
    return np.mean(fnrs), np.mean(precisions), np.mean(f1s)

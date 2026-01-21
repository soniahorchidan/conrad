"""
Utility functions for conformal risk control.

This module provides FNR computation and statistical utilities for conformal prediction.

Architecture:
- Threshold application logic lives in ThreeHopPipeline.apply_thresholds_to_scores()
- This ensures calibration and inference use the SAME prediction logic
- VectorOptimizer uses the model's static method during threshold optimization
- VectorCRC uses model.predict_with_thresholds() during inference

Key functions:
- compute_metrics: Unified metric computation (used by calibration and validation)
- binomial_upper_bound: Statistical confidence bound for CRC
- apply_vector_thresholds: DEPRECATED - use ThreeHopPipeline.apply_thresholds_to_scores()
- compute_vector_fnr: DEPRECATED - use model's method + compute_metrics()

All threshold comparisons use >= (not >) per conformal prediction theory.
"""
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from scipy.stats import beta
from scipy.optimize import brentq
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


def min_empirical_fnr_for_feasibility(n: int, alpha: float, delta: float) -> float:
    """
    Return smallest empirical FNR that could satisfy U(m,n,δ) ≤ α.
    
    Args:
        n: Number of samples
        alpha: Target false negative rate
        delta: Confidence level
        
    Returns:
        Minimum empirical false negative rate
    """
    for m in range(n + 1):
        if beta.ppf(1 - delta, m + 1, n - m) <= alpha:
            return m / n
    return 1.0  # even m=n not enough (shouldn't happen)


def validate_model_name(model_name: str) -> None:
    """
    Validate that the model name is supported.
    
    Args:
        model_name: Name of the model
        
    Raises:
        ValueError: If model name is not supported
    """
    supported_models = {
        "query2box", "transr", "ultra", "relationprediction", 
        "dbexecmodel", "multihoppredictor", "threehoppipeline",
        "twounionpipeline"
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
            
            required_keys = ['hop1', 'hop2', 'hop3']
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


def compute_metrics(preds, labels):
    """
    Compute per-query recall, precision, and F1 averaged over queries.
    Abstentions (empty predictions) contribute 0 recall and 0 precision for that query.

    Args:
        preds: list of predicted sets/lists of hop3 nodes per query
        labels: list of true sets/lists per query

    Returns:
        dict with averaged metrics and abstention stats
    """
    num_queries = len(labels)
    assert len(preds) == num_queries, "preds and labels must align"

    per_recall = []
    per_precision = []
    per_f1 = []
    abstentions = 0

    for p, L in zip(preds, labels):
        p_set = set(p) if p else set()
        L_set = set(L)

        if not p_set:
            abstentions += 1

        tp = len(p_set & L_set)
        fp = len(p_set - L_set)
        fn = len(L_set - p_set)

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        per_precision.append(prec)
        per_recall.append(rec)
        per_f1.append(f1)

    metrics = {
        "precision": float(np.mean(per_precision)),
        "recall": float(np.mean(per_recall)),          # mean per-query recall
        "f1": float(np.mean(per_f1)),
        "abstention_rate": abstentions / num_queries,
        "num_queries": num_queries,
        "num_non_abstained": num_queries - abstentions,
    }
    return metrics


def extract_ground_truth_entities(true_labels: List[Dict[int, List[int]]], 
                                 hops: Union[int, List[int]] = 3) -> List[Set[int]]:
    """
    Extract ground truth entities for each query from specified hop(s).
    
    Args:
        true_labels: True labels for each query as dict {hop: [entity_ids]}
        hops: Which hop(s) to extract. Can be:
            - Single int (e.g., 3) to extract only that hop
            - List of ints (e.g., [1, 2, 3]) to extract union of those hops
            - Default is 3 (hop3 only, since predictions are hop3 only)
        
    Returns:
        List of sets of ground truth entity IDs for each query
    """
    # Normalize hops to a list
    if isinstance(hops, int):
        hops_list = [hops]
    else:
        hops_list = list(hops)
    
    gt_entities_list = []
    
    for label_dict in true_labels:
        gt_entities = set()
        for hop in hops_list:
            gt_entities.update(label_dict.get(hop, []))
        gt_entities_list.append(gt_entities)
    
    return gt_entities_list


def apply_vector_thresholds(scores: np.ndarray, thresholds: Union[List[float], np.ndarray], 
                           mode: str = "hop3_only") -> List[List[int]]:
    """
    Apply thresholds to score tensor and extract predicted entities.
    
    DEPRECATED: Use ThreeHopPipeline.apply_thresholds_to_scores() instead.
    This function is kept for backward compatibility but delegates to the model's method.
    
    Args:
        scores: Score array of shape (n, k, d)
        thresholds: Threshold values for each hop
        mode: "hop3_only" (recommended) or "union" (deprecated)
            
    Returns:
        List of predicted entity lists, one per query
    """
    logging.warning(
        "apply_vector_thresholds is deprecated. "
        "Use ThreeHopPipeline.apply_thresholds_to_scores() instead."
    )
    
    # Delegate to the model's static method (single source of truth)
    from models.topology.model import ThreeHopPipeline
    return ThreeHopPipeline.apply_thresholds_to_scores(scores, thresholds)

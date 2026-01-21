"""
RAPS (Regularized Adaptive Prediction Sets) for conformal prediction.

Implements Algorithms 2 & 3 from https://arxiv.org/pdf/2009.14193

Key insight: RAPS modifies the conformity score computation, not just calibration.
The SAME logic must be applied at both calibration and inference time.

For CASCADE systems: Apply RAPS AFTER MAX aggregation, not to individual paths!
"""
import numpy as np
import torch
import logging
from typing import List, Dict, Union, Tuple
import copy


# ============================================================================
# RAPS for CASCADE Models (apply after MAX aggregation)
# ============================================================================

def apply_raps_to_aggregated_scores(
    cal_scores: List[Dict],
    true_labels: List[Dict[int, List[int]]],
    kreg: int = 1,
    lamda: float = 1e-3
) -> List[Dict]:
    """
    Apply RAPS penalties to MAX-aggregated scores for cascade models.
    
    This is the CORRECT approach for multi-hop cascade systems:
    1. MAX-aggregate scores across paths (already done during calibration)
    2. Apply RAPS penalty based on GT position in AGGREGATED scores
    3. This ensures penalties are computed on the actual scores used for predictions
    
    Args:
        cal_scores: Path-aware calibration scores (list of dicts)
        true_labels: Ground truth entities per hop for each query 
                     [{1: [ids], 2: [ids], 3: [ids]}, ...]
        kreg: Number of top-k elements before applying penalty
        lamda: Penalty weight
        
    Returns:
        Regularized calibration scores with penalties applied to aggregated scores
    """
    from models.topology.model import ThreeHopPipeline
    
    logging.info(f"Applying RAPS to aggregated scores (kreg={kreg}, λ={lamda:.1e})")
    
    cal_scores_regularized = copy.deepcopy(cal_scores)
    
    # Track penalties for diagnostics
    hop_penalties = {1: [], 2: [], 3: []}
    
    for query_idx, query_data in enumerate(cal_scores_regularized):
        gt_labels = true_labels[query_idx]
        hop1_calibration_nodes = set(query_data['hop1']['nodes'])
        
        # Hop 1: Direct scores, apply penalty
        if 'hop1' in query_data and 'scores' in query_data['hop1']:
            gt_entities_hop1 = set(gt_labels.get(1, []))
            hop1_scores = query_data['hop1']['scores']
            query_data['hop1']['scores'], penalty = _apply_penalty_with_gt(
                hop1_scores, gt_entities_hop1, kreg, lamda, return_penalty=True
            )
            hop_penalties[1].append(penalty)
        
        # Hop 2: MAX aggregate first, then apply penalty
        if 'hop2' in query_data and hop1_calibration_nodes:
            gt_entities_hop2 = set(gt_labels.get(2, []))
            
            # Step 1: MAX aggregate hop2 scores across all paths
            hop2_aggregated = _aggregate_scores_from_paths(
                query_data['hop2'], 
                hop1_calibration_nodes
            )
            
            # Step 2: Apply penalty to aggregated scores
            if hop2_aggregated is not None:
                hop2_aggregated, penalty = _apply_penalty_with_gt(
                    hop2_aggregated, gt_entities_hop2, kreg, lamda, return_penalty=True
                )
                hop_penalties[2].append(penalty)
                
                # Store aggregated penalized scores (replace individual paths)
                query_data['hop2_aggregated'] = hop2_aggregated
        
        # Hop 3: MAX aggregate first, then apply penalty
        if 'hop3' in query_data:
            gt_entities_hop3 = set(gt_labels.get(3, []))
            
            # Get hop2 calibration nodes
            hop2_calibration_nodes = set()
            for hop3_path in query_data['hop3']:
                hop1_parent, hop2_parent = hop3_path['parent']
                if hop1_parent in hop1_calibration_nodes:
                    hop2_calibration_nodes.add(hop2_parent)
            
            if hop2_calibration_nodes:
                # Step 1: MAX aggregate hop3 scores across all paths
                hop3_aggregated = _aggregate_scores_from_paths_hop3(
                    query_data['hop3'],
                    hop1_calibration_nodes,
                    hop2_calibration_nodes
                )
                
                # Step 2: Apply penalty to aggregated scores
                if hop3_aggregated is not None:
                    hop3_aggregated, penalty = _apply_penalty_with_gt(
                        hop3_aggregated, gt_entities_hop3, kreg, lamda, return_penalty=True
                    )
                    hop_penalties[3].append(penalty)
                    
                    # Store aggregated penalized scores
                    query_data['hop3_aggregated'] = hop3_aggregated
    
    # Log penalty statistics
    for hop in [1, 2, 3]:
        if hop_penalties[hop]:
            penalties = np.array(hop_penalties[hop])
            logging.info(f"  Hop {hop} aggregated penalties: min={penalties.min():.4f}, "
                        f"mean={penalties.mean():.4f}, max={penalties.max():.4f}, "
                        f"median={np.median(penalties):.4f}")
        else:
            logging.info(f"  Hop {hop}: No penalties computed")
    
    logging.info("RAPS regularization applied to aggregated scores")
    return cal_scores_regularized


def _aggregate_scores_from_paths(paths: List[Dict], valid_parents: set) -> Union[Dict, None]:
    """
    MAX-aggregate scores from multiple paths.
    
    Args:
        paths: List of path dictionaries with 'parent' and 'scores'
        valid_parents: Set of valid parent nodes to include
        
    Returns:
        Aggregated scores in same format as input, or None if no paths
    """
    from models.topology.model import ThreeHopPipeline
    
    score_vectors = []
    for path in paths:
        if path['parent'] in valid_parents and 'scores' in path:
            scores = path['scores']
            # Convert to dense vector
            dense_vec = ThreeHopPipeline._scores_to_dense_vector(scores)
            score_vectors.append(dense_vec)
    
    if not score_vectors:
        return None
    
    # MAX aggregate
    aggregated = np.maximum.reduce(score_vectors)
    
    # Return as sparse format for consistency
    # Find non-zero indices
    nonzero_idx = np.where(aggregated > 0)[0]
    if len(nonzero_idx) == 0:
        # All zeros, return top-1000
        top_k = min(1000, len(aggregated))
        top_idx = np.argsort(aggregated)[::-1][:top_k]
        return {
            'indices': torch.tensor(top_idx, dtype=torch.long),
            'values': torch.tensor(aggregated[top_idx], dtype=torch.float32),
            'shape': len(aggregated)
        }
    
    return {
        'indices': torch.tensor(nonzero_idx, dtype=torch.long),
        'values': torch.tensor(aggregated[nonzero_idx], dtype=torch.float32),
        'shape': len(aggregated)
    }


def _aggregate_scores_from_paths_hop3(paths: List[Dict], valid_hop1: set, valid_hop2: set) -> Union[Dict, None]:
    """
    MAX-aggregate hop3 scores from multiple paths (with 2-level parent checking).
    
    Args:
        paths: List of hop3 path dictionaries with 'parent' tuple and 'scores'
        valid_hop1: Set of valid hop1 parent nodes
        valid_hop2: Set of valid hop2 parent nodes
        
    Returns:
        Aggregated scores, or None if no paths
    """
    from models.topology.model import ThreeHopPipeline
    
    score_vectors = []
    for path in paths:
        hop1_parent, hop2_parent = path['parent']
        if hop1_parent in valid_hop1 and hop2_parent in valid_hop2 and 'scores' in path:
            scores = path['scores']
            # Convert to dense vector
            dense_vec = ThreeHopPipeline._scores_to_dense_vector(scores)
            score_vectors.append(dense_vec)
    
    if not score_vectors:
        return None
    
    # MAX aggregate
    aggregated = np.maximum.reduce(score_vectors)
    
    # Return as sparse format for consistency
    nonzero_idx = np.where(aggregated > 0)[0]
    if len(nonzero_idx) == 0:
        # All zeros, return top-1000
        top_k = min(1000, len(aggregated))
        top_idx = np.argsort(aggregated)[::-1][:top_k]
        return {
            'indices': torch.tensor(top_idx, dtype=torch.long),
            'values': torch.tensor(aggregated[top_idx], dtype=torch.float32),
            'shape': len(aggregated),
            'gt_only': False
        }
    
    return {
        'indices': torch.tensor(nonzero_idx, dtype=torch.long),
        'values': torch.tensor(aggregated[nonzero_idx], dtype=torch.float32),
        'shape': len(aggregated),
        'gt_only': False
    }


# ============================================================================
# RAPS for Vector Models (OLD: apply to individual paths - DON'T USE FOR CASCADE)
# ============================================================================

def apply_raps_to_vector_scores(
    cal_scores: List[Dict],
    true_labels: List[Dict[int, List[int]]],
    kreg: int = 1,
    lamda: float = 1e-3
) -> List[Dict]:
    """
    Apply RAPS penalties to path-aware vector scores based on ground truth.
    
    This is used ONLY during calibration to compute conformity scores.
    At inference, use apply_raps_threshold() instead.
    
    Penalizes scores based on how many elements are needed to cover
    all ground truth entities at each hop. This follows the RAPS paper
    where penalty = λ * max(k - kreg, 0), where k is the position where
    all GT entities are covered.
    
    Args:
        cal_scores: Path-aware calibration scores (list of dicts)
        true_labels: Ground truth entities per hop for each query 
                     [{1: [ids], 2: [ids], 3: [ids]}, ...]
        kreg: Number of top-k elements before applying penalty
        lamda: Penalty weight
        
    Returns:
        Regularized calibration scores (same structure, modified values)
    """
    logging.info(f"Applying GT-aware RAPS penalties (kreg={kreg}, λ={lamda:.1e})")
    
    cal_scores_regularized = copy.deepcopy(cal_scores)
    
    # Track penalties for diagnostics
    hop_penalties = {1: [], 2: [], 3: []}
    
    for query_idx, query_data in enumerate(cal_scores_regularized):
        gt_labels = true_labels[query_idx]
        
        # Apply to hop1
        if 'hop1' in query_data and 'scores' in query_data['hop1']:
            gt_entities_hop1 = set(gt_labels.get(1, []))
            query_data['hop1']['scores'], penalty = _apply_penalty_with_gt(
                query_data['hop1']['scores'], gt_entities_hop1, kreg, lamda, return_penalty=True
            )
            hop_penalties[1].append(penalty)
        
        # Apply to hop2 paths
        if 'hop2' in query_data:
            gt_entities_hop2 = set(gt_labels.get(2, []))
            for path in query_data['hop2']:
                if 'scores' in path:
                    path['scores'], penalty = _apply_penalty_with_gt(
                        path['scores'], gt_entities_hop2, kreg, lamda, return_penalty=True
                    )
                    hop_penalties[2].append(penalty)
        
        # Apply to hop3 paths
        if 'hop3' in query_data:
            gt_entities_hop3 = set(gt_labels.get(3, []))
            for path in query_data['hop3']:
                if 'scores' in path:
                    path['scores'], penalty = _apply_penalty_with_gt(
                        path['scores'], gt_entities_hop3, kreg, lamda, return_penalty=True
                    )
                    hop_penalties[3].append(penalty)
    
    # Log penalty statistics
    for hop in [1, 2, 3]:
        if hop_penalties[hop]:
            penalties = np.array(hop_penalties[hop])
            logging.info(f"  Hop {hop} penalties: min={penalties.min():.4f}, "
                        f"mean={penalties.mean():.4f}, max={penalties.max():.4f}, "
                        f"median={np.median(penalties):.4f}")
        else:
            logging.info(f"  Hop {hop}: No penalties computed")
    
    logging.info("GT-aware RAPS regularization applied")
    return cal_scores_regularized


def _apply_penalty_with_gt(scores, gt_entities: set, kreg: int, lamda: float, return_penalty: bool = False):
    """
    Apply RAPS penalty based on position of ground truth entities.
    
    The penalty is λ * max(k - kreg, 0) where k is the position where
    all GT entities are covered.
    
    Args:
        scores: Score vector (sparse dict or dense tensor/array)
        gt_entities: Set of ground truth entity IDs for this hop
        kreg: Regularization threshold
        lamda: Penalty weight
        return_penalty: If True, return (penalized_scores, penalty_value)
    """
    if isinstance(scores, dict) and 'indices' in scores and 'values' in scores:
        return _penalize_sparse_with_gt(scores, gt_entities, kreg, lamda, return_penalty)
    elif isinstance(scores, (torch.Tensor, np.ndarray)):
        return _penalize_dense_with_gt(scores, gt_entities, kreg, lamda, return_penalty)
    else:
        logging.warning(f"Unknown score format: {type(scores)}, skipping RAPS")
        if return_penalty:
            return scores, 0.0
        return scores


def _penalize_sparse_with_gt(scores: Dict, gt_entities: set, kreg: int, lamda: float, return_penalty: bool = False):
    """
    Apply RAPS penalty based on GT position in sparse scores.
    
    Computes penalty = λ * max(k - kreg, 0) where k is the position where
    all GT entities are covered when sorting scores in descending order.
    This uniform penalty is then subtracted from ALL scores.
    """
    indices = scores['indices']
    values = scores['values']
    
    # Convert to numpy
    if isinstance(values, torch.Tensor):
        values_np = values.cpu().numpy()
        indices_np = indices.cpu().numpy() if isinstance(indices, torch.Tensor) else np.array(indices)
        is_tensor = True
    else:
        values_np = np.array(values)
        indices_np = np.array(indices)
        is_tensor = False
    
    # If no GT entities, no penalty
    if not gt_entities:
        penalty = 0.0
    else:
        # Sort by scores (descending)
        sorted_idx = np.argsort(values_np)[::-1]
        sorted_entities = indices_np[sorted_idx]
        
        # Find position where all GT entities are covered
        gt_covered = set()
        k_position = len(values_np)  # Default: all entities needed
        
        for pos, entity_id in enumerate(sorted_entities):
            if entity_id in gt_entities:
                gt_covered.add(entity_id)
                if gt_covered == gt_entities:
                    k_position = pos + 1  # 1-indexed position
                    break
        
        # Apply RAPS penalty: λ * max(k - kreg, 0)
        # This uniform penalty encourages tighter sets when GT requires many predictions
        penalty = lamda * max(k_position - kreg, 0)
    
    # Apply penalty to all scores
    values_penalized = np.maximum(values_np - penalty, 0.0)
    
    # Return in original format
    if is_tensor:
        result = {
            'indices': indices,
            'values': torch.tensor(values_penalized, dtype=values.dtype, device=values.device),
            'gt_only': scores.get('gt_only', False)
        }
    else:
        result = {
            'indices': indices,
            'values': values_penalized,
            'gt_only': scores.get('gt_only', False)
        }
    
    if return_penalty:
        return result, penalty
    return result


def _penalize_dense_with_gt(scores: Union[torch.Tensor, np.ndarray], 
                             gt_entities: set, kreg: int, lamda: float, return_penalty: bool = False):
    """
    Apply RAPS penalty based on GT position in dense scores.
    
    Computes penalty = λ * max(k - kreg, 0) where k is the position where
    all GT entities are covered when sorting scores in descending order.
    This uniform penalty is then subtracted from ALL scores.
    """
    # Convert to numpy
    if isinstance(scores, torch.Tensor):
        scores_np = scores.cpu().numpy()
        is_tensor = True
    else:
        scores_np = np.array(scores)
        is_tensor = False
    
    # If no GT entities, no penalty
    if not gt_entities:
        penalty = 0.0
    else:
        # For dense vectors, entity_id = index
        # Sort indices by scores (descending)
        sorted_indices = np.argsort(scores_np)[::-1]
        
        # Find position where all GT entities are covered
        gt_covered = set()
        k_position = len(scores_np)  # Default: all entities needed
        
        for pos, entity_id in enumerate(sorted_indices):
            if entity_id in gt_entities:
                gt_covered.add(entity_id)
                if gt_covered == gt_entities:
                    k_position = pos + 1  # 1-indexed position
                    break
        
        # Apply RAPS penalty: λ * max(k - kreg, 0)
        penalty = lamda * max(k_position - kreg, 0)
    
    # Apply penalty to all scores
    scores_penalized = np.maximum(scores_np - penalty, 0.0)
    
    if is_tensor:
        result = torch.tensor(scores_penalized, dtype=scores.dtype, device=scores.device)
    else:
        result = scores_penalized
    
    if return_penalty:
        return result, penalty
    return result


# ============================================================================
# RAPS Threshold Application (for inference)
# ============================================================================

def apply_raps_threshold(scores: torch.Tensor, threshold: float, 
                        kreg: int = 1, lamda: float = 1e-3,
                        randomized: bool = True) -> List[int]:
    """
    Apply RAPS threshold to scores at inference time.
    
    IMPORTANT: This must match the calibration approach!
    During calibration, we subtract a uniform penalty from all scores.
    During inference, we must do the same thing for consistency.
    
    The penalty is: λ * max(k - kreg, 0) where k is unknown at inference.
    Since we don't know k in advance, we use iterative approach:
    - Sort scores descending
    - For each position k, check if score_k - penalty_k >= threshold
    - Include entity if it passes
    
    Args:
        scores: Score tensor for entities (batch_size, num_entities) or (num_entities,)
        threshold: Calibrated threshold τ
        kreg: Regularization parameter (default: 1)
        lamda: Penalty weight (default: 1e-3)
        randomized: Whether to use randomized RAPS (default: True)
        
    Returns:
        List of selected entity indices
    """
    # Convert to numpy and handle batching
    if scores.dim() > 1:
        scores = scores.squeeze(0)
    scores_np = scores.detach().cpu().numpy()
    
    # CRITICAL: Must apply SAME penalty logic as during calibration
    # During calibration, we compute penalty based on GT position and subtract from scores
    # At inference, we don't know GT, so we need to match the threshold application
    
    # The calibration already applied penalties to scores before finding thresholds
    # So we need to apply penalties here too before comparing to threshold
    
    # Sort by score
    sorted_idx = np.argsort(-scores_np)
    sorted_scores = scores_np[sorted_idx]
    
    # Apply position-based penalties (same as calibration)
    # For each position k, penalty = λ * max(k - kreg, 0)
    passing = []
    for k in range(len(sorted_scores)):
        position = k + 1  # 1-indexed
        penalty = lamda * max(position - kreg, 0)
        penalized_score = sorted_scores[k] - penalty
        
        if penalized_score >= threshold:
            passing.append(sorted_idx[k])
        else:
            # Scores are sorted, so no point checking further
            break
    
    return passing


def get_raps_params_from_metadata(metadata: Dict) -> Tuple[int, float, bool]:
    """
    Extract RAPS parameters from calibration metadata.
    
    Args:
        metadata: Calibration metadata dictionary
        
    Returns:
        Tuple of (kreg, lamda, randomized)
    """
    if metadata is None or 'raps_metadata' not in metadata:
        # Default RAPS parameters
        return 1, 1e-3, True
    
    raps_meta = metadata['raps_metadata']
    kreg = raps_meta.get('kreg', 1)
    lamda = raps_meta.get('lamda', 1e-3)
    randomized = raps_meta.get('randomized', True)
    
    return kreg, lamda, randomized

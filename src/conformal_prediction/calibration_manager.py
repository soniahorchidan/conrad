"""
Calibration management for conformal risk control.
"""
import numpy as np
import torch
import logging
from typing import List, Dict, Any, Optional, Union
from scipy.optimize import brentq
from .vector_optimizer import VectorOptimizer
from .utils import get_alpha_bounds, lamhat_threshold, validate_calibration_data
from .model_config import ModelConfig

# Model name to query type mapping
MODEL_TO_QUERY_TYPE = {
    "threehoppipeline": "3p",
    "twounionpipeline": "2u",
    "twointersectprojectpipeline": "2ip",
}

def get_query_type_from_model(model_name: str) -> str:
    """
    Get query type from model name.
    
    Args:
        model_name: Model name (case-insensitive)
    
    Returns:
        Query type (e.g., '3p', '2u', '2ip')
    
    Raises:
        ValueError: If model name is not recognized
    """
    model_key = model_name.lower().replace("_", "").replace("-", "")
    if model_key not in MODEL_TO_QUERY_TYPE:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported models: {list(MODEL_TO_QUERY_TYPE.keys())}"
        )
    return MODEL_TO_QUERY_TYPE[model_key]

class CalibrationManager:
    """Manages calibration process for conformal risk control."""
    
    def __init__(self, model_name: str, num_entities: Optional[int] = None):
        """
        Initialize the calibration manager.
        
        Args:
            model_name: Name of the model
            num_entities: Number of entities in the dataset. If None, will try to get from model
                or default to 14541 (fb15k-237).
        """
        self.model_name = model_name
        self.is_vector_model = ModelConfig.is_vector_model(model_name)
        self.num_entities = num_entities  # Will be set from model if available

    def compute_statistics(self, cal_scores: List) -> Dict[str, np.ndarray]:
        """
        Compute statistics for calibration scores.
        
        Args:
            cal_scores: Calibration scores (numpy array or path-aware list)
            
        Returns:
            Dictionary containing statistics
        """
        # Check if it's the new path-aware format
        if isinstance(cal_scores, list) and len(cal_scores) > 0 and isinstance(cal_scores[0], dict):
            # Path-aware format - extract basic statistics from hop1 scores
            # (Statistics are less meaningful for path-aware data, but we keep them for compatibility)
            hop1_values_list = []
            for query_data in cal_scores:
                if 'hop1' in query_data and 'scores' in query_data['hop1']:
                    scores = query_data['hop1']['scores']
                    
                    # Handle sparse dict format
                    if isinstance(scores, dict) and 'values' in scores:
                        # Sparse format: extract just the values
                        values = scores['values']
                        if isinstance(values, torch.Tensor):
                            values = values.cpu().numpy()
                        hop1_values_list.extend(values)  # Flatten across queries
                    elif isinstance(scores, torch.Tensor):
                        # Old tensor format
                        scores_np = scores.cpu().numpy()
                        hop1_values_list.extend(scores_np.flatten())
            
            if hop1_values_list:
                hop1_values = np.array(hop1_values_list)
                return {
                    "median": float(np.median(hop1_values)),
                    "mean": float(np.mean(hop1_values)),
                    "var": float(np.var(hop1_values)),
                    "format": "path_aware"
                }
            else:
                return {"median": 0.0, "mean": 0.0, "var": 0.0, "format": "path_aware_empty"}
        
        # Old format
        cal_scores_array = np.array(cal_scores)
        
        if self.is_vector_model:
            # For vector models, compute statistics per hop
            median_scores = np.median(cal_scores_array, axis=(0, 2))
            mean_scores = np.mean(cal_scores_array, axis=(0, 2))
            var_scores = np.var(cal_scores_array, axis=(0, 2))
        else:
            # For scalar models, compute overall statistics
            median_scores = np.median(cal_scores_array)
            mean_scores = np.mean(cal_scores_array)
            var_scores = np.var(cal_scores_array)

        return {
            "median": median_scores,
            "mean": mean_scores,
            "var": var_scores,
            "format": "array"
        }

    def precalibrate_thresholds(self, cal_scores: List, true_labels: List,
                                alphas: Optional[List[float]] = None) -> Dict[float, Any]:
        """
        Precalibrate thresholds for common error levels.
        
        For vector models, uses batch optimization (much faster than sequential).
        For scalar models, falls back to sequential optimization.
        
        Args:
            cal_scores: Calibration scores
            true_labels: True labels
            alphas: List of alpha values to calibrate for (default: [0.1, 0.2, 0.3, 0.4, 0.5])
            
        Returns:
            Dictionary mapping alpha values to thresholds
        """
        if alphas is None:
            alphas = np.arange(0.1, 0.61, 0.1)
        
        # For vector models, use batch optimization (much faster!)
        if self.is_vector_model:
            try:
                validate_calibration_data(cal_scores, true_labels)
                query_type = get_query_type_from_model(self.model_name)
                optimizer = VectorOptimizer(
                    cal_scores, true_labels, query_type=query_type, 
                    num_entities=self.num_entities
                )
                optimizer.print_score_distributions()
                precalibrated_thresholds = optimizer.optimize_thresholds_batch(alphas)
                logging.info(f"Batch calibration for alphas {alphas} done!")
                return precalibrated_thresholds
            except Exception as e:
                logging.warning(f"Batch calibration failed: {e}. Falling back to sequential.")
                raise e

    def create_metadata(self, cal_scores: List, true_labels: List) -> Dict[str, Any]:
        """
        Create metadata for the calibration.
        
        Args:
            cal_scores: Calibration scores
            true_labels: True labels
            
        Returns:
            Metadata dictionary
        """
        # Compute statistics
        stats = self.compute_statistics(cal_scores)
        alpha_lowerbound, alpha_upperbound = get_alpha_bounds(len(cal_scores))
        
        # Precalibrate thresholds
        precalibrated_thresholds = self.precalibrate_thresholds(cal_scores, true_labels)

        # Create metadata based on model type
        if self.is_vector_model:
            # Track number of components (3p has 3 hops, 2u has 2 branches) for downstream consumers.
            try:
                query_type = get_query_type_from_model(self.model_name)
                num_components = 2 if query_type == "2u" else 3
            except Exception:
                num_components = 3

            # Convert to list if it's a numpy array, otherwise use as-is (for scalars)
            median_val = stats["median"].tolist() if hasattr(stats["median"], "tolist") else stats["median"]
            mean_val = stats["mean"].tolist() if hasattr(stats["mean"], "tolist") else stats["mean"]
            var_val = stats["var"].tolist() if hasattr(stats["var"], "tolist") else stats["var"]
            
            return {
                "median_non_conformity_score": median_val,
                "mean_non_conformity_score": mean_val,
                "variance_non_conformity": var_val,
                "alpha_lowerbound": alpha_lowerbound,
                "alpha_upperbound": alpha_upperbound,
                "calibrated_alphas": precalibrated_thresholds,
                "vector_scores": True,
                "num_hops": num_components
            }
        else:
            return {
                "median_non_conformity_score": float(stats["median"]),
                "mean_non_conformity_score": float(stats["mean"]),
                "variance_non_conformity": float(stats["var"]),
                "alpha_lowerbound": alpha_lowerbound,
                "alpha_upperbound": alpha_upperbound,
                "calibrated_alphas": precalibrated_thresholds,
                "vector_scores": False
            }

    def calibrate(self, cal_scores: List, true_labels: List) -> Dict[str, Any]:
        """
        Perform calibration using the correct FNR metric that matches validation.
        
        Args:
            cal_scores: Calibration scores
            true_labels: True labels
            
        Returns:
            Calibration metadata
        """        
        if self.is_vector_model:
            validate_calibration_data(cal_scores, true_labels)

        metadata = self.create_metadata(cal_scores, true_labels)
        return metadata

    def optimize_thresholds(self, cal_scores: List, true_labels: List, alpha: float) -> Any:
        """
        Optimize thresholds using the correct FNR metric that matches validation.
        
        Args:
            cal_scores: Calibration scores
            true_labels: True labels
            alpha: Target false negative rate
            
        Returns:
            Optimized thresholds
        """
        logging.info(f"Optimizing for alpha {alpha:.2f} with correct FNR for model {self.model_name}")
        
        if self.is_vector_model:
            validate_calibration_data(cal_scores, true_labels)
            query_type = get_query_type_from_model(self.model_name)
            optimizer = VectorOptimizer(cal_scores, true_labels, query_type=query_type)
            thresholds = optimizer.optimize_thresholds(alpha)
            logging.info(f"Vector conformal risk control calibration for alpha {alpha:.2f} done! thresholds={thresholds}")
        else:
            thresholds = brentq(lamhat_threshold, 0, 1, args=(cal_scores, true_labels, alpha))
            logging.info(f"Calibrating for alpha {alpha:.2f} done! thresholds={thresholds}")
        
        return thresholds

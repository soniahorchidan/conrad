import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace
from typing import List, Dict, Any, Optional, Tuple
from models import MultiHopPredictor
from .calibration_data_generator import CalibrationDataGenerator


class ScoreAggregator:
    """Handles score aggregation for multi-hop predictions."""
    
    @staticmethod
    def aggregate_scores(nodes: List[int], scores: List[torch.Tensor]) -> Tuple[List[int], List[torch.Tensor]]:
        """Aggregate scores for nodes that appear multiple times."""
        if not nodes or not scores:
            return [], []
            
        result_scores = {}
        for node, score in zip(nodes, scores):
            result_scores.setdefault(node, []).append(score)
        
        final_nodes = list(result_scores.keys())
        final_scores = [max(vals) for vals in result_scores.values()]
        return final_nodes, final_scores


class BasePipeline(nn.Module):
    """
    Base class for all pipeline models that provides common functionality.
    """
    
    def __init__(self, ultra_model, dbexec_model, args: Namespace, device: str):
        super().__init__()
        self.args = Namespace()
        for key, value in vars(args).items():
            if key != "db_controller":
                setattr(self.args, key, value)
        self.device = device
        
        # MultiHopPredictor combines Ultra + DBExec internally
        self.unified_predictor = MultiHopPredictor(
            ultra_model,
            dbexec_model,
            args,
            device
        )
        self.unified_predictor.conformal_prediction = None
        
        # Initialize helper components
        calib_batch_size = getattr(args, 'calib_batch_size', 4)
        from .calibration_data_generator import CalibrationDataGenerator
        num_entities = getattr(args, 'num_entities', None)  # Get from args if available
        self.calibration_generator = CalibrationDataGenerator(
            self, device, calib_batch_size=calib_batch_size, num_entities=num_entities
        )
    
    def forward(self, graph_data, query):
        return self.unified_predictor(graph_data, query)
    
    
    @torch.no_grad()
    def generateCalibrateSamples(self, save_path: str, db_controller: Any,
                                calib_iterator: Optional[Any] = None, calibration_data_path: Optional[str] = None,
                                return_queries: bool = False) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        """Generate calibration samples using the calibration generator."""
        return self.calibration_generator.generate_calibration_samples(
            save_path, db_controller,
            calib_iterator, calibration_data_path, return_queries
        )
    
    def _extract_nodes_from_scores(self, scores: torch.Tensor, threshold: float) -> List[int]:
        """Extract nodes from scores using simple thresholding."""
        # Simple threshold-based filtering
        nonzero_indices = torch.nonzero(scores >= threshold)
        if nonzero_indices.size(0) > 0:
            node_dim = scores.dim() - 1
            return nonzero_indices[:, node_dim].tolist()
        return []
    
    @staticmethod
    def _extract_nodes_by_threshold(scores, threshold: float) -> set:
        """Extract nodes that pass threshold from score dict/tensor."""
        if isinstance(scores, dict) and 'indices' in scores and 'values' in scores:
            # Sparse format
            indices = scores['indices']
            values = scores['values']
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy()
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            return set(indices[values >= threshold])
        else:
            # Dense format
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            return set(np.where(scores >= threshold)[0])
    
    @staticmethod
    def _scores_to_dense_vector(scores, num_entities: int = 14541) -> np.ndarray:
        """Convert sparse or dense scores to dense numpy vector."""
        if isinstance(scores, dict) and 'indices' in scores and 'values' in scores:
            # Sparse format
            indices = scores['indices']
            values = scores['values']
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy()
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            
            dense = np.zeros(num_entities, dtype=np.float32)
            dense[indices] = values
            return dense
        else:
            # Already dense
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            if isinstance(scores, np.ndarray):
                return scores.astype(np.float32)
            if scores.ndim > 1:
                scores = scores.squeeze()
            # Pad or trim to num_entities
            if len(scores) < num_entities:
                padded = np.zeros(num_entities, dtype=np.float32)
                padded[:len(scores)] = scores
                return padded
            else:
                return scores[:num_entities]
    
    def _handle_confidence_modes(self, query: torch.Tensor, confidence: float, graph_data: Any,
                                 expected_length: int, error_msg: str, debug_thresholds: List[float]) -> Optional[List]:
        """
        Common confidence mode handling for all pipelines.
        
        Args:
            query: Query tensor
            confidence: Confidence level
            graph_data: Graph data
            expected_length: Expected query length for validation
            error_msg: Error message for invalid query length
            debug_thresholds: Default thresholds for debug mode (confidence=0.0)
        
        Returns:
            Predictions or None
        """
        # Validate query format
        if len(query[0]) != expected_length:
            raise ValueError(error_msg)
        
        query = query.to(self.device)
        
        # Handle different confidence modes
        if confidence == 1.0:
            raise ValueError("Confidence is set to 1.0. Returning no results.")
        elif confidence == 0.0:
            # Debug mode: use default thresholds
            import logging
            logging.info("Using normal inference without conformal prediction. DEBUG ONLY!")
            return self._predict(query, debug_thresholds, graph_data)
        else:
            # Conformal prediction mode
            if hasattr(self, 'conformal_prediction') and self.conformal_prediction is not None:
                return self.conformal_prediction.predict(query, confidence, graph_data=graph_data)
            else:
                raise ValueError(f"Conformal prediction not available for {self.__class__.__name__}")

import torch
import numpy as np
from argparse import Namespace
from typing import List, Dict, Any, Optional
from .base import BasePipeline


class TwoUnionPipeline(BasePipeline):
    """
    Pipeline for 2u (union) queries: ((anchor1, rel1), (anchor2, rel2)).
    
    Query execution:
    1. Predict entities from anchor1 via rel1 → set S1
    2. Predict entities from anchor2 via rel2 → set S2  
    3. Return union S1 ∪ S2
    """
    
    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """
        Query format: [anchor1, rel1, anchor2, rel2]
        Returns union of predictions from both branches.
        """
        return self._handle_confidence_modes(
            query, confidence, graph_data,
            expected_length=4,
            error_msg="TwoUnionPipeline expects queries of length 4: [anchor1, rel1, anchor2, rel2]",
            debug_thresholds=[0.4, 0.4]  # Two thresholds for two branches
        )
    
    @staticmethod
    def apply_thresholds_to_scores(cal_scores: List[Dict[str, Any]], thresholds: List[float],
                                   true_labels: Optional[List[Dict]] = None,
                                   num_entities: int = 14541) -> List[List[int]]:
        """
        Apply thresholds to 2u calibration scores.
        
        Args:
            cal_scores: List of score dictionaries with 'branch1' and 'branch2' keys
            thresholds: [τ1, τ2] for two branches
            true_labels: Optional ground truth (not used)
            num_entities: Number of entities (for consistency with other pipelines, not used here)
            
        Returns:
            List of prediction sets (one per query)
        """
        predictions = []
        
        for idx, query_data in enumerate(cal_scores):
            # Extract scores from both branches
            branch1_scores = query_data['branch1']['scores']
            branch2_scores = query_data['branch2']['scores']
            
            # Apply thresholds to each branch
            branch1_passing = TwoUnionPipeline._extract_nodes_by_threshold(branch1_scores, thresholds[0])
            branch2_passing = TwoUnionPipeline._extract_nodes_by_threshold(branch2_scores, thresholds[1])
            
            # Union of both branches
            union_nodes = branch1_passing | branch2_passing
            predictions.append(list(union_nodes))
        
        return predictions
    
    @torch.no_grad()
    def predict_with_thresholds(self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
                               ground_truth: Optional[List[List[int]]] = None) -> List[List[int]]:
        """
        Run 2u inference with threshold-based filtering.
        
        Args:
            query: Tensor of shape (batch_size, 4) containing [anchor1, rel1, anchor2, rel2]
            lamhat: List of 2 thresholds [τ1, τ2] for each branch
            graph_data: Graph data for inference
            ground_truth: Optional list of GT entity lists (used during calibration for filtering)
            
        Returns:
            List of predicted entity lists (union of both branches), one per query
        """
        batch_results = self._predict(query, lamhat, graph_data, ground_truth)
        
        # Extract final union nodes
        predictions = [result["union_nodes"] for result in batch_results]
        
        return predictions
    
    @torch.no_grad()
    def _predict(self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
                ground_truth: Optional[List[List[int]]] = None) -> List[Dict[str, Any]]:
        """
        Internal method that performs 2u prediction.
        
        Args:
            query: Query tensor [anchor1, rel1, anchor2, rel2]
            lamhat: [τ1, τ2] thresholds for two branches
            graph_data: Graph data
            ground_truth: Optional GT nodes for calibration filtering
            
        Returns:
            List of result dictionaries with branch scores and union nodes
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        batch_size = query.shape[0]
        batch_results = []
        
        for i in range(batch_size):
            anchor1 = query[i, 0].item()
            rel1 = query[i, 1].item()
            anchor2 = query[i, 2].item()
            rel2 = query[i, 3].item()
            
            # Branch 1: anchor1 → rel1
            source1_tensor = torch.tensor([[anchor1]], dtype=torch.long, device=self.device)
            rel1_tensor = torch.tensor([[rel1]], dtype=torch.long, device=self.device)
            scores1, _ = self.unified_predictor.predict(source1_tensor, rel1_tensor, graph_data)
            
            # Branch 2: anchor2 → rel2
            source2_tensor = torch.tensor([[anchor2]], dtype=torch.long, device=self.device)
            rel2_tensor = torch.tensor([[rel2]], dtype=torch.long, device=self.device)
            scores2, _ = self.unified_predictor.predict(source2_tensor, rel2_tensor, graph_data)
            
            # Apply thresholds to extract nodes
            threshold1 = lamhat[0] if len(lamhat) > 0 else 0.0
            threshold2 = lamhat[1] if len(lamhat) > 1 else 0.0
            
            nodes1 = self._extract_nodes_from_scores(scores1, threshold1)
            nodes2 = self._extract_nodes_from_scores(scores2, threshold2)
            
            # Union of both branches
            union_nodes = list(set(nodes1) | set(nodes2))
            
            # Apply GT filtering if provided (for calibration)
            if ground_truth is not None and i < len(ground_truth):
                # Extract ground truth from dict or use directly if list/set
                gt_data = ground_truth[i]
                if isinstance(gt_data, dict):
                    # For 2u queries: compute union of branches 1 and 2
                    branch1 = set(gt_data.get(1, []))
                    branch2 = set(gt_data.get(2, []))
                    gt_set = branch1 | branch2
                elif isinstance(gt_data, (list, set)):
                    gt_set = set(gt_data)
                else:
                    gt_set = set()
                # During calibration, restrict to GT nodes + high-scoring nodes
                scores1_np = scores1.squeeze(0).cpu().numpy()
                scores2_np = scores2.squeeze(0).cpu().numpy()
                
                # MAX the scores from both branches
                combined_scores = np.maximum(scores1_np, scores2_np)
                
                # Get min GT score from combined scores
                gt_list = list(gt_set)
                if gt_list:
                    gt_scores_combined = combined_scores[gt_list]
                    min_gt_score = float(gt_scores_combined.min()) if len(gt_scores_combined) > 0 else 0.0
                    
                    # Filter: include GT nodes + nodes scoring above min GT score in combined scores
                    filtered_nodes = []
                    for node in union_nodes:
                        if node in gt_set or combined_scores[node] >= min_gt_score:
                            filtered_nodes.append(node)
                    union_nodes = filtered_nodes
            
            batch_results.append({
                "branch1": {"nodes": nodes1, "scores": scores1},
                "branch2": {"nodes": nodes2, "scores": scores2},
                "union_nodes": union_nodes
            })
        
        return batch_results

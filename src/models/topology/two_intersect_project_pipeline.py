import torch
import numpy as np
from argparse import Namespace
from typing import List, Dict, Any, Optional
from .base import BasePipeline


class TwoIntersectProjectPipeline(BasePipeline):
    """
    Pipeline for 2ip (intersect-project) queries: ((anchor1, rel1), (anchor2, rel2), rel3).
    
    Query execution with independent branch thresholds (3D optimization):
    1. Predict entities from anchor1 via rel1 → scores_branch1
    2. Predict entities from anchor2 via rel2 → scores_branch2
    3. Apply τ_branch1 to scores_branch1 → branch1_nodes
    4. Apply τ_branch2 to scores_branch2 → branch2_nodes
    5. Compute discrete intersection: intersection_nodes = branch1_nodes ∩ branch2_nodes
    6. For each entity in intersection_nodes, predict via rel3 → MAX-aggregate projection scores
    7. Apply τ_proj threshold → final answer set
    
    This 3D formulation (τ_branch1, τ_branch2, τ_proj) allows independent pruning of noisy branches
    and maintains the nestedness property required for Conformal Risk Control.
    """
    
    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """
        Query format: [anchor1, rel1, anchor2, rel2, rel3]
        Returns predictions using independent branch thresholds and MAX-aggregation for projection.
        
        Uses 3D optimization: [τ_branch1, τ_branch2, τ_proj]
        - τ_branch1: Applied independently to branch1 scores
        - τ_branch2: Applied independently to branch2 scores
        - τ_proj: Applied to MAX-aggregated projection scores
        """
        return self._handle_confidence_modes(
            query, confidence, graph_data,
            expected_length=5,
            error_msg="TwoIntersectProjectPipeline expects queries of length 5: [anchor1, rel1, anchor2, rel2, rel3]",
            debug_thresholds=[0.4, 0.4, 0.4]  # Three thresholds: branch1, branch2, projection
        )
    
    @torch.no_grad()
    def predict_with_thresholds(self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
                               ground_truth: Optional[List[List[int]]] = None) -> List[List[int]]:
        """
        Run 2ip inference with threshold-based filtering using 3D optimization.
        
        Args:
            query: Tensor of shape (batch_size, 5) containing [anchor1, rel1, anchor2, rel2, rel3]
            lamhat: List of 3 thresholds [τ_branch1, τ_branch2, τ_proj]:
                    - τ_branch1: Applied independently to branch1 scores
                    - τ_branch2: Applied independently to branch2 scores
                    - τ_proj: Applied to MAX-aggregated projection scores
            graph_data: Graph data for inference
            ground_truth: Optional dict with GT nodes per hop (used during calibration for filtering)
            
        Returns:
            List of predicted entity lists (from projection), one per query
        """
        batch_results = self._predict(query, lamhat, graph_data, ground_truth)
        
        # Extract final projection nodes
        predictions = [result["final_nodes"] for result in batch_results]
        
        return predictions
    
    @torch.no_grad()
    def _predict(self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
                ground_truth: Optional[List[List[int]]] = None) -> List[Dict[str, Any]]:
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Avoid CUDA syncs from `.item()` when query arrives on GPU.
        query_cpu = query.detach().cpu() if isinstance(query, torch.Tensor) and query.is_cuda else query
        
        batch_size = query.shape[0]
        batch_results = []
        
        for i in range(batch_size):
            anchor1, rel1 = query_cpu[i, 0].item(), query_cpu[i, 1].item()
            anchor2, rel2 = query_cpu[i, 2].item(), query_cpu[i, 3].item()
            rel3 = query_cpu[i, 4].item()
            
            # 2. Independent Branch Thresholding (3D optimization)
            t_branch1 = lamhat[0] if len(lamhat) > 0 else 0.0
            t_branch2 = lamhat[1] if len(lamhat) > 1 else 0.0
            t_proj = lamhat[2] if len(lamhat) > 2 else 0.0
            
            # 1. Branch Execution
            source1_t = torch.tensor([[anchor1]], dtype=torch.long, device=self.device)
            rel1_t = torch.tensor([[rel1]], dtype=torch.long, device=self.device)
            scores1, _ = self.unified_predictor.predict(source1_t, rel1_t, graph_data, threshold=t_branch1)
            
            source2_t = torch.tensor([[anchor2]], dtype=torch.long, device=self.device)
            rel2_t = torch.tensor([[rel2]], dtype=torch.long, device=self.device)
            scores2, _ = self.unified_predictor.predict(source2_t, rel2_t, graph_data, threshold=t_branch2)
            
            # Apply thresholds independently to each branch
            branch1_nodes = self._extract_nodes_from_scores(scores1, t_branch1)
            branch2_nodes = self._extract_nodes_from_scores(scores2, t_branch2)
            
            # Compute discrete intersection of the two sets
            branch1_set = set(branch1_nodes)
            branch2_set = set(branch2_nodes)
            intersection_nodes = list(branch1_set & branch2_set)
            
            # Calibration-only GT filtering
            if ground_truth is not None and i < len(ground_truth):
                gt_data = ground_truth[i]
                gt_hop1 = set(gt_data.get(1, [])) if isinstance(gt_data, dict) else set()
                gt_hop2 = set(gt_data.get(2, [])) if isinstance(gt_data, dict) else set()
                gt_intersection = list(gt_hop1 & gt_hop2)
                
                if gt_intersection:
                    intersection_nodes = self._apply_2ip_gt_filtering(
                        scores1, scores2, t_branch1, t_branch2, gt_intersection
                    )
            
            # 3. Path-Aware Projection
            projection_paths = [] # CRITICAL: Store scores per parent
            final_nodes = []
            projection_scores_dict = {}
            all_proj_scores = []
            projection_nodes = []  # Initialize to handle case when intersection_nodes is empty
            
            if len(intersection_nodes) > 0:
                max_batch_size = 64
                
                for b_start in range(0, len(intersection_nodes), max_batch_size):
                    b_end = min(b_start + max_batch_size, len(intersection_nodes))
                    b_nodes = intersection_nodes[b_start:b_end]
                    
                    int_tensor = torch.tensor(b_nodes, dtype=torch.long, device=self.device).unsqueeze(1)
                    rel3_tensor = torch.tensor([[rel3]] * len(b_nodes), dtype=torch.long, device=self.device)
                    
                    p_scores, _ = self.unified_predictor.predict(int_tensor, rel3_tensor, graph_data, threshold=t_proj)
                    
                    for j, parent_id in enumerate(b_nodes):
                        s_vec = p_scores[j].detach().cpu()
                        all_proj_scores.append(s_vec)
                        
                        # Store sparse scores so the optimizer can find GT even if it failed t_proj
                        indices = torch.nonzero(s_vec).squeeze()
                        if indices.dim() == 0: indices = indices.unsqueeze(0)
                        
                        projection_paths.append({
                            'parent': parent_id,
                            'scores': {
                                'indices': indices.numpy(),
                                'values': s_vec[indices].numpy()
                            }
                        })

                # 4. Final Aggregation (for current prediction)
                if all_proj_scores:
                    max_p_scores = torch.stack(all_proj_scores).max(dim=0).values
                    final_nodes = self._extract_nodes_from_scores(max_p_scores.unsqueeze(0), t_proj)
                    
                    # Aggregate projection scores into dict format for calibration
                    # MAX-aggregate: for each node, take the maximum score across all paths
                    for proj_score_vec in all_proj_scores:
                        proj_score_vec_cpu = proj_score_vec.cpu()
                        for node_id in range(len(proj_score_vec_cpu)):
                            score_val = float(proj_score_vec_cpu[node_id].item())
                            if score_val > 0:  # Only store non-zero scores
                                if node_id not in projection_scores_dict:
                                    projection_scores_dict[node_id] = score_val
                                else:
                                    projection_scores_dict[node_id] = max(projection_scores_dict[node_id], score_val)
                    
                    # Extract projection nodes from aggregated scores (for calibration)
                    projection_nodes = self._extract_nodes_from_scores(max_p_scores.unsqueeze(0), 0.0)
                else:
                    projection_nodes = []

            # 5. Result Construction
            # Extract nodes from branch scores (for calibration) - extract all nodes (threshold=0.0)
            nodes1 = self._extract_nodes_from_scores(scores1, 0.0)
            nodes2 = self._extract_nodes_from_scores(scores2, 0.0)
            
            # Compute MIN-aggregated scores for backward compatibility (used in calibration data extraction)
            min_aggregated_scores = torch.minimum(scores1.squeeze(0), scores2.squeeze(0))
            
            batch_results.append({
                "branch1": {"nodes": nodes1, "scores": scores1},
                "branch2": {"nodes": nodes2, "scores": scores2},
                "intersection": {
                    "nodes": intersection_nodes, 
                    "scores": min_aggregated_scores.cpu().numpy()  # Keep for backward compatibility
                },
                "projection": {
                    "nodes": projection_nodes,
                    "scores": projection_scores_dict  # Dict format for calibration
                },
                "projection_paths": projection_paths,  # Keep path-aware format for path-aware processing
                "final_nodes": final_nodes
            })
        
        return batch_results
    
    def _apply_2ip_gt_filtering(self, scores1: torch.Tensor, scores2: torch.Tensor, 
                                t_branch1: float, t_branch2: float,
                                gt_intersection: List[int], max_nodes: int = 1000) -> List[int]:
        """
        Apply GT filtering for independent branch thresholds to ensure the intersection frontier 
        contains all potential true answers for calibration.
        
        Args:
            scores1: Branch 1 scores tensor
            scores2: Branch 2 scores tensor
            t_branch1: Threshold for branch 1
            t_branch2: Threshold for branch 2
            gt_intersection: List of ground truth intersection nodes
            max_nodes: Maximum number of nodes to return
            
        Returns:
            Filtered list of intersection nodes that should be preserved for calibration
        """
        # Convert scores to numpy arrays
        if isinstance(scores1, torch.Tensor):
            scores1_np = scores1.squeeze(0).cpu().numpy()
        else:
            scores1_np = np.array(scores1).flatten()
            
        if isinstance(scores2, torch.Tensor):
            scores2_np = scores2.squeeze(0).cpu().numpy()
        else:
            scores2_np = np.array(scores2).flatten()
        
        # 1. Extract nodes that pass each branch threshold independently
        branch1_passing = set(np.where(scores1_np >= t_branch1)[0])
        branch2_passing = set(np.where(scores2_np >= t_branch2)[0])
        
        # 2. Compute intersection of nodes passing both thresholds
        intersection_passing = branch1_passing & branch2_passing
        
        if not gt_intersection:
            return list(intersection_passing)
        
        # 3. Identify GT nodes that should be preserved
        gt_set = set(gt_intersection)
        gt_nodes_valid = [n for n in gt_intersection 
                         if n < len(scores1_np) and n < len(scores2_np)]
        
        if not gt_nodes_valid:
            return list(intersection_passing)
        
        # 4. For GT nodes, check if they pass both thresholds
        # If a GT node doesn't pass both thresholds, we need to ensure the optimizer
        # can see why it failed (by including nodes that score above the GT's minimum score)
        gt_branch1_scores = [scores1_np[n] for n in gt_nodes_valid]
        gt_branch2_scores = [scores2_np[n] for n in gt_nodes_valid]
        
        min_gt_branch1_score = float(min(gt_branch1_scores)) if gt_branch1_scores else 0.0
        min_gt_branch2_score = float(min(gt_branch2_scores)) if gt_branch2_scores else 0.0
        
        # 5. Include nodes that pass thresholds AND score above GT minimums
        # This ensures the optimizer sees the full context of why GT survived or failed
        filtered_nodes = []
        for node_id in intersection_passing:
            if (scores1_np[node_id] >= min_gt_branch1_score and 
                scores2_np[node_id] >= min_gt_branch2_score):
                filtered_nodes.append(node_id)
        
        # Also include GT nodes themselves if they're in the valid range
        filtered_set = set(filtered_nodes)
        filtered_set.update(gt_nodes_valid)
        filtered_nodes = list(filtered_set)
        
        # 6. Physical Constraint: Bounding the frontier for tractability
        if len(filtered_nodes) > max_nodes:
            # Score nodes by their minimum score across both branches (weakest link)
            node_scores = [(n, min(scores1_np[n], scores2_np[n])) for n in filtered_nodes]
            node_scores.sort(key=lambda x: x[1], reverse=True)
            top_nodes = [n for n, s in node_scores[:max_nodes]]
            
            # Ensure GT nodes are included if possible
            top_set = set(top_nodes)
            missing_gt = gt_set - top_set
            if len(missing_gt) > 0 and len(top_nodes) + len(missing_gt) <= max_nodes * 2:
                # Add missing GT nodes if we have room
                filtered_nodes = top_nodes + list(missing_gt)
            else:
                filtered_nodes = top_nodes
        
        return filtered_nodes
    
    @staticmethod
    def apply_thresholds_to_scores(cal_scores, thresholds, true_labels=None, num_entities: int = 14541):
        """
        Apply thresholds to 2ip calibration scores using 3D optimization.
        
        Args:
            cal_scores: List of score dictionaries with 'branch1', 'branch2', and 'projection_paths' keys
            thresholds: [τ_branch1, τ_branch2, τ_proj] - three thresholds for independent branch pruning and projection
            true_labels: Optional ground truth (not used)
            num_entities: Number of entities in the dataset
            
        Returns:
            List of prediction sets (one per query)
        """
        predictions = []
        t_branch1 = thresholds[0] if len(thresholds) > 0 else 0.0
        t_branch2 = thresholds[1] if len(thresholds) > 1 else 0.0
        t_proj = thresholds[2] if len(thresholds) > 2 else 0.0
        
        for query_data in cal_scores:
            # 1. Independent Branch Thresholding
            branch1_scores = query_data['branch1']['scores']
            branch2_scores = query_data['branch2']['scores']
            
            # Convert to dense vectors
            b1_dense = TwoIntersectProjectPipeline._scores_to_dense_vector(branch1_scores, num_entities=num_entities)
            b2_dense = TwoIntersectProjectPipeline._scores_to_dense_vector(branch2_scores, num_entities=num_entities)
            
            # Apply thresholds independently to each branch
            branch1_passing = set(np.where(b1_dense >= t_branch1)[0])
            branch2_passing = set(np.where(b2_dense >= t_branch2)[0])
            
            # Compute discrete intersection: nodes that pass BOTH thresholds
            int_passing = branch1_passing & branch2_passing
            
            if not int_passing:
                predictions.append([])
                continue
                
            # 2. Projection Stage (PATH-AWARE)
            # Only include projection paths whose parent exists in the intersection set
            proj_paths = query_data.get('projection_paths', [])
            valid_proj_vectors = []
            
            for path in proj_paths:
                if path['parent'] in int_passing:
                    vec = TwoIntersectProjectPipeline._scores_to_dense_vector(path['scores'], num_entities=num_entities)
                    valid_proj_vectors.append(vec)
            
            if not valid_proj_vectors:
                predictions.append([])
                continue
                
            # RE-AGGREGATE: MAX scores across all surviving paths
            final_max_scores = np.maximum.reduce(valid_proj_vectors)
            
            # Apply τ_proj threshold
            final_indices = np.where(final_max_scores >= t_proj)[0]
            predictions.append(final_indices.tolist())
            
        return predictions

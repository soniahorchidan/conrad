import torch
import logging
import numpy as np
from argparse import Namespace
from typing import List, Dict, Any, Optional, Tuple
from .base import BasePipeline


class ThreeHopPipeline(BasePipeline):
    """
    A wrapper pipeline model that uses MultiHopPredictor
    to approximate each hop of a 3-hop path.
    """

    def __init__(self, ultra_model, dbexec_model, args: Namespace, device: str):
        super().__init__(ultra_model, dbexec_model, args, device)
        self.max_internal_batch = getattr(args, 'max_internal_batch', 64)
        self.multihop_slack = 0.1

    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """Query format: [src, rel1, rel2, rel3]"""
        return self._handle_confidence_modes(
            query, confidence, graph_data,
            expected_length=4,
            error_msg="ThreeHopPipeline only supports 3-hop queries (len=4).",
            debug_thresholds=[0.4, 0.33, 0.6]
        )

    @staticmethod
    def apply_thresholds_to_scores(cal_scores: List[Dict[str, Any]], thresholds: List[float], 
                                   true_labels: Optional[List[Dict]] = None, 
                                   num_entities: int = 14541) -> List[List[int]]:
        """Apply thresholds to calibration scores for 3-hop cascade."""
        predictions = []
        for query_data in cal_scores:
            hop1_valid = ThreeHopPipeline._process_hop1_for_calibration(query_data, thresholds[0])
            if not hop1_valid:
                predictions.append([])
                continue
            
            hop2_valid = ThreeHopPipeline._process_hop2_for_calibration(
                query_data, thresholds[1], hop1_valid, num_entities
            )
            if not hop2_valid:
                predictions.append([])
                continue
            
            hop3_predictions = ThreeHopPipeline._process_hop3_for_calibration(
                query_data, thresholds[2], hop1_valid, hop2_valid, num_entities
            )
            predictions.append(hop3_predictions)
        return predictions
    
    @staticmethod
    def _process_hop1_for_calibration(query_data: Dict, threshold: float) -> set:
        """Extract hop1 nodes that pass threshold and are in calibration data."""
        hop1_scores = query_data['hop1']['scores']
        hop1_passing = ThreeHopPipeline._extract_nodes_by_threshold(hop1_scores, threshold)
        return hop1_passing & set(query_data['hop1']['nodes'])
    
    @staticmethod
    def _process_hop2_for_calibration(query_data: Dict, threshold: float, 
                                     hop1_valid: set, num_entities: int) -> set:
        """Aggregate hop2 scores from valid hop1 paths, apply threshold, filter by calibration nodes."""
        hop2_vectors = [
            ThreeHopPipeline._scores_to_dense_vector(hop2_path['scores'], num_entities)
            for hop2_path in query_data['hop2']
            if hop2_path['parent'] in hop1_valid
        ]
        
        if not hop2_vectors:
            return set()
        
        hop2_max = np.maximum.reduce(hop2_vectors)
        hop2_passing = set(np.where(hop2_max >= threshold)[0])
        
        # Filter to calibration-tracked nodes (from hop3 path parents)
        hop2_calibration = {
            hop3_path['parent'][1] 
            for hop3_path in query_data['hop3']
            if hop3_path['parent'][0] in hop1_valid
        }
        
        return hop2_passing & hop2_calibration
    
    @staticmethod
    def _process_hop3_for_calibration(query_data: Dict, threshold: float,
                                     hop1_valid: set, hop2_valid: set, num_entities: int) -> List[int]:
        """Aggregate hop3 scores from valid (hop1, hop2) paths and apply threshold."""
        hop3_vectors = [
            ThreeHopPipeline._scores_to_dense_vector(hop3_path['scores'], num_entities)
            for hop3_path in query_data['hop3']
            if hop3_path['parent'][0] in hop1_valid and hop3_path['parent'][1] in hop2_valid
        ]
        
        if not hop3_vectors:
            return []
        
        hop3_max = np.maximum.reduce(hop3_vectors)
        return np.where(hop3_max >= threshold)[0].tolist()
    
    @torch.no_grad()
    def predict_with_thresholds(self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
                               ground_truth_hops: Optional[List[Dict]] = None) -> List[List[int]]:
        """Run cascade inference with threshold-based filtering at each hop."""
        batch_results = self._predict(query, lamhat, graph_data, ground_truth_hops)
        return [result["hop3"]["nodes"] for result in batch_results]
    
    @torch.no_grad()
    def _predict(self, query: torch.Tensor, lamhat: List[float], graph_data: Any, 
                ground_truth_hops: Optional[List[Dict]] = None) -> List[Dict[str, Dict[str, Any]]]:
        """Internal method that performs multi-hop cascade prediction."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        use_ground_truth = ground_truth_hops is not None
        batch_size = query.shape[0]
        
        hop1_results = self._process_hop1(query, lamhat, graph_data, ground_truth_hops, use_ground_truth)
        hop2_results = self._process_hop2(hop1_results, query, lamhat, graph_data, ground_truth_hops, use_ground_truth)
        hop3_results = self._process_hop3(hop2_results, query, lamhat, graph_data, ground_truth_hops, use_ground_truth)
        
        return self._aggregate_batch_results(hop1_results, hop2_results, hop3_results, batch_size)
    
    def _process_hop1(self, query: torch.Tensor, lamhat: List[float], graph_data: Any, 
                     ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool) -> Dict[str, Any]:
        """Process hop 1 for all queries in batch (special case: single source per query)."""
        batch_size = query.shape[0]
        all_nodes, all_scores = [], []
        
        for i in range(batch_size):
            source = query[i, 0].item()
            relation = query[i, 1].item()
            
            source_tensor = torch.tensor([[source]], dtype=torch.long, device=self.device)
            rel_tensor = torch.tensor([[relation]], dtype=torch.long, device=self.device)
            scores, _ = self.unified_predictor.predict(source_tensor, rel_tensor, graph_data)
            
            threshold = self._get_threshold(lamhat, 0)
            nodes = self._extract_nodes_from_scores(scores, threshold)
            
            if use_ground_truth and self._has_gt_for_hop(ground_truth_hops, i, 1):
                scores_np = scores.squeeze(0).cpu().numpy()
                nodes = self._apply_gt_filtering(nodes, scores_np, ground_truth_hops[i][1], max_nodes=50)
            
            all_nodes.append(nodes)
            all_scores.append(scores)
        
        return {"nodes": all_nodes, "scores": all_scores}
    
    def _process_hop2(self, hop1_results: Dict[str, Any], query: torch.Tensor, lamhat: List[float], 
                     graph_data: Any, ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool) -> Dict[str, Any]:
        """Process hop 2 for all queries in batch using MAX-then-threshold aggregation."""
        return self._process_multi_source_hop(
            hop_num=2, prev_results=hop1_results, query=query, 
            relation_idx=2, lamhat=lamhat, graph_data=graph_data,
            ground_truth_hops=ground_truth_hops, use_ground_truth=use_ground_truth
        )
    
    def _process_hop3(self, hop2_results: Dict[str, Any], query: torch.Tensor, lamhat: List[float], 
                     graph_data: Any, ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool) -> Dict[str, Any]:
        """Process hop 3 for all queries in batch using MAX-then-threshold aggregation."""
        return self._process_multi_source_hop(
            hop_num=3, prev_results=hop2_results, query=query, 
            relation_idx=3, lamhat=lamhat, graph_data=graph_data,
            ground_truth_hops=ground_truth_hops, use_ground_truth=use_ground_truth,
            parent_map_builder=lambda i: self._build_hop2_to_hop1_map(
                hop2_results.get("scores", [])[i] if hop2_results.get("path_aware") else [],
                lamhat
            )
        )
    
    def _process_multi_source_hop(self, hop_num: int, prev_results: Dict[str, Any], 
                                   query: torch.Tensor, relation_idx: int, lamhat: List[float],
                                   graph_data: Any, ground_truth_hops: Optional[List[Dict]], 
                                   use_ground_truth: bool, 
                                   parent_map_builder: Optional[callable] = None) -> Dict[str, Any]:
        """Unified processing for hop2 and hop3 (multi-source hops)."""
        batch_size = query.shape[0]
        all_nodes, all_scores = [], []
        
        for i in range(batch_size):
            source_nodes = list(prev_results["nodes"][i]) if prev_results["nodes"][i] else []
            relation = query[i, relation_idx].item()
            parent_map = parent_map_builder(i) if parent_map_builder else None
            
            path_scores, _, filtered_nodes = self._process_hop_generic(
                hop_num=hop_num, source_nodes=source_nodes, relation=relation, query_idx=i,
                lamhat=lamhat, graph_data=graph_data, 
                ground_truth_hops=ground_truth_hops, use_ground_truth=use_ground_truth,
                parent_map=parent_map
            )
            
            all_nodes.append(list(filtered_nodes))
            all_scores.append(path_scores)
        
        return {"nodes": all_nodes, "scores": all_scores, "path_aware": True}
    
    def _process_hop_generic(self, hop_num: int, source_nodes: List[int], relation: int, 
                            query_idx: int, lamhat: List[float], graph_data: Any,
                            ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool,
                            parent_map: Optional[Dict] = None) -> Tuple[List[Dict], List[torch.Tensor], set]:
        """Generic hop processing that handles chunking, aggregation, and GT filtering."""
        if not source_nodes:
            return [], [], set()
        
        path_scores, score_tensors = [], []
        
        # Process sources in chunks to avoid GPU OOM
        for chunk_start in range(0, len(source_nodes), self.max_internal_batch):
            chunk_nodes = source_nodes[chunk_start:chunk_start + self.max_internal_batch]
            
            batch_source = torch.tensor([[s] for s in chunk_nodes], dtype=torch.long, device=self.device)
            batch_rel = torch.tensor([[relation]] * len(chunk_nodes), dtype=torch.long, device=self.device)
            batch_scores, _ = self.unified_predictor.predict(batch_source, batch_rel, graph_data)
            
            for idx, source in enumerate(chunk_nodes):
                scores = batch_scores[idx:idx+1]
                
                # Track parent info for path awareness
                if parent_map and source in parent_map:
                    # For hop3: track (hop1_parent, hop2_node) pairs
                    for hop1_parent in parent_map[source]:
                        path_scores.append({'parent': (hop1_parent, source), 'scores': scores})
                else:
                    # For hop2: track hop1 parent
                    path_scores.append({'parent': source, 'scores': scores})
                
                score_tensors.append(scores.squeeze(0))
        
        # Aggregate scores with MAX, apply threshold, and optionally filter by GT
        gt_nodes = None
        max_gt_nodes = {1: 50, 2: 50, 3: None}[hop_num]
        if use_ground_truth and self._has_gt_for_hop(ground_truth_hops, query_idx, hop_num):
            gt_nodes = ground_truth_hops[query_idx][hop_num]
        
        filtered_nodes = self._aggregate_and_threshold(
            score_tensors, hop_idx=hop_num-1, lamhat=lamhat, 
            gt_nodes=gt_nodes, max_gt_nodes=max_gt_nodes
        )
        
        return path_scores, score_tensors, filtered_nodes
    
    def _build_hop2_to_hop1_map(self, hop2_path_scores: List[Dict], lamhat: List[float]) -> Dict[int, List[int]]:
        """Map hop2 nodes to their hop1 parents for path tracking."""
        hop2_to_hop1 = {}
        threshold = self._get_threshold(lamhat, 1)
        
        for path_data in hop2_path_scores:
            hop1_parent = path_data['parent']
            hop2_nodes = self._extract_nodes_from_scores(path_data['scores'], threshold)
            
            for hop2_node in hop2_nodes:
                hop2_to_hop1.setdefault(hop2_node, []).append(hop1_parent)
        
        return hop2_to_hop1
    
    def _aggregate_and_threshold(self, score_tensors: List[torch.Tensor], hop_idx: int, 
                                lamhat: List[float], gt_nodes: Optional[List[int]] = None, 
                                max_gt_nodes: Optional[int] = 100) -> set:
        """Aggregate scores with MAX, apply threshold, and optionally filter by GT."""
        if not score_tensors:
            return set()
        
        max_scores = torch.stack(score_tensors, dim=0).max(dim=0).values
        threshold = self._get_threshold(lamhat, hop_idx)
        nodes = self._extract_nodes_from_scores(max_scores.unsqueeze(0), threshold)
        
        if gt_nodes is not None:
            nodes = self._apply_gt_filtering(nodes, max_scores.cpu().numpy(), gt_nodes, max_gt_nodes)
        
        return set(nodes)
    
    def _apply_gt_filtering(self, nodes: List[int], scores_np: np.ndarray, 
                           gt_nodes: List[int], max_nodes: Optional[int]) -> List[int]:
        """Apply ground truth filtering with safety limits."""
        if not gt_nodes:
            return nodes
        
        gt_scores = scores_np[gt_nodes]
        min_gt_score = float(gt_scores.min()) if len(gt_scores) > 0 else 0.0
        filtered_nodes = [n for n in nodes if scores_np[n] >= min_gt_score]
        
        if max_nodes is not None and len(filtered_nodes) > max_nodes:
            node_scores = [(n, scores_np[n]) for n in filtered_nodes]
            node_scores.sort(key=lambda x: x[1], reverse=True)
            top_nodes = [n for n, s in node_scores[:max_nodes]]
            
            # Ensure all GT nodes are included
            missing_gt = list(set(gt_nodes) - set(top_nodes))
            filtered_nodes = top_nodes + missing_gt
            
            logging.debug(f"GT filtering: top {max_nodes}={len(top_nodes)}, "
                         f"GT nodes={len(gt_nodes)}, missing GT={len(missing_gt)}, "
                         f"final={len(filtered_nodes)}")
        
        return filtered_nodes
    
    def _has_gt_for_hop(self, ground_truth_hops: Optional[List[Dict]], 
                       query_idx: int, hop_num: int) -> bool:
        """Check if ground truth data exists for a specific hop."""
        return (ground_truth_hops is not None and 
                query_idx < len(ground_truth_hops) and 
                hop_num in ground_truth_hops[query_idx])
    
    def _aggregate_batch_results(self, hop1_results: Dict[str, Any], hop2_results: Dict[str, Any], 
                               hop3_results: Dict[str, Any], total_queries: int) -> List[Dict[str, Dict[str, Any]]]:
        """Aggregate results for each query in the batch."""
        return [
            {
                "hop1": {"nodes": hop1_results["nodes"][i], "scores": hop1_results["scores"][i]},
                "hop2": {"nodes": hop2_results["nodes"][i], "scores": hop2_results["scores"][i]},
                "hop3": {"nodes": hop3_results["nodes"][i], "scores": hop3_results["scores"][i]},
            }
            for i in range(total_queries)
        ]

    def _get_threshold(self, lamhat: List[float], hop_idx: int) -> float:
        """Extract threshold for a specific hop from lamhat."""
        if isinstance(lamhat, (list, tuple)) and len(lamhat) > hop_idx:
            return lamhat[hop_idx]
        if isinstance(lamhat, (torch.Tensor, np.ndarray)) and len(lamhat) > hop_idx:
            return float(lamhat[hop_idx])
        return 0.0

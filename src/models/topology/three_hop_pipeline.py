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
    def apply_thresholds_to_scores_with_call_counts(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        """
        Apply thresholds to calibration scores and return predictions plus estimated
        neo4j/ultra call counts per query (from cascade structure).
        Returns (predictions, neo4j_per_query, ultra_per_query).
        """
        predictions = []
        neo4j_per_query = []
        ultra_per_query = []
        for query_data in cal_scores:
            hop1_valid = ThreeHopPipeline._process_hop1_for_calibration(query_data, thresholds[0])
            if not hop1_valid:
                predictions.append([])
                neo4j_per_query.append(1)
                ultra_per_query.append(1)
                continue
            n_hop2 = sum(1 for p in query_data['hop2'] if p.get('parent') in hop1_valid)
            hop2_valid = ThreeHopPipeline._process_hop2_for_calibration(
                query_data, thresholds[1], hop1_valid, num_entities
            )
            if not hop2_valid:
                predictions.append([])
                neo4j_per_query.append(1 + n_hop2)
                ultra_per_query.append(1 + n_hop2)
                continue
            n_hop3 = sum(
                1 for p in query_data['hop3']
                if p.get('parent', (None, None))[0] in hop1_valid
                and p.get('parent', (None, None))[1] in hop2_valid
            )
            hop3_predictions = ThreeHopPipeline._process_hop3_for_calibration(
                query_data, thresholds[2], hop1_valid, hop2_valid, num_entities
            )
            predictions.append(hop3_predictions)
            neo4j_per_query.append(1 + n_hop2 + n_hop3)
            ultra_per_query.append(1 + n_hop2 + n_hop3)
        return predictions, neo4j_per_query, ultra_per_query

    @staticmethod
    def apply_thresholds_to_scores_with_intermediate_sizes(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[Tuple[int, int, int]]]:
        """
        Apply thresholds and return predictions plus intermediate set sizes per query.
        Returns (predictions, list of (s1, s2, s3) per query where s_i = size at hop i).
        """
        predictions = []
        intermediate_sizes = []
        for query_data in cal_scores:
            hop1_valid = ThreeHopPipeline._process_hop1_for_calibration(query_data, thresholds[0])
            s1 = len(hop1_valid)
            if not hop1_valid:
                predictions.append([])
                intermediate_sizes.append((s1, 0, 0))
                continue
            
            hop2_valid = ThreeHopPipeline._process_hop2_for_calibration(
                query_data, thresholds[1], hop1_valid, num_entities
            )
            s2 = len(hop2_valid)
            if not hop2_valid:
                predictions.append([])
                intermediate_sizes.append((s1, s2, 0))
                continue
            
            hop3_predictions = ThreeHopPipeline._process_hop3_for_calibration(
                query_data, thresholds[2], hop1_valid, hop2_valid, num_entities
            )
            s3 = len(hop3_predictions)
            predictions.append(hop3_predictions)
            intermediate_sizes.append((s1, s2, s3))
        return predictions, intermediate_sizes
    
    @staticmethod
    def _process_hop1_for_calibration(query_data: Dict, threshold: float) -> set:
        """Extract hop1 nodes that pass threshold and are in calibration data."""
        hop1_scores = query_data['hop1']['scores']
        hop1_passing = ThreeHopPipeline._extract_nodes_by_threshold(hop1_scores, threshold)
        # TODO(sonia): maybe an issue for calibration? We could do return hop1_passing
        # return hop1_passing & set(query_data['hop1']['nodes'])
        return hop1_passing
    
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
        
        # TODO(sonia): maybe an issue for calibration? We could do return hop2_passing
        # return hop2_passing & hop2_calibration
        return hop2_passing
    
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
                               ground_truth_hops: Optional[List[Dict]] = None) -> Tuple[List[List[int]], List[int], List[int]]:
        """Run cascade inference with threshold-based filtering at each hop.
        Returns (predictions_per_query, neo4j_calls_per_query, ultra_calls_per_query)."""
        batch_size = query.shape[0] if query.dim() > 1 else 1
        if query.dim() == 1:
            query = query.unsqueeze(0)
        counts = [[0, 0] for _ in range(batch_size)]  # [neo4j, ultra] per query
        hop1_results = self._process_hop1(
            query, lamhat, graph_data, ground_truth_hops, ground_truth_hops is not None, keep_scores=False, counts=counts
        )
        hop2_results = self._process_hop2(
            hop1_results, query, lamhat, graph_data, ground_truth_hops, ground_truth_hops is not None,
            keep_path_scores=False, counts=counts,
        )
        hop3_results = self._process_hop3(
            hop2_results, query, lamhat, graph_data, ground_truth_hops, ground_truth_hops is not None,
            keep_path_scores=False, counts=counts,
        )
        neo4j_per_query = [c[0] for c in counts]
        ultra_per_query = [c[1] for c in counts]
        return hop3_results["nodes"], neo4j_per_query, ultra_per_query
    
    @torch.no_grad()
    def _predict(self, query: torch.Tensor, lamhat: List[float], graph_data: Any, 
                ground_truth_hops: Optional[List[Dict]] = None) -> List[Dict[str, Dict[str, Any]]]:
        """Internal method that performs multi-hop cascade prediction."""
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Avoid CUDA synchronization from Python `.item()` in control-flow below.
        # Keep a CPU view for extracting ints while still running scoring on GPU.
        query_cpu = query.detach().cpu() if isinstance(query, torch.Tensor) and query.is_cuda else query
        
        use_ground_truth = ground_truth_hops is not None
        batch_size = query.shape[0]
        
        hop1_results = self._process_hop1(
            query_cpu, lamhat, graph_data, ground_truth_hops, use_ground_truth, keep_scores=True
        )
        hop2_results = self._process_hop2(
            hop1_results, query_cpu, lamhat, graph_data, ground_truth_hops, use_ground_truth, keep_path_scores=True
        )
        hop3_results = self._process_hop3(
            hop2_results, query_cpu, lamhat, graph_data, ground_truth_hops, use_ground_truth, keep_path_scores=True
        )
        
        return self._aggregate_batch_results(hop1_results, hop2_results, hop3_results, batch_size)
    
    def _process_hop1(self, query: torch.Tensor, lamhat: List[float], graph_data: Any, 
                     ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool,
                     keep_scores: bool = True, counts: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """Process hop 1 for all queries in batch, batching queries with same relation type for efficiency."""
        batch_size = query.shape[0]
        all_nodes = [None] * batch_size
        all_scores = [None] * batch_size if keep_scores else None
        
        threshold = self._get_threshold(lamhat, 0)
        
        # Group queries by relation type for batching
        relation_groups = {}
        for i in range(batch_size):
            relation = query[i, 1].item()
            if relation not in relation_groups:
                relation_groups[relation] = []
            relation_groups[relation].append(i)
        
        # Process each relation group in batches
        for relation, indices in relation_groups.items():
            # Batch all queries with this relation type
            sources = [query[i, 0].item() for i in indices]
            source_tensor = torch.tensor([[s] for s in sources], dtype=torch.long, device=self.device)
            rel_tensor = torch.tensor([[relation]] * len(sources), dtype=torch.long, device=self.device)
            
            # Predict for all queries in this batch at once
            batch_scores, _, neo4j_calls, ultra_calls = self.unified_predictor.predict(
                source_tensor, rel_tensor, graph_data, threshold=threshold
            )
            if counts is not None:
                for orig_idx in indices:
                    counts[orig_idx][0] += neo4j_calls
                    counts[orig_idx][1] += ultra_calls
            
            # Extract results for each query
            for batch_idx, orig_idx in enumerate(indices):
                scores = batch_scores[batch_idx:batch_idx+1]
                nodes = self._extract_nodes_from_scores(scores, threshold)
                
                if use_ground_truth and self._has_gt_for_hop(ground_truth_hops, orig_idx, 1):
                    nodes = self._apply_gt_filtering(nodes, scores.squeeze(0), ground_truth_hops[orig_idx][1], max_nodes=50)
                
                all_nodes[orig_idx] = nodes
                if keep_scores:
                    all_scores[orig_idx] = scores
        
        if keep_scores:
            return {"nodes": all_nodes, "scores": all_scores}
        return {"nodes": all_nodes}
    
    def _process_hop2(self, hop1_results: Dict[str, Any], query: torch.Tensor, lamhat: List[float], 
                     graph_data: Any, ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool,
                     keep_path_scores: bool = True, counts: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """Process hop 2 for all queries in batch using MAX-then-threshold aggregation."""
        return self._process_multi_source_hop(
            hop_num=2, prev_results=hop1_results, query=query, 
            relation_idx=2, lamhat=lamhat, graph_data=graph_data,
            ground_truth_hops=ground_truth_hops, use_ground_truth=use_ground_truth,
            keep_path_scores=keep_path_scores, counts=counts,
        )
    
    def _process_hop3(self, hop2_results: Dict[str, Any], query: torch.Tensor, lamhat: List[float], 
                     graph_data: Any, ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool,
                     keep_path_scores: bool = True, counts: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """Process hop 3 for all queries in batch using MAX-then-threshold aggregation."""
        return self._process_multi_source_hop(
            hop_num=3, prev_results=hop2_results, query=query, 
            relation_idx=3, lamhat=lamhat, graph_data=graph_data,
            ground_truth_hops=ground_truth_hops, use_ground_truth=use_ground_truth,
            keep_path_scores=keep_path_scores,
            counts=counts,
            parent_map_builder=(
                (lambda i: self._build_hop2_to_hop1_map(
                    hop2_results.get("scores", [])[i] if hop2_results.get("path_aware") else [],
                    lamhat
                ))
                if keep_path_scores
                else None
            ),
        )
    
    def _process_multi_source_hop(self, hop_num: int, prev_results: Dict[str, Any], 
                                   query: torch.Tensor, relation_idx: int, lamhat: List[float],
                                   graph_data: Any, ground_truth_hops: Optional[List[Dict]], 
                                   use_ground_truth: bool, 
                                   parent_map_builder: Optional[callable] = None,
                                   keep_path_scores: bool = True,
                                   counts: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """Unified processing for hop2 and hop3 (multi-source hops)."""
        batch_size = query.shape[0]
        all_nodes, all_scores = [], []
        
        for i in range(batch_size):
            source_nodes = list(prev_results["nodes"][i]) if prev_results["nodes"][i] else []
            relation = query[i, relation_idx].item()
            parent_map = parent_map_builder(i) if (keep_path_scores and parent_map_builder) else None
            
            path_scores, filtered_nodes = self._process_hop_generic(
                hop_num=hop_num, source_nodes=source_nodes, relation=relation, query_idx=i,
                lamhat=lamhat, graph_data=graph_data, 
                ground_truth_hops=ground_truth_hops, use_ground_truth=use_ground_truth,
                parent_map=parent_map,
                keep_path_scores=keep_path_scores,
                counts=counts,
            )
            
            all_nodes.append(list(filtered_nodes))
            if keep_path_scores:
                all_scores.append(path_scores)
        
        if keep_path_scores:
            return {"nodes": all_nodes, "scores": all_scores, "path_aware": True}
        return {"nodes": all_nodes}
    
    # Generic hop processing used by hop2/hop3. Keeps path-aware scores for calibration,
    # but avoids materializing a giant list of score tensors by computing a running MAX.
    def _process_hop_generic(self, hop_num: int, source_nodes: List[int], relation: int, 
                            query_idx: int, lamhat: List[float], graph_data: Any,
                            ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool,
                            parent_map: Optional[Dict] = None,
                            keep_path_scores: bool = True,
                            counts: Optional[List[List[int]]] = None) -> Tuple[List[Dict], set]:
        """Chunked hop processing with running MAX + optional GT filtering."""
        if not source_nodes:
            return [], set()
        
        path_scores: List[Dict] = [] if keep_path_scores else []
        threshold = self._get_threshold(lamhat, hop_num - 1)
        max_scores: Optional[torch.Tensor] = None
        
        # Process sources in chunks to avoid GPU OOM
        for chunk_start in range(0, len(source_nodes), self.max_internal_batch):
            chunk_nodes = source_nodes[chunk_start:chunk_start + self.max_internal_batch]
            
            batch_source = torch.tensor([[s] for s in chunk_nodes], dtype=torch.long, device=self.device)
            batch_rel = torch.tensor([[relation]] * len(chunk_nodes), dtype=torch.long, device=self.device)
            batch_scores, _, neo4j_calls, ultra_calls = self.unified_predictor.predict(
                batch_source, batch_rel, graph_data, threshold=threshold
            )
            if counts is not None and query_idx < len(counts):
                counts[query_idx][0] += neo4j_calls
                counts[query_idx][1] += ultra_calls
            
            for idx, source in enumerate(chunk_nodes):
                scores = batch_scores[idx:idx+1]
                scores_1d = scores.squeeze(0)
                
                # Track parent info for path awareness
                if keep_path_scores:
                    if parent_map and source in parent_map:
                        # For hop3: track (hop1_parent, hop2_node) pairs
                        for hop1_parent in parent_map[source]:
                            path_scores.append({'parent': (hop1_parent, source), 'scores': scores})
                    else:
                        # For hop2: track hop1 parent
                        path_scores.append({'parent': source, 'scores': scores})
                
                # Running MAX aggregation (avoid stacking all tensors at end)
                if max_scores is None:
                    max_scores = scores_1d
                else:
                    max_scores = torch.maximum(max_scores, scores_1d)
        
        if max_scores is None:
            return path_scores, set()

        # Apply threshold to MAX-aggregated scores, and optionally filter by GT
        gt_nodes = None
        max_gt_nodes = {1: 50, 2: 50, 3: None}[hop_num]
        if use_ground_truth and self._has_gt_for_hop(ground_truth_hops, query_idx, hop_num):
            gt_nodes = ground_truth_hops[query_idx][hop_num]
        
        nodes = self._extract_nodes_from_scores(max_scores.unsqueeze(0), threshold)
        if gt_nodes is not None:
            nodes = self._apply_gt_filtering(
                nodes,
                max_scores,
                gt_nodes,
                max_gt_nodes,
            )
        
        return path_scores, set(nodes)
    
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
    
    def _apply_gt_filtering(self, nodes: List[int], scores: torch.Tensor, 
                           gt_nodes: List[int], max_nodes: Optional[int]) -> List[int]:
        """Apply ground truth filtering with safety limits (pure torch, no CPU transfers)."""
        if not gt_nodes:
            return nodes

        scores_1d = scores.squeeze(0) if scores.dim() > 1 else scores
        device = scores_1d.device
        num_entities = int(scores_1d.numel())

        # Clamp/validate GT indices
        gt_t = torch.as_tensor(gt_nodes, dtype=torch.long, device=device)
        gt_t = gt_t[(gt_t >= 0) & (gt_t < num_entities)]
        if gt_t.numel() == 0:
            return nodes

        min_gt_score = scores_1d[gt_t].min()

        # Filter candidate nodes by min GT score
        nodes_t = torch.as_tensor(nodes, dtype=torch.long, device=device)
        nodes_t = nodes_t[(nodes_t >= 0) & (nodes_t < num_entities)]
        if nodes_t.numel() == 0:
            return []

        node_scores = scores_1d[nodes_t]
        keep_mask = node_scores >= min_gt_score
        kept_nodes = nodes_t[keep_mask]
        if kept_nodes.numel() == 0:
            # Preserve old behavior: can return empty after GT-based filtering
            return []

        # Optional cap (keep top by score, but always include all GT nodes)
        if max_nodes is not None and kept_nodes.numel() > max_nodes:
            kept_scores = scores_1d[kept_nodes]
            k = int(min(max_nodes, kept_nodes.numel()))
            _, topk_idx = torch.topk(kept_scores, k, largest=True, sorted=True)
            top_nodes = kept_nodes[topk_idx]

            top_list = [int(x) for x in top_nodes.detach().cpu().tolist()]
            top_set = set(top_list)
            missing_gt = [int(x) for x in gt_t.detach().cpu().tolist() if int(x) not in top_set]

            filtered_nodes = top_list + missing_gt

            logging.debug(
                f"GT filtering: top {max_nodes}={len(top_list)}, "
                f"GT nodes={len(gt_t)}, missing GT={len(missing_gt)}, "
                f"final={len(filtered_nodes)}"
            )
            return filtered_nodes

        return [int(x) for x in kept_nodes.detach().cpu().tolist()]
    
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

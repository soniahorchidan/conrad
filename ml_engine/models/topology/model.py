import torch
import torch.nn as nn
import logging
import numpy as np
from argparse import Namespace
from typing import List, Dict, Any, Optional, Tuple, Union
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


class ThreeHopPipeline(nn.Module):
    """
    A wrapper pipeline model that uses MultiHopPredictor
    to approximate each hop of a 3-hop path.
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
        calib_batch_size = getattr(args, 'calib_batch_size', 4)  # Default to 4 to avoid GPU OOM
        num_entities = getattr(args, 'num_entities', None)  # Get from args if available
        self.calibration_generator = CalibrationDataGenerator(
            self, device, calib_batch_size=calib_batch_size, num_entities=num_entities
        )
        
        # Max internal batch size for hop2/hop3 predictions to avoid GPU OOM
        self.max_internal_batch = getattr(args, 'max_internal_batch', 64)
        
        # Slack parameter for dynamic threshold calculation
        self.multihop_slack = 0.1
        
        # RAPS parameters (set during calibration)
        self.use_raps = False
        self.raps_kreg = 1
        self.raps_lamda = 1e-3
        self.raps_randomized = True 

    def set_raps_params(self, use_raps: bool = True, kreg: int = 1, lamda: float = 1e-3, randomized: bool = True):
        """
        Set RAPS parameters for inference.
        
        This should be called after calibration to enable RAPS at inference time.
        
        Args:
            use_raps: Whether to use RAPS
            kreg: Regularization parameter
            lamda: Penalty weight
            randomized: Whether to use randomized RAPS
        """
        self.use_raps = use_raps
        self.raps_kreg = kreg
        self.raps_lamda = lamda
        self.raps_randomized = randomized
        logging.info(f"RAPS parameters set: use_raps={use_raps}, kreg={kreg}, λ={lamda}, randomized={randomized}")
    
    def preprocess(self, general_args: Namespace, args: Namespace):
        # If MultiHopPredictor has its own preprocess logic, call it here
        return self.unified_predictor.preprocess(general_args, args)

    def postprocess(self, general_args: Namespace, args: Namespace):
        return self.unified_predictor.postprocess(general_args, args)

    def forward(self, graph_data, query):
        """
        Just forward pass. You can make this return embeddings or predictions.
        """
        return self.unified_predictor(graph_data, query)

    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """
        Query format: [src, rel1, rel2, rel3]
        Enforces exactly 3-hop inference using MultiHopPredictor.
        Uses conformal risk control for threshold calibration.
        Returns final results as list of entity indices.
        """
        # Validate query format
        if len(query[0]) != 4:
            raise ValueError("ThreeHopPipeline only supports 3-hop queries (len=4).")
        
        query = query.to(self.device)
        
        # Handle different confidence modes
        if confidence == 1.0:
            raise ValueError("Confidence is set to 1.0. Returning no results.")
        elif confidence == 0.0:
            # Debug mode: use default thresholds
            logging.info("Using normal inference without conformal prediction. DEBUG ONLY!")
            debug_thresholds = [0.4, 0.33, 0.6]
            return self._predict(query, debug_thresholds, graph_data)
        else:
            # Conformal prediction mode
            if hasattr(self, 'conformal_prediction') and self.conformal_prediction is not None:
                return self.conformal_prediction.predict(query, confidence, graph_data=graph_data)
            else:
                raise ValueError("Conformal prediction not available for ThreeHopPipeline")


    @staticmethod
    def apply_thresholds_to_scores(cal_scores: List[Dict[str, Any]], thresholds: List[float], 
                                   true_labels: Optional[List[Dict]] = None, 
                                   num_entities: int = 14541) -> List[List[int]]:
        predictions = []
        
        for idx, query_data in enumerate(cal_scores):
            # Step 1: Apply τ1 to hop1 scores
            hop1_scores = query_data['hop1']['scores']
            
            # Check if hop1 uses GT-only format
            hop1_is_gt_only = isinstance(hop1_scores, dict) and hop1_scores.get('gt_only', False)
            
            # Both GT-only and regular scores use same extraction logic
            hop1_passing = ThreeHopPipeline._extract_nodes_by_threshold(
                hop1_scores, thresholds[0]
            )
            
            # Use ALL calibration-tracked nodes (GT + high-scoring from min-GT-score logic)
            # These are the paths we explored during calibration data generation
            hop1_calibration_nodes = set(query_data['hop1']['nodes'])
            
            # CRITICAL: Filter to only nodes that actually pass the threshold
            # This simulates what happens during real inference
            hop1_valid_nodes = hop1_passing & hop1_calibration_nodes
            
            if not hop1_valid_nodes:
                # No hop1 nodes pass threshold
                predictions.append([])
                continue
            
            # Step 2: Collect ALL hop2 scores from THRESHOLD-PASSING hop1 paths, MAX aggregate, then threshold
            hop2_score_vectors = []
            hop2_parents_collected = set()  # Track which hop1 nodes contributed scores
            hop2_is_gt_only = False
            
            for hop2_path in query_data['hop2']:
                hop1_parent = hop2_path['parent']
                
                # Only process if this hop1 node PASSED the threshold
                if hop1_parent in hop1_valid_nodes:
                    hop2_scores = hop2_path['scores']
                    
                    # Check if this is GT-only sparse format
                    if isinstance(hop2_scores, dict) and hop2_scores.get('gt_only', False):
                        hop2_is_gt_only = True
                    
                    hop2_score_vec = ThreeHopPipeline._scores_to_dense_vector(hop2_scores, num_entities=num_entities)
                    hop2_score_vectors.append(hop2_score_vec)
                    hop2_parents_collected.add(hop1_parent)
            
            if not hop2_score_vectors:
                # No hop2 paths to process
                predictions.append([])
                continue
            
            # MAX aggregate hop2 scores across all paths
            hop2_max_scores = np.maximum.reduce(hop2_score_vectors)
            
            # Apply threshold to hop2 to get passing hop2 nodes
            if hop2_is_gt_only:
                hop2_gt_entities = np.where(hop2_max_scores > 0)[0]
                hop2_passing = set(hop2_gt_entities[hop2_max_scores[hop2_gt_entities] >= thresholds[1]])
            else:
                hop2_passing = set(np.where(hop2_max_scores >= thresholds[1])[0])
            
            # Extract all hop2 nodes that were used during calibration (from hop3 path parents)
            hop2_calibration_nodes = set()
            for hop3_path in query_data['hop3']:
                hop1_parent, hop2_parent = hop3_path['parent']
                if hop1_parent in hop1_valid_nodes:  # Also fixed: use hop1_valid_nodes
                    hop2_calibration_nodes.add(hop2_parent)
            
            # CRITICAL: Filter to only hop2 nodes that pass threshold
            hop2_valid_nodes = hop2_passing & hop2_calibration_nodes
            
            if not hop2_valid_nodes:
                # No hop2 nodes pass threshold
                predictions.append([])
                continue
            
            # Step 3: Collect ALL hop3 scores from THRESHOLD-PASSING (hop1, hop2) paths, MAX aggregate, then threshold
            hop3_score_vectors = []
            hop3_is_gt_only = False
            
            for hop3_path in query_data['hop3']:
                hop1_parent, hop2_parent = hop3_path['parent']
                
                # Only process if both parents PASSED thresholds
                if hop1_parent in hop1_valid_nodes and hop2_parent in hop2_valid_nodes:
                    hop3_scores = hop3_path['scores']
                    
                    # Check if this is GT-only sparse format
                    if isinstance(hop3_scores, dict) and hop3_scores.get('gt_only', False):
                        hop3_is_gt_only = True
                    
                    hop3_score_vec = ThreeHopPipeline._scores_to_dense_vector(hop3_scores, num_entities=num_entities)
                    hop3_score_vectors.append(hop3_score_vec)
            
            if not hop3_score_vectors:
                # No hop3 paths to process
                predictions.append([])
                continue
            
            # MAX aggregate hop3 scores across all paths
            hop3_max_scores = np.maximum.reduce(hop3_score_vectors)
            
            # Apply threshold - if GT-only, only consider non-zero positions
            if hop3_is_gt_only:
                # For GT-only scores, only evaluate GT entities (non-zero positions)
                hop3_gt_entities = np.where(hop3_max_scores > 0)[0]
                hop3_passing = set(hop3_gt_entities[hop3_max_scores[hop3_gt_entities] >= thresholds[2]])
            else:
                hop3_passing = set(np.where(hop3_max_scores >= thresholds[2])[0])
            
            predictions.append(list(hop3_passing))
        
        return predictions
    
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
            
            # Create dense vector
            dense = np.zeros(num_entities, dtype=np.float32)
            dense[indices] = values
            return dense
        else:
            # Already dense
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            if scores.ndim > 1:
                scores = scores.squeeze()
            # Pad or trim to num_entities
            if len(scores) < num_entities:
                padded = np.zeros(num_entities, dtype=np.float32)
                padded[:len(scores)] = scores
                return padded
            else:
                return scores[:num_entities]
    
    @torch.no_grad()
    def predict_with_thresholds(self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
                               ground_truth_hops: Optional[List[Dict]] = None) -> List[List[int]]:
        """
        Run cascade inference with threshold-based filtering at each hop.
        
        This is the SINGLE SOURCE OF TRUTH for threshold-based prediction used by both:
        - Calibration: Optimizes thresholds by evaluating different lamhat values
        - Inference: Uses calibrated thresholds to make predictions
        
        The cascade process:
        1. Start with anchor node from query
        2. Run 1-hop inference with unified model → get scores for all entities
        3. Apply lamhat[0] threshold (>=) → filter to get hop1 nodes
        4. For each hop1 node: run 1-hop inference with relation2 → get scores
        5. Take max scores across all paths → apply lamhat[1] threshold → get hop2 nodes
        6. For each hop2 node: run 1-hop inference with relation3 → get scores  
        7. Take max scores across all paths → apply lamhat[2] threshold → get hop3 nodes
        8. Return hop3 nodes as final predictions
        
        Args:
            query: Tensor of shape (batch_size, 4) containing [anchor, rel1, rel2, rel3]
            lamhat: List of 3 thresholds [τ1, τ2, τ3] for each hop
            graph_data: Graph data for inference
            ground_truth_hops: Optional dict {hop: [entity_ids]} to restrict intermediate nodes
                              (used during calibration to avoid computational explosion)
        
        Returns:
            List of predicted entity lists (hop3 nodes only), one per query
            
        Note: Both calibration and validation use ONLY hop3 outputs for FNR computation.
        """
        # Run the full cascade
        batch_results = self._predict(query, lamhat, graph_data, ground_truth_hops)
        
        # Extract ONLY hop3 nodes (final answers) for FNR computation
        predictions = [result["hop3"]["nodes"] for result in batch_results]
        
        return predictions
    
    @torch.no_grad()
    def _predict(self, query: torch.Tensor, lamhat: List[float], graph_data: Any, 
                ground_truth_hops: Optional[List[Dict]] = None) -> List[Dict[str, Dict[str, Any]]]:
        """
        Internal method that performs multi-hop cascade prediction.
        
        Returns detailed results for all hops. Use predict_with_thresholds() for the
        public API that returns only final hop3 predictions.
        
        During calibration: If ground_truth_hops is provided, restricts intermediate hops 
        to ground-truth nodes to avoid computational explosion.
        
        During inference: Uses acceptance thresholds (lamhat) for filtering at each hop.
        """
        use_ground_truth = ground_truth_hops is not None
        total_queries = len(query)
        
        # Process all three hops sequentially
        hop1_results = self._process_hop1(query, lamhat, graph_data, ground_truth_hops, use_ground_truth)
        hop2_results = self._process_hop2(hop1_results, query, lamhat, graph_data, ground_truth_hops, use_ground_truth)
        hop3_results = self._process_hop3(hop2_results, query, lamhat, graph_data, ground_truth_hops, use_ground_truth)
        
        # Aggregate results for each query
        return self._aggregate_batch_results(hop1_results, hop2_results, hop3_results, total_queries)
    
    def _apply_gt_filtering(self, nodes: List[int], scores_np: np.ndarray, 
                           gt_nodes: List[int], max_nodes: Optional[int]) -> List[int]:
        """
        Apply ground truth filtering with safety limits.
        
        Ensures all GT nodes are included, even if they're not in the top-scoring nodes.
        """
        gt_scores = scores_np[gt_nodes]
        min_gt_score = float(gt_scores.min()) if len(gt_scores) > 0 else 0.0
        
        # Filter nodes by min GT score
        filtered_nodes = [n for n in nodes if scores_np[n] >= min_gt_score]
        
        # Apply safety limit if max_nodes is specified
        if max_nodes is not None and len(filtered_nodes) > max_nodes:
            node_scores = [(n, scores_np[n]) for n in filtered_nodes]
            node_scores.sort(key=lambda x: x[1], reverse=True)
            top_nodes = [n for n, s in node_scores[:max_nodes]]
            
            # Ensure all GT nodes are included, even if not in top max_nodes
            gt_set = set(gt_nodes)
            top_set = set(top_nodes)
            missing_gt_nodes = list(gt_set - top_set)  # GT nodes not in top 50
            
            # Combine: top nodes + missing GT nodes
            filtered_nodes = top_nodes + missing_gt_nodes
            
            logging.debug(f"GT filtering: top {max_nodes}={len(top_nodes)}, "
                         f"GT nodes={len(gt_nodes)}, missing GT={len(missing_gt_nodes)}, "
                         f"final={len(filtered_nodes)}")
        
        return filtered_nodes
    
    def _process_hop(self, hop_num: int, source_nodes: List[int], relation: int, 
                            query_idx: int, lamhat: List[float], graph_data: Any,
                            ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool,
                            parent_map: Optional[Dict] = None) -> Tuple[List[Dict], List[torch.Tensor], set]:
        """
        Generic hop processing that handles chunking, aggregation, and GT filtering.
        
        Args:
            hop_num: Hop number (1, 2, or 3) for threshold indexing
            source_nodes: Source node IDs to process
            relation: Relation ID to use
            query_idx: Index of current query in batch
            lamhat: Threshold list
            graph_data: Graph data
            ground_truth_hops: Optional GT hops for calibration filtering
            use_ground_truth: Whether to apply GT filtering
            parent_map: Optional map from node -> parent(s) for path tracking
            
        Returns:
            Tuple of (path_scores, score_tensors, filtered_nodes_set)
        """
        if not source_nodes:
            return [], [], set()
        
        # Process sources in chunks
        query_path_scores = []
        score_tensors = []
        
        for chunk_start in range(0, len(source_nodes), self.max_internal_batch):
            chunk_nodes = source_nodes[chunk_start:chunk_start + self.max_internal_batch]
            
            batch_source = torch.tensor([[s] for s in chunk_nodes], dtype=torch.long, device=self.device)
            batch_rel = torch.tensor([[relation]] * len(chunk_nodes), dtype=torch.long, device=self.device)
            batch_scores, _ = self.unified_predictor.predict(batch_source, batch_rel, graph_data)
            
            for idx, source in enumerate(chunk_nodes):
                scores = batch_scores[idx:idx+1]
                
                # Determine parent info for path tracking
                if parent_map and source in parent_map:
                    parents = parent_map[source]
                    # For hop3, parents is a list of hop1 ancestors
                    for hop1_parent in parents:
                        query_path_scores.append({'parent': (hop1_parent, source), 'scores': scores})
                else:
                    # For hop2 (no parent_map), use source as parent
                    query_path_scores.append({'parent': source, 'scores': scores})
                
                score_tensors.append(scores.squeeze(0))
        
        # Aggregate and threshold
        gt_nodes = None
        max_gt_nodes = {1: 50, 2: 50, 3: None}[hop_num]  # No GT filtering for hop3
        if use_ground_truth and ground_truth_hops and query_idx < len(ground_truth_hops) and hop_num in ground_truth_hops[query_idx]:
            gt_nodes = ground_truth_hops[query_idx][hop_num]
        
        filtered_nodes = self._aggregate_and_threshold(
            score_tensors, hop_idx=hop_num-1, lamhat=lamhat, 
            gt_nodes=gt_nodes, max_gt_nodes=max_gt_nodes
        )
        
        return query_path_scores, score_tensors, filtered_nodes
    
    def _process_hop1(self, query: torch.Tensor, lamhat: List[float], graph_data: Any, 
                     ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool) -> Dict[str, Any]:
        """Process hop 1 for all queries in batch (special case: single source per query)."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        batch_size = query.shape[0]
        all_nodes = []
        all_scores = []
        
        for i in range(batch_size):
            source = query[i, 0].item()
            relation = query[i, 1].item()
            
            # Hop1 is special: single source from query, no aggregation needed
            source_tensor = torch.tensor([[source]], dtype=torch.long, device=self.device)
            rel_tensor = torch.tensor([[relation]], dtype=torch.long, device=self.device)
            scores, _ = self.unified_predictor.predict(source_tensor, rel_tensor, graph_data)
            
            threshold = self._get_threshold(lamhat, 0)
            nodes = self._extract_nodes_from_scores(scores, threshold)
            
            # Apply GT filtering if in calibration mode
            if use_ground_truth and ground_truth_hops and i < len(ground_truth_hops) and 1 in ground_truth_hops[i]:
                scores_np = scores.squeeze(0).cpu().numpy()
                nodes = self._apply_gt_filtering(nodes, scores_np, ground_truth_hops[i][1], max_nodes=50)
            
            all_nodes.append(nodes)
            all_scores.append(scores)
        
        return {"nodes": all_nodes, "scores": all_scores}
    
    def _aggregate_and_threshold(self, score_tensors: List[torch.Tensor], hop_idx: int, 
                                lamhat: List[float], gt_nodes: Optional[List[int]] = None, 
                                max_gt_nodes: Optional[int] = 100) -> set:
        """
        Aggregate scores with MAX, apply threshold, and optionally filter by GT.
        
        Returns:
            Set of node indices that pass the threshold
        """
        if not score_tensors:
            return set()
        
        # MAX aggregate across paths
        max_scores = torch.stack(score_tensors, dim=0).max(dim=0).values
        threshold = self._get_threshold(lamhat, hop_idx)
        nodes = self._extract_nodes_from_scores(max_scores.unsqueeze(0), threshold)
        
        # Apply GT filtering if provided
        if gt_nodes is not None:
            max_scores_np = max_scores.cpu().numpy()
            nodes = self._apply_gt_filtering(nodes, max_scores_np, gt_nodes, max_gt_nodes)
        
        return set(nodes)
    
    def _process_hop2(self, hop1_results: Dict[str, Any], query: torch.Tensor, lamhat: List[float], 
                     graph_data: Any, ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool) -> Dict[str, Any]:
        """Process hop 2 for all queries in batch using MAX-then-threshold aggregation."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        batch_size = query.shape[0]
        all_nodes = []
        all_scores = []
        
        for i in range(batch_size):
            source_nodes = list(hop1_results["nodes"][i]) if hop1_results["nodes"][i] else []
            relation = query[i, 2].item()
            
            # Use generic hop processing
            query_path_scores, _, filtered_nodes = self._process_hop(
                hop_num=2, source_nodes=source_nodes, relation=relation, query_idx=i,
                lamhat=lamhat, graph_data=graph_data, 
                ground_truth_hops=ground_truth_hops, use_ground_truth=use_ground_truth
            )
            
            all_nodes.append(list(filtered_nodes))
            all_scores.append(query_path_scores)
        
        return {"nodes": all_nodes, "scores": all_scores, "path_aware": True}
    
    def _build_hop2_to_hop1_map(self, hop2_path_scores: List[Dict], lamhat: List[float]) -> Dict[int, List[int]]:
        """Map hop2 nodes to their hop1 parents for path tracking."""
        hop2_to_hop1 = {}
        for path_data in hop2_path_scores:
            hop1_parent = path_data['parent']
            path_scores = path_data['scores']
            hop2_nodes_from_path = self._extract_nodes_from_scores(path_scores, self._get_threshold(lamhat, 1))
            for hop2_node in hop2_nodes_from_path:
                if hop2_node not in hop2_to_hop1:
                    hop2_to_hop1[hop2_node] = []
                hop2_to_hop1[hop2_node].append(hop1_parent)
        return hop2_to_hop1
    
    def _process_hop3(self, hop2_results: Dict[str, Any], query: torch.Tensor, lamhat: List[float], 
                     graph_data: Any, ground_truth_hops: Optional[List[Dict]], use_ground_truth: bool) -> Dict[str, Any]:
        """Process hop 3 for all queries in batch using MAX-then-threshold aggregation."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        batch_size = query.shape[0]
        all_nodes = []
        all_scores = []
        
        for i in range(batch_size):
            source_nodes = list(hop2_results["nodes"][i]) if hop2_results["nodes"][i] else []
            relation = query[i, 3].item()
            hop2_path_scores = hop2_results["scores"][i] if "path_aware" in hop2_results else []
            
            # Build parent map for hop1 ancestry tracking
            hop2_to_hop1 = self._build_hop2_to_hop1_map(hop2_path_scores, lamhat) if hop2_path_scores else {}
            
            # Use generic hop processing with parent map for path tracking
            query_path_scores, _, filtered_nodes = self._process_hop(
                hop_num=3, source_nodes=source_nodes, relation=relation, query_idx=i,
                lamhat=lamhat, graph_data=graph_data, 
                ground_truth_hops=ground_truth_hops, use_ground_truth=use_ground_truth,
                parent_map=hop2_to_hop1
            )
            
            all_nodes.append(list(filtered_nodes))
            all_scores.append(query_path_scores)
        
        return {"nodes": all_nodes, "scores": all_scores, "path_aware": True}
    
    def _aggregate_batch_results(self, hop1_results: Dict[str, Any], hop2_results: Dict[str, Any], 
                               hop3_results: Dict[str, Any], total_queries: int) -> List[Dict[str, Dict[str, Any]]]:
        """Aggregate results for each query in the batch."""
        batch_results = []
        for i in range(total_queries):
            # For threshold-based inference, hop3_results["nodes"] already contains the final
            # thresholded nodes. hop3_results["scores"] contains path score dictionaries.
            # We don't need to aggregate - just use the nodes directly.
            final_nodes = hop3_results["nodes"][i]
            final_scores = hop3_results["scores"][i]
            
            batch_results.append({
                "hop1": {"nodes": hop1_results["nodes"][i], "scores": hop1_results["scores"][i]},
                "hop2": {"nodes": hop2_results["nodes"][i], "scores": hop2_results["scores"][i]},
                "hop3": {"nodes": final_nodes, "scores": final_scores},
            })
        
        return batch_results

    def _get_threshold(self, lamhat: List[float], hop_idx: int) -> float:
        """Extract threshold for a specific hop from lamhat."""
        if isinstance(lamhat, (list, tuple, torch.Tensor, np.ndarray)) and len(lamhat) > hop_idx:
            return lamhat[hop_idx]
        else:
            return 0.0

    def _extract_nodes_from_scores(self, scores: torch.Tensor, threshold: float) -> List[int]:
        """
        Extract nodes from scores using either RAPS or simple thresholding.
        
        If RAPS is enabled (via set_raps_params), uses RAPS Algorithm 3.
        Otherwise, uses simple threshold-based filtering.
        
        Args:
            scores: Score tensor for entities
            threshold: Calibrated threshold
            
        Returns:
            List of selected entity indices
        """
        if self.use_raps:
            # Use RAPS threshold application (Algorithm 3 from paper)
            from conformal_prediction.raps import apply_raps_threshold
            return apply_raps_threshold(
                scores, 
                threshold, 
                kreg=self.raps_kreg,
                lamda=self.raps_lamda,
                randomized=self.raps_randomized
            )
        else:
            # Simple threshold-based filtering (standard conformal prediction)
            nonzero_indices = torch.nonzero(scores >= threshold)
            if nonzero_indices.size(0) > 0:
                node_dim = scores.dim() - 1
                return nonzero_indices[:, node_dim].tolist()
            return []
    
    @staticmethod
    def _extract_nodes_by_threshold(scores, threshold: float) -> set:
        """
        Extract nodes that pass threshold from score dict/tensor (for static contexts).
        Handles both sparse (dict) and dense (tensor/array) formats.
        """
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



    @torch.no_grad()
    def generateCalibrateSamples(self, save_path: str, db_controller: Any, vector_db_controller: Any,
                                calib_iterator: Optional[Any] = None, calibration_data_path: Optional[str] = None, 
                                return_queries: bool = False) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        """
        Produces (scores, answers) for 3-hop queries using the ThreeHopPipeline.
        Consumes directly from calib_iterator which yields (query, target, graph_data).
        
        Args:
            save_path: Path for saving/loading data
            db_controller: Database controller (used if calib_iterator is None and calibration_data_path is None)
            vector_db_controller: Vector database controller
            calib_iterator: Optional pre-created iterator
            calibration_data_path: Optional path to saved calibration data files
            return_queries: Whether to return queries along with scores and answers
        
        Returns vector-valued nonconformity scores: [hop1_score, hop2_score, hop3_score]
        """
        return self.calibration_generator.generate_calibration_samples(
            save_path, db_controller, vector_db_controller, 
            calib_iterator, calibration_data_path, return_queries
        )

    def load_all_components(self, load_path: str, ignore_components: List[str] = None) -> None:
        """Load all model components from the specified path."""
        if ignore_components is None:
            ignore_components = []
        return self.unified_predictor.load_all_components(load_path, ignore_components)

    def trainModel(self, general_args: Namespace, args: Namespace) -> None:
        """Train the model using the provided arguments."""
        return self.unified_predictor.trainModel(general_args, args)


class TwoUnionPipeline(nn.Module):
    """
    Pipeline for 2u (union) queries: ((anchor1, rel1), (anchor2, rel2)).
    
    Query execution:
    1. Predict entities from anchor1 via rel1 → set S1
    2. Predict entities from anchor2 via rel2 → set S2  
    3. Return union S1 ∪ S2
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
        
        # RAPS parameters
        self.use_raps = False
        self.raps_kreg = 1
        self.raps_lamda = 1e-3
        self.raps_randomized = True
    
    def set_raps_params(self, use_raps: bool = True, kreg: int = 1, lamda: float = 1e-3, randomized: bool = True):
        """Set RAPS parameters for inference."""
        self.use_raps = use_raps
        self.raps_kreg = kreg
        self.raps_lamda = lamda
        self.raps_randomized = randomized
        logging.info(f"RAPS parameters set: use_raps={use_raps}, kreg={kreg}, λ={lamda}, randomized={randomized}")
    
    def preprocess(self, general_args: Namespace, args: Namespace):
        return self.unified_predictor.preprocess(general_args, args)
    
    def postprocess(self, general_args: Namespace, args: Namespace):
        return self.unified_predictor.postprocess(general_args, args)
    
    def forward(self, graph_data, query):
        """Forward pass."""
        return self.unified_predictor(graph_data, query)
    
    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """
        Query format: [anchor1, rel1, anchor2, rel2]
        Returns union of predictions from both branches.
        """
        # Validate query format
        if len(query[0]) != 4:
            raise ValueError("TwoUnionPipeline expects queries of length 4: [anchor1, rel1, anchor2, rel2]")
        
        query = query.to(self.device)
        
        # Handle different confidence modes
        if confidence == 1.0:
            raise ValueError("Confidence is set to 1.0. Returning no results.")
        elif confidence == 0.0:
            # Debug mode: use default thresholds
            logging.info("Using normal inference without conformal prediction. DEBUG ONLY!")
            debug_thresholds = [0.4, 0.4]  # Two thresholds for two branches
            return self._predict(query, debug_thresholds, graph_data)
        else:
            # Conformal prediction mode
            if hasattr(self, 'conformal_prediction') and self.conformal_prediction is not None:
                return self.conformal_prediction.predict(query, confidence, graph_data=graph_data)
            else:
                raise ValueError("Conformal prediction not available for TwoUnionPipeline")
    
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
            if scores.ndim > 1:
                scores = scores.squeeze()
            # Pad or trim to num_entities
            if len(scores) < num_entities:
                padded = np.zeros(num_entities, dtype=np.float32)
                padded[:len(scores)] = scores
                return padded
            else:
                return scores[:num_entities]
    
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
    
    def _extract_nodes_from_scores(self, scores: torch.Tensor, threshold: float) -> List[int]:
        """Extract nodes from scores using either RAPS or simple thresholding."""
        if self.use_raps:
            from conformal_prediction.raps import apply_raps_threshold
            return apply_raps_threshold(
                scores,
                threshold,
                kreg=self.raps_kreg,
                lamda=self.raps_lamda,
                randomized=self.raps_randomized
            )
        else:
            # Simple threshold-based filtering
            nonzero_indices = torch.nonzero(scores >= threshold)
            if nonzero_indices.size(0) > 0:
                node_dim = scores.dim() - 1
                return nonzero_indices[:, node_dim].tolist()
            return []
    
    @torch.no_grad()
    def generateCalibrateSamples(self, save_path: str, db_controller: Any, vector_db_controller: Any,
                                calib_iterator: Optional[Any] = None, calibration_data_path: Optional[str] = None,
                                return_queries: bool = False) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        """
        Produces (scores, answers) for 2u queries.
        Returns vector-valued nonconformity scores: [branch1_score, branch2_score]
        """
        return self.calibration_generator.generate_calibration_samples(
            save_path, db_controller, vector_db_controller,
            calib_iterator, calibration_data_path, return_queries
        )
    
    def load_all_components(self, load_path: str, ignore_components: List[str] = None) -> None:
        """Load all model components from the specified path."""
        if ignore_components is None:
            ignore_components = []
        return self.unified_predictor.load_all_components(load_path, ignore_components)
    
    def trainModel(self, general_args: Namespace, args: Namespace) -> None:
        """Train the model using the provided arguments."""
        return self.unified_predictor.trainModel(general_args, args)


class TwoIntersectProjectPipeline(nn.Module):
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
        
        # RAPS parameters
        self.use_raps = False
        self.raps_kreg = 1
        self.raps_lamda = 1e-3
        self.raps_randomized = True
    
    def set_raps_params(self, use_raps: bool = True, kreg: int = 1, lamda: float = 1e-3, randomized: bool = True):
        """Set RAPS parameters for inference."""
        self.use_raps = use_raps
        self.raps_kreg = kreg
        self.raps_lamda = lamda
        self.raps_randomized = randomized
        logging.info(f"RAPS parameters set: use_raps={use_raps}, kreg={kreg}, λ={lamda}, randomized={randomized}")
    
    def preprocess(self, general_args: Namespace, args: Namespace):
        return self.unified_predictor.preprocess(general_args, args)
    
    def postprocess(self, general_args: Namespace, args: Namespace):
        return self.unified_predictor.postprocess(general_args, args)
    
    def forward(self, graph_data, query):
        """Forward pass."""
        return self.unified_predictor(graph_data, query)
    
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
        # Validate query format
        if len(query[0]) != 5:
            raise ValueError("TwoIntersectProjectPipeline expects queries of length 5: [anchor1, rel1, anchor2, rel2, rel3]")
        
        query = query.to(self.device)
        
        # Handle different confidence modes
        if confidence == 1.0:
            raise ValueError("Confidence is set to 1.0. Returning no results.")
        elif confidence == 0.0:
            # Debug mode: use default thresholds (3D: τ_branch1, τ_branch2, τ_proj)
            logging.info("Using normal inference without conformal prediction. DEBUG ONLY!")
            debug_thresholds = [0.4, 0.4, 0.4]  # Three thresholds: branch1, branch2, projection
            return self._predict(query, debug_thresholds, graph_data)
        else:
            # Conformal prediction mode
            if hasattr(self, 'conformal_prediction') and self.conformal_prediction is not None:
                return self.conformal_prediction.predict(query, confidence, graph_data=graph_data)
            else:
                raise ValueError("Conformal prediction not available for TwoIntersectProjectPipeline")
    
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
            if scores.ndim > 1:
                scores = scores.squeeze()
            # Pad or trim to num_entities
            if len(scores) < num_entities:
                padded = np.zeros(num_entities, dtype=np.float32)
                padded[:len(scores)] = scores
                return padded
            else:
                return scores[:num_entities]
    
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
        
        batch_size = query.shape[0]
        batch_results = []
        
        for i in range(batch_size):
            anchor1, rel1 = query[i, 0].item(), query[i, 1].item()
            anchor2, rel2 = query[i, 2].item(), query[i, 3].item()
            rel3 = query[i, 4].item()
            
            # 1. Branch Execution
            source1_t = torch.tensor([[anchor1]], dtype=torch.long, device=self.device)
            rel1_t = torch.tensor([[rel1]], dtype=torch.long, device=self.device)
            scores1, _ = self.unified_predictor.predict(source1_t, rel1_t, graph_data)
            
            source2_t = torch.tensor([[anchor2]], dtype=torch.long, device=self.device)
            rel2_t = torch.tensor([[rel2]], dtype=torch.long, device=self.device)
            scores2, _ = self.unified_predictor.predict(source2_t, rel2_t, graph_data)
            
            # 2. Independent Branch Thresholding (3D optimization)
            t_branch1 = lamhat[0] if len(lamhat) > 0 else 0.0
            t_branch2 = lamhat[1] if len(lamhat) > 1 else 0.0
            t_proj = lamhat[2] if len(lamhat) > 2 else 0.0
            
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
                    
                    p_scores, _ = self.unified_predictor.predict(int_tensor, rel3_tensor, graph_data)
                    
                    for j, parent_id in enumerate(b_nodes):
                        s_vec = p_scores[j].cpu()
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
    
    def _extract_nodes_from_scores(self, scores: torch.Tensor, threshold: float) -> List[int]:
        """Extract nodes from scores using either RAPS or simple thresholding."""
        if self.use_raps:
            from conformal_prediction.raps import apply_raps_threshold
            return apply_raps_threshold(
                scores,
                threshold,
                kreg=self.raps_kreg,
                lamda=self.raps_lamda,
                randomized=self.raps_randomized
            )
        else:
            # Simple threshold-based filtering
            nonzero_indices = torch.nonzero(scores >= threshold)
            if nonzero_indices.size(0) > 0:
                node_dim = scores.dim() - 1
                return nonzero_indices[:, node_dim].tolist()
            return []


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
            return np.array(scores, dtype=np.float32)
    
    @torch.no_grad()
    def generateCalibrateSamples(self, save_path: str, db_controller: Any, vector_db_controller: Any,
                                calib_iterator: Optional[Any] = None, calibration_data_path: Optional[str] = None,
                                return_queries: bool = False) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        """
        Produces (scores, answers) for 2ip queries.
        Returns vector-valued nonconformity scores: [branch1_score, branch2_score, projection_score]
        """
        return self.calibration_generator.generate_calibration_samples(
            save_path, db_controller, vector_db_controller,
            calib_iterator, calibration_data_path, return_queries
        )
    
    def load_all_components(self, load_path: str, ignore_components: List[str] = None) -> None:
        """Load all model components from the specified path."""
        if ignore_components is None:
            ignore_components = []
        return self.unified_predictor.load_all_components(load_path, ignore_components)
    
    def trainModel(self, general_args: Namespace, args: Namespace) -> None:
        """Train the model using the provided arguments."""
        return self.unified_predictor.trainModel(general_args, args)
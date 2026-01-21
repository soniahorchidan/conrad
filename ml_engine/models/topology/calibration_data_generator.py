"""
Calibration data generation utilities for ThreeHopPipeline.
"""
import torch
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .model import ThreeHopPipeline
from argparse import Namespace
from tqdm import tqdm
from .dataloader import DataIterator
from .calibration_sampler import FileBasedDataIterator
from ..multihop.model import MultiHopPredictor


class CalibrationDataGenerator:
    """Handles calibration data generation for ThreeHopPipeline."""
    
    def __init__(self, model: 'ThreeHopPipeline', device: str, top_k: int = 1000, 
                 max_gt_size: int = 1000, calib_batch_size: int = 8, num_entities: Optional[int] = None):
        self.model = model
        self.device = device
        # Use provided num_entities or get from model args, default to 14541 for backward compatibility
        if num_entities is not None:
            self.num_entities = num_entities
        elif hasattr(model, 'args') and hasattr(model.args, 'num_entities'):
            self.num_entities = model.args.num_entities
        else:
            # Fallback: try to get from graph_data if available
            # Otherwise use default (will be updated when dataset is known)
            self.num_entities = 14541  # Default for fb15k-237
            logging.warning(f"num_entities not provided, using default: {self.num_entities}")
        self.top_k = top_k  # Only persist top-K scores per hop to reduce memory
        self.max_gt_size = max_gt_size  # Skip queries with >max_gt_size GT hop3 entities
        self.calib_batch_size = calib_batch_size  # Batch size for calibration processing (reduced for GPU memory)
    
    def generate_calibration_samples(self, save_path: str, db_controller: Any, 
                                   vector_db_controller: Any, calib_iterator: Optional[Any] = None, 
                                   calibration_data_path: Optional[str] = None, 
                                   return_queries: bool = False) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        """
        Generate calibration samples for conformal prediction.
        
        Args:
            save_path: Path for saving/loading data
            db_controller: Database controller
            vector_db_controller: Vector database controller
            calib_iterator: Optional pre-created iterator
            calibration_data_path: Optional path to saved calibration data files
            return_queries: Whether to return queries along with scores and answers
        
        Returns:
            Tuple of (scores, answers, queries) where queries is None if return_queries=False
        """
        logging.info("Starting calibration sample generation")
        logging.info(f"Calibration config - Max GT size: {self.max_gt_size}, Top-K: {self.top_k}, Batch size: {self.calib_batch_size}")
        
        self._setup_calibration_args()
        calib_iterator = self._get_calibration_iterator(
            calib_iterator, calibration_data_path, save_path, db_controller
        )
        
        all_scores, all_answers, all_queries = self._process_calibration_batches(calib_iterator)
        
        if not all_scores:
            raise RuntimeError("No calibration samples were produced from calib_iterator.")
        
        logging.info(f"Generated {len(all_scores)} calibration queries")
        
        logging.debug(f"Generated calibration data: {len(all_scores)} queries with path-aware scores, "
                     f"answers count={len(all_answers)}, queries count={len(all_queries)}")
        
        if return_queries:
            return all_scores, all_answers, all_queries
        
        return all_scores, all_answers, None
    
    def _setup_calibration_args(self) -> None:
        """Setup calibration-specific arguments."""
        self.model.args.batch_size = self.calib_batch_size
        self.model.args.conformal_prediction = Namespace()
        self.model.args.conformal_prediction.calib_size = 0.5
        self.model.args.max_num_ans = 1000
        self.model.args.query_generator_log_ratio = 0.3
    
    def _get_calibration_iterator(self, calib_iterator: Optional[Any], 
                                 calibration_data_path: Optional[str], 
                                 save_path: str, db_controller: Any) -> Any:
        """Get the appropriate calibration iterator."""
        if calib_iterator is not None:
            return calib_iterator
        
        if calibration_data_path is not None:
            # Use file-based iterator from saved calibration data
            # Load graph_data from database if not cached in the calibration directory
            from .dataloader import DataIterator as DI
            graph_data = DI.get_graph_data(db_controller, self.device)
            
            return FileBasedDataIterator(
                calibration_data_path,
                batch_size=self.model.args.batch_size,
                device=self.device,
                graph_data=graph_data  # Pass graph_data from database
            )
        else:
            # Fall back to database-based iterator
            return DataIterator(
                self.model.args,
                save_path,
                db_controller,
                self.device,
                "calib",
                set(),
            )
    
    def _process_calibration_batches(self, calib_iterator: Any) -> Tuple[List[torch.Tensor], Any, Any]:
        """Process all calibration batches."""
        all_scores = []
        all_answers = []
        all_queries = []
        
        calib_list = list(calib_iterator)
        logging.info(f"Processing {len(calib_list)} calibration batches")
        
        total_filtered = 0
        total_processed = 0
        
        model_class_name = self.model.__class__.__name__
        is_2u = model_class_name == "TwoUnionPipeline"
        is_2ip = model_class_name == "TwoIntersectProjectPipeline"
        # 2u and 2ip use 2D optimization (τ_intersect/τ_branch, τ_proj)
        # 3p uses 3D optimization (τ1, τ2, τ3)
        zero_thresholds = [0.0, 0.0] if (is_2u or is_2ip) else [0.0, 0.0, 0.0]

        for batch_idx, data in enumerate(tqdm(calib_list)):
            # Handle both 3-tuple and 4-tuple formats
            if len(data) == 4:
                # TODO(Sonia): intermediate results are passed in and, so we dont need to return them separately in the iteraror.
                query, ans, graph_data, _ = data
            else:
                query, ans, graph_data = data
            
            graph_data = graph_data.to(self.device)
            query = query.to(self.device)
            
            ans_list = ans if isinstance(ans, list) else [ans]
            kept_indices = []
            batch_filtered = 0
            
            # When batch_size=1, ensure we process truly one query at a time for maximum memory efficiency
            # Even if iterator gives us a batch, process each query individually
            if self.calib_batch_size == 1:
                # Ensure query is 2D [batch_size, query_dim]
                if query.dim() == 1:
                    query = query.unsqueeze(0)
                
                # Process each query individually
                num_queries = query.shape[0]
                for query_idx in range(num_queries):
                    single_query = query[query_idx:query_idx+1].clone()
                    single_ans = [ans_list[query_idx]] if query_idx < len(ans_list) else ans_list
                    
                    # Predict single query with immediate cleanup
                    try:
                        single_pred_out = self.model._predict(single_query, zero_thresholds, graph_data, single_ans)
                        
                        if single_pred_out:
                            pred_out = single_pred_out[0]
                            gt_labels = ans_list[query_idx] if query_idx < len(ans_list) else None
                            
                            try:
                                if is_2u:
                                    gt_size = len(gt_labels) if isinstance(gt_labels, (list, set)) else 0
                                    if gt_size > self.max_gt_size:
                                        batch_filtered += 1
                                        continue
                                    vector_nc_scores = self._extract_2u_scores(pred_out, gt_labels)
                                elif is_2ip:
                                    if isinstance(gt_labels, dict):
                                        gt_final = gt_labels.get(3, gt_labels.get('final', []))
                                        gt_size = len(gt_final)
                                    else:
                                        gt_size = len(gt_labels) if isinstance(gt_labels, (list, set)) else 0
                                    if gt_size > self.max_gt_size:
                                        batch_filtered += 1
                                        continue
                                    vector_nc_scores = self._extract_2ip_scores(pred_out, gt_labels)
                                else:
                                    if gt_labels and 3 in gt_labels:
                                        gt_hop3_size = len(gt_labels[3])
                                        if gt_hop3_size > self.max_gt_size:
                                            batch_filtered += 1
                                            continue
                                    vector_nc_scores = self._extract_vector_scores(pred_out, gt_labels)
                                
                                all_scores.append(vector_nc_scores)
                                kept_indices.append(query_idx)
                                if query_idx < len(ans_list):
                                    all_answers.append(ans_list[query_idx])
                                all_queries.append(single_query[0].cpu())
                            finally:
                                del pred_out
                    finally:
                        del single_query, single_pred_out
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                
                # Skip the batch processing loop below
                total_filtered += batch_filtered
                total_processed += len(kept_indices)
                # Clear graph_data reference
                del graph_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            
            # Predict with GT restriction (use zero thresholds) - for batch_size > 1
            batch_pred_out = self.model._predict(query, zero_thresholds, graph_data, ans)
            
            for idx, pred_out in enumerate(batch_pred_out):
                gt_labels = ans_list[idx] if idx < len(ans_list) else None
                
                try:
                    if is_2u:
                        # For 2u, GT labels are expected to be a list/set of entities (union output).
                        gt_size = len(gt_labels) if isinstance(gt_labels, (list, set)) else 0
                        if gt_size > self.max_gt_size:
                            logging.debug(
                                f"Filtering query {idx} - GT size {gt_size} > {self.max_gt_size}"
                            )
                            batch_filtered += 1
                            continue
                        vector_nc_scores = self._extract_2u_scores(pred_out, gt_labels)
                    elif is_2ip:
                        # For 2ip, GT labels can be dict {1: branch1, 2: branch2, 3: final} or list/set
                        if isinstance(gt_labels, dict):
                            gt_final = gt_labels.get(3, gt_labels.get('final', []))
                            gt_size = len(gt_final)
                        else:
                            gt_size = len(gt_labels) if isinstance(gt_labels, (list, set)) else 0
                        if gt_size > self.max_gt_size:
                            logging.debug(
                                f"Filtering query {idx} - GT size {gt_size} > {self.max_gt_size}"
                            )
                            batch_filtered += 1
                            continue
                        vector_nc_scores = self._extract_2ip_scores(pred_out, gt_labels)
                    else:
                        # For 3p: Filter queries with too many GT hop3 entities
                        if gt_labels and 3 in gt_labels:
                            gt_hop3_size = len(gt_labels[3])
                            if gt_hop3_size > self.max_gt_size:
                                logging.debug(f"Filtering query {idx} - GT hop3 size {gt_hop3_size} > {self.max_gt_size}")
                                batch_filtered += 1
                                continue

                        vector_nc_scores = self._extract_vector_scores(pred_out, gt_labels)

                    all_scores.append(vector_nc_scores)
                    kept_indices.append(idx)
                finally:
                    # Clear pred_out immediately after extraction to free GPU memory
                    del pred_out
                    # Clear cache after each query when batch_size=1
                    if self.calib_batch_size == 1 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            total_filtered += batch_filtered
            total_processed += len(kept_indices)
            
            # Append answers and queries for kept indices
            for idx in kept_indices:
                if idx < len(ans_list):
                    all_answers.append(ans_list[idx])
                
                if isinstance(query, torch.Tensor) and query.dim() > 1:
                    if idx < query.shape[0]:
                        all_queries.append(query[idx].cpu() if query.is_cuda else query[idx])
                else:
                    all_queries.append(query.cpu() if isinstance(query, torch.Tensor) and query.is_cuda else query)
            
            # Clear GPU memory after each batch to prevent accumulation
            del batch_pred_out
            # Move query and graph_data off GPU if they're tensors
            if isinstance(query, torch.Tensor) and query.is_cuda:
                query = query.cpu()
            if hasattr(graph_data, 'to') and hasattr(graph_data, 'device') and str(graph_data.device).startswith('cuda'):
                # graph_data might be a Data object, be careful
                pass  # Don't move graph_data as it might be reused
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # Periodic more aggressive cleanup every 5 batches (more frequent)
            if (batch_idx + 1) % 5 == 0 and torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        logging.info(f"Calibration processing complete - Processed: {total_processed}, Filtered: {total_filtered}")
        return all_scores, all_answers, all_queries

    def _extract_2u_scores(self, pred_out: Dict[str, Any], gt_labels: Optional[Any] = None) -> Dict[str, Any]:
        """
        Extract branch-wise calibration scores for 2u (TwoUnionPipeline).

        pred_out is expected to contain:
          {
            "branch1": {"nodes": [...], "scores": Tensor[1, E]},
            "branch2": {"nodes": [...], "scores": Tensor[1, E]},
            "union_nodes": [...]
          }

        Returns:
          {
            "branch1": {"nodes": [...], "scores": <sparse top-k dict>},
            "branch2": {"nodes": [...], "scores": <sparse top-k dict>}
          }
        """
        scores1 = pred_out["branch1"]["scores"]
        scores2 = pred_out["branch2"]["scores"]

        # Build a must-include set similar to the 3p path-aware extraction:
        # include GT + nodes scoring above the minimum GT score under MAX aggregation.
        must_include = set()
        if gt_labels is not None:
            if isinstance(gt_labels, set):
                gt = [int(x) for x in gt_labels]
            elif isinstance(gt_labels, list):
                gt = [int(x) for x in gt_labels]
            elif isinstance(gt_labels, dict):
                # Best-effort: union over all dict values.
                gt = []
                for v in gt_labels.values():
                    if isinstance(v, (list, set)):
                        gt.extend(int(x) for x in v)
            else:
                gt = []

            if gt:
                combined = torch.maximum(scores1.squeeze(0), scores2.squeeze(0))
                valid = [i for i in gt if 0 <= int(i) < combined.size(0)]
                if valid:
                    idx_tensor = torch.tensor(valid, device=combined.device, dtype=torch.long)
                    min_gt = float(combined[idx_tensor].min().item())
                    hi = torch.nonzero(combined >= min_gt, as_tuple=False).flatten().tolist()
                    must_include.update(int(i) for i in hi)
                    must_include.update(int(i) for i in valid)

        b1_processed = self._process_single_hop_scores(
            scores1, must_include=must_include if must_include else None
        )
        b2_processed = self._process_single_hop_scores(
            scores2, must_include=must_include if must_include else None
        )

        return {
            "branch1": {"nodes": pred_out["branch1"].get("nodes", []), "scores": b1_processed},
            "branch2": {"nodes": pred_out["branch2"].get("nodes", []), "scores": b2_processed},
        }

    def _extract_2ip_scores(self, pred_out: Dict[str, Any], gt_labels: Optional[Any] = None) -> Dict[str, Any]:
        """
        Extract calibration scores for 2ip (TwoIntersectProjectPipeline) using independent branch thresholds.

        The 2ip pipeline uses 3D optimization:
        - τ_branch1: Applied independently to branch1 scores
        - τ_branch2: Applied independently to branch2 scores
        - τ_proj: Applied to MAX-aggregated projection scores
        
        Intersection is computed as discrete intersection: nodes passing both τ_branch1 AND τ_branch2.

        pred_out is expected to contain:
          {
            "branch1": {"nodes": [...], "scores": Tensor[1, E]},
            "branch2": {"nodes": [...], "scores": Tensor[1, E]},
            "intersection": {"nodes": [...], "scores": Tensor[E]},  # MIN-aggregated (for backward compatibility)
            "projection": {"nodes": [...], "scores": dict {node_id: score}},  # MAX-aggregated
            "final_nodes": [...]
          }

        Returns:
          {
            "branch1": {"nodes": [...], "scores": <sparse top-k dict>},  # Individual branch scores
            "branch2": {"nodes": [...], "scores": <sparse top-k dict>},  # Individual branch scores
            "intersection": {"nodes": [...], "scores": <sparse top-k dict>},  # MIN-aggregated (for backward compatibility)
            "projection": {"nodes": [...], "scores": <sparse top-k dict>}  # MAX-aggregated
          }
        """
        # Extract branch scores
        scores1 = pred_out.get("branch1", {}).get("scores")
        scores2 = pred_out.get("branch2", {}).get("scores")
        
        if scores1 is None or scores2 is None:
            logging.warning("2ip: Missing branch scores in pred_out")
            return {
                "branch1": {"nodes": [], "scores": {'indices': torch.empty(0, dtype=torch.long), 
                                                     'values': torch.empty(0, dtype=torch.float32), 'shape': 0}},
                "branch2": {"nodes": [], "scores": {'indices': torch.empty(0, dtype=torch.long), 
                                                     'values': torch.empty(0, dtype=torch.float32), 'shape': 0}},
                "intersection": {"nodes": [], "scores": {'indices': torch.empty(0, dtype=torch.long), 
                                                          'values': torch.empty(0, dtype=torch.float32), 'shape': 0}},
                "projection": {"nodes": [], "scores": {'indices': torch.empty(0, dtype=torch.long), 
                                                        'values': torch.empty(0, dtype=torch.float32), 'shape': 0}},
            }
        
        # Extract intersection scores (MIN-aggregated) if available, otherwise compute
        if "intersection" in pred_out and "scores" in pred_out["intersection"]:
            intersection_scores = pred_out["intersection"]["scores"]
            if isinstance(intersection_scores, torch.Tensor):
                intersection_scores_tensor = intersection_scores
            else:
                # Convert to tensor if needed
                intersection_scores_tensor = torch.tensor(intersection_scores, dtype=torch.float32)
        else:
            # Compute MIN aggregation from branch scores
            scores1_squeezed = scores1.squeeze(0) if scores1.dim() > 1 else scores1
            scores2_squeezed = scores2.squeeze(0) if scores2.dim() > 1 else scores2
            intersection_scores_tensor = torch.minimum(scores1_squeezed, scores2_squeezed)
        
        # Extract projection scores (MAX-aggregated)
        projection_data = pred_out.get("projection", {})
        projection_scores = projection_data.get("scores", {})
        projection_nodes = projection_data.get("nodes", [])
        
        # Build must-include sets based on GT labels
        intersection_must_include = set()
        projection_must_include = set()
        
        if gt_labels is not None:
            if isinstance(gt_labels, dict):
                # GT format: {1: branch1_gt, 2: branch2_gt, 3: final_gt} or {'final': final_gt}
                gt_branch1 = set(gt_labels.get(1, []))
                gt_branch2 = set(gt_labels.get(2, []))
                gt_intersection = gt_branch1 & gt_branch2  # Intersection GT is the overlap
                gt_final = set(gt_labels.get(3, gt_labels.get('final', [])))
            elif isinstance(gt_labels, (list, set)):
                # Simple list of final entities
                gt_intersection = set()
                gt_final = set(gt_labels)
            else:
                gt_intersection = set()
                gt_final = set()
            
            # For intersection: include GT intersection nodes + nodes scoring above min GT score
            if gt_intersection:
                valid_gt = [int(idx) for idx in gt_intersection if 0 <= int(idx) < intersection_scores_tensor.size(0)]
                if valid_gt:
                    idx_tensor = torch.tensor(valid_gt, device=intersection_scores_tensor.device, dtype=torch.long)
                    min_gt_score = float(intersection_scores_tensor[idx_tensor].min().item())
                    high_score_indices = torch.nonzero(
                        intersection_scores_tensor >= min_gt_score, as_tuple=False
                    ).flatten().tolist()
                    intersection_must_include.update(int(idx) for idx in high_score_indices)
                    intersection_must_include.update(int(idx) for idx in valid_gt)
            
            # For projection: include GT final nodes + nodes scoring above min GT score
            if gt_final and isinstance(projection_scores, dict):
                gt_scores = [projection_scores.get(int(node), 0.0) for node in gt_final]
                if gt_scores:
                    min_gt_proj_score = min(gt_scores)
                    for node_id, score in projection_scores.items():
                        if score >= min_gt_proj_score:
                            projection_must_include.add(int(node_id))
                    projection_must_include.update(int(node) for node in gt_final)
        
        # Process branch scores (for backward compatibility, though not used in 2D optimization)
        b1_processed = self._process_single_hop_scores(
            scores1, must_include=intersection_must_include if intersection_must_include else None
        )
        b2_processed = self._process_single_hop_scores(
            scores2, must_include=intersection_must_include if intersection_must_include else None
        )
        
        # Process intersection scores (MIN-aggregated)
        if intersection_scores_tensor.dim() == 1:
            intersection_scores_tensor = intersection_scores_tensor.unsqueeze(0)
        intersection_processed = self._process_single_hop_scores(
            intersection_scores_tensor, 
            must_include=intersection_must_include if intersection_must_include else None
        )
        
        # Process projection scores (convert dict to tensor, then process)
        if isinstance(projection_scores, dict) and projection_scores:
            projection_tensor = torch.zeros(self.num_entities, dtype=torch.float32)
            for node_id, score in projection_scores.items():
                if 0 <= int(node_id) < self.num_entities:
                    projection_tensor[int(node_id)] = float(score)
            projection_processed = self._process_single_hop_scores(
                projection_tensor.unsqueeze(0),
                must_include=projection_must_include if projection_must_include else None
            )
        else:
            projection_processed = {'indices': torch.empty(0, dtype=torch.long), 
                                    'values': torch.empty(0, dtype=torch.float32), 'shape': 0}
        
        # Extract intersection nodes from pred_out
        intersection_nodes = pred_out.get("intersection", {}).get("nodes", [])
        
        return {
            "branch1": {"nodes": pred_out.get("branch1", {}).get("nodes", []), "scores": b1_processed},
            "branch2": {"nodes": pred_out.get("branch2", {}).get("nodes", []), "scores": b2_processed},
            "intersection": {"nodes": intersection_nodes, "scores": intersection_processed},
            "projection": {"nodes": projection_nodes, "scores": projection_processed},
            "projection_paths": pred_out.get("projection_paths", []),  # ADD THIS LINE - preserve path-aware data
        }
        
    def _extract_vector_scores(self, pred_out: Dict[str, Dict[str, Any]], 
                               gt_labels: Optional[Dict[int, List[int]]] = None) -> Dict[str, Any]:
        """
        Extract path-aware calibration data from prediction output.
        
        This preserves the cascade structure: which hop1 nodes led to which hop2 nodes, etc.
        This is critical because thresholds at early hops affect what nodes are available
        for later hops.
        
        Args:
            pred_out: Prediction output from model
            gt_labels: Optional ground truth labels {hop: [entity_ids]} for GT-only hop3 filtering
        
        Returns dict with path structure:
        {
            'hop1': {'nodes': [...], 'scores': tensor (D,)},
            'hop2': [{'parent': hop1_node, 'scores': tensor (D,)} for each hop1 node],
            'hop3': [{'parent': (hop1_node, hop2_node), 'scores': tensor (D,)} for each path]
        }
        """
        # Extract hop data
        hop1_nodes = pred_out["hop1"]["nodes"]
        hop1_scores = pred_out["hop1"]["scores"]
        hop2_scores_list = pred_out["hop2"]["scores"] if "hop2" in pred_out else []
        hop3_scores_list = pred_out["hop3"]["scores"] if "hop3" in pred_out else []
        
        gt_labels = gt_labels or {}
        gt_hop1 = set(gt_labels.get(1, []))
        gt_hop2 = set(gt_labels.get(2, []))
        gt_hop3 = set(gt_labels.get(3, []))
        
        min_gt_thresholds = self._compute_min_gt_thresholds(pred_out, gt_labels)
        hop1_threshold = min_gt_thresholds.get(1)
        hop2_threshold = min_gt_thresholds.get(2)
        hop3_threshold = min_gt_thresholds.get(3)
        
        # Build hop1 data - use top-K format like hop2/hop3
        hop1_scores_flat = hop1_scores.squeeze(0)
        
        # Collect must_include nodes (GT + high-scoring nodes above threshold)
        must_include = set()
        if hop1_threshold is not None:
            high_score_indices = torch.nonzero(
                hop1_scores_flat >= hop1_threshold, as_tuple=False
            ).flatten().tolist()
            must_include.update(int(idx) for idx in high_score_indices)
        if gt_hop1:
            must_include.update(int(idx) for idx in gt_hop1 if 0 <= idx < hop1_scores_flat.size(0))
        
        # Use same format as hop2/hop3: top-K + must_include
        hop1_processed_scores = self._process_single_hop_scores(
            hop1_scores,
            must_include=must_include if must_include else None
        )
        
        hop1_data = {
            'nodes': hop1_nodes,
            'scores': hop1_processed_scores
        }
        
        # Build hop2 data - keep all GT nodes and any node scoring above the GT minimum
        hop2_data = []
        
        # Check if hop2_scores_list is a list (for 3p queries) or a tensor (for 2ip/2u queries)
        if isinstance(hop2_scores_list, list):
            for path_info in hop2_scores_list:
                if isinstance(path_info, dict) and 'parent' in path_info and 'scores' in path_info:
                    path_scores = path_info['scores']
                    must_include = set(gt_hop2)
                    
                    if hop2_threshold is not None:
                        path_flat = path_scores.squeeze(0)
                        high_score_indices = torch.nonzero(
                            path_flat >= hop2_threshold, as_tuple=False
                        ).flatten().tolist()
                        must_include.update(int(idx) for idx in high_score_indices)
                    
                    processed_scores = self._process_single_hop_scores(
                        path_scores,
                        must_include=must_include if must_include else None
                    )
                    
                    hop2_data.append({
                        'parent': int(path_info['parent']),
                        'scores': processed_scores
                    })
                elif isinstance(path_info, torch.Tensor):
                    path_scores = path_info
                    must_include = set(gt_hop2)
                    
                    if hop2_threshold is not None:
                        path_flat = path_scores.squeeze(0)
                        high_score_indices = torch.nonzero(
                            path_flat >= hop2_threshold, as_tuple=False
                        ).flatten().tolist()
                        must_include.update(int(idx) for idx in high_score_indices)
                    
                    parent_idx = len(hop2_data)
                    hop2_data.append({
                        'parent': int(hop1_nodes[parent_idx]) if parent_idx < len(hop1_nodes) else -1,
                        'scores': self._process_single_hop_scores(
                            path_scores,
                            must_include=must_include if must_include else None
                        )
                    })
        elif isinstance(hop2_scores_list, torch.Tensor):
            # For 2ip queries: hop2_scores_list is a single tensor, not a list
            # Process it as a single branch score
            must_include = set(gt_hop2)
            
            if hop2_threshold is not None:
                hop2_flat = hop2_scores_list.squeeze(0) if hop2_scores_list.dim() > 1 else hop2_scores_list
                high_score_indices = torch.nonzero(
                    hop2_flat >= hop2_threshold, as_tuple=False
                ).flatten().tolist()
                must_include.update(int(idx) for idx in high_score_indices)
            
            # For 2ip, hop2_data should be in the same format as hop1
            hop2_data = {
                'nodes': pred_out.get("hop2", {}).get("nodes", []),
                'scores': self._process_single_hop_scores(
                    hop2_scores_list,
                    must_include=must_include if must_include else None
                )
            }
        
        # Build hop3 data with faithful score retention
        # For 2ip queries, hop3_scores_list is a dict, not a list
        if isinstance(hop3_scores_list, dict):
            # For 2ip: hop3 is the final aggregated projection result
            # Convert dict of {node_id: score} to a proper format
            must_include = set(gt_hop3)
            
            # Get nodes and convert to tensor format
            hop3_nodes = pred_out.get("hop3", {}).get("nodes", [])
            if hop3_nodes and hop3_scores_list:
                # Create a full score tensor
                hop3_scores_tensor = torch.zeros(self.num_entities, dtype=torch.float32)
                for node_id, score in hop3_scores_list.items():
                    if 0 <= node_id < self.num_entities:
                        hop3_scores_tensor[node_id] = score
                
                if hop3_threshold is not None:
                    high_score_indices = torch.nonzero(
                        hop3_scores_tensor >= hop3_threshold, as_tuple=False
                    ).flatten().tolist()
                    must_include.update(int(idx) for idx in high_score_indices)
                
                hop3_data = {
                    'nodes': hop3_nodes,
                    'scores': self._process_single_hop_scores(
                        hop3_scores_tensor.unsqueeze(0),
                        must_include=must_include if must_include else None
                    )
                }
            else:
                hop3_data = {'nodes': [], 'scores': torch.zeros(self.top_k, dtype=torch.float32)}
        else:
            # For 3p: hop3 is path-aware
            hop3_data = self._extract_hop3_path_data(
                hop3_scores_list,
                hop3_threshold,
                gt_hop3
            )
        
        return {
            'hop1': hop1_data,
            'hop2': hop2_data,
            'hop3': hop3_data
        }
    
    def _extract_hop3_path_data(self,
                                hop3_scores_list: List[Dict[str, Any]],
                                hop3_threshold: Optional[float],
                                gt_hop3_entities: Optional[set]) -> List[Dict[str, Any]]:
        """
        Extract hop3 data with parent path information.
        """
        hop3_data = []
        
        for path_info in hop3_scores_list:
            if isinstance(path_info, dict) and 'parent' in path_info and 'scores' in path_info:
                scores_tensor = path_info['scores']
                must_include = set(gt_hop3_entities) if gt_hop3_entities else set()
                
                if hop3_threshold is not None:
                    scores_flat = scores_tensor.squeeze(0)
                    high_score_indices = torch.nonzero(
                        scores_flat >= hop3_threshold, as_tuple=False
                    ).flatten().tolist()
                    must_include.update(int(idx) for idx in high_score_indices)
                
                processed_scores = self._process_single_hop_scores(
                    scores_tensor,
                    must_include=must_include if must_include else None
                )
                
                hop3_data.append({
                    'parent': path_info['parent'],
                    'scores': processed_scores
                })
        
        return hop3_data
    
    def _compute_min_gt_thresholds(self, pred_out: Dict[str, Dict[str, Any]],
                                   gt_labels: Dict[int, List[int]]) -> Dict[int, float]:
        """Compute per-hop minimum GT score thresholds under MAX aggregation."""
        thresholds: Dict[int, float] = {}
        
        if not gt_labels:
            return thresholds
        
        def _valid_indices(indices: Iterable[int], limit: int) -> List[int]:
            return [int(idx) for idx in indices if 0 <= int(idx) < limit]
        
        # Hop1 uses direct scores
        gt_hop1 = gt_labels.get(1, [])
        if gt_hop1:
            hop1_scores = pred_out["hop1"]["scores"].squeeze(0)
            valid = _valid_indices(gt_hop1, hop1_scores.size(0))
            if valid:
                idx_tensor = torch.tensor(valid, device=hop1_scores.device)
                thresholds[1] = float(hop1_scores[idx_tensor].min().item())
        
        # Hop2 uses MAX aggregation across paths
        gt_hop2 = gt_labels.get(2, [])
        if gt_hop2:
            hop2_scores_list = pred_out["hop2"]["scores"] if "hop2" in pred_out else []
            hop2_tensors = []
            for path_info in hop2_scores_list:
                if isinstance(path_info, dict) and 'scores' in path_info:
                    hop2_tensors.append(path_info['scores'])
                elif isinstance(path_info, torch.Tensor):
                    hop2_tensors.append(path_info)
            if hop2_tensors:
                hop2_aggregated = self._process_multi_hop_scores(hop2_tensors)
                valid = _valid_indices(gt_hop2, hop2_aggregated.size(0))
                if valid:
                    idx_tensor = torch.tensor(valid, device=hop2_aggregated.device)
                    thresholds[2] = float(hop2_aggregated[idx_tensor].min().item())
        
        # Hop3 uses MAX aggregation across paths
        gt_hop3 = gt_labels.get(3, [])
        if gt_hop3:
            hop3_scores_list = pred_out["hop3"]["scores"] if "hop3" in pred_out else []
            hop3_tensors = []
            for path_info in hop3_scores_list:
                if isinstance(path_info, dict) and 'scores' in path_info:
                    hop3_tensors.append(path_info['scores'])
                elif isinstance(path_info, torch.Tensor):
                    hop3_tensors.append(path_info)
            if hop3_tensors:
                hop3_aggregated = self._process_multi_hop_scores(hop3_tensors)
                valid = _valid_indices(gt_hop3, hop3_aggregated.size(0))
                if valid:
                    idx_tensor = torch.tensor(valid, device=hop3_aggregated.device)
                    thresholds[3] = float(hop3_aggregated[idx_tensor].min().item())
        
        return thresholds
    
    def _process_hop_scores_to_entity_dim(self, hop_scores: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Process hop scores to ensure consistent entity dimension (14541).
        
        Args:
            hop_scores: Either a single tensor (hop1) or list of tensors (hop2, hop3)
            
        Returns:
            Tensor of shape (14541,) with scores for all entities
        """
        if isinstance(hop_scores, torch.Tensor):
            # Single tensor case (hop1)
            return self._process_single_hop_scores(hop_scores)
        elif isinstance(hop_scores, list):
            # List of tensors case (hop2, hop3) - use max aggregation
            return self._process_multi_hop_scores(hop_scores)
        else:
            raise ValueError(f"Unexpected hop_scores type: {type(hop_scores)}")
    
    def _process_single_hop_scores(self, hop_scores: torch.Tensor, top_k: Optional[int] = None,
                                   must_include: Optional[Iterable[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Process single hop scores to a sparse top-K representation to reduce memory.
        Returns a dict with { 'indices': Tensor[K], 'values': Tensor[K], 'shape': int }.
        """
        hop_scores_flat = hop_scores.squeeze(0)
        total_size = min(hop_scores_flat.size(0), self.num_entities)
        effective_top_k = self.top_k if top_k is None else top_k
        k = min(effective_top_k, total_size)
        
        if total_size == 0 or k == 0:
            indices_tensor = torch.empty(0, dtype=torch.long)
            values_tensor = torch.empty(0, dtype=torch.float32)
        else:
            top_values, top_indices = torch.topk(
                hop_scores_flat[:total_size], k, largest=True, sorted=True
            )
            score_map = {
                int(idx): float(val)
                for idx, val in zip(top_indices.detach().cpu().tolist(), top_values.detach().cpu().tolist())
            }
            
            if must_include:
                for raw_idx in must_include:
                    idx = int(raw_idx)
                    if idx < 0 or idx >= total_size:
                        continue
                    val = float(hop_scores_flat[idx].item())
                    if idx in score_map:
                        score_map[idx] = max(score_map[idx], val)
                    else:
                        score_map[idx] = val
            
            sorted_items = sorted(score_map.items())
            indices_tensor = torch.tensor([idx for idx, _ in sorted_items], dtype=torch.long)
            values_tensor = torch.tensor([val for _, val in sorted_items], dtype=torch.float32)
        
        return {
            'indices': indices_tensor,
            'values': values_tensor,
            'shape': total_size
        }
    
    def _process_multi_hop_scores(self, hop_scores: List[torch.Tensor]) -> torch.Tensor:
        """Process multiple hop scores using elementwise max aggregation."""
        if not hop_scores:
            return torch.zeros(self.num_entities, dtype=torch.float32, device=self.device)
        
        hop_tensors = [s.squeeze(0) if s.dim() > 1 else s for s in hop_scores]
        padded = [
            torch.cat([t[:self.num_entities], torch.zeros(self.num_entities - t.size(0), device=t.device)]) 
            if t.size(0) < self.num_entities else t[:self.num_entities]
            for t in hop_tensors
        ]
        stacked = torch.stack(padded, dim=0)  # [num_hops, num_entities]
        return stacked.max(dim=0).values  # elementwise max across hops

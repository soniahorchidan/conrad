"""
Vector conformal optimization for multi-component models.

This optimizer is designed to be generic and query-type agnostic. It focuses solely
on the optimization algorithm (building threshold chains, evaluating FNR, finding
optimal thresholds) while delegating execution semantics to pipeline models.

Architecture:
- VectorOptimizer: Generic optimization algorithm (black-box approach)
- Pipeline models (ThreeHopPipeline, TwoUnionPipeline): Query-specific execution logic
  - ThreeHopPipeline: Implements cascading for 3-hop queries
  - TwoUnionPipeline: Implements union for 2-union queries
  
The optimizer treats pipelines as black boxes and calls their methods to apply thresholds.
Each pipeline is responsible for its own execution semantics (cascade, union, intersection, etc.).
"""
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional
from joblib import Parallel, delayed
from .utils import binomial_upper_bound, compute_fnr_metrics
import os

# Configuration for different query types
QUERY_TYPE_CONFIG = {
    '3p': {
        'k': 3,
        'component_keys': ['hop1', 'hop2', 'hop3'],
        'component_type': 'hop',
        'has_path_dependencies': True,
        'final_component_idx': 3,  # For ground truth extraction
        'description': '3-hop queries with sequential path dependencies'
    },
    '2u': {
        'k': 2,
        'component_keys': ['branch1', 'branch2'],
        'component_type': 'branch',
        'has_path_dependencies': False,
        'final_component_idx': None,  # Compute union of branches 1 and 2
        'description': '2-union queries with independent branches'
    },
    '2ip': {
        'k': 3,
        'component_keys': ['branch1', 'branch2', 'projection'],
        'component_type': 'intersect_project',
        'has_path_dependencies': True, 
        'final_component_idx': 3,  # Ground truth uses key 3 (or 'final') for final projection results
        'description': '2-intersect-project queries with 3D optimization: τ_branch1, τ_branch2 for independent branch pruning, τ_proj for projection'
    },
}

class VectorOptimizer:
    """
    Generic vector conformal optimization for multi-component models.
    
    This optimizer is query-type agnostic and treats pipeline models as black boxes.
    It focuses solely on the CRC optimization algorithm:
    - Building candidate threshold chains
    - Evaluating empirical FNR for each threshold
    - Finding optimal thresholds that satisfy risk constraints
    
    Pipeline Interface (expected methods):
    -----------------------------------
    Pipeline models should implement these methods for optimal performance:
    
    1. apply_thresholds_to_scores(cal_scores, thresholds, true_labels) -> List[List[int]]
       - Apply thresholds to calibration data and return predictions
       - Required for all pipelines
       
    2. apply_thresholds_to_scores_cached(dense_cache, cal_scores, thresholds, component_keys) -> List[List[int]]
       - Optimized version using pre-computed dense vectors
       - Optional (falls back to non-cached version)
       - Dramatically speeds up optimization (1000+ threshold evaluations)
       
    3. precompute_dense_vectors_for_optimization(cal_scores, component_keys) -> List[Dict]
       - Pre-compute and cache dense score vectors
       - Optional (uses generic fallback if not provided)
       - Each pipeline can optimize caching for its specific execution pattern
    
    The optimizer does NOT implement query-specific execution logic (cascade, union, etc.).
    All execution semantics are delegated to the pipeline models.
    """

    def __init__(self, cal_scores: List[Dict], true_labels: List[Dict], 
                 query_type: str, pipeline_model=None, num_entities: Optional[int] = None):
        """
        Initialize the vector optimizer.
        
        Args:
            cal_scores: List of calibration data dicts (one per query).
                For 3p queries: {'hop1': {...}, 'hop2': [{...}, ...], 'hop3': [{...}, ...]}
                For 2u queries: {'branch1': {...}, 'branch2': {...}}
            true_labels: True labels for each query.
                For 3p: dict with keys {1, 2, 3} mapping to [entity_ids]
                For 2u: list of [entity_ids] or dict with final output key
            query_type: Query type identifier (e.g., '3p', '2u', '2ip')
            pipeline_model: Optional pipeline model class (e.g., ThreeHopPipeline, TwoUnionPipeline)
                Used for apply_thresholds_to_scores() method.
            num_entities: Number of entities in the dataset. If None, will try to get from pipeline_model
                or default to 14541 (fb15k-237).
        """
        self.cal_scores = cal_scores
        self.true_labels = true_labels
        self.n = len(cal_scores)
        self.pipeline_model = pipeline_model
        
        # Determine num_entities
        if num_entities is not None:
            self.num_entities = num_entities
        elif pipeline_model is not None and hasattr(pipeline_model, 'calibration_generator'):
            self.num_entities = pipeline_model.calibration_generator.num_entities
        elif pipeline_model is not None and hasattr(pipeline_model, 'args') and hasattr(pipeline_model.args, 'num_entities'):
            self.num_entities = pipeline_model.args.num_entities
        else:
            self.num_entities = 14541  # Default for fb15k-237
            logging.warning(f"num_entities not provided, using default: {self.num_entities}")
        
        # Load configuration for this query type
        if query_type not in QUERY_TYPE_CONFIG:
            raise ValueError(
                f"Unknown query type '{query_type}'. "
                f"Supported types: {list(QUERY_TYPE_CONFIG.keys())}"
            )
        
        config = QUERY_TYPE_CONFIG[query_type]
        self.query_type = query_type
        self.k = config['k']
        self.component_keys = config['component_keys']
        self.component_type = config['component_type']
        self.has_path_dependencies = config['has_path_dependencies']
        self.final_component_idx = config['final_component_idx']
        
        logging.info(f"Loaded {self.n} {query_type} calibration queries with {self.k} components")
        logging.info(f"Description: {config['description']}")
        logging.info(f"Component keys: {self.component_keys}")
        
        # Import the appropriate pipeline model if not provided
        if self.pipeline_model is None:
            self._load_default_pipeline_model()
        
        # Pre-compute dense vectors once to avoid redundant conversions during optimization
        self._precompute_dense_vectors()
        logging.info(f"Pre-computed dense score vectors for faster threshold evaluation")
    
    def _load_default_pipeline_model(self):
        """Load the default pipeline model based on query type."""
        if self.query_type == '3p':
            from models.topology.model import ThreeHopPipeline
            self.pipeline_model = ThreeHopPipeline
            logging.info("Using ThreeHopPipeline for 3p queries")
        elif self.query_type == '2u':
            from models.topology.model import TwoUnionPipeline
            self.pipeline_model = TwoUnionPipeline
            logging.info("Using TwoUnionPipeline for 2u queries")
        elif self.query_type == '2ip':
            from models.topology.model import TwoIntersectProjectPipeline
            self.pipeline_model = TwoIntersectProjectPipeline
            logging.info("Using TwoIntersectProjectPipeline for 2ip queries")
        else:
            raise ValueError(f"No default pipeline model for query type '{self.query_type}'")
    
    def print_score_distributions(self):
        """
        Print the distribution of scores in the calibration data dictionaries.
        Shows statistics for all components (hops/branches).
        Uses MAX aggregation for components with paths.
        """
        component_scores_lists = {key: [] for key in self.component_keys}
        component_path_counts = {key: 0 for key in self.component_keys}
        
        for query_data in self.cal_scores:
            for comp_key in self.component_keys:
                comp_data = query_data[comp_key]
                
                # Handle different data structures
                if isinstance(comp_data, dict) and 'scores' in comp_data:
                    # Single score vector (e.g., first hop or branch)
                    scores = comp_data['scores']
                    dense = self.pipeline_model._scores_to_dense_vector(scores, num_entities=self.num_entities)
                    component_scores_lists[comp_key].append(dense)
                    component_path_counts[comp_key] += 1
                    
                elif isinstance(comp_data, list):
                    # Multiple paths (e.g., hop2, hop3) - MAX aggregate
                    score_vectors = []
                    for path_data in comp_data:
                        if isinstance(path_data, dict) and 'scores' in path_data:
                            scores = path_data['scores']
                            dense = self.pipeline_model._scores_to_dense_vector(scores, num_entities=self.num_entities)
                            score_vectors.append(dense)
                            component_path_counts[comp_key] += 1
                    
                    if score_vectors:
                        max_aggregated = np.maximum.reduce(score_vectors)
                        component_scores_lists[comp_key].append(max_aggregated)
        
        # Stack and flatten scores
        component_scores_flat = {}
        for comp_key in self.component_keys:
            scores_list = component_scores_lists[comp_key]
            if scores_list:
                stacked = np.stack(scores_list)
                component_scores_flat[comp_key] = stacked.flatten()
            else:
                component_scores_flat[comp_key] = np.array([])
        
        # Print distributions
        logging.info("=" * 80)
        logging.info("SCORE DISTRIBUTIONS IN CALIBRATION DATA (MAX Aggregated)")
        logging.info("=" * 80)
        logging.info(f"Total queries: {self.n}")
        logging.info(f"Total entities per query: 14,541")
        logging.info(f"Component type: {self.component_type}")
        
        for idx, comp_key in enumerate(self.component_keys, start=1):
            scores_array = component_scores_flat[comp_key]
            path_count = component_path_counts[comp_key]
            
            if len(scores_array) == 0:
                logging.info(f"\n{comp_key}: No scores found")
                continue
            
            # Only count non-zero scores for statistics
            nonzero_scores = scores_array[scores_array > 0]
            
            logging.info(f"{comp_key} Score Distribution (MAX Aggregated):")
            logging.info(f"  Total score entries: {len(scores_array):,} ({self.n} queries × 14,541 entities)")
            logging.info(f"  Non-zero scores: {len(nonzero_scores):,}")
            logging.info(f"  Zero scores: {len(scores_array) - len(nonzero_scores):,}")
            logging.info(f"  Path count: {path_count}")
            
            if len(nonzero_scores) > 0:
                logging.info(f"  Min: {nonzero_scores.min():.6f}")
                logging.info(f"  Max: {nonzero_scores.max():.6f}")
                logging.info(f"  Mean: {nonzero_scores.mean():.6f}")
                logging.info(f"  Median: {np.median(nonzero_scores):.6f}")
                logging.info(f"  Std: {nonzero_scores.std():.6f}")
                logging.info(f"  25th percentile: {np.percentile(nonzero_scores, 25):.6f}")
                logging.info(f"  75th percentile: {np.percentile(nonzero_scores, 75):.6f}")
                logging.info(f"  90th percentile: {np.percentile(nonzero_scores, 90):.6f}")
                logging.info(f"  95th percentile: {np.percentile(nonzero_scores, 95):.6f}")
                logging.info(f"  99th percentile: {np.percentile(nonzero_scores, 99):.6f}")
        
        logging.info("=" * 80)

    def _precompute_dense_vectors(self):
        """
        Pre-compute dense score vectors for all components across all queries.
        This is done ONCE during initialization to avoid redundant conversions
        during threshold optimization (which evaluates 1000+ threshold candidates).
        
        Delegates to pipeline model for query-type-specific caching logic.
        
        Stores pre-computed dense vectors in self.dense_cache.
        """        
        # Delegate to pipeline model if it provides custom caching
        if hasattr(self.pipeline_model, 'precompute_dense_vectors_for_optimization'):
            self.dense_cache = self.pipeline_model.precompute_dense_vectors_for_optimization(
                self.cal_scores,
                self.component_keys
            )
        else:
            # Generic fallback based on has_path_dependencies flag
            if self.has_path_dependencies:
                self._precompute_dense_vectors_with_dependencies()
            else:
                self._precompute_dense_vectors_independent()
    
    def _precompute_dense_vectors_independent(self):
        """
        Pre-compute dense vectors for independent components (e.g., 2u branches).
        No path tracking needed since components are evaluated independently then combined.
        """
        self.dense_cache = []
        
        for query_idx, query_data in enumerate(self.cal_scores):
            query_cache = {}
            
            for comp_key in self.component_keys:
                comp_data = query_data[comp_key]
                
                # For independent components, expect dict with 'scores' key
                if isinstance(comp_data, dict) and 'scores' in comp_data:
                    scores = comp_data['scores']
                    is_gt_only = isinstance(scores, dict) and scores.get('gt_only', False)
                    dense = self.pipeline_model._scores_to_dense_vector(scores, num_entities=self.num_entities)
                    
                    query_cache[f'{comp_key}_dense'] = dense
                    query_cache[f'{comp_key}_is_gt_only'] = is_gt_only
                    if 'nodes' in comp_data:
                        query_cache[f'{comp_key}_calibration_nodes'] = set(comp_data['nodes'])
                else:
                    raise ValueError(f"Expected dict with 'scores' for independent component {comp_key}, got {type(comp_data)}")
            
            self.dense_cache.append(query_cache)
    
    def _precompute_dense_vectors_with_dependencies(self):
        """
        Pre-compute dense vectors for sequential components with path dependencies (e.g., 3p hops).
        Maintains parent-child relationships for threshold-dependent path filtering.
        """
        self.dense_cache = []
        
        for query_idx, query_data in enumerate(self.cal_scores):
            query_cache = {}
            
            # Process each component
            for comp_idx, comp_key in enumerate(self.component_keys):
                comp_data = query_data[comp_key]
                
                # Initialize component cache
                query_cache[f'{comp_key}_calibration_nodes'] = set()
                query_cache[f'{comp_key}_dense_vectors'] = []
                query_cache[f'{comp_key}_parents'] = []
                query_cache[f'{comp_key}_is_gt_only'] = False
                query_cache[f'{comp_key}_max_scores'] = None
                
                # Check for pre-aggregated scores (from RAPS)
                aggregated_key = f'{comp_key}_aggregated'
                is_already_aggregated = aggregated_key in query_data
                
                if is_already_aggregated:
                    # Use pre-aggregated scores
                    query_cache[f'{comp_key}_max_scores'] = self.pipeline_model._scores_to_dense_vector(
                        query_data[aggregated_key], num_entities=self.num_entities
                    )
                    query_cache[f'{comp_key}_is_gt_only'] = query_data[aggregated_key].get('gt_only', False)
                    
                elif isinstance(comp_data, dict) and 'scores' in comp_data:
                    # First component or single-path component
                    scores = comp_data['scores']
                    if isinstance(scores, dict) and scores.get('gt_only', False):
                        query_cache[f'{comp_key}_is_gt_only'] = True
                    dense = self.pipeline_model._scores_to_dense_vector(scores, num_entities=self.num_entities)
                    query_cache[f'{comp_key}_dense'] = dense
                    
                    if 'nodes' in comp_data:
                        query_cache[f'{comp_key}_calibration_nodes'] = set(comp_data['nodes'])
                
                elif isinstance(comp_data, list):
                    # Multi-path component (e.g., hop2, hop3)
                    # Need to track which paths are valid based on parent components
                    for path_data in comp_data:
                        if not isinstance(path_data, dict) or 'scores' not in path_data:
                            continue
                        
                        scores = path_data['scores']
                        parent = path_data.get('parent')
                        
                        # Check if parent is valid
                        if comp_idx > 0 and parent is not None:
                            prev_comp_key = self.component_keys[comp_idx - 1]
                            prev_calibration_nodes = query_cache.get(f'{prev_comp_key}_calibration_nodes', set())
                            
                            # Handle tuple parents (e.g., (hop1_parent, hop2_parent))
                            if isinstance(parent, tuple):
                                # Check if first parent is in previous component's calibration nodes
                                if prev_calibration_nodes and parent[0] not in prev_calibration_nodes:
                                    continue
                            elif isinstance(parent, (int, np.integer)):
                                # Single parent
                                if prev_calibration_nodes and parent not in prev_calibration_nodes:
                                    continue
                        
                        # Valid path - add to cache
                        if isinstance(scores, dict) and scores.get('gt_only', False):
                            query_cache[f'{comp_key}_is_gt_only'] = True
                        
                        dense = self.pipeline_model._scores_to_dense_vector(scores, num_entities=self.num_entities)
                        query_cache[f'{comp_key}_dense_vectors'].append(dense)
                        query_cache[f'{comp_key}_parents'].append(parent)
                    
                    # Pre-compute MAX aggregated scores (threshold-independent)
                    if query_cache[f'{comp_key}_dense_vectors']:
                        query_cache[f'{comp_key}_max_scores'] = np.maximum.reduce(
                            query_cache[f'{comp_key}_dense_vectors']
                        )
                
                # Extract calibration nodes for next component
                # (nodes that should be tracked through the cascade)
                if comp_idx < len(self.component_keys) - 1:
                    next_comp_key = self.component_keys[comp_idx + 1]
                    next_comp_data = query_data.get(next_comp_key)
                    
                    if isinstance(next_comp_data, list):
                        for path_data in next_comp_data:
                            if isinstance(path_data, dict) and 'parent' in path_data:
                                parent = path_data['parent']
                                # Extract relevant parent for this component
                                if isinstance(parent, tuple):
                                    # Multi-level parent - use the one corresponding to this component
                                    relevant_parent = parent[comp_idx] if comp_idx < len(parent) else parent[0]
                                else:
                                    relevant_parent = parent
                                
                                # Check if this parent is reachable from previous components
                                if comp_idx == 0:
                                    query_cache[f'{comp_key}_calibration_nodes'].add(relevant_parent)
                                else:
                                    prev_comp_key = self.component_keys[comp_idx - 1]
                                    prev_calib_nodes = query_cache.get(f'{prev_comp_key}_calibration_nodes', set())
                                    if not prev_calib_nodes or relevant_parent in prev_calib_nodes or comp_idx == 0:
                                        query_cache[f'{comp_key}_calibration_nodes'].add(relevant_parent)
            
            self.dense_cache.append(query_cache)

    # TODO(sonia): is this even needed?
    def _extract_gt_scores(self) -> np.ndarray:
        """
        Extracts scores into a matrix for threshold chain building.
        Uses MAX aggregation for components with multiple paths.
        
        Returns:
            np.ndarray of shape (N, k, num_entities) where:
            - N = number of queries
            - k = number of components
            - num_entities = number of entities
        """
        scores_matrix = np.zeros((self.n, self.k, self.num_entities), dtype=np.float32)
        
        for query_idx, query_data in enumerate(self.cal_scores):
            for comp_idx, comp_key in enumerate(self.component_keys):
                comp_data = query_data[comp_key]
                
                if isinstance(comp_data, dict) and 'scores' in comp_data:
                    # Single score vector
                    dense = self.pipeline_model._scores_to_dense_vector(comp_data['scores'], num_entities=self.num_entities)
                    scores_matrix[query_idx, comp_idx, :min(len(dense), self.num_entities)] = dense[:self.num_entities]
                    
                elif isinstance(comp_data, list) and self.has_path_dependencies:
                    # Multiple paths - filter by valid parents and MAX aggregate
                    score_vectors = []
                    
                    # Get calibration nodes from previous component
                    if comp_idx > 0:
                        prev_comp_key = self.component_keys[comp_idx - 1]
                        prev_comp_data = query_data[prev_comp_key]
                        if isinstance(prev_comp_data, dict) and 'nodes' in prev_comp_data:
                            calibration_nodes = set(prev_comp_data['nodes'])
                        else:
                            calibration_nodes = None
                    else:
                        calibration_nodes = None
                    
                    for path_data in comp_data:
                        if not isinstance(path_data, dict) or 'scores' not in path_data:
                            continue
                        
                        parent = path_data.get('parent')
                        
                        # Check if path is valid based on parent
                        is_valid = True
                        if calibration_nodes is not None and parent is not None:
                            if isinstance(parent, tuple):
                                # Multi-level parent - check first element
                                is_valid = parent[0] in calibration_nodes
                            elif isinstance(parent, (int, np.integer)):
                                is_valid = parent in calibration_nodes
                        
                        if is_valid:
                            vec = self.pipeline_model._scores_to_dense_vector(path_data['scores'], num_entities=self.num_entities)
                            score_vectors.append(vec)
                    
                    if score_vectors:
                        max_aggregated = np.maximum.reduce(score_vectors)
                        scores_matrix[query_idx, comp_idx, :min(len(max_aggregated), self.num_entities)] = max_aggregated[:self.num_entities]
                
                elif isinstance(comp_data, list) and not self.has_path_dependencies:
                    # Independent components with multiple items (shouldn't happen for 2u, but handle it)
                    score_vectors = []
                    for item in comp_data:
                        if isinstance(item, dict) and 'scores' in item:
                            vec = self.pipeline_model._scores_to_dense_vector(item['scores'], num_entities=self.num_entities)
                            score_vectors.append(vec)
                    
                    if score_vectors:
                        max_aggregated = np.maximum.reduce(score_vectors)
                        scores_matrix[query_idx, comp_idx, :min(len(max_aggregated), self.num_entities)] = max_aggregated[:self.num_entities]
                
        return scores_matrix

    def _build_candidate_thresholds(self, calibration_scores, num_levels: int = 100) -> List[np.ndarray]:
        quantiles = np.linspace(0.0, 1.0, num_levels)**3
        chain = []

        logging.info(f"Building candidate thresholds with {num_levels} levels...")
        
        # 1. Validation
        for j in range(self.k):
            if not np.any(np.isfinite(calibration_scores[:, j])):
                raise ValueError(f"No finite scores for hop {j+1}") 

        if self.k == 3:
            if self.query_type == '2ip':
                # For 2ip: branch1, branch2 (symmetric), projection (depends on intersection)
                hop_exponents = [0.85, 0.85, 1.02]
                logging.info(f"Using 2ip specialized exponents: {hop_exponents}")
            else:
                # For 3p: sequential hops
                hop_exponents = [0.85, 1.30, 1.02]
                logging.info(f"Using 3-hop specialized exponents: {hop_exponents}")
        elif self.k == 2:
            # For 2u: two independent branches
            hop_exponents = [0.9, 0.9]
            logging.info(f"Using 2u specialized exponents: {hop_exponents}")
        else:
            # Fallback for k=1 or k>3
            hop_exponents = [1.0] * self.k
            logging.info(f"Using fallback balanced exponents for k={self.k}")

        # Aggregate using MAX per query (shape: (N, k, num_entities) -> (N, k))
        aggregated_scores = np.max(calibration_scores, axis=2)
        
        # 3. Generate all candidates
        for i in range(num_levels):
            q = quantiles[i]
            thresholds = np.array([
                np.quantile(aggregated_scores[:, j][np.isfinite(aggregated_scores[:, j])], q ** hop_exponents[j])
                for j in range(self.k)
            ])
            chain.append(thresholds)

        logging.info("Generated initial chain. Filtering...")
        
        # Round to handle floating point noise
        rounded_chain = [np.round(t, 10) for t in chain]
        
        # Keep only the absolute origin OR vectors where ALL components are > 0
        filtered_chain = []
        for i, vec in enumerate(rounded_chain):
            is_origin = np.all(vec == 0)
            is_interior = np.all(vec > 0)
            
            # We only add the origin ONCE (at the start)
            if is_origin:
                if len(filtered_chain) == 0:
                    filtered_chain.append(chain[i])
            elif is_interior:
                filtered_chain.append(chain[i])

        # Ensure the final [1, 1, 1] is there
        if not np.all(filtered_chain[-1] == 1):
            filtered_chain.append(np.ones(self.k))

        final_chain = np.vstack([np.zeros(self.k), filtered_chain])
        # Deduplicate in case the first entry was already very close to zero
        _, indices = np.unique(final_chain.round(8), axis=0, return_index=True)
        final_chain = final_chain[np.argsort(indices)]

        logging.info(f"Final filtered chain levels: {len(final_chain)}")
        logging.info(f"Final chain: {final_chain}")
        return final_chain

    def _get_gt_scores_for_component(self, calibration_scores: List[Dict], component_idx: int) -> np.ndarray:
        """
        Extract scores for ground truth entities for a specific component.
        
        Args:
            calibration_scores: List of calibration data dicts
            component_idx: Index of component
                - For 2ip: 0=branch1, 1=branch2, 2=projection
                - For 3p: 0=hop1, 1=hop2, 2=hop3
                - For 2u: 0=branch1, 1=branch2
            
        Returns:
            Array of GT scores for this component across all queries
        """
        comp_key = self.component_keys[component_idx]
        gt_scores = []
        
        for query_idx, query_data in enumerate(calibration_scores):
            # Get component data
            comp_data = query_data.get(comp_key, {})
            
            # Convert to dense vector
            if isinstance(comp_data, dict) and 'scores' in comp_data:
                scores = comp_data['scores']
                dense = self.pipeline_model._scores_to_dense_vector(scores, num_entities=self.num_entities)
            elif isinstance(comp_data, list):
                # Multiple paths - MAX aggregate
                score_vectors = []
                for path_data in comp_data:
                    if isinstance(path_data, dict) and 'scores' in path_data:
                        path_scores = path_data['scores']
                        path_dense = self.pipeline_model._scores_to_dense_vector(path_scores, num_entities=self.num_entities)
                        score_vectors.append(path_dense)
                if score_vectors:
                    dense = np.maximum.reduce(score_vectors)
                else:
                    dense = np.zeros(self.num_entities)
            else:
                dense = np.zeros(self.num_entities)
            
            # Get GT entities for this component
            if query_idx < len(self.true_labels):
                label = self.true_labels[query_idx]
                
                if self.query_type == '2ip':
                    # For 2ip with 3D optimization: component_idx 0=branch1, 1=branch2, 2=projection
                    if component_idx == 0:  # Branch 1
                        if isinstance(label, dict):
                            gt_entities = label.get(1, [])
                        else:
                            gt_entities = []
                    elif component_idx == 1:  # Branch 2
                        if isinstance(label, dict):
                            gt_entities = label.get(2, [])
                        else:
                            gt_entities = []
                    else:  # component_idx == 2, Projection
                        # GT projection = final entities (key 3 or 'final')
                        if isinstance(label, dict):
                            gt_entities = label.get(3, label.get('final', []))
                        elif isinstance(label, (list, set)):
                            gt_entities = list(label) if isinstance(label, set) else label
                        else:
                            gt_entities = []
                elif self.query_type == '3p':
                    # For 3p, component_idx maps to hop number (1-indexed)
                    hop_num = component_idx + 1
                    if isinstance(label, dict):
                        gt_entities = label.get(hop_num, [])
                    else:
                        gt_entities = []
                elif self.query_type == '2u':
                    # For 2u, component_idx 0 = branch1, 1 = branch2
                    branch_num = component_idx + 1
                    if isinstance(label, dict):
                        gt_entities = label.get(branch_num, [])
                    else:
                        gt_entities = []
                else:
                    gt_entities = []
                
                # Extract scores for GT entities
                for entity_id in gt_entities:
                    if 0 <= entity_id < len(dense):
                        score = dense[entity_id]
                        if score > 0:  # Only include non-zero scores
                            gt_scores.append(score)
            else:
                # No GT for this query
                pass
        
        return np.array(gt_scores) if gt_scores else np.array([])
    
    def _get_all_scores_for_component(self, calibration_scores: List[Dict], component_idx: int) -> np.ndarray:
        """
        Extract all non-zero scores for a specific component across all queries.
        
        Args:
            calibration_scores: List of calibration data dicts
            component_idx: Index of component
            
        Returns:
            Array of all non-zero scores for this component
        """
        comp_key = self.component_keys[component_idx]
        all_scores = []
        
        for query_data in calibration_scores:
            comp_data = query_data.get(comp_key, {})
            
            # Convert to dense vector
            if isinstance(comp_data, dict) and 'scores' in comp_data:
                scores = comp_data['scores']
                dense = self.pipeline_model._scores_to_dense_vector(scores, num_entities=self.num_entities)
            elif isinstance(comp_data, list):
                # Multiple paths - MAX aggregate
                score_vectors = []
                for path_data in comp_data:
                    if isinstance(path_data, dict) and 'scores' in path_data:
                        path_scores = path_data['scores']
                        path_dense = self.pipeline_model._scores_to_dense_vector(path_scores, num_entities=self.num_entities)
                        score_vectors.append(path_dense)
                if score_vectors:
                    dense = np.maximum.reduce(score_vectors)
                else:
                    dense = np.zeros(self.num_entities)
            else:
                dense = np.zeros(self.num_entities)
            
            # Extract all non-zero scores
            nonzero_scores = dense[dense > 0]
            all_scores.extend(nonzero_scores.tolist())
        
        return np.array(all_scores) if all_scores else np.array([])

    def _evaluate_thresholds(self, thresholds: np.ndarray, alpha: float, delta: float) -> Tuple[bool, float, float]:
        """
        Evaluate a single threshold configuration.
        
        Args:
            thresholds: Threshold values
            alpha: Target false negative rate
            delta: Confidence level
            
        Returns:
            Tuple of (is_feasible, empirical_fnr, upper_bound)
        """
        fnr = self._compute_fnr_for_thresholds(thresholds)
        m = int(round(fnr * self.n))
        ucb = binomial_upper_bound(m, self.n, delta)
        
        return ucb <= alpha, fnr, ucb

    def optimize_thresholds(self, alpha: float = 0.1, delta: float = 0.05, num_levels: int = 25, chain_type: str = "balanced") -> np.ndarray:
        logging.error("optimize_thresholds is not implemented")
        pass

    def optimize_thresholds_batch(self, alphas: List[float], delta: float = 0.05) -> Dict[float, np.ndarray]:
        calibration_scores = self._extract_gt_scores()
        chain = self._build_candidate_thresholds(calibration_scores)
        n = self.n
        B = 1.0
        
        memo = {}  # Cache FNR results to avoid re-computing for different alphas

        def get_adjusted_risk(idx):
            if idx not in memo:
                # Compute ONLY when requested
                fnr = self._compute_fnr_for_thresholds_cached(chain[idx])
                memo[idx] = (n / (n + 1)) * fnr + (B / (n + 1))
            return memo[idx]

        best_thresholds_dict = {}
        
        # We can narrow the search range for subsequent alphas if they are sorted
        search_low = 0 
        
        for alpha in sorted(alphas):
            logging.info(f"Processing alpha={alpha}")
            low = search_low
            high = len(chain) - 1
            best_idxForAlpha = 0
            
            # Binary search for the largest index (strictest threshold) where risk <= alpha
            while low <= high:
                mid = (low + high) // 2
                if get_adjusted_risk(mid) <= alpha:
                    best_idxForAlpha = mid
                    low = mid + 1  # Look for a stricter threshold
                else:
                    high = mid - 1
            
            best_thresholds_dict[alpha] = chain[best_idxForAlpha]
            logging.info(f"α={alpha} -> Optimal τ: {best_thresholds_dict[alpha]}, Adjusted Risk: {get_adjusted_risk(best_idxForAlpha):.4f}")
            # Optimization: Since alphas are sorted, the next alpha (larger) 
            # will have a best_idx >= current best_idx
            search_low = best_idxForAlpha 

        return best_thresholds_dict
    
    def _compute_fnr_for_thresholds_cached(self, thresholds: np.ndarray) -> float:
        """
        Compute FNR using pipeline-specific execution logic.
        
        The pipeline model is responsible for implementing its own threshold application
        logic (cascade for 3p, union for 2u, etc.). The optimizer treats it as a black box.
        
        Args:
            thresholds: Threshold values for each component (τ1, τ2, ..., τk)
            
        Returns:
            Mean false negative rate across queries
        """
        # Delegate to pipeline model's cached threshold application
        if hasattr(self.pipeline_model, 'apply_thresholds_to_scores_cached'):
            predictions = self.pipeline_model.apply_thresholds_to_scores_cached(
                self.dense_cache,
                self.cal_scores,
                thresholds,
                self.component_keys
            )
        else:
            # Fallback to non-cached version
            predictions = self.pipeline_model.apply_thresholds_to_scores(
                self.cal_scores,
                thresholds,
                self.true_labels,
                num_entities=self.num_entities
            )
        
        # Extract ground truth using configured final component index
        ground_truth = self._extract_ground_truth()
        
        # Compute FNR using the unified metric
        fnr, _, _ = compute_fnr_metrics(predictions, ground_truth)
        return fnr
    
    def _extract_ground_truth(self) -> List[List[int]]:
        """
        Extract ground truth labels for FNR computation.
        
        For union queries (final_component_idx=None), computes union of branches 1 and 2.
        For other queries, extracts from the specified component index.
        
        Returns:
            List of ground truth entity lists (one per query)
        """
        ground_truth = []
        for idx, label in enumerate(self.true_labels):
            if isinstance(label, dict):
                if self.final_component_idx is None:
                    # For union queries: compute union of branches 1 and 2
                    branch1 = set(label.get(1, []))
                    branch2 = set(label.get(2, []))
                    gt_entities = list(branch1 | branch2)
                else:
                    # For other queries: extract from specified component
                    gt_entities = label.get(self.final_component_idx, [])
            elif isinstance(label, (list, set)):
                gt_entities = list(label) if isinstance(label, set) else label
            else:
                gt_entities = []
            
            ground_truth.append(gt_entities)
        
        return ground_truth
            
    def _compute_fnr_for_thresholds(self, thresholds: np.ndarray) -> float:
        """   
        Args:
            thresholds: Threshold values for each component (τ1, τ2, ..., τk)
            
        Returns:
            Mean false negative rate across queries
        """
        # Use the model's apply_thresholds_to_scores for consistency
        predictions = self.pipeline_model.apply_thresholds_to_scores(
            self.cal_scores, 
            thresholds, 
            self.true_labels,
            num_entities=self.num_entities
        )
        
        # Extract ground truth (final component's entities)
        ground_truth = []
        for label in self.true_labels:
            if isinstance(label, dict):
                # Use configured final_component_idx
                if self.final_component_idx is not None:
                    gt_entities = label.get(self.final_component_idx, [])
                else:
                    # For union/intersection, try common keys
                    gt_entities = label.get('union', label.get(max(label.keys()) if label.keys() else 0, []))
            elif isinstance(label, list):
                gt_entities = label
            else:
                gt_entities = []
            ground_truth.append(gt_entities)
        
        # Compute FNR using the unified metric
        fnr, _, _ = compute_fnr_metrics(predictions, ground_truth)
        return fnr
"""
Vector conformal optimization for multi-component models.

The optimizer is query-type agnostic and delegates query execution semantics
(cascade, union, intersect-project) to pipeline models via
`pipeline_model.apply_thresholds_to_scores(...)`.
"""
import numpy as np
import logging
from typing import List, Dict, Optional
from .utils import compute_fnr_metrics

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
    
    Supported query types: '3p', '2u', '2ip'.

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
            query_type: Query type identifier (e.g., '3p', '2u', '2ip')
            pipeline_model: Optional pipeline model class (e.g., ThreeHopPipeline, TwoUnionPipeline)
                Used for apply_thresholds_to_scores() method.
            num_entities: Number of entities in the dataset.
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
                    # Not supported in this project (2u uses dict scores per branch).
                    raise ValueError(
                        f"Unexpected list data for independent component '{comp_key}'. "
                        f"Expected dict with 'scores'."
                    )
                else:
                    raise ValueError(
                        f"Unsupported calibration score format for component '{comp_key}': {type(comp_data)}"
                    )
                
        return scores_matrix

    def _build_candidate_thresholds(self, calibration_scores, num_levels: int = 100) -> List[np.ndarray]:
        quantiles = np.linspace(0.0, 1.0, num_levels)**3
        chain = []

        logging.info(f"Building candidate thresholds with {num_levels} levels...")
        
        # 1. Validation
        for j in range(self.k):
            if not np.any(np.isfinite(calibration_scores[:, j])):
                raise ValueError(f"No finite scores for hop {j+1}") 

        # if self.query_type == '2ip':
        #     hop_exponents = [0.85, 0.85, 1.02]
        # elif self.query_type == '3p':
        #     hop_exponents = [0.85, 1.30, 1.02]
        # elif self.query_type == '2u':
        #     hop_exponents = [0.9, 0.9]
        # else:
        #     raise ValueError(f"Unsupported query_type='{self.query_type}'. Supported: '2ip', '3p', '2u'.")

        # logging.info(f"Using {self.query_type} specialized exponents: {hop_exponents}")

        # Aggregate using MAX per query (shape: (N, k, num_entities) -> (N, k))
        aggregated_scores = np.max(calibration_scores, axis=2)

        # 1. Calculate the 'Quality' of each hop based on GT scores
        # High mean score = High confidence = We can afford to be stricter (Lower Exponent)
        hop_means = []
        for j in range(self.k):
            # Get valid GT scores for this hop
            valid_scores = aggregated_scores[:, j]
            valid_scores = valid_scores[np.isfinite(valid_scores)]
            
            if len(valid_scores) > 0:
                mean_score = np.mean(valid_scores)
            else:
                mean_score = 0.5 # Fallback
            
            # Clip to avoid division by zero or extreme outliers
            mean_score = np.clip(mean_score, 0.1, 0.99)
            hop_means.append(mean_score)
        
        hop_means = np.array(hop_means)
        
        # 2. Compute Auto-Exponents: Inverse Proportionality
        # Base idea: Exponent ~ (Global Average / Hop Average)
        # If Hop 1 has mean 0.9 and Global is 0.6 -> Exp = 0.66 (Strict)
        # If Hop 2 has mean 0.3 and Global is 0.6 -> Exp = 2.0 (Loose)
        
        global_mean = np.mean(hop_means)
        
        # Add a 'dampening factor' (power) to control how aggressive the tuning is.
        # power=1.0 is linear. power=0.0 makes all exponents 1.0 (uniform).
        # power=0.5 is a safe, conservative setting for VLDB.
        tuning_aggression = 1.0 
        
        hop_exponents = (global_mean / hop_means) ** tuning_aggression
        
        # 3. Safety Constraints
        # Cap exponents to prevent them from going wild (e.g., 0.01 or 10.0)
        hop_exponents = np.clip(hop_exponents, 0.5, 2.0)
        logging.info(f"Auto-tuned exponents based on score means {np.round(hop_means, 3)}: {np.round(hop_exponents, 3)}")
        
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


    def optimize_thresholds_batch(self, alphas: List[float]) -> Dict[float, np.ndarray]:
        calibration_scores = self._extract_gt_scores()
        chain = self._build_candidate_thresholds(calibration_scores)
        n = self.n
        B = 1.0
        
        memo = {}  # Cache FNR results to avoid re-computing for different alphas

        def get_adjusted_risk(idx):
            if idx not in memo:
                # Compute ONLY when requested
                fnr = self._compute_fnr_for_thresholds(chain[idx])
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
            logging.info(f"alpha={alpha} -> Optimal lamhat: {best_thresholds_dict[alpha]}, Adjusted Risk: {get_adjusted_risk(best_idxForAlpha):.4f}")
            # Optimization: Since alphas are sorted, the next alpha (larger) 
            # will have a best_idx >= current best_idx
            search_low = best_idxForAlpha 

        return best_thresholds_dict
    
    def _compute_fnr_for_thresholds(self, thresholds: np.ndarray) -> float:
        """
        Compute FNR using pipeline-specific execution logic.
        
        The pipeline model is responsible for implementing its own threshold application
        logic (cascade for 3p, union for 2u, etc.). The optimizer treats it as a black box.
        
        Args:
            thresholds: Threshold values for each component (τ1, τ2, ..., τk)
            
        Returns:
            Mean false negative rate across queries
        """
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
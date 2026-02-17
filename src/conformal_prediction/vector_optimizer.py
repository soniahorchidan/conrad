"""
Vector conformal optimization for multi-component models.

The optimizer is query-type agnostic and delegates query execution semantics
(cascade, union, intersect-project) to pipeline models via
`pipeline_model.apply_thresholds_to_scores(...)`.

Scalarization strategies (for building the threshold chain in quantile space):
1. uniform: [1, 1, 1] equal thresholds in quantile space (pooled scores).
2. loose_early_tight_late: [1.5, 1.0, 0.5] — permissive first hop, aggressive last hop.
3. tight_early_loose_late: [0.5, 1.0, 1.5] — the reverse.
4. balanced_tight: [0.7, 0.7, 0.7] — uniformly aggressive.
5. balanced_loose: [1.5, 1.5, 1.5] — uniformly permissive.
6. quantile_neutral: [1, 1, 1] with raw quantile mapping (no exponentiation).
"""
import numpy as np
import logging
import random
from typing import List, Dict, Optional, Tuple, Any, Union
from .utils import compute_fnr_metrics

# Scalarization strategy names and config: 'pooled' | 'raw_quantile' | list of exponents (per component)
# Strategies are query-type-specific to ensure semantic correctness
SCALARIZATION_STRATEGIES: Dict[str, Dict[str, Union[str, List[float]]]] = {
    "uniform": {
        "3p": "pooled",
        "2u": "pooled",
        "2ip": "pooled",
    },
    "loose_early_tight_late": {
        "3p": [1.5, 1.0, 0.5],      # loose hop1, medium hop2, tight hop3
        "2u": [1.5, 1.5],            # loose branch1, tight branch2
        "2ip": [1.5, 1.5, 0.5],      # loose branch1, medium branch2, tight projection
    },
    "tight_early_loose_late": {
        "3p": [0.5, 1.0, 1.5],       # tight hop1, medium hop2, loose hop3
        "2u": [0.5, 0.5],            # tight branch1, loose branch2
        "2ip": [0.5, 0.5, 1.5],      # tight branch1, medium branch2, loose projection
    },
    "balanced_tight": {
        "3p": [0.7, 0.7, 0.7],
        "2u": [0.7, 0.7],
        "2ip": [0.7, 0.7, 0.7],
    },
    "balanced_loose": {
        "3p": [1.5, 1.5, 1.5],
        "2u": [1.5, 1.5],
        "2ip": [1.5, 1.5, 1.5],
    },
    "quantile_neutral": {
        "3p": "raw_quantile",
        "2u": "raw_quantile",
        "2ip": "raw_quantile",
    },
}
# Backward compatibility: 'quantile' -> quantile_neutral
STRATEGY_ALIASES = {"quantile": "quantile_neutral"}

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
            true_labels: True labels for each query (one dict per query).
                For 3p: dict with keys 1, 2, 3 mapping to lists of entity IDs at each hop.
                So true_labels[i][1] = GT entities at hop1, [2] = hop2, [3] = hop3 (final answers).
                Index alignment: scores_matrix[query_idx, hop_idx, :] corresponds to
                true_labels[query_idx][hop_idx + 1] (hop_idx 0→hop1, 1→hop2, 2→hop3).
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
        Does not use true_labels; only cal_scores.

        Returns:
            np.ndarray of shape (N, k, num_entities) where:
            - N = number of queries
            - k = number of components (3 for 3p: hop1, hop2, hop3)
            - num_entities = number of entities

        Key mapping (3p): scores_matrix[query_idx, hop_idx, entity_id] is the unified
        score of that entity at that hop. Per-hop ground truth is true_labels[query_idx][hop_idx + 1]
        (keys 1, 2, 3 for hop1, hop2, hop3). FNR is computed only on the final answer
        (true_labels[i][final_component_idx], i.e. key 3 for 3p) in _extract_ground_truth().
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

    def _get_exponents_for_strategy(self, strategy: str) -> Optional[List[float]]:
        """Resolve strategy to a list of k exponents (for q^e_j per component). None = raw q."""
        strategy = STRATEGY_ALIASES.get(strategy, strategy)
        if strategy not in SCALARIZATION_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Supported: {list(SCALARIZATION_STRATEGIES.keys())}"
            )
        # Get query-type-specific config
        config = SCALARIZATION_STRATEGIES[strategy].get(self.query_type)
        if config is None:
            raise ValueError(
                f"Strategy '{strategy}' not defined for query type '{self.query_type}'. "
                f"Available types: {list(SCALARIZATION_STRATEGIES[strategy].keys())}"
            )
        if config == "pooled":
            return None  # handled separately (single pooled quantile)
        if config == "raw_quantile":
            return [1.0] * self.k  # q^1 = q, i.e. raw quantile per component
        # list of exponents: should match self.k exactly
        exponents = list(config)
        if len(exponents) != self.k:
            raise ValueError(
                f"Strategy '{strategy}' for query type '{self.query_type}' has {len(exponents)} exponents, "
                f"but query type requires {self.k} components. Expected: {self.k}"
            )
        return exponents

    def _build_candidate_thresholds(self, calibration_scores, num_levels: int = 100,
                                    strategy: str = 'quantile_neutral') -> np.ndarray:
        strategy = STRATEGY_ALIASES.get(strategy, strategy)
        quantiles = np.linspace(0.0, 1.0, num_levels) ** 3
        chain = []

        logging.info(f"Building candidate thresholds with {num_levels} levels, strategy={strategy}...")

        # 1. Validation
        for j in range(self.k):
            if not np.any(np.isfinite(calibration_scores[:, j])):
                raise ValueError(f"No finite scores for hop {j+1}")

        # Aggregate using MAX per query (shape: (N, k, num_entities) -> (N, k))
        aggregated_scores = np.max(calibration_scores, axis=2)

        # 2. Strategy-specific chain generation
        strategy_config = SCALARIZATION_STRATEGIES.get(strategy, {}).get(self.query_type)
        if strategy == "uniform" or strategy_config == "pooled":
            # Uniform: same threshold for all components (pooled scores)
            all_finite_scores = aggregated_scores[np.isfinite(aggregated_scores)]
            for i in range(num_levels):
                q = quantiles[i]
                threshold = np.quantile(all_finite_scores, q)
                thresholds = np.full(self.k, threshold)
                chain.append(thresholds)

        elif strategy == "quantile_neutral" or strategy_config == "raw_quantile":
            # Quantile-neutral: raw quantile per component (no exponent)
            for i in range(num_levels):
                q = quantiles[i]
                thresholds = np.array([
                    np.quantile(aggregated_scores[:, j][np.isfinite(aggregated_scores[:, j])], q)
                    for j in range(self.k)
                ])
                chain.append(thresholds)

        else:
            # Exponent-based: q^exponents[j] per component
            exponents = self._get_exponents_for_strategy(strategy)
            if exponents is None:
                raise ValueError(f"Strategy '{strategy}' resolved to None exponents")
            logging.info(f"Using exponents for {strategy}: {exponents}")
            for i in range(num_levels):
                q = quantiles[i]
                thresholds = np.array([
                    np.quantile(
                        aggregated_scores[:, j][np.isfinite(aggregated_scores[:, j])],
                        min(max(q ** exponents[j], 0.0), 1.0)
                    )
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


    def _compute_gt_median_ranks(self, calibration_scores) -> np.ndarray:
        """
        For each hop, compute the median rank of ground truth entities
        among all candidates, based on the unified scores.
        
        A low median rank means the model places GT entities near the top
        (hop is "easy"). A high median rank means GT entities are buried
        in the ranking (hop is "hard").
        
        Args:
            calibration_scores: shape (N, k, num_entities)
        
        Returns:
            np.ndarray of shape (k,) with median GT rank per hop
        """
        median_ranks = np.zeros(self.k)

        for j in range(self.k):
            all_gt_ranks = []

            for query_idx in range(self.n):
                # Get scores for this query and hop
                scores = calibration_scores[query_idx, j, :]  # shape: (num_entities,)

                # Get ground truth entities for this hop
                label = self.true_labels[query_idx]
                if isinstance(label, dict):
                    gt_entities = label.get(j + 1, [])  # hop_idx 0 -> key 1, etc.
                else:
                    continue

                if len(gt_entities) == 0:
                    continue

                # Rank all entities by descending score
                # rank 0 = highest score
                ranked_indices = np.argsort(-scores)
                rank_of_entity = np.empty(len(scores), dtype=int)
                rank_of_entity[ranked_indices] = np.arange(len(scores))

                # Collect ranks of GT entities
                for entity_id in gt_entities:
                    if entity_id < len(scores):
                        all_gt_ranks.append(rank_of_entity[entity_id])

            if all_gt_ranks:
                median_ranks[j] = np.median(all_gt_ranks)
            else:
                median_ranks[j] = len(scores) / 2  # fallback: assume middle

            logging.info(
                f"Hop {j+1}: median GT rank = {median_ranks[j]:.1f}, "
                f"num GT samples = {len(all_gt_ranks)}"
            )

        return median_ranks


    def _split_calibration_data(
        self,
        cal_scores: List[Dict],
        true_labels: List[Dict],
        fraction_opt: float = 0.5,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Split calibration data into optimization and evaluation halves."""
        n = len(cal_scores)
        if n != len(true_labels):
            raise ValueError("cal_scores and true_labels must have the same length")
        indices = list(range(n))
        random.Random(seed).shuffle(indices)
        n_opt = max(1, int(n * fraction_opt))
        opt_idx = indices[:n_opt]
        eval_idx = indices[n_opt:]
        opt_scores = [cal_scores[i] for i in opt_idx]
        opt_labels = [true_labels[i] for i in opt_idx]
        eval_scores = [cal_scores[i] for i in eval_idx]
        eval_labels = [true_labels[i] for i in eval_idx]
        logging.info(f"Split calibration: n_opt={len(opt_scores)}, n_eval={len(eval_scores)}")
        return opt_scores, opt_labels, eval_scores, eval_labels

    def estimate_cost_from_cal_scores(
        self,
        eval_cal_scores: List[Dict],
        eval_true_labels: List[Dict],
        lambda_vec: np.ndarray,
        w_neo4j: float = 1.0,
        w_ultra: float = 10.0,
    ) -> Dict[str, float]:
        """
        Estimate average query cost and quality under a calibrated threshold vector
        using precomputed calibration scores (no live execution).

        Returns:
            dict with avg_cost, avg_neo4j_calls, avg_ultra_calls, avg_precision, avg_recall
        """
        if not hasattr(self.pipeline_model, "apply_thresholds_to_scores_with_call_counts"):
            logging.warning("Pipeline has no apply_thresholds_to_scores_with_call_counts; using zero call counts")
            preds = self.pipeline_model.apply_thresholds_to_scores(
                eval_cal_scores, lambda_vec.tolist(), eval_true_labels, self.num_entities
            )
            nq = len(preds)
            total_neo4j = total_ultra = 0
        else:
            preds, neo4j_list, ultra_list = self.pipeline_model.apply_thresholds_to_scores_with_call_counts(
                eval_cal_scores, lambda_vec.tolist(), eval_true_labels, self.num_entities
            )
            nq = len(preds)
            total_neo4j = sum(neo4j_list)
            total_ultra = sum(ultra_list)
        total_cost = w_neo4j * total_neo4j + w_ultra * total_ultra
        ground_truth = self._extract_ground_truth_from_labels(eval_true_labels)
        total_precision = 0.0
        total_recall = 0.0
        for i in range(nq):
            y_pred = set(preds[i]) if preds[i] else set()
            y_true = set(ground_truth[i]) if ground_truth[i] else set()
            if y_true:
                total_recall += len(y_pred & y_true) / len(y_true)
            if y_pred:
                total_precision += len(y_pred & y_true) / len(y_pred)
        return {
            "avg_cost": total_cost / nq if nq else 0.0,
            "avg_neo4j_calls": total_neo4j / nq if nq else 0.0,
            "avg_ultra_calls": total_ultra / nq if nq else 0.0,
            "avg_precision": total_precision / nq if nq else 0.0,
            "avg_recall": total_recall / nq if nq else 0.0,
        }

    def estimate_precision_from_cal_scores(
        self,
        eval_cal_scores: List[Dict],
        eval_true_labels: List[Dict],
        lambda_vec: np.ndarray,
    ) -> Dict[str, float]:
        """
        Estimate average precision/tightness (sum of prediction set sizes at each step)
        under a calibrated threshold vector using precomputed calibration scores.

        Returns:
            dict with avg_sum_set_sizes (sum of intermediate + final set sizes per query),
            and optionally per-step averages for logging.
        """
        if not hasattr(self.pipeline_model, "apply_thresholds_to_scores_with_intermediate_sizes"):
            logging.warning("Pipeline has no apply_thresholds_to_scores_with_intermediate_sizes; using fallback")
            preds = self.pipeline_model.apply_thresholds_to_scores(
                eval_cal_scores, lambda_vec.tolist(), eval_true_labels, self.num_entities
            )
            # Fallback: just use final prediction sizes
            total_sum_sizes = sum(len(p) if p else 0 for p in preds)
            nq = len(preds)
            return {
                "avg_sum_set_sizes": total_sum_sizes / nq if nq else 0.0,
            }
        
        preds, intermediate_sizes = self.pipeline_model.apply_thresholds_to_scores_with_intermediate_sizes(
            eval_cal_scores, lambda_vec.tolist(), eval_true_labels, self.num_entities
        )
        nq = len(preds)
        
        # Sum of set sizes per query (s1 + s2 + ... + s_final)
        total_sum_sizes = sum(sum(sizes) for sizes in intermediate_sizes)
        
        # Also compute per-step averages for logging
        if intermediate_sizes and len(intermediate_sizes[0]) > 0:
            num_steps = len(intermediate_sizes[0])
            step_totals = [0.0] * num_steps
            for sizes in intermediate_sizes:
                for i, s in enumerate(sizes):
                    step_totals[i] += s
            step_avgs = [t / nq if nq else 0.0 for t in step_totals]
        else:
            step_avgs = []
        
        return {
            "avg_sum_set_sizes": total_sum_sizes / nq if nq else 0.0,
            "step_avg_sizes": step_avgs,
        }

    def _extract_ground_truth_from_labels(self, labels: List[Dict]) -> List[List[int]]:
        """Extract ground truth entity lists from label dicts (for eval set)."""
        ground_truth = []
        for label in labels:
            if isinstance(label, dict):
                if self.final_component_idx is None:
                    branch1 = set(label.get(1, []))
                    branch2 = set(label.get(2, []))
                    gt_entities = list(branch1 | branch2)
                else:
                    gt_entities = label.get(self.final_component_idx, [])
            elif isinstance(label, (list, set)):
                gt_entities = list(label) if isinstance(label, set) else label
            else:
                gt_entities = []
            ground_truth.append(gt_entities)
        return ground_truth

    def optimize_thresholds_batch(self, alphas: List[float],
                                  strategy: str = 'uniform') -> Dict[float, np.ndarray]:
        strategy = STRATEGY_ALIASES.get(strategy, strategy)
        calibration_scores = self._extract_gt_scores()
        chain = self._build_candidate_thresholds(calibration_scores, strategy=strategy)
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

    def optimize_thresholds_batch_with_strategy_selection(
        self,
        alphas: List[float],
        fraction_opt: float = 0.5,
        objective: str = "precision",
        w_neo4j: float = 1.0,
        w_ultra: float = 10.0,
        seed: int = 42,
    ) -> Tuple[Dict[float, np.ndarray], Dict[float, str]]:
        """
        Optimize using fraction_opt of calibration data and pick the scalarization strategy
        that is valid and optimizes the chosen objective on the evaluation set.

        Args:
            alphas: List of alpha values to calibrate for
            fraction_opt: Fraction of calibration data used for optimization (rest for eval)
            objective: "cost" to minimize invocations (w_neo4j * neo4j + w_ultra * ultra),
                       "precision" to minimize sum of prediction set sizes at each step
            w_neo4j: Cost weight per Neo4j call (only used if objective="cost")
            w_ultra: Cost weight per ULTRA call (only used if objective="cost")
            seed: Random seed for data splitting

        Returns:
            best_thresholds_dict: mapping alpha -> chosen lambda vector
            chosen_strategy_per_alpha: mapping alpha -> strategy name
        """
        opt_scores, opt_labels, eval_scores, eval_labels = self._split_calibration_data(
            self.cal_scores, self.true_labels, fraction_opt=fraction_opt, seed=seed
        )
        opt_optimizer = VectorOptimizer(
            opt_scores, opt_labels,
            query_type=self.query_type,
            pipeline_model=self.pipeline_model,
            num_entities=self.num_entities,
        )
        # Eval-side optimizer only for estimate_cost (needs _extract_ground_truth_from_labels and pipeline)
        eval_optimizer = VectorOptimizer(
            eval_scores, eval_labels,
            query_type=self.query_type,
            pipeline_model=self.pipeline_model,
            num_entities=self.num_entities,
        )
        strategies = list(SCALARIZATION_STRATEGIES.keys())
        # Per alpha: list of (strategy, lambda_vec, cost_dict)
        results_per_alpha: Dict[float, List[Tuple[str, np.ndarray, Dict[str, float]]]] = {
            alpha: [] for alpha in alphas
        }
        for strategy in strategies:
            logging.info(f"Strategy '{strategy}': optimizing on opt half...")
            try:
                thresholds_dict = opt_optimizer.optimize_thresholds_batch(alphas, strategy=strategy)
            except Exception as e:
                logging.warning(f"Strategy '{strategy}' failed: {e}")
                continue
            for alpha in alphas:
                lambda_vec = thresholds_dict[alpha]
                if objective == "cost":
                    metric_dict = eval_optimizer.estimate_cost_from_cal_scores(
                        eval_scores, eval_labels, lambda_vec, w_neo4j=w_neo4j, w_ultra=w_ultra
                    )
                    metric_key = "avg_cost"
                elif objective == "precision":
                    metric_dict = eval_optimizer.estimate_precision_from_cal_scores(
                        eval_scores, eval_labels, lambda_vec
                    )
                    metric_key = "avg_sum_set_sizes"
                else:
                    raise ValueError(f"Unknown objective '{objective}'. Supported: 'cost', 'precision'")
                results_per_alpha[alpha].append((strategy, lambda_vec, metric_dict))
        best_thresholds_dict = {}
        chosen_strategy_per_alpha = {}
        for alpha in alphas:
            candidates = results_per_alpha[alpha]
            if not candidates:
                logging.warning(f"No valid strategy for alpha={alpha}; using uniform on full data")
                best_thresholds_dict[alpha] = self.optimize_thresholds_batch([alpha], strategy="uniform")[alpha]
                chosen_strategy_per_alpha[alpha] = "uniform"
                continue
            # Pick valid strategy that minimizes the chosen objective
            if objective == "cost":
                best = min(candidates, key=lambda x: x[2]["avg_cost"])
                best_thresholds_dict[alpha] = best[1]
                chosen_strategy_per_alpha[alpha] = best[0]
                logging.info(
                    f"alpha={alpha} -> chosen strategy={best[0]} (cost), "
                    f"avg_cost={best[2]['avg_cost']:.2f}, avg_neo4j={best[2]['avg_neo4j_calls']:.2f}, "
                    f"avg_ultra={best[2]['avg_ultra_calls']:.2f}"
                )
            elif objective == "precision":
                best = min(candidates, key=lambda x: x[2]["avg_sum_set_sizes"])
                best_thresholds_dict[alpha] = best[1]
                chosen_strategy_per_alpha[alpha] = best[0]
                step_avgs = best[2].get("step_avg_sizes", [])
                step_info = f", step_avgs={step_avgs}" if step_avgs else ""
                logging.info(
                    f"alpha={alpha} -> chosen strategy={best[0]} (precision), "
                    f"avg_sum_set_sizes={best[2]['avg_sum_set_sizes']:.2f}{step_info}"
                )
            else:
                raise ValueError(f"Unknown objective '{objective}'")
        return best_thresholds_dict, chosen_strategy_per_alpha

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
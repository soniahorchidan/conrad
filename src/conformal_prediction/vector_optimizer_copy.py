"""
Vector conformal optimization for multi-hop models.
"""
import numpy as np
import torch
import logging
from typing import List, Dict
from scipy.optimize import brentq
from .utils import compute_metrics


class VectorOptimizer:
    """
    Handles vector conformal optimization for multi-hop models.
    Implements Option 2B (nested chain search) for CRC-style recall control.
    
    The optimizer uses ThreeHopPipeline.apply_thresholds_to_scores() to ensure
    the SAME threshold application logic is used during both:
    - Calibration (optimizing thresholds)
    - Inference (making predictions)
    
    Only hop3 nodes (final answers) are used for FNR computation, matching
    what the model returns during inference.
    """

    def __init__(self, cal_scores: List[Dict], true_labels: List[Dict]):
        """
        Initialize the vector optimizer.
        
        Args:
            cal_scores: List of path-aware calibration data dicts (one per query).
                Each dict contains hop1/hop2/hop3 data with path connectivity.
                This structure preserves how thresholds at early hops affect later hops.
            true_labels: True labels for each query as list of dicts {hop: [entity_ids]}
        """
        self.cal_scores = cal_scores  # Now a list of dicts, not numpy array
        self.true_labels = true_labels
        self.n = len(cal_scores)
        self.k = 3  # Number of hops (always 3 for three-hop queries)
        
        logging.info(f"Loaded {self.n} path-aware calibration queries with {self.k} hops")
        
        # Pre-compute dense vectors once to avoid redundant conversions during optimization
        self._precompute_dense_vectors()
        logging.info(f"Pre-computed dense score vectors for faster threshold evaluation")

    def _precompute_dense_vectors(self):
        """
        Pre-compute dense score vectors for all paths across all queries.
        This is done ONCE during initialization to avoid redundant conversions
        during threshold optimization (which evaluates 25+ threshold candidates).
        
        Stores pre-computed dense vectors in self.dense_cache.
        """
        from models.topology.model import ThreeHopPipeline
        
        self.dense_cache = []
        
        for query_idx, query_data in enumerate(self.cal_scores):
            query_cache = {
                'hop1_calibration_nodes': set(query_data['hop1']['nodes']),
                'hop2_dense_vectors': [],
                'hop2_parents': [],
                'hop3_dense_vectors': [],
                'hop3_parents': [],
                'hop2_calibration_nodes': set(),
                'hop3_is_gt_only': False,
            }
            
            # Check if RAPS already computed aggregated scores
            hop2_already_aggregated = 'hop2_aggregated' in query_data
            hop3_already_aggregated = 'hop3_aggregated' in query_data
            
            if hop2_already_aggregated:
                # Use pre-aggregated hop2 scores from RAPS
                query_cache['hop2_max_scores'] = ThreeHopPipeline._scores_to_dense_vector(
                    query_data['hop2_aggregated']
                )
                query_cache['hop2_dense_vectors'] = []  # Not needed
                query_cache['hop2_parents'] = []
            else:
                # Compute MAX aggregation from individual paths
                for hop2_path in query_data['hop2']:
                    hop1_parent = hop2_path['parent']
                    if hop1_parent in query_cache['hop1_calibration_nodes']:
                        hop2_scores = hop2_path['scores']
                        hop2_dense = ThreeHopPipeline._scores_to_dense_vector(hop2_scores)
                        query_cache['hop2_dense_vectors'].append(hop2_dense)
                        query_cache['hop2_parents'].append(hop1_parent)
                
                # Pre-compute MAX aggregated scores (these are threshold-independent)
                if query_cache['hop2_dense_vectors']:
                    query_cache['hop2_max_scores'] = np.maximum.reduce(query_cache['hop2_dense_vectors'])
                else:
                    query_cache['hop2_max_scores'] = None
            
            # Extract hop2 calibration nodes from hop3 paths
            for hop3_path in query_data['hop3']:
                hop1_parent, hop2_parent = hop3_path['parent']
                if hop1_parent in query_cache['hop1_calibration_nodes']:
                    query_cache['hop2_calibration_nodes'].add(hop2_parent)
            
            if hop3_already_aggregated:
                # Use pre-aggregated hop3 scores from RAPS
                query_cache['hop3_max_scores'] = ThreeHopPipeline._scores_to_dense_vector(
                    query_data['hop3_aggregated']
                )
                query_cache['hop3_dense_vectors'] = []  # Not needed
                query_cache['hop3_parents'] = []
                query_cache['hop3_is_gt_only'] = query_data['hop3_aggregated'].get('gt_only', False)
            else:
                # Compute MAX aggregation from individual paths
                for hop3_path in query_data['hop3']:
                    hop1_parent, hop2_parent = hop3_path['parent']
                    if (hop1_parent in query_cache['hop1_calibration_nodes'] and 
                        hop2_parent in query_cache['hop2_calibration_nodes']):
                        hop3_scores = hop3_path['scores']
                        
                        # Check if GT-only format
                        if isinstance(hop3_scores, dict) and hop3_scores.get('gt_only', False):
                            query_cache['hop3_is_gt_only'] = True
                        
                        hop3_dense = ThreeHopPipeline._scores_to_dense_vector(hop3_scores)
                        query_cache['hop3_dense_vectors'].append(hop3_dense)
                        query_cache['hop3_parents'].append((hop1_parent, hop2_parent))
                
                # Pre-compute MAX aggregated scores (these are threshold-independent)
                if query_cache['hop3_dense_vectors']:
                    query_cache['hop3_max_scores'] = np.maximum.reduce(query_cache['hop3_dense_vectors'])
                else:
                    query_cache['hop3_max_scores'] = None
            
            self.dense_cache.append(query_cache)

    def optimize_thresholds(self, alpha: float = 0.1) -> np.ndarray:
        """
        Optimize thresholds using standard Conformal Risk Control.
        
        Standard CRC guarantee: E[loss(C(X_{n+1}), Y_{n+1})] <= alpha.
        Uses a scalar threshold λ shared across all hops, found via Brent root finding.
        
        Args:
            alpha: Target false negative rate
            
        Returns:
            Optimal threshold array
        """
        # Sanity check: compute empirical FNR with [0, 0, 0] thresholds
        # This should predict everything (all scores >= 0), so FNR should be ~0
        zero_thresholds = [0.0, 0.0, 0.0]
        zero_fnr = self._compute_fnr_for_thresholds(zero_thresholds)
        logging.info(f"Sanity check - FNR with [0, 0, 0] thresholds: {zero_fnr:.4f} (expected: 0.0)")
        assert abs(zero_fnr - 0.0) < 1e-6, f"Sanity check failed: FNR expected 0.0 but got {zero_fnr:.4f}"

        one_thresholds = [1.0, 1.0, 1.0]
        one_fnr = self._compute_fnr_for_thresholds(one_thresholds)
        logging.info(f"Sanity check - FNR with [1, 1, 1] thresholds: {one_fnr:.4f} (expected: 1.0)")
        assert abs(one_fnr - 1.0) < 1e-6, f"Sanity check failed: FNR expected 1.0 but got {one_fnr:.4f}"
        
        logging.info(f"Standard CRC optimization: target α={alpha} (E[FNR] <= α)")

        pac_alpha = ((self.n + 1) / self.n) * alpha - 1 / (self.n + 1)
        pac_alpha = max(pac_alpha, 0.0)
        logging.info(f"Using PAC-feasible α_pac={pac_alpha:.6f} for n={self.n}")

        def fnr_minus_target(lam: float) -> float:
            thresholds = [lam] * self.k
            fnr = self._compute_fnr_for_thresholds(thresholds)
            logging.debug(f"Evaluating λ={lam:.6f}: Empirical FNR={fnr:.6f}, Target={pac_alpha:.6f}")
            return fnr - pac_alpha

        lower, upper = 0.0, 1.0
        f_lower = fnr_minus_target(lower)
        f_upper = fnr_minus_target(upper)

        if f_lower >= 0:
            logging.warning(
                "Empirical FNR at λ=0 already exceeds target; returning zero thresholds."
            )
            lamhat = lower
        elif f_upper <= 0:
            logging.warning(
                "Empirical FNR at λ=1 still below target; returning unit thresholds."
            )
            lamhat = upper
        else:
            lamhat = brentq(fnr_minus_target, lower, upper, xtol=1e-6, rtol=1e-6, maxiter=200)

        best_thresholds = np.array([lamhat] * self.k, dtype=float)
        final_fnr = self._compute_fnr_for_thresholds(best_thresholds)

        logging.info(
            f"Final λ*: {best_thresholds}, Empirical FNR={final_fnr:.6f}, Target α_pac={pac_alpha:.6f}"
        )

        return best_thresholds
    
    def optimize_thresholds_batch(self, alphas: List[float]) -> Dict[float, np.ndarray]:
        """
        Optimize scalar CRC thresholds for multiple alpha values.
        
        Applies the same Brent root-finding routine used in `optimize_thresholds`
        independently for each requested alpha and replicates the scalar λ across
        the three hops.
        
        Args:
            alphas: List of target false negative rates (e.g., [0.2, 0.3, 0.4, 0.5])
            
        Returns:
            Dictionary mapping alpha -> optimal threshold array
        """
        zero_thresholds = [0.0, 0.0, 0.0]
        zero_fnr = self._compute_fnr_for_thresholds(zero_thresholds)
        logging.info(f"Sanity check - FNR with [0, 0, 0] thresholds: {zero_fnr:.4f} (expected: 0.0)")
        
        best_thresholds_dict: Dict[float, np.ndarray] = {}
        
        for alpha in sorted(alphas):
            logging.info(f"Finding optimal thresholds for α={alpha:.3f}")
            
            pac_alpha = ((self.n + 1) / self.n) * alpha - 1 / (self.n + 1)
            pac_alpha = max(pac_alpha, 0.0)
            logging.info(
                f"Using PAC-feasible α_pac={pac_alpha:.6f} for n={self.n} (original α={alpha:.6f})"
            )

            def fnr_minus_target(lam: float) -> float:
                thresholds = [lam] * self.k
                fnr = self._compute_fnr_for_thresholds(thresholds)
                logging.debug(f"  Evaluating λ={lam:.6f}: Empirical FNR={fnr:.6f}, Target={pac_alpha:.6f}")
                return fnr - pac_alpha

            lower, upper = 0.0, 1.0
            f_lower = fnr_minus_target(lower)
            f_upper = fnr_minus_target(upper)

            if f_lower >= 0:
                logging.warning(
                    f"  Empirical FNR at λ=0 already exceeds target; returning zero thresholds for α={alpha:.3f}."
                )
                lamhat = lower
            elif f_upper <= 0:
                logging.warning(
                    f"  Empirical FNR at λ=1 still below target; returning unit thresholds for α={alpha:.3f}."
                )
                lamhat = upper
            else:
                lamhat = brentq(fnr_minus_target, lower, upper, xtol=1e-6, rtol=1e-6, maxiter=200)

            thresholds_vec = np.array([lamhat] * self.k, dtype=float)
            final_fnr = self._compute_fnr_for_thresholds(thresholds_vec)

            logging.info(
                f"  Final λ*={lamhat:.6f} → thresholds={thresholds_vec}, Empirical FNR={final_fnr:.6f}, Target α_pac={pac_alpha:.6f}"
            )
            
            best_thresholds_dict[alpha] = thresholds_vec
        
        return best_thresholds_dict
    
    def _compute_fnr_for_thresholds(self, thresholds: np.ndarray) -> float:
        """
        Compute multilabel empirical FNR for calibration data using given thresholds.
        
        CRITICAL: Must simulate the full cascade with threshold filtering at each hop,
        matching what happens during real inference via predict_with_thresholds().
        
        Args:
            thresholds: Threshold values for each hop (τ1, τ2, τ3)
            
        Returns:
            Mean false negative rate across queries
        """
        # Use the model's apply_thresholds_to_scores for consistency
        from models.topology.model import ThreeHopPipeline
        predictions = ThreeHopPipeline.apply_thresholds_to_scores(
            self.cal_scores,
            thresholds,
            None,  # simulate inference: no GT restriction
            use_calib_restriction=False
        )
        
        # Extract ground truth (hop3 only - predictions are hop3 entities)
        ground_truth = []
        for label_dict in self.true_labels:
            # Only use hop3 entities since predictions only contain hop3 entities
            gt_entities = label_dict.get(3, [])
            ground_truth.append(gt_entities)
        
        # Compute FNR using the unified metric
        metrics = compute_metrics(predictions, ground_truth)
        return 1 - metrics['recall']

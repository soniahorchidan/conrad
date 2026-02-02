import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from typing import Optional
import logging
import threading


class MultiHopPredictor(nn.Module):
    """
    Builds one unified score s*(x,y) on the union of ULTRA and Neo4j candidates:
    - Neo4j confidences → map to [neoj_mini, 1]
      - ULTRA scores → map to [0, neoj_mini)
      - overlap uses max
    Works for 1-hop or k-hop the same way per hop, since you already handle the DAG outside.
    """

    def __init__(self, ultra_model, neo4j_model, args: Namespace, device: str):
        super().__init__()
        self.ultra = ultra_model
        self.neo4j = neo4j_model
        self.args = args
        self.device = device

    @torch.no_grad()
    def _unified_scores_from_raw(self, ultra_scores: torch.Tensor, neo_scores: torch.Tensor, min_neo4j: float = 0.5):
            """
            Ensures ULTRA < Neo4j strictly, but removes 'moats' and 'ties' 
            to allow Conformal Risk Control to find precise thresholds.
            """
            neo_mask = neo_scores > 0
            
            # Neo4j scores are already in [0.5, 1.0] typically, but can be lower if popularity data is missing.
            # Clamp to [0.5, 1.0] to ensure they stay above ULTRA scores [0, 0.499]
            neo_final = torch.clamp(neo_scores, min_neo4j, 1.0) 
            
            # Normalize ULTRA to [0, 0.499] globally
            ultra_final = ultra_scores * 0.499
            
            # Combine
            unified = torch.where(neo_mask, neo_final, ultra_final)
            
            # add small jitter to break ties
            # 1e-7 is large enough to break ties but too small to flip the ULTRA/Neo hierarchy
            jitter = torch.rand_like(unified) * 1e-7
            unified = torch.clamp(unified + jitter, 0, 1.0)

            return unified

    @torch.no_grad()
    def _should_skip_ultra(self, threshold: Optional[float]) -> bool:
        """
        Determines if ULTRA inference should be skipped.
        
        Args:
            threshold: Optional threshold value
            
        Returns:
            True if threshold > 0.5 (since all Ultra scores are < 0.5)
        """
        return threshold is not None and threshold > 0.5

    @torch.no_grad()
    def _run_neo4j_query(self, query: torch.Tensor):
        """
        Runs Neo4j query synchronously.
        
        Args:
            query: Concatenated head and relation tensor
            
        Returns:
            Tuple of (results, scores) from Neo4j
        """
        if hasattr(self.neo4j, 'predict_batched'):
            return self.neo4j.predict_batched(query)
        else:
            return self.neo4j.predict(query)

    @torch.no_grad()
    def _get_num_entities(self, graph_data, neo4j_results=None) -> int:
        """
        Determines the number of entities (V) from graph_data or Neo4j results.
        
        Args:
            graph_data: Graph data for ULTRA model
            neo4j_results: Optional Neo4j results to infer V from
            
        Returns:
            Number of entities
        """
        # Try to get V from graph_data first
        if hasattr(graph_data, 'num_nodes'):
            return graph_data.num_nodes
        elif hasattr(graph_data, 'x') and graph_data.x is not None:
            return graph_data.x.shape[0]
        
        # Infer V from Neo4j results if available
        if neo4j_results is not None:
            if neo4j_results and any(neo4j_results):
                max_entity = max([max(r) if r else 0 for r in neo4j_results])
                default_v = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else 14541
                return max(max_entity + 1, default_v)
        
        # Fallback to graph_data.num_nodes or default
        if hasattr(graph_data, 'num_nodes'):
            return graph_data.num_nodes
        return 14541  # Default entity count for fb15k-237

    @torch.no_grad()
    def _run_ultra_inference(self, graph_data, query: torch.Tensor, skip_ultra: bool, 
                            threshold: Optional[float], neo4j_results) -> torch.Tensor:
        """
        Runs ULTRA inference or creates zero tensor if skipped.
        
        Args:
            graph_data: Graph data for ULTRA model
            query: Concatenated head and relation tensor
            skip_ultra: Whether to skip ULTRA inference
            threshold: Optional threshold value (for logging)
            neo4j_results: Neo4j results (used to infer V if needed)
            
        Returns:
            ULTRA raw scores tensor [B, V]
        """
        if skip_ultra:
            V = self._get_num_entities(graph_data, neo4j_results)
            B = query.shape[0]
            ultra_raw = torch.zeros((B, V), device=self.device)
            
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Skipping Ultra inference for threshold {threshold:.4f} > 0.5 (V={V})")
        else:
            # Run Ultra inference
            # Ensure model is in eval mode for correct inference behavior
            # This is especially important for UltraQuery checkpoints
            if hasattr(self.ultra, 'eval'):
                self.ultra.eval()
            ultra_raw = self.ultra.forward(graph_data, query)  # [B, V]
        
        return ultra_raw

    @torch.no_grad()
    def _combine_neo4j_results(self, ultra_raw: torch.Tensor, neo4j_results, 
                               neo4j_scores) -> torch.Tensor:
        """
        Combines Neo4j results into a probability tensor matching ultra_raw shape.
        
        Args:
            ultra_raw: ULTRA raw scores tensor [B, V]
            neo4j_results: List of entity ID lists per batch
            neo4j_scores: List of score dictionaries per batch
            
        Returns:
            Probability tensor [B, V] with Neo4j scores filled in
        """
        B, V = ultra_raw.size()
        device = ultra_raw.device
        prob = torch.zeros((B, V), device=device, dtype=ultra_raw.dtype)

        # Neo4j results arrive as Python lists/dicts. Avoid per-element GPU writes from Python,
        # which are extremely slow. Instead, do one indexed assignment per batch row.
        for b in range(B):
            ids = neo4j_results[b]
            if not ids:
                continue
            # ids: List[int], neo4j_scores[b]: Dict[int, float]
            idx = torch.as_tensor(ids, dtype=torch.long, device=device)
            vals = torch.as_tensor([neo4j_scores[b][int(i)] for i in ids], dtype=ultra_raw.dtype, device=device)
            prob[b].index_put_((idx,), vals, accumulate=False)
        return prob

    # TODO(sonia): delete this function
    @torch.no_grad()
    def _print_score_distribution(self, scores: torch.Tensor, name: str):
        """
        Prints distribution statistics for a score tensor.
        
        Args:
            scores: Score tensor of any shape
            name: Name to display in the output
        """
        scores_flat = scores.flatten().cpu().float()
        non_zero = scores_flat[scores_flat > 0]
        
        print(f"\n=== {name} Score Distribution ===")
        print(f"Total values: {len(scores_flat)}")
        print(f"Non-zero values: {len(non_zero)}")
        print(f"Min: {scores_flat.min().item():.6f}")
        print(f"Max: {scores_flat.max().item():.6f}")
        print(f"Mean: {scores_flat.mean().item():.6f}")
        print(f"Std: {scores_flat.std().item():.6f}")
        print(f"Median: {scores_flat.median().item():.6f}")
        
        if len(non_zero) > 0:
            print(f"\nNon-zero statistics:")
            print(f"  Min: {non_zero.min().item():.6f}")
            print(f"  Max: {non_zero.max().item():.6f}")
            print(f"  Mean: {non_zero.mean().item():.6f}")
            print(f"  Std: {non_zero.std().item():.6f}")
            print(f"  Median: {non_zero.median().item():.6f}")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles (all values):")
        for p in percentiles:
            val = torch.quantile(scores_flat, p / 100.0).item()
            print(f"  {p}th: {val:.6f}")
        
        if len(non_zero) > 0:
            print(f"\nPercentiles (non-zero values):")
            for p in percentiles:
                val = torch.quantile(non_zero, p / 100.0).item()
                print(f"  {p}th: {val:.6f}")
        print("=" * 40)

    @torch.no_grad()
    def predict(self, head: torch.Tensor, relation: torch.Tensor, graph_data, threshold: Optional[float] = None):
        """
        Runtime scoring-only path that returns ids above a given threshold.
        This mirrors your ULTRA.predict style but uses the unified score.
        
        Args:
            head: Head entity tensor
            relation: Relation tensor
            graph_data: Graph data for ULTRA model
            threshold: Optional threshold value. If > 0.5, Ultra inference is skipped
                      since all Ultra scores are < 0.5 and won't pass the threshold.
        """
        query = torch.cat([head, relation], dim=-1)
        
        # Determine if ULTRA should be skipped
        skip_ultra = self._should_skip_ultra(threshold)
        
        # Run Neo4j query
        neo4j_results, neo4j_scores = self._run_neo4j_query(query)
        
        # Run ULTRA inference
        ultra_raw = self._run_ultra_inference(graph_data, query, skip_ultra, threshold, neo4j_results)
        
        # Combine Neo4j results into probability tensor
        prob = self._combine_neo4j_results(ultra_raw, neo4j_results, neo4j_scores)
        
        # # Print distributions of raw scores
        # self._print_score_distribution(ultra_raw, "Raw ULTRA")
        # self._print_score_distribution(prob, "Raw Neo4j")
        
        # Compute unified scores
        min_neo4j = 0.5
        s_unified = self._unified_scores_from_raw(ultra_raw, prob, min_neo4j)
        
        # Keep scores on-device (GPU) to avoid sync/copies in hot loops.
        # Callers can move to CPU only for the small outputs they need.
        return s_unified, min_neo4j
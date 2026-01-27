import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from typing import Optional
import logging


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
            # This turns the 'stairs' into a 'ramp'
            # 1e-7 is large enough to break ties but too small to flip the ULTRA/Neo hierarchy
            jitter = torch.rand_like(unified) * 1e-7
            unified = torch.clamp(unified + jitter, 0, 1.0)

            return unified

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
    
        if hasattr(self.neo4j, 'predict_batched'):
            results, scores = self.neo4j.predict_batched(query)
        else:
            results, scores = self.neo4j.predict(query)

        # Optimization: Skip Ultra inference if threshold > 0.5
        # Since Neo4j scores are in [0.5, 1.0] and Ultra scores are in [0, 0.499],
        # if threshold > 0.5, only Neo4j predictions will pass the threshold.
        skip_ultra = threshold is not None and threshold > 0.5
        
        if skip_ultra:
            # We need to get V (vocabulary size) to create ultra_raw with correct shape
            # First, try to get it from graph_data if available
            if hasattr(graph_data, 'num_nodes'):
                V = graph_data.num_nodes
            elif hasattr(graph_data, 'x') and graph_data.x is not None:
                V = graph_data.x.shape[0]
            else:
                # Fallback: infer from results or use default
                # Estimate V from the max entity ID in results, with a reasonable minimum
                if results and any(results):
                    max_entity = max([max(r) if r else 0 for r in results])
                    V = max(max_entity + 1, 14541)  # At least default for fb15k-237
                else:
                    V = 14541  # Default entity count for fb15k-237
            
            B = query.shape[0]
            ultra_raw = torch.zeros((B, V), device=self.device)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Skipping Ultra inference for threshold {threshold:.4f} > 0.5 (V={V})")
        else:
            ultra_raw = self.ultra.forward(graph_data, query)          # [B, V]
        
        B, V = ultra_raw.size()
        prob = torch.zeros((B, V), device=ultra_raw.device)
        for b in range(B):
            for a in results[b]:
                prob[b, a] = scores[b][a]

        min_neo4j = 0.5
        s_unified = self._unified_scores_from_raw(ultra_raw, prob, min_neo4j)
        
        return s_unified.cpu().float(), min_neo4j
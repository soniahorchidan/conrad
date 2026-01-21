import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import logging


# helper: safe per-row min-max to [a,b]
def _normalize_rowwise(x, a=0.0, b=1.0, mask=None):
    """
    Rescale values in x (already known to be in [0, 1]) into [a, b].

    If mask is given:
      - only entries where mask=True are rescaled
      - masked-out entries become 0
    """
    y = a + (b - a) * x

    if mask is not None:
        # keep only valid entries, fill others with 0
        y = torch.where(mask, y, torch.zeros_like(y))

    return y


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
            
            # 1. Normalize Neo4j to [0.5, 1.0] globally
            # We assume raw Neo4j is roughly [0, 1]. If not, divide by a global max.
            neo_global_max = 1.0 
            neo_norm = torch.clamp(neo_scores / neo_global_max, 0, 1.0)
            # Using sqrt helps spread out the high-confidence scores
            neo_final = 0.5 + (torch.sqrt(neo_norm) * 0.499) 
            
            # 2. Normalize ULTRA to [0, 0.499] globally
            # This ensures ULTRA is ALWAYS below Neo4j without needing row-wise logic
            ultra_final = ultra_scores * 0.499
            
            # 3. Combine
            unified = torch.where(neo_mask, neo_final, ultra_final)
            
            # 4. THE CRITICAL FIX: Add infinitesimal jitter
            # This turns the 'stairs' into a 'ramp'
            # 1e-7 is large enough to break ties but too small to flip the ULTRA/Neo hierarchy
            jitter = torch.rand_like(unified) * 1e-7
            unified = torch.clamp(unified + jitter, 0, 1.0)

            return unified

    def load_all_components(self, load_path, ignore_components: list = []):
        if (
            self.conformal_prediction is not None
            and "conformal_prediction" not in ignore_components
        ):
            logging.info("Loading conformal prediction components...")
            self.conformal_prediction.load_all_components(load_path)


    @torch.no_grad()
    def predict(self, head: torch.Tensor, relation: torch.Tensor, graph_data):
        """
        Runtime scoring-only path that returns ids above a given threshold.
        This mirrors your ULTRA.predict style but uses the unified score.
        """
        query = torch.cat([head, relation], dim=-1)
    
        if hasattr(self.neo4j, 'predict_batched'):
            results, scores = self.neo4j.predict_batched(query)
        else:
            results, scores = self.neo4j.predict(query)

        ultra_raw = self.ultra.forward(graph_data, query)          # [B, V]
        
        B, V = ultra_raw.size()
        prob = torch.zeros((B, V), device=ultra_raw.device)
        for b in range(B):
            for a in results[b]:
                prob[b, a] = scores[b][a]

        min_neo4j = 0.5
        s_unified = self._unified_scores_from_raw(ultra_raw, prob, min_neo4j)
        
        return s_unified.cpu().float(), min_neo4j
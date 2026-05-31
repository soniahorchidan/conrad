import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base import BasePipeline


class TwoIntersectPipeline(BasePipeline):
    """
    Pipeline for 2i (intersection) queries: ((anchor1, rel1), (anchor2, rel2)).

    Query execution:
    1. Predict entities from anchor1 via rel1 → S1 (filtered by τ1)
    2. Predict entities from anchor2 via rel2 → S2 (filtered by τ2)
    3. Return S1 ∩ S2
    """

    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """Query format: [anchor1, rel1, anchor2, rel2]"""
        return self._handle_confidence_modes(
            query, confidence, graph_data,
            expected_length=4,
            error_msg="TwoIntersectPipeline expects queries of length 4: [anchor1, rel1, anchor2, rel2]",
            debug_thresholds=[0.4, 0.4],
        )

    @staticmethod
    def apply_thresholds_to_scores(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> List[List[int]]:
        predictions = []
        for query_data in cal_scores:
            b1 = TwoIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch1']['scores'], thresholds[0]
            )
            b2 = TwoIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch2']['scores'], thresholds[1]
            )
            predictions.append(list(b1 & b2))
        return predictions

    @staticmethod
    def apply_thresholds_to_scores_with_call_counts(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        predictions = TwoIntersectPipeline.apply_thresholds_to_scores(
            cal_scores, thresholds, true_labels, num_entities
        )
        n = len(predictions)
        return predictions, [2] * n, [2] * n

    @staticmethod
    def apply_thresholds_to_scores_with_intermediate_sizes(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[Tuple[int, int, int]]]:
        predictions, intermediate_sizes = [], []
        for query_data in cal_scores:
            b1 = TwoIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch1']['scores'], thresholds[0]
            )
            b2 = TwoIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch2']['scores'], thresholds[1]
            )
            intersection = b1 & b2
            predictions.append(list(intersection))
            intermediate_sizes.append((len(b1), len(b2), len(intersection)))
        return predictions, intermediate_sizes

    @torch.no_grad()
    def predict_with_thresholds(
        self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
        ground_truth: Optional[List[List[int]]] = None,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        batch_results, neo4j_per_query, ultra_per_query = self._predict(
            query, lamhat, graph_data, ground_truth
        )
        predictions = [r["intersection_nodes"] for r in batch_results]
        return predictions, neo4j_per_query, ultra_per_query

    @torch.no_grad()
    def _predict(
        self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
        ground_truth: Optional[List[List[int]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query_cpu = query.detach().cpu() if isinstance(query, torch.Tensor) and query.is_cuda else query

        batch_size = query.shape[0]
        batch_results = []
        neo4j_per_query = [0] * batch_size
        ultra_per_query = [0] * batch_size

        for i in range(batch_size):
            anchor1, rel1 = query_cpu[i, 0].item(), query_cpu[i, 1].item()
            anchor2, rel2 = query_cpu[i, 2].item(), query_cpu[i, 3].item()

            t1 = lamhat[0] if len(lamhat) > 0 else 0.0
            t2 = lamhat[1] if len(lamhat) > 1 else 0.0

            src1 = torch.tensor([[anchor1]], dtype=torch.long, device=self.device)
            r1 = torch.tensor([[rel1]], dtype=torch.long, device=self.device)
            scores1, _, n1, u1 = self.unified_predictor.predict(src1, r1, graph_data, threshold=t1)
            neo4j_per_query[i] += n1
            ultra_per_query[i] += u1

            src2 = torch.tensor([[anchor2]], dtype=torch.long, device=self.device)
            r2 = torch.tensor([[rel2]], dtype=torch.long, device=self.device)
            scores2, _, n2, u2 = self.unified_predictor.predict(src2, r2, graph_data, threshold=t2)
            neo4j_per_query[i] += n2
            ultra_per_query[i] += u2

            nodes1 = set(self._extract_nodes_from_scores(scores1, t1))
            nodes2 = set(self._extract_nodes_from_scores(scores2, t2))
            intersection = list(nodes1 & nodes2)

            batch_results.append({
                "branch1": {"nodes": list(nodes1), "scores": scores1},
                "branch2": {"nodes": list(nodes2), "scores": scores2},
                "intersection_nodes": intersection,
            })

        return batch_results, neo4j_per_query, ultra_per_query

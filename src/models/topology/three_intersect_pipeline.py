import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base import BasePipeline


class ThreeIntersectPipeline(BasePipeline):
    """
    Pipeline for 3i (three-way intersection) queries:
    ((anchor1, rel1), (anchor2, rel2), (anchor3, rel3)).

    Query execution:
    1. Predict from anchor1 via rel1 → S1 (filtered by τ1)
    2. Predict from anchor2 via rel2 → S2 (filtered by τ2)
    3. Predict from anchor3 via rel3 → S3 (filtered by τ3)
    4. Return S1 ∩ S2 ∩ S3
    """

    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """Query format: [anchor1, rel1, anchor2, rel2, anchor3, rel3]"""
        return self._handle_confidence_modes(
            query, confidence, graph_data,
            expected_length=6,
            error_msg="ThreeIntersectPipeline expects queries of length 6: [anchor1, rel1, anchor2, rel2, anchor3, rel3]",
            debug_thresholds=[0.4, 0.4, 0.4],
        )

    @staticmethod
    def apply_thresholds_to_scores(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> List[List[int]]:
        predictions = []
        for query_data in cal_scores:
            b1 = ThreeIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch1']['scores'], thresholds[0]
            )
            b2 = ThreeIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch2']['scores'], thresholds[1]
            )
            b3 = ThreeIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch3']['scores'], thresholds[2]
            )
            predictions.append(list(b1 & b2 & b3))
        return predictions

    @staticmethod
    def apply_thresholds_to_scores_with_call_counts(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        predictions = ThreeIntersectPipeline.apply_thresholds_to_scores(
            cal_scores, thresholds, true_labels, num_entities
        )
        n = len(predictions)
        return predictions, [3] * n, [3] * n

    @staticmethod
    def apply_thresholds_to_scores_with_intermediate_sizes(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[Tuple[int, int, int, int]]]:
        predictions, intermediate_sizes = [], []
        for query_data in cal_scores:
            b1 = ThreeIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch1']['scores'], thresholds[0]
            )
            b2 = ThreeIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch2']['scores'], thresholds[1]
            )
            b3 = ThreeIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch3']['scores'], thresholds[2]
            )
            intersection = b1 & b2 & b3
            predictions.append(list(intersection))
            intermediate_sizes.append((len(b1), len(b2), len(b3), len(intersection)))
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
            anchor3, rel3 = query_cpu[i, 4].item(), query_cpu[i, 5].item()

            t1 = lamhat[0] if len(lamhat) > 0 else 0.0
            t2 = lamhat[1] if len(lamhat) > 1 else 0.0
            t3 = lamhat[2] if len(lamhat) > 2 else 0.0

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

            src3 = torch.tensor([[anchor3]], dtype=torch.long, device=self.device)
            r3 = torch.tensor([[rel3]], dtype=torch.long, device=self.device)
            scores3, _, n3, u3 = self.unified_predictor.predict(src3, r3, graph_data, threshold=t3)
            neo4j_per_query[i] += n3
            ultra_per_query[i] += u3

            nodes1 = set(self._extract_nodes_from_scores(scores1, t1))
            nodes2 = set(self._extract_nodes_from_scores(scores2, t2))
            nodes3 = set(self._extract_nodes_from_scores(scores3, t3))
            intersection = list(nodes1 & nodes2 & nodes3)

            batch_results.append({
                "branch1": {"nodes": list(nodes1), "scores": scores1},
                "branch2": {"nodes": list(nodes2), "scores": scores2},
                "branch3": {"nodes": list(nodes3), "scores": scores3},
                "intersection_nodes": intersection,
            })

        return batch_results, neo4j_per_query, ultra_per_query

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .three_hop_pipeline import ThreeHopPipeline


class TwoHopPipeline(ThreeHopPipeline):
    """
    Pipeline for 2p (two-hop path) queries: (anchor, rel1, rel2).

    Query execution:
    1. Predict entities from anchor via rel1 → S1 (filtered by τ1)
    2. For each s in S1, predict via rel2 → MAX-aggregate → final (filtered by τ2)
    """

    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """Query format: [src, rel1, rel2]"""
        return self._handle_confidence_modes(
            query, confidence, graph_data,
            expected_length=3,
            error_msg="TwoHopPipeline only supports 2-hop queries (len=3).",
            debug_thresholds=[0.4, 0.6],
        )

    @staticmethod
    def apply_thresholds_to_scores(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> List[List[int]]:
        predictions = []
        for query_data in cal_scores:
            hop1_valid = TwoHopPipeline._process_hop1_for_calibration(query_data, thresholds[0])
            if not hop1_valid:
                predictions.append([])
                continue
            preds = TwoHopPipeline._process_hop2_as_final_for_calibration(
                query_data, thresholds[1], hop1_valid, num_entities
            )
            predictions.append(preds)
        return predictions

    @staticmethod
    def apply_thresholds_to_scores_with_call_counts(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        predictions, neo4j_per_query, ultra_per_query = [], [], []
        for query_data in cal_scores:
            hop1_valid = TwoHopPipeline._process_hop1_for_calibration(query_data, thresholds[0])
            if not hop1_valid:
                predictions.append([])
                neo4j_per_query.append(1)
                ultra_per_query.append(1)
                continue
            n_hop2 = sum(1 for p in query_data['hop2'] if p.get('parent') in hop1_valid)
            preds = TwoHopPipeline._process_hop2_as_final_for_calibration(
                query_data, thresholds[1], hop1_valid, num_entities
            )
            predictions.append(preds)
            neo4j_per_query.append(1 + n_hop2)
            ultra_per_query.append(1 + n_hop2)
        return predictions, neo4j_per_query, ultra_per_query

    @staticmethod
    def apply_thresholds_to_scores_with_intermediate_sizes(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        predictions, intermediate_sizes = [], []
        for query_data in cal_scores:
            hop1_valid = TwoHopPipeline._process_hop1_for_calibration(query_data, thresholds[0])
            s1 = len(hop1_valid)
            if not hop1_valid:
                predictions.append([])
                intermediate_sizes.append((s1, 0))
                continue
            preds = TwoHopPipeline._process_hop2_as_final_for_calibration(
                query_data, thresholds[1], hop1_valid, num_entities
            )
            predictions.append(preds)
            intermediate_sizes.append((s1, len(preds)))
        return predictions, intermediate_sizes

    @staticmethod
    def _process_hop2_as_final_for_calibration(
        query_data: Dict, threshold: float, hop1_valid: set, num_entities: int,
    ) -> List[int]:
        """MAX-aggregate hop2 scores from valid hop1 parents, apply threshold, return as final predictions."""
        hop2_vectors = [
            TwoHopPipeline._scores_to_dense_vector(p['scores'], num_entities)
            for p in query_data['hop2']
            if p['parent'] in hop1_valid
        ]
        if not hop2_vectors:
            return []
        hop2_max = np.maximum.reduce(hop2_vectors)
        return np.where(hop2_max >= threshold)[0].tolist()

    @torch.no_grad()
    def predict_with_thresholds(
        self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
        ground_truth_hops: Optional[List[Dict]] = None,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        batch_size = query.shape[0] if query.dim() > 1 else 1
        if query.dim() == 1:
            query = query.unsqueeze(0)
        counts = [[0, 0] for _ in range(batch_size)]
        hop1_results = self._process_hop1(
            query, lamhat, graph_data, ground_truth_hops,
            ground_truth_hops is not None, keep_scores=False, counts=counts,
        )
        hop2_results = self._process_hop2(
            hop1_results, query, lamhat, graph_data, ground_truth_hops,
            ground_truth_hops is not None, keep_path_scores=False, counts=counts,
        )
        return hop2_results["nodes"], [c[0] for c in counts], [c[1] for c in counts]

    @torch.no_grad()
    def _predict(
        self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
        ground_truth_hops: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Dict[str, Any]]]:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query_cpu = query.detach().cpu() if isinstance(query, torch.Tensor) and query.is_cuda else query
        use_ground_truth = ground_truth_hops is not None
        batch_size = query.shape[0]

        hop1_results = self._process_hop1(
            query_cpu, lamhat, graph_data, ground_truth_hops, use_ground_truth, keep_scores=True,
        )
        hop2_results = self._process_hop2(
            hop1_results, query_cpu, lamhat, graph_data, ground_truth_hops, use_ground_truth,
            keep_path_scores=True,
        )
        return [
            {
                "hop1": {"nodes": hop1_results["nodes"][i], "scores": hop1_results["scores"][i]},
                "hop2": {"nodes": hop2_results["nodes"][i], "scores": hop2_results["scores"][i]},
            }
            for i in range(batch_size)
        ]

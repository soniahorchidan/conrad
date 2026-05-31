import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .three_hop_pipeline import ThreeHopPipeline


class ProjectIntersectPipeline(ThreeHopPipeline):
    """
    Pipeline for pi (project-intersect) queries:
    (anchor1, rel1, rel2, anchor2, rel3)

    Query execution:
    1. chain hop1: anchor1 → rel1 → S_int  (filtered by τ_chain1)
    2. chain hop2: each s ∈ S_int → rel2 → scores; MAX-aggregate → S_chain  (filtered by τ_chain2)
    3. 1p branch:  anchor2 → rel3 → S_1p   (filtered by τ_1p)
    4. Return S_chain ∩ S_1p

    Thresholds: [τ_chain1, τ_chain2, τ_1p]

    The chain hop1/hop2 reuse ThreeHopPipeline's _process_hop1/_process_hop2 since the
    query layout [anchor1, rel1, rel2, ...] puts the chain anchor/relations at indices 0-2,
    exactly where those methods expect them.
    """

    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """Query format: [anchor1, rel1, rel2, anchor2, rel3]"""
        return self._handle_confidence_modes(
            query, confidence, graph_data,
            expected_length=5,
            error_msg="ProjectIntersectPipeline expects queries of length 5: [anchor1, rel1, rel2, anchor2, rel3]",
            debug_thresholds=[0.4, 0.4, 0.4],
        )

    @staticmethod
    def apply_thresholds_to_scores(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> List[List[int]]:
        t_chain1 = thresholds[0] if len(thresholds) > 0 else 0.0
        t_chain2 = thresholds[1] if len(thresholds) > 1 else 0.0
        t_1p     = thresholds[2] if len(thresholds) > 2 else 0.0

        predictions = []
        for query_data in cal_scores:
            # Step 1: chain hop1
            chain_hop1_valid = ProjectIntersectPipeline._extract_nodes_by_threshold(
                query_data['chain_hop1']['scores'], t_chain1
            )
            if not chain_hop1_valid:
                predictions.append([])
                continue

            # Step 2: chain hop2 — path-aware, MAX-aggregate over survivors
            hop2_vectors = [
                ProjectIntersectPipeline._scores_to_dense_vector(p['scores'], num_entities)
                for p in query_data['chain_hop2']
                if p['parent'] in chain_hop1_valid
            ]
            if not hop2_vectors:
                predictions.append([])
                continue
            chain_result_valid = set(
                np.where(np.maximum.reduce(hop2_vectors) >= t_chain2)[0]
            )

            # Step 3: 1p branch
            branch_1p_valid = ProjectIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch_1p']['scores'], t_1p
            )

            # Step 4: intersection
            predictions.append(list(chain_result_valid & branch_1p_valid))

        return predictions

    @staticmethod
    def apply_thresholds_to_scores_with_call_counts(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        t_chain1 = thresholds[0] if len(thresholds) > 0 else 0.0
        t_chain2 = thresholds[1] if len(thresholds) > 1 else 0.0
        t_1p     = thresholds[2] if len(thresholds) > 2 else 0.0

        predictions, neo4j_per_query, ultra_per_query = [], [], []
        for query_data in cal_scores:
            chain_hop1_valid = ProjectIntersectPipeline._extract_nodes_by_threshold(
                query_data['chain_hop1']['scores'], t_chain1
            )
            # 1 call for chain hop1 + 1 call for 1p branch
            base_calls = 2
            if not chain_hop1_valid:
                predictions.append([])
                neo4j_per_query.append(base_calls)
                ultra_per_query.append(base_calls)
                continue

            n_chain2 = sum(1 for p in query_data['chain_hop2'] if p.get('parent') in chain_hop1_valid)
            hop2_vectors = [
                ProjectIntersectPipeline._scores_to_dense_vector(p['scores'], num_entities)
                for p in query_data['chain_hop2']
                if p['parent'] in chain_hop1_valid
            ]
            if not hop2_vectors:
                predictions.append([])
                neo4j_per_query.append(base_calls + n_chain2)
                ultra_per_query.append(base_calls + n_chain2)
                continue

            chain_result_valid = set(
                np.where(np.maximum.reduce(hop2_vectors) >= t_chain2)[0]
            )
            branch_1p_valid = ProjectIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch_1p']['scores'], t_1p
            )
            predictions.append(list(chain_result_valid & branch_1p_valid))
            total_calls = base_calls + n_chain2
            neo4j_per_query.append(total_calls)
            ultra_per_query.append(total_calls)

        return predictions, neo4j_per_query, ultra_per_query

    @staticmethod
    def apply_thresholds_to_scores_with_intermediate_sizes(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[Tuple[int, int, int, int]]]:
        t_chain1 = thresholds[0] if len(thresholds) > 0 else 0.0
        t_chain2 = thresholds[1] if len(thresholds) > 1 else 0.0
        t_1p     = thresholds[2] if len(thresholds) > 2 else 0.0

        predictions, intermediate_sizes = [], []
        for query_data in cal_scores:
            chain_hop1_valid = ProjectIntersectPipeline._extract_nodes_by_threshold(
                query_data['chain_hop1']['scores'], t_chain1
            )
            s1 = len(chain_hop1_valid)
            if not chain_hop1_valid:
                predictions.append([])
                intermediate_sizes.append((s1, 0, 0, 0))
                continue

            hop2_vectors = [
                ProjectIntersectPipeline._scores_to_dense_vector(p['scores'], num_entities)
                for p in query_data['chain_hop2']
                if p['parent'] in chain_hop1_valid
            ]
            if not hop2_vectors:
                predictions.append([])
                intermediate_sizes.append((s1, 0, 0, 0))
                continue

            chain_result_valid = set(
                np.where(np.maximum.reduce(hop2_vectors) >= t_chain2)[0]
            )
            s2 = len(chain_result_valid)

            branch_1p_valid = ProjectIntersectPipeline._extract_nodes_by_threshold(
                query_data['branch_1p']['scores'], t_1p
            )
            s3 = len(branch_1p_valid)

            intersection = list(chain_result_valid & branch_1p_valid)
            predictions.append(intersection)
            intermediate_sizes.append((s1, s2, s3, len(intersection)))

        return predictions, intermediate_sizes

    @torch.no_grad()
    def predict_with_thresholds(
        self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
        ground_truth_hops: Optional[List[Dict]] = None,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        batch_size = query.shape[0] if query.dim() > 1 else 1
        if query.dim() == 1:
            query = query.unsqueeze(0)

        t_1p = lamhat[2] if len(lamhat) > 2 else 0.0
        counts = [[0, 0] for _ in range(batch_size)]

        # Chain: reuse ThreeHopPipeline's hop1/hop2 processing (indices 0,1,2 of query)
        chain_hop1 = self._process_hop1(
            query, lamhat, graph_data, None, False, keep_scores=False, counts=counts,
        )
        chain_hop2 = self._process_hop2(
            chain_hop1, query, lamhat, graph_data, None, False,
            keep_path_scores=False, counts=counts,
        )

        predictions = []
        query_cpu = query.detach().cpu() if isinstance(query, torch.Tensor) and query.is_cuda else query
        for i in range(batch_size):
            anchor2 = query_cpu[i, 3].item()
            rel3    = query_cpu[i, 4].item()
            src = torch.tensor([[anchor2]], dtype=torch.long, device=self.device)
            rel = torch.tensor([[rel3]], dtype=torch.long, device=self.device)
            scores_1p, _, n1p, u1p = self.unified_predictor.predict(src, rel, graph_data, threshold=t_1p)
            counts[i][0] += n1p
            counts[i][1] += u1p

            nodes_1p = set(self._extract_nodes_from_scores(scores_1p, t_1p))
            chain_nodes = set(chain_hop2["nodes"][i])
            predictions.append(list(chain_nodes & nodes_1p))

        return predictions, [c[0] for c in counts], [c[1] for c in counts]

    @torch.no_grad()
    def _predict(
        self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
        ground_truth_hops: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query_cpu = query.detach().cpu() if isinstance(query, torch.Tensor) and query.is_cuda else query
        batch_size = query.shape[0]
        t_1p = lamhat[2] if len(lamhat) > 2 else 0.0
        # Calibration-time GT filtering for the chain (mirrors ThreeHopPipeline).
        # Without it, at threshold=0 hop1 retains ~all entities and hop2 fires one
        # ULTRA call per hop1 entity, which is catastrophic at calibration scale.
        # ground_truth_hops keys per query: 1 -> chain intermediates, 2 -> chain result.
        use_ground_truth = ground_truth_hops is not None

        chain_hop1 = self._process_hop1(
            query_cpu, lamhat, graph_data, ground_truth_hops, use_ground_truth, keep_scores=True,
        )
        chain_hop2 = self._process_hop2(
            chain_hop1, query_cpu, lamhat, graph_data, ground_truth_hops, use_ground_truth,
            keep_path_scores=True,
        )

        batch_results = []
        for i in range(batch_size):
            anchor2 = query_cpu[i, 3].item()
            rel3    = query_cpu[i, 4].item()
            src = torch.tensor([[anchor2]], dtype=torch.long, device=self.device)
            rel = torch.tensor([[rel3]], dtype=torch.long, device=self.device)
            scores_1p, _, _, _ = self.unified_predictor.predict(src, rel, graph_data, threshold=t_1p)
            nodes_1p = self._extract_nodes_from_scores(scores_1p, t_1p)

            chain_nodes = set(chain_hop2["nodes"][i])
            intersection = list(chain_nodes & set(nodes_1p))

            batch_results.append({
                "chain_hop1": {
                    "nodes": chain_hop1["nodes"][i],
                    "scores": chain_hop1["scores"][i],
                },
                "chain_hop2": {
                    "nodes": chain_hop2["nodes"][i],
                    "scores": chain_hop2["scores"][i],
                },
                "branch_1p": {"nodes": nodes_1p, "scores": scores_1p},
                "intersection_nodes": intersection,
            })

        return batch_results

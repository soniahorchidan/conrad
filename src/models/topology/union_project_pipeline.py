import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base import BasePipeline


class UnionProjectPipeline(BasePipeline):
    """
    Pipeline for up (union-project) queries:
    ((anchor1, rel1), (anchor2, rel2), rel3)

    Query execution:
    1. Predict from anchor1 via rel1 → S1 (filtered by τ1)
    2. Predict from anchor2 via rel2 → S2 (filtered by τ2)
    3. union_nodes = S1 ∪ S2
    4. For each u ∈ union_nodes, predict via rel3 → MAX-aggregate → final (filtered by τ_proj)

    Thresholds: [τ1, τ2, τ_proj]
    """

    @torch.no_grad()
    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        """Query format: [anchor1, rel1, anchor2, rel2, rel3]"""
        return self._handle_confidence_modes(
            query, confidence, graph_data,
            expected_length=5,
            error_msg="UnionProjectPipeline expects queries of length 5: [anchor1, rel1, anchor2, rel2, rel3]",
            debug_thresholds=[0.4, 0.4, 0.4],
        )

    @staticmethod
    def apply_thresholds_to_scores(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> List[List[int]]:
        t1     = thresholds[0] if len(thresholds) > 0 else 0.0
        t2     = thresholds[1] if len(thresholds) > 1 else 0.0
        t_proj = thresholds[2] if len(thresholds) > 2 else 0.0

        predictions = []
        for query_data in cal_scores:
            b1_dense = UnionProjectPipeline._scores_to_dense_vector(
                query_data['branch1']['scores'], num_entities=num_entities
            )
            b2_dense = UnionProjectPipeline._scores_to_dense_vector(
                query_data['branch2']['scores'], num_entities=num_entities
            )
            union_passing = (
                set(np.where(b1_dense >= t1)[0]) | set(np.where(b2_dense >= t2)[0])
            )
            if not union_passing:
                predictions.append([])
                continue

            proj_paths = query_data.get('projection_paths', [])
            valid_proj_vectors = [
                UnionProjectPipeline._scores_to_dense_vector(p['scores'], num_entities=num_entities)
                for p in proj_paths
                if p['parent'] in union_passing
            ]
            if not valid_proj_vectors:
                predictions.append([])
                continue

            final_max = np.maximum.reduce(valid_proj_vectors)
            predictions.append(np.where(final_max >= t_proj)[0].tolist())

        return predictions

    @staticmethod
    def apply_thresholds_to_scores_with_call_counts(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        t1     = thresholds[0] if len(thresholds) > 0 else 0.0
        t2     = thresholds[1] if len(thresholds) > 1 else 0.0
        t_proj = thresholds[2] if len(thresholds) > 2 else 0.0

        predictions, neo4j_per_query, ultra_per_query = [], [], []
        for query_data in cal_scores:
            b1_dense = UnionProjectPipeline._scores_to_dense_vector(
                query_data['branch1']['scores'], num_entities=num_entities
            )
            b2_dense = UnionProjectPipeline._scores_to_dense_vector(
                query_data['branch2']['scores'], num_entities=num_entities
            )
            union_passing = (
                set(np.where(b1_dense >= t1)[0]) | set(np.where(b2_dense >= t2)[0])
            )
            if not union_passing:
                predictions.append([])
                neo4j_per_query.append(2)
                ultra_per_query.append(2)
                continue

            proj_paths = query_data.get('projection_paths', [])
            valid_proj_vectors = [
                UnionProjectPipeline._scores_to_dense_vector(p['scores'], num_entities=num_entities)
                for p in proj_paths
                if p['parent'] in union_passing
            ]
            n_proj = len(union_passing)
            if not valid_proj_vectors:
                predictions.append([])
                neo4j_per_query.append(2 + n_proj)
                ultra_per_query.append(2 + n_proj)
                continue

            final_max = np.maximum.reduce(valid_proj_vectors)
            predictions.append(np.where(final_max >= t_proj)[0].tolist())
            neo4j_per_query.append(2 + n_proj)
            ultra_per_query.append(2 + n_proj)

        return predictions, neo4j_per_query, ultra_per_query

    @staticmethod
    def apply_thresholds_to_scores_with_intermediate_sizes(
        cal_scores: List[Dict[str, Any]], thresholds: List[float],
        true_labels: Optional[List[Dict]] = None,
        num_entities: int = 14541,
    ) -> Tuple[List[List[int]], List[Tuple[int, int, int, int]]]:
        t1     = thresholds[0] if len(thresholds) > 0 else 0.0
        t2     = thresholds[1] if len(thresholds) > 1 else 0.0
        t_proj = thresholds[2] if len(thresholds) > 2 else 0.0

        predictions, intermediate_sizes = [], []
        for query_data in cal_scores:
            b1_dense = UnionProjectPipeline._scores_to_dense_vector(
                query_data['branch1']['scores'], num_entities=num_entities
            )
            b2_dense = UnionProjectPipeline._scores_to_dense_vector(
                query_data['branch2']['scores'], num_entities=num_entities
            )
            b1_set = set(np.where(b1_dense >= t1)[0])
            b2_set = set(np.where(b2_dense >= t2)[0])
            union_passing = b1_set | b2_set
            s1, s2, s3 = len(b1_set), len(b2_set), len(union_passing)
            if not union_passing:
                predictions.append([])
                intermediate_sizes.append((s1, s2, s3, 0))
                continue

            proj_paths = query_data.get('projection_paths', [])
            valid_proj_vectors = [
                UnionProjectPipeline._scores_to_dense_vector(p['scores'], num_entities=num_entities)
                for p in proj_paths
                if p['parent'] in union_passing
            ]
            if not valid_proj_vectors:
                predictions.append([])
                intermediate_sizes.append((s1, s2, s3, 0))
                continue

            final_max = np.maximum.reduce(valid_proj_vectors)
            final_indices = np.where(final_max >= t_proj)[0]
            predictions.append(final_indices.tolist())
            intermediate_sizes.append((s1, s2, s3, len(final_indices)))

        return predictions, intermediate_sizes

    @torch.no_grad()
    def predict_with_thresholds(
        self, query: torch.Tensor, lamhat: List[float], graph_data: Any,
        ground_truth: Optional[List[List[int]]] = None,
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        batch_results, neo4j_per_query, ultra_per_query = self._predict(
            query, lamhat, graph_data, ground_truth
        )
        predictions = [r["final_nodes"] for r in batch_results]
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
            rel3 = query_cpu[i, 4].item()

            t1     = lamhat[0] if len(lamhat) > 0 else 0.0
            t2     = lamhat[1] if len(lamhat) > 1 else 0.0
            t_proj = lamhat[2] if len(lamhat) > 2 else 0.0

            src1 = torch.tensor([[anchor1]], dtype=torch.long, device=self.device)
            r1   = torch.tensor([[rel1]], dtype=torch.long, device=self.device)
            scores1, _, n1, u1 = self.unified_predictor.predict(src1, r1, graph_data, threshold=t1)
            neo4j_per_query[i] += n1
            ultra_per_query[i] += u1

            src2 = torch.tensor([[anchor2]], dtype=torch.long, device=self.device)
            r2   = torch.tensor([[rel2]], dtype=torch.long, device=self.device)
            scores2, _, n2, u2 = self.unified_predictor.predict(src2, r2, graph_data, threshold=t2)
            neo4j_per_query[i] += n2
            ultra_per_query[i] += u2

            nodes1 = set(self._extract_nodes_from_scores(scores1, t1))
            nodes2 = set(self._extract_nodes_from_scores(scores2, t2))
            union_nodes = list(nodes1 | nodes2)

            # Calibration GT filtering (union of GT from both branches).
            # Cap the frontier so the rel3 projection step doesn't fire against
            # ~14541 entities at threshold=0 — mirrors 2ip's _apply_2ip_gt_filtering.
            if ground_truth is not None and i < len(ground_truth):
                gt_data = ground_truth[i]
                if isinstance(gt_data, dict):
                    gt_set = set(gt_data.get(1, [])) | set(gt_data.get(2, []))
                elif isinstance(gt_data, (list, set)):
                    gt_set = set(gt_data)
                else:
                    gt_set = set()
                if gt_set:
                    scores1_np = scores1.squeeze(0).cpu().numpy()
                    scores2_np = scores2.squeeze(0).cpu().numpy()
                    combined = np.maximum(scores1_np, scores2_np)
                    gt_list = [g for g in gt_set if 0 <= g < len(combined)]
                    if gt_list:
                        min_gt = float(combined[gt_list].min())
                        union_nodes = [n for n in union_nodes if n in gt_set or combined[n] >= min_gt]

                    # Bound frontier for tractability (matches 2ip pattern, max_nodes=1000).
                    max_nodes = 1000
                    if len(union_nodes) > max_nodes:
                        gt_in_union = [n for n in union_nodes if n in gt_set]
                        non_gt = [n for n in union_nodes if n not in gt_set]
                        non_gt.sort(key=lambda n: float(combined[n]), reverse=True)
                        budget = max(0, max_nodes - len(gt_in_union))
                        union_nodes = gt_in_union + non_gt[:budget]

            # Final-answer GT used as must-include for projection_paths top-K below.
            proj_gt_set: set = set()
            if ground_truth is not None and i < len(ground_truth):
                gt_data_p = ground_truth[i]
                if isinstance(gt_data_p, dict):
                    proj_gt_set = set(gt_data_p.get(3, [])) | set(gt_data_p.get("final", []))
                elif isinstance(gt_data_p, (list, set)):
                    proj_gt_set = set(gt_data_p)

            # Projection over union
            projection_paths = []
            all_proj_scores = []
            projection_scores_dict = {}

            if union_nodes:
                max_batch = 64
                for b_start in range(0, len(union_nodes), max_batch):
                    b_nodes = union_nodes[b_start:b_start + max_batch]
                    int_t = torch.tensor(b_nodes, dtype=torch.long, device=self.device).unsqueeze(1)
                    r3_t  = torch.tensor([[rel3]] * len(b_nodes), dtype=torch.long, device=self.device)
                    p_scores, _, np_, up_ = self.unified_predictor.predict(
                        int_t, r3_t, graph_data, threshold=t_proj
                    )
                    neo4j_per_query[i] += np_
                    ultra_per_query[i] += up_

                    for j, parent_id in enumerate(b_nodes):
                        s_vec = p_scores[j].detach().cpu()
                        all_proj_scores.append(s_vec)
                        # Top-K compression: ULTRA's unified scores are strictly positive for
                        # every entity (sigmoid + jitter), so torch.nonzero(s_vec) would store
                        # all num_entities entries per path. With up to 1000 parents per query
                        # × 4000 calibration queries, that exhausts host RAM on large graphs
                        # (NELL 75k, YAGO 123k entities). Match the top-K=1000 convention
                        # used by _process_single_hop_scores; pin GT via must-include.
                        k_eff = min(1000, s_vec.numel())
                        top_v, top_i = torch.topk(s_vec, k_eff, largest=True, sorted=False)
                        score_map: Dict[int, float] = {
                            int(idx): float(val)
                            for idx, val in zip(top_i.tolist(), top_v.tolist())
                        }
                        for raw_idx in proj_gt_set:
                            idx_int = int(raw_idx)
                            if 0 <= idx_int < s_vec.size(0) and idx_int not in score_map:
                                score_map[idx_int] = float(s_vec[idx_int].item())
                        sorted_items = sorted(score_map.items())
                        projection_paths.append({
                            'parent': parent_id,
                            'scores': {
                                'indices': np.array([i for i, _ in sorted_items], dtype=np.int64),
                                'values': np.array([v for _, v in sorted_items], dtype=np.float32),
                            },
                        })

            final_nodes = []
            max_p = None
            if all_proj_scores:
                max_p = torch.stack(all_proj_scores).max(dim=0).values
                final_nodes = self._extract_nodes_from_scores(max_p.unsqueeze(0), t_proj)
                # Vectorized: build projection_scores_dict from max-over-paths,
                # replaces an O(num_paths × num_entities) Python loop that ran one
                # tensor .item() per cell (~10–100s per query at calibration).
                max_p_cpu = max_p.detach().cpu()
                nz_mask = max_p_cpu > 0
                if bool(nz_mask.any()):
                    nz_idx = torch.nonzero(nz_mask, as_tuple=False).flatten()
                    projection_scores_dict = dict(zip(
                        nz_idx.tolist(),
                        max_p_cpu[nz_idx].tolist(),
                    ))

            nodes1_all = self._extract_nodes_from_scores(scores1, 0.0)
            nodes2_all = self._extract_nodes_from_scores(scores2, 0.0)

            batch_results.append({
                "branch1": {"nodes": nodes1_all, "scores": scores1},
                "branch2": {"nodes": nodes2_all, "scores": scores2},
                "union_nodes": union_nodes,
                "projection": {
                    "nodes": self._extract_nodes_from_scores(max_p.unsqueeze(0), 0.0)
                              if max_p is not None else [],
                    "scores": projection_scores_dict,
                },
                "projection_paths": projection_paths,
                "final_nodes": final_nodes,
            })

        return batch_results, neo4j_per_query, ultra_per_query

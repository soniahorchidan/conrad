import torch
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from argparse import Namespace
import torch.nn as nn


class NonVector3HopNeural(nn.Module):
    """
    A wrapper model that uses Ultra model for
    to approximate each hop of a 3-hop path.
    Uses non vector CRC calibration procedure to calibrate a threshold on one hop,
    that is applied for each hop of when prediting with confidence.

    Implemented methods:
    - score_hop: run ULTRA for a single-hop (used for calibration)
    - predict_with_threshold: sequentially run ULTRA per hop and apply per-hop thresholding
    - generate_calibrate_samples: create calibration samples from file-based iterator
    """

    def __init__(self, ultra_model, args: Namespace, device: str):
        super().__init__()
        # store references to underlying ULTRA model and runtime config
        self.ultra = ultra_model
        self.args = args
        self.device = device
        self.max_internal_batch = getattr(args, 'max_internal_batch', 64)

    def get_name(self) -> str:
        return "nonvector3hopneural"

    @torch.no_grad()
    def score_hop(self, query: torch.Tensor, graph_data: Any) -> torch.Tensor:
        """
        Use internal ultra model to run predictions.
        Run a single calibration and return all scores.
        
        :param self: self
        :param query: Query format: (entity, rel)
        :type query: torch.Tensor
        :param graph_data: GraphData similar to pytorch geometric graph data
        :type graph_data: Any
        :return: Raw scores of the underlying ultra model inference
        :rtype: Any
        """
        if hasattr(self.ultra, 'eval'):
            self.ultra.eval()

        # Normalize query shape to [B, D]
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # ULTRA expects (head, rel) for a single hop; if query is longer take first two columns
        if query.shape[1] > 2:
            query_for_hop = query[:, :2]
        else:
            query_for_hop = query

        # Move query / graph_data to device
        if query_for_hop.device != torch.device(self.device):
            query_for_hop = query_for_hop.to(self.device)
        graph_data_dev = graph_data.to(self.device) if hasattr(graph_data, 'to') else graph_data

        scores = self.ultra.forward(graph_data_dev, query_for_hop)
        # ULTRA.forward may return (scores, intermediate); ensure tensor
        if isinstance(scores, tuple) or isinstance(scores, list):
            scores = scores[0]
        scores = torch.sigmoid(scores)
        return scores

    def predict(self, query: torch.Tensor, confidence: float, graph_data: Any) -> Optional[List]:
        alpha = round(1 - confidence, 4)
        threshold = self.conformal_prediction.get_threshold(alpha)
        return self.predict_with_threshold(query, threshold, graph_data)

    @torch.no_grad()
    def predict_with_threshold(self, query: torch.Tensor, lamhat: float, graph_data: Any) -> List[List[int]]:
        """
        Use internal ultra model to run predictions.
        If query has more than one relation the ultra model is called iteratively using
        using the results of the previous hop as the source nodes for the next hop.
        The confidence threshold is applied on the scores of the previous hop,
        resticting the nodes used a sources nodes for the next hop.
        Usually this method is used to predict 3 hop queries.
        
        :param self: self
        :param query: Query format: (entity, (rel1, rel2, rel3))
        :type query: torch.Tensor
        :param confidence: Confidence threshold applied to model scores
        :type confidence: float
        :param graph_data: GraphData similar to pytorch geometric graph data
        :type graph_data: Any
        :return: Predicted nodes given query and confidence threshold
        :rtype: Any
        """
        if hasattr(self.ultra, 'eval'):
            self.ultra.eval()

        if query.dim() == 1:
            query = query.unsqueeze(0)

        batch_size = query.shape[0]
        num_hops = query.shape[1] - 1
        graph_data_dev = graph_data.to(self.device) if hasattr(graph_data, 'to') else graph_data

        all_predictions: List[List[int]] = []

        for b in range(batch_size):
            row = query[b]
            head = int(row[0].item())
            current_entities = [head]
            final_predictions: List[int] = []

            for hop_idx in range(num_hops):
                rel_id = int(row[hop_idx + 1].item())

                if not current_entities:
                    break

                # Prepare batched queries for this hop
                src_count = len(current_entities)
                # chunk if too large
                scores_list = []
                for start in range(0, src_count, self.max_internal_batch):
                    chunk = current_entities[start : start + self.max_internal_batch]
                    hop_queries = torch.tensor([[s, rel_id] for s in chunk], dtype=torch.long, device=self.device)
                    hop_scores = self.ultra.forward(graph_data_dev, hop_queries)
                    if isinstance(hop_scores, tuple) or isinstance(hop_scores, list):
                        hop_scores = hop_scores[0]
                    hop_scores = torch.sigmoid(hop_scores)

                    # Ensure shape [N_chunk, V]
                    if hop_scores.dim() == 1:
                        hop_scores = hop_scores.unsqueeze(0)
                    scores_list.append(hop_scores)

                # Concatenate chunked source-score tensors -> [src_count, V]
                hop_scores_full = torch.cat(scores_list, dim=0) if len(scores_list) > 1 else scores_list[0]

                if hop_idx == num_hops - 1:
                    # Last hop: collect predicted target nodes from all source entities
                    preds = set()
                    for src_row in hop_scores_full:
                        nodes = self._extract_nodes_from_scores(src_row, lamhat)
                        preds.update(nodes)
                    final_predictions = sorted(preds)
                else:
                    # Intermediate hop: aggregate across sources and filter for next hop
                    aggregated = hop_scores_full.max(dim=0)[0]  # [V]
                    next_entities = self._extract_nodes_from_scores(aggregated, lamhat)
                    current_entities = next_entities
                    # If no nodes remain, stop early
                    if not current_entities:
                        break

            all_predictions.append(final_predictions)

        return all_predictions


    def _extract_nodes_from_scores(self, scores: torch.Tensor, threshold: float) -> List[int]:
        """Extract nodes from scores using simple thresholding.

        Kept as a small helper to match pipeline behavior and to keep the
        non-vector implementation consistent with the rest of the codebase.
        """
        if scores is None:
            return []
        # Ensure 1-D tensor
        if scores.dim() > 1:
            scores = scores.squeeze()
        nonzero_indices = torch.nonzero(scores >= threshold, as_tuple=True)[0]
        if nonzero_indices.numel() > 0:
            return nonzero_indices.tolist()
        return []


    @torch.no_grad()
    def generateCalibrateSamples(self, save_path: str, db_controller: Any,
                                   calib_iterator: Optional[Any] = None,
                                   calibration_data_path: Optional[str] = None,
                                   return_queries: bool = False) -> Tuple[List[Any], List[Any], Optional[torch.Tensor]]:
        """
        Produce calibration data (path-aware format) for the non-vector model.
        Returns: (cal_scores, true_labels) or (cal_scores, true_labels, queries) when requested.
        Each cal_score entry is a dict with 'hop1': {'scores': Tensor} (CPU tensor).
        """
        # lazy imports to avoid import cycles
        from models.topology.calibration_sampler import FileBasedDataIterator
        from utils import get_graph
        from tqdm import tqdm

        if calib_iterator is None:
            if calibration_data_path is None:
                raise ValueError("Either calib_iterator or calibration_data_path must be provided")
            # Ensure we have a graph_data instance for ULTRA
            graph_data = get_graph(db_controller, self.device, augment_inverse_edges=True, relation_graph=True)
            calib_iterator = FileBasedDataIterator(calibration_data_path=calibration_data_path,
                                                   batch_size=8, device=self.device, hop_level=1, graph_data=graph_data)

        all_scores = []
        all_labels = []
        all_queries = []

        for batch in tqdm(calib_iterator):
            # batch: (queries, answers_by_hop, graph_data, interm_hop_data)
            queries_batch, answers_batch, graph_data_batch, _ = batch

            if queries_batch.dim() == 1:
                queries_batch = queries_batch.unsqueeze(0)

            scores_batch = self.score_hop(queries_batch, graph_data_batch)  # [B, V]
            # print("scores", scores_batch)

            for i in range(scores_batch.shape[0]):
                scores_row = scores_batch[i].cpu()
                # Path-aware format: only hop1 is relevant for non-vector CRC
                all_scores.append(scores_row)

                # answers_batch[i] is a dict per hop -> extract hop1 answers
                hop1_answers = []
                ans_info = answers_batch[i]
                if isinstance(ans_info, dict):
                    hop1_answers = ans_info.get(1, [])
                elif isinstance(ans_info, list):
                    # fallback: assume this is already hop1 answers
                    hop1_answers = ans_info
                all_labels.append(hop1_answers)

                if return_queries:
                    all_queries.append(queries_batch[i].cpu())

        if return_queries:
            return all_scores, all_labels, torch.stack(all_queries) if all_queries else None
        return all_scores, all_labels, None
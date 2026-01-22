import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..common import ModelUtils
from .models import UltraModel, RelationProjection
from .dataloader import DataIterator
from utils import save_component
from argparse import Namespace
from tqdm import tqdm
import os
import logging

MODEL_CONFIG = {
    "QueryNBFNet": {
        "aggregate_func": "sum",
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "input_dim": 64,
        "layer_norm": True,
        "message_func": "distmult",
        "short_cut": True,
    },
    "RelNBFNet": {
        "aggregate_func": "sum",
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "input_dim": 64,
        "layer_norm": True,
        "message_func": "distmult",
        "short_cut": True,
    },
}


class ULTRA(nn.Module, ModelUtils):

    def __init__(self, args: Namespace, num_relations: int, device: str):
        super(ULTRA, self).__init__()
        self.args = Namespace()
        for key, value in vars(args).items():
            if key != "db_controller":
                setattr(self.args, key, value)
        self.device = device

        self.model = RelationProjection(
            UltraModel(
                rel_model_cfg=MODEL_CONFIG["RelNBFNet"],
                entity_model_cfg=MODEL_CONFIG["QueryNBFNet"],
            ),
            args.msp_threshold,
        )
        self.non_overlap_set = set()

    def predict(
        self, head: torch.Tensor, relation: torch.Tensor, confidence: float, graph_data
    ):
        self.eval()
        query = torch.cat([head, relation], dim=-1)
        prob = self.forward(graph_data, query)
        
        error_intervals = []
        if confidence < 0 or confidence > 1:
            logging.info("The confidence value needs to be between 0 and 1.")
            raise ValueError("The confidence value needs to be between 0 and 1.")
        if confidence == 1.0:
            logging.info("Confidence is set to 1.0. Returning no results.")
            raise ValueError(
                "Raising error because confidence is set to 1.0. Returning no results."
            )
        elif confidence == 0.0:
            acceptance_threshold = self.args.acceptance_threshold
            logging.info(f"ULTRA prediction threshold set to {acceptance_threshold}. ")
            logging.info(
                f"Confidence is set to 0. Using normal inference without conformal prediction. "
            )
            batch_size = prob.shape[0]
            all_results = []
            for i in range(batch_size):
                thresolded_results = torch.nonzero(
                    prob[i] > acceptance_threshold, as_tuple=True
                )[0].tolist()
                # NOTE: ULTRA can estimate the cardinality of the queries via score thresholding.
                # We want to let the model complete the graph, so it should return more results
                # than the true cardinality of the query. Therefore, we convey to return 50% more.
                # Also, we ensure we always return at least 3 results.
                # Context: ISSUE-834
                cardinality = len(thresolded_results)
                num_returned = max(3, int(1.5 * cardinality))
                _, results = torch.topk(prob, num_returned)
                all_results.append(results[0].tolist())
        else:
            raise ValueError(
                f"ULTRA.predict() does not support conformal prediction. "
                f"Confidence level {confidence} is not supported. "
                f"Use confidence=0.0 for threshold-based inference, or use pipeline models "
                f"(ThreeHopPipeline, TwoUnionPipeline, TwoIntersectProjectPipeline) for conformal prediction."
            )

        return all_results, error_intervals

    def load_weights(self, load_path: str):
        """Load ULTRA model weights from the specified path."""
        logging.info("Loading ULTRA model weights...")
        if load_path.endswith(".pth"):
            # If the path is a file, we assume it's the main model weights
            main_path = load_path
        else:
            main_path = os.path.join(load_path, "ultra", "main.pth")
        
        # Validate file exists and has content
        if not os.path.exists(main_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at: {main_path}\n"
                f"Load path was: {load_path}"
            )
        
        file_size = os.path.getsize(main_path)
        logging.info(f"Loading checkpoint from: {main_path} (size: {file_size} bytes)")
        
        if file_size == 0:
            raise ValueError(
                f"Checkpoint file is empty (0 bytes) at: {main_path}\n"
                f"The file may be corrupted or was not saved properly."
            )
        
        try:
            self.load_state_dict(torch.load(main_path, map_location=self.device))
            logging.info("Successfully loaded ULTRA model weights")
        except EOFError as e:
            raise EOFError(
                f"Failed to load checkpoint - file appears corrupted or truncated.\n"
                f"File: {main_path}\n"
                f"Size: {file_size} bytes\n"
                f"Original error: {str(e)}\n"
                f"You may need to re-train or re-download the model checkpoint."
            ) from e

    def forward(self, graph_data, query, return_intermediate=False):
        h_prob = F.one_hot(query[:, 0], graph_data.num_nodes).float()
        num_hops = query.shape[1] - 1
        
        if return_intermediate:
            intermediate_scores = []
        
        for hop in range(num_hops):
            h_prob = self.model(graph_data, h_prob, query[:, hop + 1])
            if return_intermediate:
                intermediate_scores.append(h_prob.clone())
        
        # h_prob = F.sigmoid(h_prob)
        if return_intermediate:
            return h_prob, intermediate_scores
        return h_prob

    def generateCalibrateSamples(
        self, save_path, db_controller, calib_iterator=None, return_queries=False, return_intermediate_scores=False
    ):

        if return_intermediate_scores:
            logging.info("Returning intermediate scores for ULTRA...")
        else:
            logging.info("Not returning intermediate scores for ULTRA...")

        self.calib_iterator = DataIterator(
            self.args,
            save_path,
            db_controller,
            self.device,
            "calib",
            self.non_overlap_set,
        )

        self.eval()
        scores, queries, answers = [], [], []
        intermediate_scores_list = [] if return_intermediate_scores else None

        with torch.no_grad():
            logging.info(
                "Calibration/Validation data does not exist. Start to prepare it"
            )
            calib_list = list(self.calib_iterator)

            for data in tqdm(calib_list):
                query, ans, graph_data = data
                query = query.to(self.device)
                graph_data = graph_data.to(self.device)

                if return_intermediate_scores:
                    prob, intermediate_scores = self.forward(graph_data, query, return_intermediate=True)
                    intermediate_scores_list.append(intermediate_scores)
                else:
                    prob = self.forward(graph_data, query)

                scores.append(prob)
                if return_queries:
                    queries.append(query)  # Each of shape [batch_sz, query_length], but query_length varies
                answers.extend(ans)

            scores = torch.cat(scores, dim=0).cpu().float()

            if return_intermediate_scores:
                # intermediate_scores_list is a list of lists: [batch][hop]
                # Need to reorganize: [hop][batch] then concatenate
                num_hops = len(intermediate_scores_list[0]) if intermediate_scores_list else 0
                intermediate_by_hop = []
                for hop_idx in range(num_hops):
                    hop_scores = [batch_scores[hop_idx] for batch_scores in intermediate_scores_list]
                    intermediate_by_hop.append(torch.cat(hop_scores, dim=0).cpu().float())
                intermediate_scores_list = intermediate_by_hop

            if return_queries:
                # Pad query tensors along the query length dimension (D) so they can be stacked
                max_D = max(q.shape[1] for q in queries)
                queries_padded = [F.pad(q, (0, max_D - q.shape[1]), value=-1) for q in queries]
                queries = torch.cat(queries_padded, dim=0).cpu()  # shape: [num_batches * batch_sz, max_query_length]

        if return_intermediate_scores:
            if return_queries:
                return scores, answers, queries, intermediate_scores_list
            return scores, answers, intermediate_scores_list
        elif return_queries:
            return scores, answers, queries
        return scores, answers
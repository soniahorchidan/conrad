import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from ..common import ModelUtils
from .models import UltraModel, RelationProjection
from .dataloader import DataIterator
from utils import save_component, AverageMeter, AverageMeterDict, calc_metrics_batch
from argparse import Namespace
from tqdm import tqdm, trange
import os
import time
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

    @staticmethod
    def preprocess(general_args: Namespace, args: Namespace):
        logging.info("Skipping preprocess for ULTRA.")

    @staticmethod
    def postprocess(model, general_args: Namespace, args: Namespace):
        if os.path.exists(os.path.join(general_args.save_path, "ultra", "main.pth")):
            logging.info(
                "Skipping save for ULTRA because the model is saved in the training loop."
            )
        else:
            logging.info("Saving model weights...")
            save_component(general_args.save_path, "ultra", "main", model.state_dict())
        # TODO: save embeddings to vector database for fast inference
        logging.info("ULTRA does not write embeddings to vector database for now.")
        logging.info("Postprocess for ULTRA is done.")

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
                bounds = self.conformal_prediction.predict_with_error_tagging(
                    None, torch.tensor(prob[i]).unsqueeze(0).T, len(results[0].tolist())
                )
                error_intervals.extend(bounds)
        else:
            logging.info("Using conformal prediction.")
            results = self.conformal_prediction.predict(None, prob, confidence, query)
            all_results = [r.tolist() for r in results]
            logging.info("Returning results from ULTRA.")
            logging.info(f"Results: {all_results}")

        return all_results, error_intervals

    def load_all_components(self, load_path, ignore_components: list = []):
        if load_path is not None:
            # load model weights
            logging.info("Loading ULTRA model weights...")
            if load_path.endswith(".pth"):
                # If the path is a file, we assume it's the main model weights
                main_path = load_path
            else:
                main_path = os.path.join(load_path, "ultra", "main.pth")
            self.load_state_dict(torch.load(main_path, map_location=self.device))
        else:
            logging.info("No load path provided, skipping loading of model weights")
        # if (
        #     self.conformal_prediction is not None
        #     and "conformal_prediction" not in ignore_components
        # ):
        #     logging.info("Loading conformal prediction components...")
        #     self.conformal_prediction.load_all_components(load_path)

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

    def trainModel(self, general_args, args):
        if args.max_steps == 0:
            logging.info("Max steps is 0. Skipping training.")
            return

        optimizer = torch.optim.AdamW(self.parameters(), lr=args.learning_rate)
        self.val_iterator = DataIterator(
            args,
            general_args.save_path,
            general_args.db_controller,
            self.device,
            mode="val",
        )
        self.non_overlap_set = (
            self.val_iterator.get_non_overlap_dataset() | self.non_overlap_set
        )

        train_iterator = DataIterator(
            args,
            general_args.save_path,
            general_args.db_controller,
            self.device,
            mode="train",
            non_overlap_dataset=self.non_overlap_set,
        )

        self.best_mrr = 0
        self.metadata = {}
        self.early_stopping_counter = 0

        loss_meter = AverageMeter()
        for epoch in range(0, args.max_steps):
            self.train()

            data = next(train_iterator)
            query, target, graph_data = data
            query = query.to(self.device)
            target = target.to(self.device)
            graph_data = graph_data.to(self.device)

            pred = self.forward(graph_data, query)

            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

            is_positive = target > 0.5
            is_negative = target <= 0.5
            num_positive = is_positive.sum(dim=-1)
            num_negative = is_negative.sum(dim=-1)

            neg_weight = torch.zeros_like(pred)
            neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(
                num_positive
            )

            neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(
                num_negative
            )
            loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            loss = loss.mean()
            loss_meter.update(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % args.log_steps_freq == 0:
                logging.info("Epoch {} | Loss: {:.4f}".format(epoch, loss_meter.avg))
                loss_meter.reset()

            if epoch % args.valid_steps_freq == 0:
                val_mrr, metadata = self.validate(general_args, args)
                logging.info(f"Validation MRR at epoch {epoch}: {val_mrr}.")
                if val_mrr < self.best_mrr:
                    self.early_stopping_counter += 1
                    logging.info(
                        f"MRR increased. Early stopping counter is now: {self.early_stopping_counter}."
                    )
                else:
                    logging.info(
                        "Best mrr achieved. The early stopping counter is reset to 0."
                    )
                    self.best_mrr = val_mrr
                    self.metadata = metadata
                    self.early_stopping_counter = 0
                    logging.info("Saving model weights...")
                    save_component(
                        general_args.save_path, "ultra", "main", self.state_dict()
                    )

            if self.early_stopping_counter > args.early_stopping_threshold:
                logging.info("Early stopping...")
                break
        return self.metadata

    def validate(self, general_args, args):
        logging.info("Start validating...")
        if not hasattr(self, "val_iterator"):
            logging.info("Validation iterator does not exist. Creating a new one.")
            self.val_iterator = DataIterator(
                args,
                general_args.save_path,
                general_args.db_controller,
                self.device,
                mode="val",
            )

        val_meter = AverageMeterDict()
        self.eval()
        with torch.no_grad():
            for data in self.val_iterator:
                query, answers, graph_data = data
                query = query.to(self.device)
                graph_data = graph_data.to(self.device)

                prob = self.forward(graph_data, query)
                
                metrics = calc_metrics_batch(prob, answers)
                val_meter.update(metrics, len(answers))

            logging.info(f"Validation HITS@1: {val_meter.avg['HITS@1']}")
            logging.info(f"Validation HITS@3: {val_meter.avg['HITS@3']}")
            logging.info(f"Validation HITS@10: {val_meter.avg['HITS@10']}")
            logging.info(f"Validation MRR: {val_meter.avg['MRR']}")

            metadata = {
                "Validation HITS@1": val_meter.avg["HITS@1"],
                "Validation HITS@3": val_meter.avg["HITS@3"],
                "Validation HITS@10": val_meter.avg["HITS@10"],
                "Validation MRR": val_meter.avg["MRR"],
            }

            self.val_iterator.reset()
        self.entity_embedding_lazy = None
        return val_meter.avg["MRR"], metadata



    def generateCalibrateSamples(
        self, save_path, db_controller, vector_db_controller, calib_iterator=None, return_queries=False, return_intermediate_scores=False
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
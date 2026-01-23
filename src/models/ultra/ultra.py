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
        self.use_multi_gpu = getattr(args, 'use_multi_gpu', True)  # Default to True
        self.num_gpus = 0
        self._is_parallel = False

        self.model = RelationProjection(
            UltraModel(
                rel_model_cfg=MODEL_CONFIG["RelNBFNet"],
                entity_model_cfg=MODEL_CONFIG["QueryNBFNet"],
            ),
            args.msp_threshold,
        )
        self.non_overlap_set = set()

    def setup_multi_gpu(self):
        """Setup multi-GPU support if available and enabled."""
        if not self.use_multi_gpu:
            self.num_gpus = 1 if torch.cuda.is_available() and self.device.startswith('cuda') else 0
            logging.info(f"Multi-GPU disabled. Using {self.num_gpus} GPU(s).")
            return False
        
        if not torch.cuda.is_available():
            self.num_gpus = 0
            logging.info("CUDA not available. Using CPU.")
            return False
    
        self.num_gpus = torch.cuda.device_count()
        
        if self.num_gpus > 1:
            logging.info(f"Setting up multi-GPU support for ULTRA: {self.num_gpus} GPUs available")
            # Wrap the model with DataParallel
            self.model = torch.nn.DataParallel(self.model)
            self._is_parallel = True
            logging.info(f"ULTRA model wrapped with DataParallel. Using {self.num_gpus} GPUs for inference.")
            return True
        else:
            logging.info(f"Only 1 GPU available. Using single GPU for ULTRA inference.")
            self.num_gpus = 1
            return False

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
        """Load ULTRA model weights from the specified path.
        
        Supports both custom checkpoints and official ULTRA/UltraQuery checkpoints.
        Handles different checkpoint formats:
        - Direct state dict (.pth file with state_dict)
        - Checkpoint dict with 'model' or 'state_dict' keys
        - DataParallel wrapped checkpoints (strips 'module.' prefix)
        
        Examples:
        - Official UltraQuery checkpoint: load_path="/path/to/ultraquery/" (looks for ultraquery.pth in that directory)
          (Set msp_threshold=0.0 in args for UltraQuery checkpoints)
        - Official ULTRA checkpoint: load_path="/path/to/checkpoint_dir/" (looks for .pth files in that directory)
          (Set msp_threshold=0.8 or higher for vanilla ULTRA checkpoints)
        - Custom checkpoint: load_path="/path/to/checkpoint_dir" (looks for ultra/main.pth)
        """
        logging.info("Loading ULTRA model weights...")
        if load_path.endswith(".pth"):
            # If the path is a file, we assume it's the main model weights (backward compatibility)
            main_path = load_path
        elif os.path.exists(os.path.join(load_path, "ultra", "main.pth")):
            # Standard checkpoint structure: ultra/main.pth
            main_path = os.path.join(load_path, "ultra", "main.pth")
        else:
            # Look for .pth files in the directory (e.g., ultraquery.pth, ultra_4g.pth)
            pth_files = [f for f in os.listdir(load_path) if f.endswith(".pth")]
            if not pth_files:
                raise FileNotFoundError(
                    f"No checkpoint file found in {load_path}.\n"
                    f"Expected either: ultra/main.pth or a .pth file (e.g., ultraquery.pth)"
                )
            if len(pth_files) > 1:
                logging.warning(f"Multiple .pth files found in {load_path}: {pth_files}. Using {pth_files[0]}")
            main_path = os.path.join(load_path, pth_files[0])
            logging.info(f"Found checkpoint file: {main_path}")
        
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
            checkpoint = torch.load(main_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Check if it's a checkpoint dict with nested state_dict
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    logging.info("Found checkpoint with 'model' key, extracting state dict")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    logging.info("Found checkpoint with 'state_dict' key, extracting state dict")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logging.info("Found checkpoint with 'model_state_dict' key, extracting state dict")
                else:
                    # Assume it's already a state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove DataParallel wrapper if present (strips 'module.' prefix)
            if any(key.startswith('module.') for key in state_dict.keys()):
                logging.info("Detected DataParallel wrapper, removing 'module.' prefix")
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('module.', '', 1)  # Remove first occurrence only
                    new_state_dict[new_key] = value
                state_dict = new_state_dict
            
            # Try to load the state dict
            try:
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logging.warning(f"Missing keys when loading checkpoint: {missing_keys[:5]}..." 
                                  if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}..." 
                                  if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
                logging.info("Successfully loaded ULTRA model weights")
            except RuntimeError as e:
                # If strict loading fails, try to match keys more flexibly
                logging.warning(f"Strict loading failed: {str(e)}")
                logging.info("Attempting flexible key matching...")
                
                # Get model state dict keys
                model_keys = set(self.state_dict().keys())
                checkpoint_keys = set(state_dict.keys())
                
                # Try to find matching keys (handle common prefix differences)
                matched_state_dict = {}
                for ckpt_key in checkpoint_keys:
                    # Try exact match first
                    if ckpt_key in model_keys:
                        matched_state_dict[ckpt_key] = state_dict[ckpt_key]
                    else:
                        # Try removing common prefixes
                        for prefix in ['model.', 'ultra.', '']:
                            stripped_key = ckpt_key.replace(prefix, '', 1) if prefix else ckpt_key
                            if stripped_key in model_keys:
                                matched_state_dict[stripped_key] = state_dict[ckpt_key]
                                logging.debug(f"Mapped checkpoint key '{ckpt_key}' -> '{stripped_key}'")
                                break
                
                if matched_state_dict:
                    self.load_state_dict(matched_state_dict, strict=False)
                    logging.info(f"Successfully loaded {len(matched_state_dict)}/{len(checkpoint_keys)} weights with flexible matching")
                else:
                    raise RuntimeError(
                        f"Could not match any checkpoint keys to model keys.\n"
                        f"Model keys (first 10): {list(model_keys)[:10]}\n"
                        f"Checkpoint keys (first 10): {list(checkpoint_keys)[:10]}\n"
                        f"Please check that the checkpoint matches the model architecture."
                    )
                    
        except EOFError as e:
            raise EOFError(
                f"Failed to load checkpoint - file appears corrupted or truncated.\n"
                f"File: {main_path}\n"
                f"Size: {file_size} bytes\n"
                f"Original error: {str(e)}\n"
                f"You may need to re-train or re-download the model checkpoint."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {main_path}.\n"
                f"Error: {str(e)}\n"
                f"Please ensure the checkpoint is compatible with this model architecture."
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
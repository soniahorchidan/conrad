import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


def get_x(emb, x):
    return x


class TwoIntersectProjectPipelineCalibProbDatasetVal(Dataset):
    """
    Custom dataloader for TwoIntersectProjectPipeline vector conformal risk control.
    
    This dataloader is designed to handle the specific data structure
    produced by TwoIntersectProjectPipeline.generateCalibrateSamples():
    - scores: tensor of shape [num_queries, 3, num_entities] 
      where the 3 components are: hop1 (branch1), hop2 (branch2), hop3 (projection)
    - answers: list of lists, where each inner list contains answer entities for that query
    - queries: list of query tensors
    
    Similar to ThreeHopPipeline but with different semantics:
    - hop1 and hop2 are independent branches (not sequential)
    - hop3 is the projection from the intersection of hop1 and hop2
    """
    
    def __init__(self, data, db_controller, vectordb_controller):
        self.data = data
        self.len = len(data[0])  # data[0] is scores, data[1] is answers, data[2] is queries
        self.db_controller = db_controller
        self.vectordb_controller = vectordb_controller

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        scores, answers, queries = self.data
        x = get_x(None, scores[idx])  # scores[idx] has shape [3, num_entities]
        answer = answers[idx]  # answers[idx] could be a list or dict depending on data source
        query = queries[idx]   # queries[idx] is the query tensor for this query

        # Handle different answer formats
        if isinstance(answer, dict):
            # FileBasedDataIterator format: answer is a dict with hop levels as keys
            # Keep all hop answers for vector optimization
            answer_dict = {}
            for hop_level, hop_answer in answer.items():
                # Convert from set to list if needed
                if isinstance(hop_answer, set):
                    hop_answer = list(hop_answer)
                answer_dict[hop_level] = hop_answer
            answer_tensor = answer_dict
        else:
            # Standard format: answer is already a list
            answer_list = answer
            # Ensure answer_list is a list (not a set or other type)
            if isinstance(answer_list, set):
                answer_list = list(answer_list)
            elif not isinstance(answer_list, list):
                answer_list = []

            # Convert answer to tensor - this should be a 1D tensor of answer entity IDs
            try:
                answer_tensor = torch.tensor(answer_list)
            except Exception as e:
                print(f"Error converting answer to tensor: {e}, answer: {answer_list}")
                answer_tensor = torch.tensor([])

        return x, answer_tensor, query

    def collate_fn(self, data):
        # Keep batch_size=1 behavior (matches threehoppipeline vector CRC usage).
        x, y, q = data[0]
        return x, y, q




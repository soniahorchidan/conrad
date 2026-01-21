import torch
from torch.utils.data import Dataset


def get_x(emb, x):
    # For TwoUnionPipeline we represent one sample as a dense tensor of shape [2, num_entities]:
    # [branch1_scores, branch2_scores]
    return x


def _scores_to_dense(scores, num_entities: int) -> torch.Tensor:
    """
    Convert scores to a dense 1D tensor of length num_entities.

    Supports:
    - dense torch.Tensor of shape [1, E] or [E]
    - sparse dict format: {'indices': Tensor[K], 'values': Tensor[K], 'shape': int}
    
    Args:
        scores: Scores in sparse dict or dense tensor format
        num_entities: Number of entities (required)
    """
    
    if isinstance(scores, dict) and "indices" in scores and "values" in scores:
        indices = scores["indices"]
        values = scores["values"]

        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        else:
            indices = indices.to(dtype=torch.long)

        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32)
        else:
            values = values.to(dtype=torch.float32)

        dense = torch.zeros(num_entities, dtype=torch.float32)
        if indices.numel() > 0:
            dense[indices] = values
        return dense

    if isinstance(scores, torch.Tensor):
        s = scores.squeeze()
        s = s.to(dtype=torch.float32)
        if s.numel() < num_entities:
            out = torch.zeros(num_entities, dtype=torch.float32)
            out[: s.numel()] = s
            return out
        return s[:num_entities]

    raise TypeError(f"Unsupported score type: {type(scores)}")


class TwoUnionPipelineCalibProbDatasetVal(Dataset):
    """
    Validation dataset for vector CRC on 2u queries.

    Expects (cal_scores, answers, queries) where each cal_scores[i] is:
      {
        'branch1': {'nodes': [...], 'scores': <tensor or sparse dict>},
        'branch2': {'nodes': [...], 'scores': <tensor or sparse dict>}
      }
    and answers[i] is a list/set of GT entity ids (union output).
    """

    def __init__(self, data, db_controller, num_entities: int):
        self.data = data
        self.len = len(data[0])
        self.db_controller = db_controller
        # Store num_entities for use in __getitem__
        self.num_entities = num_entities

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        cal_scores, answers, queries = self.data
        qd = cal_scores[idx]

        b1 = _scores_to_dense(qd["branch1"]["scores"], num_entities=self.num_entities)
        b2 = _scores_to_dense(qd["branch2"]["scores"], num_entities=self.num_entities)
        x = get_x(None, torch.stack([b1, b2], dim=0))

        answer = answers[idx]
        if isinstance(answer, set):
            answer = list(answer)
        if not isinstance(answer, list):
            answer = []
        answer_tensor = torch.tensor(answer, dtype=torch.long)

        query = queries[idx]
        return x, answer_tensor, query

    def collate_fn(self, data):
        # Keep batch_size=1 behavior (matches threehoppipeline vector CRC usage).
        x, y, q = data[0]
        return x, y, q



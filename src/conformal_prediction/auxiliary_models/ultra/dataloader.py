import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


def get_x(emb, x):
    return x


class CalibProbDataset(Dataset):
    def __init__(self, data, num_negative_samples, db_controller):
        self.data = data
        self.len = len(data[0])
        self.num_negative_samples = num_negative_samples
        self.db_controller = db_controller

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        scores, answer = (self.data[i][idx] for i in range(len(self.data)))
        x = get_x(None, scores)
        return x, answer

    def collate_fn(self, data):
        x = torch.cat([_[0] for _ in data], dim=0)
        y = torch.cat([_[1] for _ in data], dim=0)
        return x, y


class CalibProbDatasetVal(CalibProbDataset):
    def __init__(self, data, db_controller):
        super(CalibProbDatasetVal, self).__init__(
            data, 0, db_controller
        )

    def __getitem__(self, idx):
        scores, answer, query = (self.data[i][idx] for i in range(len(self.data)))
        x = get_x(None, scores)
        return x, torch.tensor(answer), query

    def collate_fn(self, data):
        data = data[0]
        return data[0], data[1], data[2]

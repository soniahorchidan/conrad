import torch.nn as nn
import torch.nn.functional as F

class CalibProbModel(nn.Module):
    def __init__(self):
        super(CalibProbModel, self).__init__()

    def forward(self, x):
        # ensure scores lie inside [0, 1]
        return F.sigmoid(x)

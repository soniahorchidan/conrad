import torch
import logging
import numpy as np


def shape_input(inputs, graph_data, device):

    query = torch.from_numpy(inputs["data"]).to(device)
    confidence = (
        inputs["confidence"]
        if isinstance(inputs["confidence"], float)
        else float(inputs["confidence"])
    )

    head = query[:, 0].unsqueeze(1)
    relation = query[:, 1:]

    params = {
        "head": head,
        "relation": relation,
        "confidence": confidence,
        "graph_data": graph_data,
    }

    return params

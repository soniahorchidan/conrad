import logging
import numpy as np
import torch
from . import ConformalRiskControl

# Dataset statistics: number of entities and relations for each dataset
# Relations count includes inverse relations (e.g., fb15k-237 has 237 relations, 474 with inverses)
DATASET_STATISTICS = {
    "fb15k-237": {
        "num_entities": 14541,
        "num_relations": 237,  # Original relations (474 with inverse edges)
    },
    "nell-955": {
        "num_entities": 75494,  # From terminal output: "75494 nodes"
        "num_relations": 955,   # Original relations (1910 with inverse edges if added)
    },
}


def get_dataset_statistics(dataset: str):
    """
    Get dataset statistics (num_entities, num_relations) for a given dataset.
    
    Args:
        dataset: Dataset name (e.g., "fb15k-237", "nell-955")
    
    Returns:
        dict with "num_entities" and "num_relations" keys
    
    Raises:
        ValueError: If dataset is not supported
    """
    if dataset not in DATASET_STATISTICS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Supported datasets: {list(DATASET_STATISTICS.keys())}"
        )
    return DATASET_STATISTICS[dataset]


class ConformalRiskControlValidator(object):
    def __init__(self, crc: ConformalRiskControl):
        self.crc = crc

    def __call__(
        self,
        calib_iterator,
        entity_embedding,
        confidence,
        save_path,
        db_controller,
        vectordb_controller,
    ):

        val_data = self.crc.generateCalibrateSamples_fn(
            save_path, db_controller, vectordb_controller, None, True
        )

        val_data_size = len(val_data[0])
        x, answers = [], []
        predictions = []
        queries = []
        for i in range(val_data_size):
            data_single = [val_data[j][i] for j in range(len(val_data))]
            x_single = data_single[0]
            ans = data_single[1]
            query = data_single[2]
            if len(x_single) == 1:
                x_single = x_single[0]
            x.append(x_single)
            answers.append(ans)
            queries.append(query)

        predictions = self.crc.predict(entity_embedding, x, confidence, queries)
        avg_pred_sz = np.mean([len(pred) for pred in predictions])
        fn = false_negative_rate(predictions, answers)

        logging.info(f"Confidence: {confidence:.6f}")
        logging.info(f"Validation False Negative Rate: {fn:.6f}")
        logging.info(f"Average prediction set size: {avg_pred_sz:.2f}")
        return fn, confidence


def false_negative_rate(preds, y):
    ovrlp = []
    for i in range(len(preds)):
        gt_labels = set(int(x) for x in y[i])
        pred_labels = set(int(x) for x in preds[i])
        intersection = len(pred_labels & gt_labels)

        if len(gt_labels) > 0:
            overlap_ratio = 1 - intersection / len(gt_labels)
        else:
            overlap_ratio = 0  # avoid division by zero
        ovrlp.append(overlap_ratio)
    return np.mean(ovrlp)
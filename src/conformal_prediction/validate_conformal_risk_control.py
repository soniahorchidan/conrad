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
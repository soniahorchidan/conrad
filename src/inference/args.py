import argparse


def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Log Model Script")
    parser.add_argument("--device", type=str)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--log_path", type=str, default="./artifacts/inference_logs",
                       help="Path for log files (default: ./artifacts/inference_logs)")
    parser.add_argument("--args_file", help="Path to JSON file containing arguments (optional)")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--neo4j_host", type=str, default="localhost",
                       help="Neo4j host (default: localhost)")
    parser.add_argument("--neo4j_bolt_port", type=int, default=7687,
                       help="Neo4j bolt port (default: 7687)")
    parser.add_argument("--node_unique_id", type=str, default="id",
                       help="Node unique identifier field (default: id)")
    parser.add_argument("--relation_unique_id", type=str, default="type",
                       help="Relation unique identifier field (default: type)")
    parser.add_argument("--model_to_infer", help="The model to infer")

    # Topology models (ThreeHopPipeline, TwoUnionPipeline, etc.)
    parser.add_argument("--calib_batch_size", type=int, default=4,
                       help="Batch size for calibration processing (default: 4)")
    
    # Multi-GPU and batching arguments
    parser.add_argument("--use-multi-gpu", dest="use_multi_gpu", type=str_to_bool, default=True,
                       help="Use all available GPUs for inference (default: True, pass False to disable)")
    parser.add_argument("--inference-batch-size", dest="inference_batch_size", type=int, default=8,
                       help="Batch size for inference queries (default: 8)")


    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args(args_list)[0]
    return args

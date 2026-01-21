import argparse


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


    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args(args_list)[0]
    return args

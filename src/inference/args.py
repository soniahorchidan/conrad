import argparse


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Log Model Script")
    parser.add_argument("--device", type=str)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--args_file", help="Path to JSON file containing arguments")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--code_paths", type=str, nargs="+")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--neo4j_host", type=str)
    parser.add_argument("--neo4j_bolt_port", type=int)
    parser.add_argument("--kuzu_database_path", type=str)
    parser.add_argument(
        "--db_backend", type=str, default="neo4j", choices=["kuzu", "neo4j"]
    )
    parser.add_argument("--node_unique_id", type=str)
    parser.add_argument("--relation_unique_id", type=str)
    parser.add_argument("--model_to_infer", help="The model to infer")
    parser.add_argument("--EXPERIMENTAL_vectordb_backend", type=str)
    parser.add_argument("--vectordb_path", type=str)
    parser.add_argument("--wheel_file", help="The path to the bindings library")
    ## model specific arguments
    # Query2Box
    parser.add_argument("--query2box_slack", type=float)
    parser.add_argument("--query2box_collection", type=str)

    # ULTRA
    parser.add_argument("--ultra_pred_threshold", type=float)

    # TransR
    parser.add_argument("--transr_collection", type=str)

    # Topology models (ThreeHopPipeline, TwoUnionPipeline, etc.)
    parser.add_argument("--calib_batch_size", type=int, default=4,
                       help="Batch size for calibration processing (default: 4)")

    # Remote checkpoints [Folder]
    parser.add_argument("--remote_ckpt_url", type=str)
    parser.add_argument("--remote_ckpt_save_path", type=str)

    # Dropbox Authentication
    parser.add_argument("--app_key", type=str)
    parser.add_argument("--app_secret", type=str)
    parser.add_argument("--refresh_token", type=str)

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args(args_list)[0]
    return args

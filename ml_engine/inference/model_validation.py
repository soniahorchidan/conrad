import logging
import torch
import os
from utils import merge_args, set_logger, parse_time, set_global_seed
from . import ModelFactory, parse_args_inference
from utils import set_logger
from graph_handler import KuzuBackendDBController
from graph_handler import Neo4JBackendDBController
from argparse import Namespace


class ModelValidation(object):
    def __init__(self):
        inf_args = merge_args(parse_args_inference, "args_file", ["core", "vectordb", "inference_engine", "dropbox"])
        assert inf_args.load_path is not None, "Load path must be provided"

        if inf_args.model_to_infer.lower() == "ultra" and inf_args.device == "mps":
            inf_args.device = "cpu"
            logging.info("Switching to CPU for ULTRA model inference.")

        set_global_seed(inf_args.seed)

        time_str = parse_time()

        inf_args.log_path = os.path.join(
            inf_args.log_path, f"Validation_{inf_args.model_to_infer}_{time_str}"
        )
    
        os.makedirs(inf_args.log_path, exist_ok=True)
        set_logger(inf_args.log_path, "validation.log", True)

        self.save_path = inf_args.log_path

        model_factory = ModelFactory(inf_args)
        model_factory.prepare_model()

        self.model = model_factory.model.to(inf_args.device)
        self.model_args = model_factory.model_args

        # Initialize database controller based on the specified backend
        if inf_args.db_backend == "kuzu":
            self.db_controller = KuzuBackendDBController(inf_args.kuzu_database_path)
        elif inf_args.db_backend == "neo4j":
            self.db_controller = Neo4JBackendDBController(
                f"neo4j://{inf_args.neo4j_host}:{inf_args.neo4j_bolt_port}",
                inf_args.node_unique_id,
                inf_args.relation_unique_id,
            )

    def validate(self):
        logging.info("Validating the model...")
        general_args = Namespace(
            db_controller=self.db_controller,
            save_path=self.save_path,
        )
    
        self.model.validate(general_args, self.model_args)

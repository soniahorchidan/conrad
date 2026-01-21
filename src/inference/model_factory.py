import os
import json
import models
import torch
from graph_handler import Neo4JBackendDBController
from argparse import Namespace
from models import (
    ultra_parse_args,
    ModelUtils,
    MODEL_DEPENDENCIES,
    NUMNODES_USAGE,
)
from conformal_prediction import conformal_prediction_parse_args, VectorConformalRiskControl
from utils import args2sequence
import logging


class ModelFactory:
    def __init__(self, inf_args):
        self.model_to_infer = inf_args.model_to_infer
        self.inf_args = inf_args
        # Extract model-specific arguments from model arguments
        if self.model_to_infer.lower() in ["ultra", "dbexecmodel", "multihoppredictor", "threehoppipeline", "twounionpipeline", "twointersectprojectpipeline"]:
            self.model_parser = ultra_parse_args
            # mps not supported for ULTRA
            if inf_args.device == "mps":
                inf_args.device = "cpu"
        else:
            raise ValueError(f"Unsupported model type: {self.model_to_infer}")
        
    def prepare_model(self):
        """
        Prepare the model for logging.
        """
        logging.info(f"Preparing model {self.model_to_infer} for inference")
        
        # Models are expected to be already present in artifacts/snapshots
        if self.inf_args.load_path is None:
            raise ValueError(f"load_path must be specified. Models should be present in artifacts/snapshots")

        self.model = self.load_model()
        self.model.load_all_components(self.inf_args.load_path)

    def load_model(self):
        if self.model_to_infer.lower() not in ["dbexecmodel", "threehoppipeline", "twounionpipeline", "twointersectprojectpipeline"]:
            if self.model_to_infer.lower() == "multihoppredictor":
                model_args_dict_path = os.path.join(
                    self.inf_args.load_path, f"ultra_args.json"
                )
            else:
                model_args_dict_path = os.path.join(
                    self.inf_args.load_path, f"{self.model_to_infer.lower()}_args.json"
                )
            model_args_dict = json.load(open(model_args_dict_path, "r"))
            self.model_args = self.model_parser(args2sequence(model_args_dict))
            if "conformal_prediction" in model_args_dict:
                self.model_args.conformal_prediction = conformal_prediction_parse_args(
                    args2sequence(model_args_dict["conformal_prediction"])
                )
        else:
            model_args_dict = {
                "conformal_prediction": {
                    "negative_sample_size": 128,
                    "hidden_dim": 200,
                    "batch_size": 1024,
                    "lr": 1e-3,
                    "train_epochs": 50,
                    "calib_size": 0.5,
                    "log_epochs_freq": 10,
                    "val_epochs_freq": 10,
                    "train_size": 0.1,
                    "val_size": 0.4
                    }
            }

            self.model_args = self.model_parser(args2sequence(model_args_dict))
            if "conformal_prediction" in model_args_dict:
                self.model_args.conformal_prediction = conformal_prediction_parse_args(
                    args2sequence(model_args_dict["conformal_prediction"])
                )


        # Initialize database controller - only Neo4j supported
        if self.inf_args.db_backend != "neo4j":
            raise ValueError(f"Only Neo4j backend is supported. Got: {self.inf_args.db_backend}")
        self.model_args.db_controller = Neo4JBackendDBController(
            f"neo4j://{self.inf_args.neo4j_host}:{self.inf_args.neo4j_bolt_port}",
            self.inf_args.node_unique_id,
            self.inf_args.relation_unique_id,
        )

        # get the number of relations
        num_relations = self.model_args.db_controller.get_num_relations()
        num_nodes = len(self.model_args.db_controller.get_all_node_ids())
        self.model_args.db_controller.close()

        # Check for model dependencies (none expected for conrad models)
        dependencies = MODEL_DEPENDENCIES.get(self.model_to_infer, [])
        if dependencies:
            raise ValueError(f"Model {self.model_to_infer} has unsupported dependencies: {dependencies}")

        # tentative fix for non-inductive models
        if self.inf_args.model_to_infer in NUMNODES_USAGE:
            self.model_args.num_nodes = num_nodes
       

        if self.model_to_infer.lower() in ["threehoppipeline", "twounionpipeline", "twointersectprojectpipeline"]:
            # Load ULTRA model (once)
            inf_args_ultra = Namespace(**vars(self.inf_args))
            inf_args_ultra.model_to_infer = "ULTRA"
            model_factory_ultra = ModelFactory(inf_args_ultra)
            model_factory_ultra.prepare_model()

            # Instantiate DBExecModel (rule-based, no weights to load)
            inf_args_dbexec = Namespace(**vars(self.inf_args))
            inf_args_dbexec.model_to_infer = "DBExecModel"
            model_factory_dbexec = ModelFactory(inf_args_dbexec)
            model_factory_dbexec.prepare_model()

            # Create pipeline model (preserve original casing for class name)
            pipeline_model_name = self.inf_args.model_to_infer
            model: ModelUtils = getattr(models, pipeline_model_name)(
                model_factory_ultra.model, model_factory_dbexec.model, self.inf_args, self.inf_args.device
            )
        
        else:    
            # Initialize model
            model: ModelUtils = getattr(models, self.inf_args.model_to_infer)(
                self.model_args, num_relations, self.inf_args.device
            )
        # Initialize conformal prediction (only for pipeline models)
        try:
            if self.model_to_infer.lower() in ["threehoppipeline", "twounionpipeline", "twointersectprojectpipeline"]:
                # Create proper args for VectorConformalRiskControl
                vector_crc_args = conformal_prediction_parse_args({})
                vector_crc_args.load_path = self.inf_args.load_path
                
                # Pass RAPS parameters from inf_args
                vector_crc_args.disable_raps = getattr(self.inf_args, 'disable_raps', False)
                vector_crc_args.raps_lamda = getattr(self.inf_args, 'raps_lamda', 1e-3)
                vector_crc_args.raps_kreg = getattr(self.inf_args, 'raps_kreg', 1)
                
                # Pass dataset for dataset-aware caching
                vector_crc_args.dataset = getattr(self.inf_args, 'dataset', None)
                
                model.conformal_prediction = VectorConformalRiskControl(
                    vector_crc_args,
                    model.generateCalibrateSamples,
                    self.inf_args.model_to_infer.lower(),
                    self.inf_args.device,
                    model,  # Pass the actual model instance
                )
            else:
                logging.info(f"Conformal prediction not supported for model {self.model_to_infer}. "
                           "Only pipeline models (ThreeHopPipeline, TwoUnionPipeline, TwoIntersectProjectPipeline) are supported.")
        except ValueError as e:
            logging.info(f"Conformal Prediction Initialization Error: {e}. Continuing "
                         "without Conformal Prediction.")
        
        # Move model to device
        model = model.to(self.inf_args.device)
        return model

    def get_model_specific_args(self):
        model_specific_args = Namespace()
        for key, value in self.inf_args.__dict__.items():
            if self.inf_args.model_to_infer.lower() in key:
                new_key = key.split("_")[1]
                setattr(model_specific_args, new_key, value)
        return model_specific_args

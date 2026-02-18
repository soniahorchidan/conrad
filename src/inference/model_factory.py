import os
import json
import models
import torch
from graph_handler import Neo4JBackendDBController
from argparse import Namespace
from models import (
    ultra_parse_args,
    ModelUtils,
)
from conformal_prediction import conformal_prediction_parse_args, VectorConformalRiskControl, NonVectorCRC
from utils import args2sequence
import logging


class ModelFactory:
    def __init__(self, inf_args):
        self.model_to_infer = inf_args.model_to_infer
        self.inf_args = inf_args
        # Extract model-specific arguments from model arguments
        if self.model_to_infer.lower() in ["ultra", "dbexecmodel", "multihoppredictor", "threehoppipeline", "twounionpipeline", "twointersectprojectpipeline", "nonvector3hopneural"]:
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
        # Load model weights if the model supports it
        if hasattr(self.model, 'load_weights'):
            self.model.load_weights(self.inf_args.load_path)
        
        # Set model to eval mode for inference (critical for UltraQuery checkpoints)
        # This ensures dropout and batch norm layers behave correctly during inference
        if hasattr(self.model, 'eval'):
            self.model.eval()
            logging.info("Model set to eval mode for inference")
        
        # For pipeline models, also set the underlying ULTRA model to eval mode
        if hasattr(self.model, 'unified_predictor') and hasattr(self.model.unified_predictor, 'ultra'):
            if hasattr(self.model.unified_predictor.ultra, 'eval'):
                self.model.unified_predictor.ultra.eval()
                logging.info("Underlying ULTRA model set to eval mode")

    def load_model(self):
        if self.model_to_infer.lower() not in ["dbexecmodel", "threehoppipeline", "twounionpipeline", "twointersectprojectpipeline", "nonvector3hopneural"]:
            # Check for ultra_args.json first (standard checkpoint structure)
            if self.model_to_infer.lower() == "multihoppredictor":
                model_args_dict_path = os.path.join(
                    self.inf_args.load_path, f"ultra_args.json"
                )
            else:
                model_args_dict_path = os.path.join(
                    self.inf_args.load_path, f"{self.model_to_infer.lower()}_args.json"
                )
            
            if os.path.exists(model_args_dict_path):
                # Standard checkpoint with args file
                logging.info(f"Loading model args from: {model_args_dict_path}")
                model_args_dict = json.load(open(model_args_dict_path, "r"))
                self.model_args = self.model_parser(args2sequence(model_args_dict))
            else:
                # No args file found - assume this is an official checkpoint directory (e.g., ultraquery/)
                # Use default args or args from inf_args
                logging.info(f"No args file found at {model_args_dict_path}, using default/model args from inf_args")
                model_args_dict = {}
                # Copy relevant args from inf_args if available
                if hasattr(self.inf_args, 'msp_threshold'):
                    model_args_dict['msp_threshold'] = self.inf_args.msp_threshold
                if hasattr(self.inf_args, 'acceptance_threshold'):
                    model_args_dict['acceptance_threshold'] = self.inf_args.acceptance_threshold
                if hasattr(self.inf_args, 'batch_size'):
                    model_args_dict['batch_size'] = self.inf_args.batch_size
                if hasattr(self.inf_args, 'hidden_dim'):
                    model_args_dict['hidden_dim'] = self.inf_args.hidden_dim
                # Use defaults for other required args
                self.model_args = self.model_parser(args2sequence(model_args_dict))
        else:
            # For pipeline models, create minimal model_args (they don't use most of it)
            # Pipeline models use inf_args directly and conformal prediction is initialized separately
            model_args_dict = {}
            self.model_args = self.model_parser(args2sequence(model_args_dict))


        # Initialize Neo4j database controller
        self.model_args.db_controller = Neo4JBackendDBController(
            f"neo4j://{self.inf_args.neo4j_host}:{self.inf_args.neo4j_bolt_port}",
            self.inf_args.node_unique_id,
            self.inf_args.relation_unique_id,
        )

        # get the number of relations
        num_relations = self.model_args.db_controller.get_num_relations()
        num_nodes = len(self.model_args.db_controller.get_all_node_ids())
        self.model_args.db_controller.close()

        # Set num_nodes for all models
        self.model_args.num_nodes = num_nodes
       
        # Pass use_multi_gpu flag to model args (default to True if not specified)
        if not hasattr(self.inf_args, 'use_multi_gpu'):
            self.inf_args.use_multi_gpu = True
        self.model_args.use_multi_gpu = self.inf_args.use_multi_gpu
        
        # Pass msp_threshold from inf_args if available (important for UltraQuery checkpoints)
        if hasattr(self.inf_args, 'msp_threshold'):
            self.model_args.msp_threshold = self.inf_args.msp_threshold

        if self.model_to_infer.lower() in ["threehoppipeline", "twounionpipeline", "twointersectprojectpipeline"]:
            # Load ULTRA model (once)
            inf_args_ultra = Namespace(**vars(self.inf_args))
            inf_args_ultra.model_to_infer = "ULTRA"
            model_factory_ultra = ModelFactory(inf_args_ultra)
            model_factory_ultra.prepare_model()
            
            # Setup multi-GPU for ULTRA model
            if hasattr(model_factory_ultra.model, 'setup_multi_gpu'):
                model_factory_ultra.model.setup_multi_gpu()
                num_gpus_used = model_factory_ultra.model.num_gpus
                logging.info(f"ULTRA inference configured: Using {num_gpus_used} GPU(s)")

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
        elif self.model_to_infer.lower() in ["nonvector3hopneural"]:
            # Load ULTRA model (once)
            inf_args_ultra = Namespace(**vars(self.inf_args))
            inf_args_ultra.model_to_infer = "ULTRA"
            model_factory_ultra = ModelFactory(inf_args_ultra)
            model_factory_ultra.prepare_model()
            
            # Setup multi-GPU for ULTRA model
            if hasattr(model_factory_ultra.model, 'setup_multi_gpu'):
                model_factory_ultra.model.setup_multi_gpu()
                num_gpus_used = model_factory_ultra.model.num_gpus
                logging.info(f"ULTRA inference configured: Using {num_gpus_used} GPU(s)")
            
            model: ModelUtils = getattr(models, self.inf_args.model_to_infer)(
                model_factory_ultra.model, self.model_args, self.inf_args.device
            )
        else:    
            # Initialize model
            model: ModelUtils = getattr(models, self.inf_args.model_to_infer)(
                self.model_args, num_relations, self.inf_args.device
            )
            
            # Setup multi-GPU for direct ULTRA model
            if hasattr(model, 'setup_multi_gpu'):
                model.setup_multi_gpu()
                num_gpus_used = model.num_gpus
                logging.info(f"ULTRA inference configured: Using {num_gpus_used} GPU(s)")
        # Initialize conformal prediction (only for pipeline models)
        try:
            if self.model_to_infer.lower() in ["threehoppipeline", "twounionpipeline", "twointersectprojectpipeline"]:
                # Create proper args for VectorConformalRiskControl
                vector_crc_args = conformal_prediction_parse_args({})
                vector_crc_args.load_path = self.inf_args.load_path
                
                # Pass dataset for dataset-aware caching
                vector_crc_args.dataset = getattr(self.inf_args, 'dataset', None)
                
                model.conformal_prediction = VectorConformalRiskControl(
                    vector_crc_args,
                    model.generateCalibrateSamples,
                    self.inf_args.model_to_infer.lower(),
                    self.inf_args.device,
                    model,  # Pass the actual model instance
                )
            elif self.model_to_infer.lower() in ["nonvector3hopneural"]:
                crc_args = conformal_prediction_parse_args({})
                crc_args.load_path = self.inf_args.load_path

                # Pass dataset for dataset-aware caching
                crc_args.dataset = getattr(self.inf_args, 'dataset', None)
                
                model.conformal_prediction = NonVectorCRC(
                    args=crc_args, generateCalibrateSamples_fn=model.generateCalibrateSamples, model_name=self.inf_args.model_to_infer.lower(), device=self.inf_args.device
                )
            else:
                logging.info(f"Conformal prediction not supported for model {self.model_to_infer}. "
                           "Only pipeline models (ThreeHopPipeline, TwoUnionPipeline, TwoIntersectProjectPipeline) are supported.")
        except ValueError as e:
            logging.info(f"Conformal Prediction Initialization Error: {e}. Continuing "
                         "without Conformal Prediction.")
        
        # Move model to device
        model = model.to(self.inf_args.device)
        
        # Log final GPU usage summary
        if torch.cuda.is_available() and self.inf_args.device.startswith('cuda'):
            num_gpus_available = torch.cuda.device_count()
            if hasattr(model, 'unified_predictor') and hasattr(model.unified_predictor, 'ultra'):
                if hasattr(model.unified_predictor.ultra, 'num_gpus'):
                    num_gpus_used = model.unified_predictor.ultra.num_gpus
                    logging.info(f"GPU Usage Summary")
                    logging.info(f"GPUs available: {num_gpus_available}")
                    logging.info(f"GPUs used for ULTRA inference: {num_gpus_used}")
                    logging.info(f"Multi-GPU enabled: {model.unified_predictor.ultra.use_multi_gpu}")
            elif hasattr(model, 'num_gpus'):
                num_gpus_used = model.num_gpus
                logging.info(f"GPU Usage Summary")
                logging.info(f"GPUs available: {num_gpus_available}")
                logging.info(f"GPUs used for ULTRA inference: {num_gpus_used}")
                logging.info(f"Multi-GPU enabled: {model.use_multi_gpu}")
        
        return model

    def get_model_specific_args(self):
        model_specific_args = Namespace()
        for key, value in self.inf_args.__dict__.items():
            if self.inf_args.model_to_infer.lower() in key:
                new_key = key.split("_")[1]
                setattr(model_specific_args, new_key, value)
        return model_specific_args

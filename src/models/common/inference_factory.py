import numpy as np
import logging
import torch
from models import (
    ultra_shape_input,
    relationprediction_shape_input,
)

from graph_handler import KuzuBackendDBController
from graph_handler import Neo4JBackendDBController
from utils import get_graph


class InferenceFactory:
    def __init__(self, inference_args, model_specific_args):
        self.model_to_infer = inference_args.model_to_infer
        self.model_specific_args = model_specific_args
        self.db_backend = inference_args.db_backend
        self.device = inference_args.device

        # Initialize database controller based on the specified backend
        if self.db_backend == "kuzu":
            db_controller = KuzuBackendDBController(inference_args.kuzu_database_path)
        elif self.db_backend == "neo4j":
            db_controller = Neo4JBackendDBController(
                f"neo4j://{inference_args.neo4j_host}:{inference_args.neo4j_bolt_port}",
                inference_args.node_unique_id,
                inference_args.relation_unique_id,
            )

        if self.model_to_infer.lower() == "ultra":
            self.shape_input_fn = ultra_shape_input
        elif self.model_to_infer.lower() == "dbexecmodel":
            self.shape_input_fn = ultra_shape_input
        else:
            raise KeyError(f"Unsupported model type: {self.model}")

        self.input_fn_args = self.shape_input_fn.__code__.co_varnames

        self.inited_inputs = init_inputs(
            db_controller, self.device, self.input_fn_args
        )

    def shape_input(self, inputs):
        """
        Shapes the input data based on the specified model type.

        Parameters:
        inputs (dict): The input data that needs to be shaped.

        Returns:
        The shaped input data as returned by the corresponding shaping function.

        Raises:
        KeyError: If the model type is not supported, an error message is logged.
        """
        inputs["model_specific_args"] = self.model_specific_args

        self.inited_inputs["inputs"] = inputs

        try:
            return self.shape_input_fn(**self.inited_inputs)
        except Exception as e:
            print(f"Error while shaping the input: {str(e)}")
            raise


def init_inputs(db_controller, device, input_fn_args):
    inputs = {}
    if "db_controller" in input_fn_args:
        inputs["db_controller"] = db_controller
    if "device" in input_fn_args:
        inputs["device"] = device
    if "all_node_ids" in input_fn_args:
        inputs["all_node_ids"] = db_controller.get_all_node_ids()
    if "num2id" in input_fn_args:
        all_node_ids = (
            db_controller.get_all_node_ids()
            if "all_node_ids" not in inputs
            else inputs["all_node_ids"]
        )
        inputs["num2id"] = {i: id_ for i, id_ in enumerate(all_node_ids)}
    if "entity_embedding" in input_fn_args:
        all_node_ids = (
            db_controller.get_all_node_ids()
            if "all_node_ids" not in inputs
            else inputs["all_node_ids"]
        )
        inputs["entity_embedding"] = torch.zeros(len(all_node_ids), 1, dtype=torch.float32, device=device)
    if "all_relations" in input_fn_args:
        inputs["all_relations"] = db_controller.get_all_relations()
    if "graph_data" in input_fn_args:
        inputs["graph_data"] = get_graph(db_controller, device, True, True)
    return inputs

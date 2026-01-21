import logging
import uvicorn
import numpy as np
import torch
import itertools
import sys
import os
import argparse
import pyarrow as pa
from utils import merge_args, set_logger, parse_time, set_global_seed
from . import ModelFactory, parse_args_inference
from fastapi import FastAPI, Request
from pydantic import BaseModel
from models import VECTORDB_USAGE
from models.common.inference_factory import InferenceFactory
from vectordb import VectorDBControllerFactory
from utils import set_logger


class FastAPIApp(object):
    def __init__(self, inf_args, host="0.0.0.0", port=8000):
        self.app = FastAPI(debug=True)
        self.host = host
        self.port = port
        self.inference_factory, self.model, self.slack = self.init_inference(inf_args)
        self.setup_routes()

    def init_inference(self, inf_args):
        inf_args = merge_args(
            parse_args_inference,
            "args_file",
            ["core", "vectordb", "inference_engine", "dropbox"],
            inf_args,
        )
        assert inf_args.load_path is not None, "Load path must be provided"

        if inf_args.model_to_infer.lower() == "ultra" and inf_args.device == "mps":
            inf_args.device = "cpu"
            logging.info("Switching to CPU for ULTRA model inference.")

        set_global_seed(inf_args.seed)

        time_str = parse_time()
        os.makedirs(inf_args.log_path, exist_ok=True)
        set_logger(
            inf_args.log_path, f"deploy_{inf_args.model_to_infer}_{time_str}.log", True
        )

        model_factory = ModelFactory(inf_args)
        model_factory.prepare_model()

        hidden_dim = model_factory.model_args.hidden_dim

        model_specific_args = model_factory.get_model_specific_args()

        model = model_factory.model.to(inf_args.device)
        slack = None

        if inf_args.model_to_infer.lower() == "query2box":
            slack = model_specific_args.slack
            logging.info("Query2Box slack set to %s", slack)

        if inf_args.EXPERIMENTAL_vectordb_backend == "faiss":
            logging.info("Using experimental FAISS vector search.")

        if VECTORDB_USAGE.get(inf_args.model_to_infer, False):
            logging.info("VectorDB used for this model.")
            try:
                vectordb_controller = VectorDBControllerFactory.get_controller(
                    inf_args.EXPERIMENTAL_vectordb_backend,
                    inf_args.vectordb_path,
                    model_specific_args.collection,
                    hidden_dim,
                )
            except Exception as e:
                logging.error(
                    f"Error occurred while creating vectordb controller object: {e}",
                    exc_info=True,
                )
                sys.exit(1)
        else:
            logging.info("VectorDB not used for this model.")
            vectordb_controller = None

        inference_factory = InferenceFactory(
            inf_args, vectordb_controller, model_specific_args
        )
        return inference_factory, model, slack

    def setup_routes(self):
        @self.app.post("/invocations")
        async def predict(request: Request):
            """
            Model inference endpoint. Accepts Arrow stream or JSON format.
            Arrow stream: Stream of Arrow IPC messages with "data" column and optional "confidence" metadata.
            JSON: {"inputs": {"data": [[1,2,3]], "confidence": 0.95}}
            """
            ARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"
            JSON_CONTENT_TYPE = "application/json"

            logging.info("Received request for inference.")
            inputs_dict: dict = {}

            if request.headers.get("Content-Type") == ARROW_CONTENT_TYPE:
                body = await request.body()
                logging.info("Processing request with Arrow stream format.")
                # Open the Arrow stream
                reader = pa.ipc.open_stream(body)

                # Convert all batches to a single table
                table = reader.read_all()

                # Extract individual columns by name
                if self.slack:
                    inputs_dict["slack"] = self.slack
                if "data" in table.schema.names:
                    inputs_dict["data"] = np.array(table.column("data").to_pylist())
                if table.schema.metadata:
                    metadata_dict = table.schema.metadata
                    if b"confidence" in metadata_dict:
                        inputs_dict["confidence"] = float(metadata_dict[b"confidence"])
            elif request.headers.get("Content-Type") == JSON_CONTENT_TYPE:
                logging.info("Processing request with JSON format.")
                inputs_dict = await request.json()
                inputs_dict = inputs_dict["inputs"]
                logging.info("Received JSON inputs: %s", inputs_dict)
                if self.slack:
                    inputs_dict["slack"] = self.slack
                if "data" in inputs_dict:
                    inputs_dict["data"] = np.array(inputs_dict["data"])
            else:
                logging.error(
                    "Unsupported Content-Type: %s. Only Arrow stream and JSON are supported.",
                    request.headers.get("Content-Type"),
                )
                return {"error": "Unsupported Content-Type"}, 415

            logging.info("Inputs received for inference: %s", inputs_dict)
            params = self.inference_factory.shape_input(inputs_dict)

            outputs = {}
            # Run model inference.
            with torch.no_grad():
                try:
                    out, confidence_bounds = self.model.predict(**params)
                    prediction_result = out.tolist() if hasattr(out, "tolist") else out
                    bound_result = (
                        confidence_bounds.tolist()
                        if hasattr(confidence_bounds, "tolist")
                        else confidence_bounds
                    )
                    out = {
                        "predictions": prediction_result,
                        "confidence_bounds": bound_result,
                    }
                    outputs = {"predictions": out}
                except Exception as e:
                    logging.error(
                        "Error occurred while attempting to infer: ", exc_info=True
                    )

            # TODO: Set Content-Type to 'application/vnd.apache.arrow.stream' after QueryResult in OrbDB is updated to return Arrow-formatted data.
            return outputs

        @self.app.get("/")
        async def root():
            return {
                "Name": "Orb DB ML Engine Prediction API",
                "description": "This API is used to make predictions using the Orb DB ML Engine",
            }

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)


def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI Server")
    parser.add_argument("--port", type=int, default=8000)
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    args, unknown = parse_args()

    app_instance = FastAPIApp(inf_args=unknown, port=args.port)
    app_instance.run()

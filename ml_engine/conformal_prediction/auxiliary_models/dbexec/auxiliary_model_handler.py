import torch.nn as nn
import torch.nn.functional as F
import logging
from argparse import Namespace
from ..common import CalibProbModelHandler


class DBExecCalibProbModelHandler(CalibProbModelHandler):
    def __init__(self, args, model_name, device):
        super(DBExecCalibProbModelHandler, self).__init__(args, model_name, device)
        logging.info("Auxiliary model not required.")

    @staticmethod
    def postprocess(model, general_args: Namespace, args: Namespace):
        logging.info(
            "Skipping save for auxiliary model because it is saved in the training loop."
        )

    def load_all_components(self, load_path, ignore_components: list = []):
        pass

    def train(
        self, train_data, val_data, save_path, db_controller, vectordb_controller
    ):
        pass

    def validate(self, val_dataloader):
        pass

    def predict(self, x):
        return x

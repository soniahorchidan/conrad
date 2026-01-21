import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from argparse import Namespace
from ..common import CalibProbModelHandler
from .auxiliary_model import CalibProbModel
from .dataloader import CalibProbDataset, CalibProbDatasetVal


class UltraCalibProbModelHandler(CalibProbModelHandler):
    def __init__(self, args, model_name, device):
        super(UltraCalibProbModelHandler, self).__init__(args, model_name, device)
        logging.info("Auxiliary model not required.")
        self.aux_model = CalibProbModel().to(self.device)

    @staticmethod
    def postprocess(model, general_args: Namespace, args: Namespace):
        logging.info(
            "Skipping save for auxiliary model because it is saved in the training loop."
        )

    def load_all_components(self, load_path, ignore_components: list = []):
        pass

    def train(
        self, train_data, val_data, save_path, db_controller
    ):
        pass

    def validate(self, val_dataloader):
        pass

    def predict(self, x):
        self.aux_model.eval()
        with torch.no_grad():
            output = self.aux_model(x)
        return output

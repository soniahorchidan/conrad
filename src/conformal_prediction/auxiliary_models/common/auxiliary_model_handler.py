import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import Namespace
from utils import AverageMeter, save_component
import logging
import os


class CalibProbModelHandler(object):
    def __init__(self, args, model_name, device):
        self.args = args
        self.model_name = model_name
        self.device = device
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.negative_sample_size = args.negative_sample_size
        self.train_epochs = args.train_epochs
        self.aux_model = None

    @staticmethod
    def postprocess(model, general_args: Namespace, args: Namespace):
        logging.info(
            "Skipping save for auxiliary model because it is saved in the training loop."
        )

    def load_all_components(self, load_path, ignore_components: list = []):
        if load_path is not None:
            # load auxiliary model weights
            logging.info("Loading auxiliary model weights...")
            aux_model_path = os.path.join(
                load_path, f"risk_control_{self.model_name}", "auxiliary_model.pth"
            )
            self.aux_model.load_state_dict(
                torch.load(aux_model_path, map_location=self.device)
            )
            logging.info("Auxiliary model weights loaded")
        else:
            logging.info(
                "No load path provided, skipping loading of auxiliary model weights"
            )

    def train(
        self, train_data, val_data, save_path, db_controller
    ):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

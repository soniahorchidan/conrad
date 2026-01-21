from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any


class ModelUtils(ABC):
    @abstractmethod
    def trainModel(self, general_args: Namespace, model_args: Namespace):
        """
        Train the model
        :param general_args: Namespace, the general arguments for all models
        :param model_args: Namespace, the model specific arguments
        """
        pass

    @abstractmethod
    def load_all_components(self, load_path: str, ignore_components: list = []):
        """
        Load all components for the model. Recursively load all components for the model.
        :param load_path: str, the path to load the components from
        :param ignore_components: list, the components to ignore
        """
        pass

    @abstractmethod
    def generateCalibrateSamples(
        self,
        save_path: str,
        db_controller,
        calib_iterator=None,
    ):
        """
        Generate calibration samples
        :param save_path: str, the path to save the calibration samples
        :param db_controller: the database controller
        :param calib_iterator: the calibration iterator, if it exists
        :return: a tuple of sliceable objects
        """
        pass

    @staticmethod
    @abstractmethod
    def preprocess(general_args: Namespace, args: Namespace):
        """
        Preprocess the model
        :param general_args: Namespace, the general arguments for all models
        :param args: Namespace, the model specific arguments
        """
        pass

    @staticmethod
    @abstractmethod
    def postprocess(model, general_args: Namespace, args: Namespace):
        pass

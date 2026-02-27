from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any

# TODO(sonia): Remove this class and use the calibration_data_generator class instead
class ModelUtils(ABC):
    pass
    # @abstractmethod
    # def generateCalibrateSamples(
    #     self,
    #     save_path: str,
    #     db_controller,
    #     calib_iterator=None,
    # ):
    #     """
    #     Generate calibration samples
    #     :param save_path: str, the path to save the calibration samples
    #     :param db_controller: the database controller
    #     :param calib_iterator: the calibration iterator, if it exists
    #     :return: a tuple of sliceable objects
    #     """
    #     pass

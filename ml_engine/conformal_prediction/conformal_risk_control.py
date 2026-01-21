import numpy as np
import torch
import os
import logging
import json
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from scipy.optimize import brentq
from torch.utils.data import DataLoader
from .auxiliary_models import (
    UltraCalibProbModelHandler,
    UltraCalibProbDatasetVal,
    get_x_ultra,
    DBExecCalibProbModelHandler,
    DBExecCalibProbDatasetVal,
    get_x_dbexec,
)
from utils import save_component

class ConformalRiskControl(object):
    def __init__(self, args, generateCalibrateSamples_fn, model_name, device):
        self.args = args
        self.generateCalibrateSamples_fn = generateCalibrateSamples_fn
        self.model_name = model_name
        self.device = device
        self.batch_size = self.args.batch_size

        if model_name == "ultra":
            self.cali_dataset = UltraCalibProbDatasetVal
            self.get_x_fn = get_x_ultra
            self.calib_model_handler = UltraCalibProbModelHandler(
                self.args, model_name, device
            )
        elif model_name == "dbexecmodel":
            self.cali_dataset = DBExecCalibProbDatasetVal
            self.get_x_fn = get_x_dbexec
            self.calib_model_handler = DBExecCalibProbModelHandler(
                self.args, model_name, device
            )
        elif model_name == "multihoppredictor":
            self.cali_dataset = UltraCalibProbDatasetVal
            self.get_x_fn = get_x_ultra
            self.calib_model_handler = DBExecCalibProbModelHandler(
                self.args, model_name, device
            )
        else:
            raise ValueError("Model not supported")

    def _load_calibration_data(self, load_path, hop_name):
        model_dir = os.path.join(load_path, f"risk_control_{self.model_name}_{hop_name}")
        if os.path.exists(model_dir):
            logging.info(f"Loading calibration scores, true labels and metadata for {hop_name} from {load_path}")

            cal_scores_path = os.path.join(model_dir, "cal_scores.pth")
            if os.path.exists(cal_scores_path):
                setattr(self, f"cal_scores_{hop_name}", torch.load(cal_scores_path))

            true_labels_path = os.path.join(model_dir, "true_labels.pth")
            if os.path.exists(true_labels_path):
                setattr(self, f"true_labels_{hop_name}", torch.load(true_labels_path))

            metadata_path = os.path.join(load_path, f"metadata_crc_{hop_name}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as file:
                    metadata = json.load(file)[self.model_name.lower()]["calibrate"]
                    setattr(self, f"metadata_{hop_name}", metadata)
        else:
            logging.info(f"Calibration scores for {hop_name} not found. Recalibration required.")


    def load_all_components(self, load_path, ignore_components: list = []):
        if load_path is not None:
            self._load_calibration_data(load_path, "1hop")
            self._load_calibration_data(load_path, "2hop")
            self._load_calibration_data(load_path, "3hop")
        else:
            logging.info("No load path provided, skipping loading of calibration scores")

        self.calib_model_handler.load_all_components(load_path)
        logging.info("Conformal risk control components loaded")


    @staticmethod
    def postprocess(risk_control, general_args: Namespace, args: Namespace):
        logging.info("Saving calibration scores")

        save_component(
            general_args.load_path,
            f"risk_control_{risk_control.model_name}",
            "cal_scores",
            risk_control.cal_scores,
        )
        save_component(
            general_args.load_path,
            f"risk_control_{risk_control.model_name}",
            "cal_queries",
            risk_control.cal_queries,
        )
        save_component(
            general_args.load_path,
            f"risk_control_{risk_control.model_name}",
            "true_labels",
            risk_control.true_labels,
        )
        calib_model_handler = risk_control.calib_model_handler
        calib_model_handler.postprocess(calib_model_handler, general_args, args)

    def split_data(self, full_calib_data):
        """
        Split the data into:
        - calibration data
        - training data for the auxiliary model
        - validation data for the auxiliary model
        """

        # NOTE(sonia): full_calib_data contains the sampled one-hop, two-hop, three-hop queries

        print("DEBUG:: hardcoding train/calib/valid size since they seem to be read from IDK where!!!")
        print("DEBUG:: initial train_size:", self.args.train_size)
        print("DEBUG:: initial val_size:", self.args.val_size)
        print("DEBUG:: initial calib_size:", self.args.calib_size)
        # Hardcoding the sizes to avoid reading from config files
        self.args.train_size = 0
        self.args.val_size = 0
        self.args.calib_size = 1

        # Split the data into calibration and training/validation data
        data_size = len(full_calib_data[0])
        train_size = int(data_size * self.args.train_size)
        val_size = int(data_size * self.args.val_size)
        calib_size = data_size - train_size - val_size

        print("DEBUG:: Data size:", data_size)
        print("DEBUG:: Calibration size:", calib_size)
        print("DEBUG:: Train size:", train_size)
        print("DEBUG:: Validation size:", val_size)

        def _split_data(data, indices):
            res = tuple()
            for x in data:
                if isinstance(x, list):
                    res += ([x[i] for i in indices],)
                else:
                    res += (x[indices],)
            return res

        indices = np.arange(data_size)
        np.random.shuffle(indices)
        calib_indices = indices[:calib_size]
        train_indices = indices[calib_size : calib_size + train_size]
        val_indices = indices[calib_size + train_size :]

        calib_data = _split_data(full_calib_data, calib_indices)
        train_data = _split_data(full_calib_data, train_indices)
        val_data = _split_data(full_calib_data, val_indices)

        logging.info(f"Calibration set size: {len(calib_indices)}")
        logging.info(f"Train set size: {len(train_indices)}")
        logging.info(f"Validation set size: {len(val_indices)}")

        return calib_data, train_data, val_data

    def prepare_data(self, save_path, db_controller, vectordb_controller):
        logging.info("Generating calibration data")
        full_calib_data = self.generateCalibrateSamples_fn(
            save_path, db_controller, vectordb_controller, None, True
        )
        logging.info("Splitting data")
        self.calib_data, self.train_data, self.val_data = self.split_data(
            full_calib_data
        )

    def train(self, save_path, db_controller, vectordb_controller):
        logging.info("Training auxiliary model")
        self.calib_model_handler.train(
            self.train_data,
            self.val_data,
            save_path,
            db_controller,
            vectordb_controller,
        )

    def calibrate(self, db_controller, vectordb_controller, threshold=-1):
        logging.info("Calibrating model")
        self.cal_scores = []
        self.true_labels = []
        self.cal_queries = []

        cali_dataset = self.cali_dataset(
            self.calib_data, db_controller, vectordb_controller
        )
        logging.info("Calibration data size: {}".format(len(cali_dataset)))
        cali_dataloader = DataLoader(
            cali_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=cali_dataset.collate_fn,
        )

        for x, y, query in cali_dataloader:
            with torch.no_grad():
                output = self.calib_model_handler.predict(x)
            cal_scores = output.cpu().numpy().reshape(-1)
            self.cal_queries.append(query)
            self.cal_scores.append(cal_scores)
            self.true_labels.append(y)

        non_conformity_scores = self.cal_scores
        median_nc_score = np.median(non_conformity_scores)
        mean_nc_score = np.mean(non_conformity_scores)
        var_nc_score = np.var(non_conformity_scores)

        logging.info(f"Median non-conformity score: {median_nc_score:.6f}")
        logging.info(f"Mean non-conformity score: {mean_nc_score:.6f}")
        logging.info(f"Variance of non-conformity: {var_nc_score:.6f}")

        alpha_lowerbound, alpha_upperbound = get_alpha_bounds(len(self.cal_scores))

        # find optimal lamhat values for common error thresholds
        precalibrated_thresholds = {}

        for alpha in np.arange(0.1, 0.51, 0.1):
            lamhat = self.optimize(alpha)
            precalibrated_thresholds[alpha] = lamhat

        metadata = {
            "median_non_conformity_score": float(median_nc_score),
            "mean_non_conformity_score": float(mean_nc_score),
            "variance_non_conformity": float(var_nc_score),
            "alpha_lowerbound": alpha_lowerbound,
            "alpha_upperbound": alpha_upperbound,
            "calibrated_alphas": precalibrated_thresholds,
        }
        return metadata

    def prepare_calibrate(
        self, save_path, db_controller, vectordb_controller, threshold=-1
    ):
        self.prepare_data(save_path, db_controller, vectordb_controller)
        self.train(save_path, db_controller, vectordb_controller)
        self.metadata = self.calibrate(db_controller, vectordb_controller, threshold)
        return self.metadata

    def optimize(self, alpha):
        lamhat = brentq(
            lamhat_threshold, 0, 1, args=(self.cal_scores, self.true_labels, alpha)
        )
        logging.info(f"Calibrating for alpha {alpha:.2f} done! lamhat={lamhat}")
        return lamhat

    def predict(self, entity_embedding, x, confidence, query=None):
        alpha = round(1 - confidence, 4)
        q_length = len(query[0]) if query is not None else -1
        if q_length == 2:
            # cal_scores = self.cal_scores_1hop
            metadata = self.metadata_1hop
        elif q_length == 3:
            # cal_scores = self.cal_scores_2hop
            metadata = self.metadata_2hop
        elif q_length == 4:
            # cal_scores = self.cal_scores_3hop
            metadata = self.metadata_3hop      
        # alpha_lowerbound, alpha_upperbound = get_alpha_bounds(len(cal_scores))
        # if alpha < alpha_lowerbound or alpha > alpha_upperbound:
        #     raise ValueError(
        #         f"Confidence level must be between {1 - alpha_upperbound} and {1- alpha_lowerbound}"
        #     )

        is_optimised = False
        if metadata is not None:
            calibrated = metadata.get("calibrated_alphas", {})
            if calibrated:
                # find a key whose float value equals alpha
                match_key = next(
                    (k for k in calibrated
                    if abs(float(k) - alpha) < 1e-9),
                    None
                )
                if match_key is not None:
                    lamhat = float(calibrated[match_key])
                else:
                    lamhat = 0.0

                if lamhat > 0:
                    logging.info(
                        f"Loading lamhat {lamhat} already optimized for alpha {alpha}."
                    )
                    is_optimised = True
            else:
                self.metadata["calibrated_alphas"] = {}
        if not is_optimised:
            logging.info(
                f"Lamhat not found for alpha {alpha}. Optimizing..."
            )
            lamhat = self.optimize(alpha)
            metadata["calibrated_alphas"][str(alpha)] = lamhat
        out = []
        for x_i in x:
            with torch.no_grad():
                output = self.calib_model_handler.predict(x_i)
            output = output.detach().cpu().view(-1)
            preds = (output >= lamhat).nonzero(as_tuple=True)[0]
            out.append(preds)
        return out

    # NOTE: This metric is not really sounds; it's just an indication of how confident the
    # model is, but it does not correlate with the CRC confidence levels the user sets.
    # For example, if the user wants at least 90% confidence, the prediction set size will
    # be large, and will include classes with low sigmoid score to make sure the coverage
    # validity is satisfied. The errors in this method do not really correlate to that. We
    # might need to do something like Platt scaling to get mathematically accurate
    # intervals. Another option is to not expose these absolute values, but to tag them
    # with "LOW", "MEDIUM", "HIGH".
    def predict_with_error_tagging(self, entity_embedding, x, pred_length=-1):
        is_optimised = False
        if hasattr(self, "metadata") and self.metadata is not None:
            if "calibrated_alphas" in self.metadata:
                is_optimised = True
            else:
                self.metadata["calibrated_alphas"] = {}
        confidence_bounds = []
        for i in range(x.shape[1]):
            feat = self.get_x_fn(entity_embedding, x[:, i])
            with torch.no_grad():
                output = self.calib_model_handler.predict(feat)
            output = output.detach().cpu().numpy().reshape(-1)
            # CRC always assumes higher is better
            if pred_length >= 0:
                nonconf_score, _ = torch.topk(torch.tensor(output), pred_length)
            else:
                nonconf_score, _ = torch.max(torch.tensor(output), dim=0)
                nonconf_score = [nonconf_score]
            if is_optimised:
                all_bounds = []
                for s in nonconf_score:
                    bounds = self.find_error_interval(s)
                    if bounds:
                        min_alpha, max_alpha = bounds
                        all_bounds.append((min_alpha, max_alpha))
                confidence_bounds.append(all_bounds)

        return confidence_bounds

    def find_error_interval(self, score):
        alphas = self.metadata["calibrated_alphas"]
        keys = sorted(alphas, key=float)
        min_alpha = None
        max_alpha = None

        if score <= alphas[keys[0]]:
            min_alpha = 0
            max_alpha = float(keys[0])
        elif score >= alphas[keys[-1]]:
            min_alpha = float(keys[-1])
            max_alpha = 1
        else:
            for i in range(len(keys) - 1):
                if float(alphas[keys[i]]) <= score <= float(alphas[keys[i + 1]]):
                    min_alpha = float(keys[i])
                    max_alpha = float(keys[i + 1])

        return round(min_alpha, 2), round(max_alpha, 2)


def false_negative_rate(scores, y, lam):
    ovrlp = []
    for i in range(len(scores)):
        preds = np.where(scores[i] >= lam)[0]
        gt_labels = set(y[i].tolist())
        intersection = len(set(preds) & gt_labels)
        if len(gt_labels) > 0:
            overlap_ratio = 1 - intersection / len(gt_labels)
        else:
            overlap_ratio = 0  # avoid division by zero
        ovrlp.append(overlap_ratio)
    return np.mean(ovrlp)


def lamhat_threshold(lam, cal_scores, y, alpha):
    n = len(cal_scores)
    fnr = false_negative_rate(cal_scores, y, lam)
    return fnr - ((n + 1) / n * alpha - 1 / (n + 1))


def get_alpha_bounds(n):
    alpha_lowerbound = n / ((n + 1) ** 2)
    alpha_upperbound = n * (n + 2) / ((n + 1) ** 2)
    return alpha_lowerbound, alpha_upperbound

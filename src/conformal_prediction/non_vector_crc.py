import numpy as np
import os
import logging
import json
from argparse import Namespace
from typing import Any, Dict, Callable
from scipy.optimize import brentq
from datetime import datetime

class NonVectorCRC(object):
    """
    Classic CRC procedure for node prediction. But extended for three hop queries using Bonferroni Inequality
    ONLY SUPPORTS THREE HOP QUERIES. Specifically NonVector3HopNeural model.
    """
    def __init__(self, args: Namespace, generateCalibrateSamples_fn: Callable, model_name: str, device: str):
        self.args = args
        self.generate_calibrate_samples = generateCalibrateSamples_fn
        self.device = device
        self.model_name = model_name
        self.alphas = np.round(np.arange(0.1, 0.51, 0.1), 1)

        # The whole cache process is a little improvised. The cache is cleared from 'run_crc_non_vector_auto.py'
        self.cache_dir = f"../artifacts/non_vector_crc_cache"
        self.is_calibrated = False
        self.cache_version = "v2"

        self.calibration_results = self._load_or_initialize_cache()
    
    def _get_cache_path(self) -> str:
        """Get the cache file path for this model."""
        return os.path.join(self.cache_dir, "calibration_results.json")
    
    def _load_or_initialize_cache(self) -> Dict[str, Any]:
        """Load cache if it exists and cache_version matches, otherwise initialize empty cache."""
        cache_path = self._get_cache_path()
        
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                if cache_data.get("cache_version") == self.cache_version and cache_data.get("alphas") == self.alphas.tolist():
                    logging.info(f"Loaded valid calibration cache for cache_version {self.cache_version}")
                    cache_data["calibrated_alphas"] = {float(k): float(v) for k, v in cache_data["calibrated_alphas"].items()}
                    self.is_calibrated = True
                    return cache_data
                else:
                    logging.info(f"Cache cache_version mismatch (cached: {cache_data.get('cache_version')}, current: {self.cache_version}). Resetting cache.")
        except Exception as e:
            logging.warning(f"Failed to load calibration cache: {e}")
        
        return {
            "cache_version": self.cache_version,
            "model_name": self.model_name,
            "time_stamp": datetime.now().isoformat(),
            "alphas": self.alphas.tolist(),
            "bonferroni_alphas": [],
            "calibrated_alphas": {}
        }
    
    def _save_cache(self):
        """Save calibration results to disk cache."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = self._get_cache_path()
            with open(cache_path, 'w') as f:
                json.dump(self.calibration_results, f, indent=2)
            logging.info(f"Calibration results cached to {cache_path}")
        except Exception as e:
            logging.warning(f"Failed to save calibration cache: {e}")
        
    def get_threshold(self, alpha):
        assert self.is_calibrated, "Model is not calibrated yet"
        return self.calibration_results["calibrated_alphas"][alpha]
    
    def prepare_calibrate(self, save_path: str, db_controller: Any, 
                         calibration_data_path: str, threshold: float = -1) -> Dict[str, Any]:
        """
        Prepare and perform calibration for three hops using bonferroni inequality.
        
        Args:
            save_path: Path to save data
            db_controller: Database controller
            calibration_data_path: Optional path to saved calibration data
            
        Returns:
            Calibration metadata
        """
        if self.is_calibrated:
            logging.info("Already calibrated. Skipping calibration process...")
            return self.calibration_results

        logging.info("Starting prepare_calibrate...")
        cal_scores, true_labels, cal_queries = self.prepare_data(save_path, db_controller, calibration_data_path)
        
        logging.info("Starting CRC optimization...")
        for alpha in self.alphas:
            # alpha levels are updated for a 3 hop union bound
            alpha_three_hop = self.bonferroni_inequality_3_hops(alpha)
            # classic 1 hop CRC optimization on the updated alpha levels
            lamhat = self.optimize(alpha=alpha_three_hop, cal_scores=cal_scores, true_labels=true_labels)
            self.calibration_results["bonferroni_alphas"].append(alpha_three_hop)
            self.calibration_results["calibrated_alphas"][alpha] = lamhat
        
        self.is_calibrated = True
        self._save_cache()
        
        logging.info(f"Calibration for model {self.model_name} done!")
        return self.calibration_results
    

    def optimize(self, alpha, cal_scores, true_labels):
        lamhat = brentq(
            self.lamhat_threshold, 0, 1, args=(cal_scores, true_labels, alpha)
        )
        logging.info(f"Calibrating for alpha {alpha:.2f} done! lamhat={lamhat}")
        return lamhat
    
    def bonferroni_inequality_3_hops(self, alpha):
        """
        Union bound on confidence over three hops means we need to apply
        the cube root of the total confidence to each individual hop.
        Note: really this is model/pipeline specific and should live elsewhere.
        """
        confidence = round(1 - alpha, 4)
        union_bound_confidence = np.cbrt(confidence) # cube root
        return round(1 - union_bound_confidence, 4)
    
    def lamhat_threshold(self, lam, cal_scores, y, alpha):
        n = len(cal_scores)
        fnr = self.false_negative_rate(cal_scores, y, lam)
        return fnr - ((n + 1) / n * alpha - 1 / (n + 1))

    def false_negative_rate(self, scores, y, lam):
        ovrlp = []
        for i in range(len(scores)):
            preds = np.where(scores[i] >= lam)[0]
            gt_labels = set(y[i])
            intersection = len(set(preds) & gt_labels)
            if len(gt_labels) > 0:
                overlap_ratio = 1 - intersection / len(gt_labels)
            else:
                overlap_ratio = 0  # avoid division by zero
            ovrlp.append(overlap_ratio)
        return np.mean(ovrlp)
    
    def prepare_data(self, save_path: str, db_controller: Any, 
                    calibration_data_path: str) -> Any:
        """
        Prepare calibration data with caching support.
        
        Args:
            save_path: Path to save data
            db_controller: Database controller
            calibration_data_path: Optional path to saved calibration data
            
        Returns:
            Any: returns loaded data
        """
        logging.info(f"Preparing to calibrate model {self.model_name}")
        cal_scores, true_labels, cal_queries = self.generate_calibrate_samples(
            save_path=save_path, db_controller=db_controller, 
            calib_iterator=None, calibration_data_path=calibration_data_path, 
            return_queries=True)

        return cal_scores, true_labels, cal_queries
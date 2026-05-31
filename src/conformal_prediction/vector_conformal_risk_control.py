import torch
import logging
import hashlib
import json
from argparse import Namespace
from typing import List, Dict, Any, Optional
from .data_manager import CalibrationDataManager
from .calibration_manager import CalibrationManager
from .utils import validate_model_name
from utils import save_component


class VectorConformalRiskControl:
    """
    Main class for vector conformal risk control.
    
    This class provides a clean interface for conformal risk control,
    separating concerns into data management, calibration, and prediction.
    """

    def __init__(self, args: Namespace, generateCalibrateSamples_fn: callable, 
                 model_name: str, device: str, model: Optional[Any] = None):
        """
        Initialize the vector conformal risk control.
        
        Args:
            args: Configuration arguments
            generateCalibrateSamples_fn: Function to generate calibration samples
            model_name: Name of the model
            device: Device to use
            model: Optional model instance
        """
        self.args = args
        self.generateCalibrateSamples_fn = generateCalibrateSamples_fn
        self.model_name = model_name
        self.device = device
        self.batch_size = getattr(args, 'batch_size', 16)
        self.model = model
        
        # Validate model name
        validate_model_name(model_name)
        
        # Initialize components
        load_path = getattr(args, 'load_path', None)
        self.data_manager = CalibrationDataManager(model_name, load_path)
        
        # Get num_entities from args, model, or default
        num_entities = None
        if hasattr(args, 'num_entities') and args.num_entities is not None:
            num_entities = args.num_entities
        elif model is not None and hasattr(model, 'calibration_generator'):
            num_entities = model.calibration_generator.num_entities
        elif model is not None and hasattr(model, 'args') and hasattr(model.args, 'num_entities'):
            num_entities = model.args.num_entities
        
        self.calibration_manager = CalibrationManager(model_name, num_entities=num_entities)
        
        # Cache configuration
        self._config_hash = self._generate_config_hash()

    def _generate_config_hash(self) -> str:
        """
        Generate a hash of the configuration for caching purposes.
        
        Returns:
            Configuration hash string
        """
        # Create a dictionary of relevant configuration parameters
        config_dict = {
            'model_name': self.model_name,
            'batch_size': getattr(self.args, 'batch_size', 16),
            'device': self.device,
            'max_calibration_queries': getattr(self.args, 'max_calibration_queries', None),
            'calib_size': getattr(self.args, 'calib_size', None),
            'load_path': getattr(self.args, 'load_path', None),
            'dataset': getattr(self.args, 'dataset', None)  # Include dataset for dataset-aware caching
        }
        
        # Convert to JSON string and hash
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def prepare_data(self, save_path: str, db_controller: Any, 
                    calibration_data_path: Optional[str] = None) -> bool:
        """
        Prepare calibration data with caching support.
        
        Args:
            save_path: Path to save data
            db_controller: Database controller
            calibration_data_path: Optional path to saved calibration data
            
        Returns:
            bool: True if data was loaded from cache, False if new data was generated
        """
        logging.info("Attempting to load calibration data from cache...")
        if self.data_manager.load_cached_calibration_data(self._config_hash, "default"):
            logging.info("Successfully loaded calibration data from cache")
            return True
        
        # Generate new calibration data
        if calibration_data_path is not None:
            logging.info(f"Using saved calibration data from: {calibration_data_path}")
            full_calib_data = self.generateCalibrateSamples_fn(
                save_path, db_controller, 
                calib_iterator=None, calibration_data_path=calibration_data_path, 
                return_queries=True
            )
        else:
            logging.info("Generating calibration data from database")
            full_calib_data = self.generateCalibrateSamples_fn(
                save_path, db_controller, None, None, return_queries=True
            )
        
        cal_scores, true_labels, cal_queries = full_calib_data
        self.data_manager.set_calibration_data(cal_scores, true_labels, cal_queries)
        
        return False

    def calibrate(self) -> Dict[str, Any]:
        """
        Perform calibration using the correct FNR metric.
        
        Returns:
            Calibration metadata
        """
        cal_scores, true_labels, _ = self.data_manager.get_calibration_data()
        
        if not self.data_manager.has_calibration_data():
            raise ValueError("No calibration data available. Call prepare_data first.")
        
        return self.calibration_manager.calibrate(cal_scores, true_labels)

    def prepare_calibrate(self, save_path: str, db_controller: Any, 
                         threshold: float = -1, 
                         calibration_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare and perform calibration.
        
        Args:
            save_path: Path to save data
            db_controller: Database controller
            threshold: Threshold value
            calibration_data_path: Optional path to saved calibration data
            
        Returns:
            Calibration metadata
        """
        logging.info("Starting prepare_calibrate...")
        data_loaded_from_cache = self.prepare_data(save_path, db_controller, calibration_data_path)
        
        # Save to cache right before calibration starts (only if we generated new data)
        if not data_loaded_from_cache and self.data_manager.load_path is not None:
            logging.info("Saving calibration data to cache...")
            self.data_manager.save_calibration_data_cache(self._config_hash, "default")
        
        logging.info(f"Preparing to calibrate model {self.model_name}")
        metadata = self.calibrate()
        
        logging.info(f"Calibration for model {self.model_name} done!")
        self.metadata = metadata
        return metadata

    def optimize(self, alpha: float) -> Any:
        """
        Optimize thresholds for a given alpha using the correct FNR metric.
        
        Args:
            alpha: Target false negative rate
            
        Returns:
            Optimized thresholds
        """
        cal_scores, true_labels, _ = self.data_manager.get_calibration_data()
        return self.calibration_manager.optimize_thresholds(cal_scores, true_labels, alpha)

    def predict(
        self,
        x: List,
        confidence: float,
        graph_data: Optional[Any] = None,
        ground_truth_hops: Optional[List] = None,
    ) -> List:
        """
        Make predictions with conformal risk control.
        
        Args:
            x: Input data
            confidence: Confidence level
            graph_data: Optional graph data
            ground_truth_hops: Optional ground truth hops (used for calibration/debug)
            
        Returns:
            List of predictions
        """
        alpha = round(1 - confidence, 4)

        # Get or optimize thresholds
        thresholds = self._get_or_optimize_thresholds(alpha)
        
        # Validate model and graph data
        if self.model is None:
            raise ValueError(f"{self.model_name} model not provided to VectorConformalRiskControl")
        
        if graph_data is None:
            raise ValueError(f"graph_data is required for {self.model_name} prediction but was not provided")
        
        # Use the model's unified prediction method
        # This ensures consistent threshold application for both calibration and inference
        predictions = []
        neo4j_calls_list = []
        ultra_calls_list = []
        
        if ground_truth_hops is None:
            for x_i in x:
                with torch.no_grad():
                    result = self.model.predict_with_thresholds(x_i.unsqueeze(0), thresholds, graph_data, None)
                    pred_list, neo4j_per_query, ultra_per_query = result[0], result[1], result[2]
                    predictions.append(pred_list[0])
                    neo4j_calls_list.append(neo4j_per_query[0] if neo4j_per_query else 0)
                    ultra_calls_list.append(ultra_per_query[0] if ultra_per_query else 0)
        else:
            for x_i, gt in zip(x, ground_truth_hops):
                with torch.no_grad():
                    result = self.model.predict_with_thresholds(x_i.unsqueeze(0), thresholds, graph_data, [gt])
                    pred_list, neo4j_per_query, ultra_per_query = result[0], result[1], result[2]
                    predictions.append(pred_list[0])
                    neo4j_calls_list.append(neo4j_per_query[0] if neo4j_per_query else 0)
                    ultra_calls_list.append(ultra_per_query[0] if ultra_per_query else 0)
        
        metadata = {"neo4j_calls": neo4j_calls_list, "ultra_calls": ultra_calls_list}
        return predictions, metadata
    
    def _get_or_optimize_thresholds(self, alpha: float) -> Any:
        """
        Get existing thresholds or optimize new ones.
        
        Args:
            alpha: Target false negative rate
            
        Returns:
            Threshold values
        """

        if alpha == 0.0:
            # Confidence=1.0 would normally be handled earlier, but keep a sane default.
            # 2-component pipelines (2u, 2p, 2i) get 2 thresholds; all others get 3.
            two_component_models = {"twounionpipeline", "twohoppipeline", "twointersectpipeline"}
            default = [0.45, 0.45] if self.model_name in two_component_models else [0.45, 0.45, 0.45]
            logging.info(f"Alpha is 0.0, returning default thresholds {default}. DEBUG ONLY!")
            return default

        # Try to get from metadata first
        metadata = getattr(self, 'metadata', {})
        calibrated = metadata.get("calibrated_alphas", {})
        
        # Find matching alpha
        match_key = next((k for k in calibrated if abs(float(k) - alpha) < 1e-9), None)
        
        if match_key is not None:
            thresholds = calibrated[match_key]
            if thresholds is not None:
                return thresholds
        
        # Optimize new thresholds
        logging.info(f"Thresholds not found for alpha {alpha}. Optimizing...")
        thresholds = self.optimize(alpha)
        
        # Store in metadata
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        if "calibrated_alphas" not in self.metadata:
            self.metadata["calibrated_alphas"] = {}
        self.metadata["calibrated_alphas"][str(alpha)] = thresholds
        
        return thresholds
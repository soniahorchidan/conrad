"""
Data management for conformal risk control.
"""
import os
import json
import logging
import torch
import pickle
from typing import List, Dict, Any, Optional, Tuple
from .utils import is_vector_model


class CalibrationDataManager:
    """Manages calibration data loading and saving."""
    
    def __init__(self, model_name: str, load_path: Optional[str] = None):
        """
        Initialize the data manager.
        
        Args:
            model_name: Name of the model
            load_path: Path to load calibration data from
        """
        self.model_name = model_name
        self.load_path = load_path
        self.cal_scores = None
        self.true_labels = None
        self.cal_queries = None
        self.metadata = {}
        
        # For vector CRC caching
        self.is_vector_model = is_vector_model(model_name)

    def set_calibration_data(self, cal_scores: List, true_labels: List, cal_queries: List) -> None:
        """
        Set calibration data from external source.
        
        Args:
            cal_scores: Calibration scores
            true_labels: True labels
            cal_queries: Calibration queries
        """
        self.cal_scores = cal_scores
        self.true_labels = true_labels
        self.cal_queries = cal_queries

    def get_calibration_data(self) -> Tuple[List, List, List]:
        """
        Get current calibration data.
        
        Returns:
            Tuple of (cal_scores, true_labels, cal_queries)
        """
        return self.cal_scores, self.true_labels, self.cal_queries

    def has_calibration_data(self) -> bool:
        """
        Check if calibration data is available.
        
        Returns:
            True if calibration data is available
        """
        return self.cal_scores is not None and self.true_labels is not None

    def _generate_cache_key(self, config_hash: str, cache_suffix: str = "default") -> str:
        """
        Generate a cache key for the calibration data.
        
        Args:
            config_hash: Hash of the configuration
            cache_suffix: Suffix for the cache key
            
        Returns:
            Cache key string
        """
        key_components = [
            self.model_name,
            config_hash,
            cache_suffix
        ]
        return "_".join(key_components)

    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the cache file path for the given key.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        if self.load_path is None:
            raise ValueError("Load path must be set to use caching")
        
        cache_dir = os.path.join(self.load_path, "vector_crc_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{cache_key}.pkl")

    def load_cached_calibration_data(self, config_hash: str, cache_suffix: str = "default") -> bool:
        """
        Load cached calibration data if available.
        
        Args:
            config_hash: Hash of the configuration
            cache_suffix: Suffix for the cache key
            
        Returns:
            True if cached data was loaded successfully
        """
        if not self.is_vector_model or self.load_path is None:
            return False
            
        cache_key = self._generate_cache_key(config_hash, cache_suffix)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            logging.info(f"No cached calibration data found at {cache_path}")
            return False
        
        try:
            logging.info(f"Loading cached calibration data from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Validate cached data
            if self._validate_cached_data(cached_data):
                self.cal_scores = cached_data['cal_scores']
                self.true_labels = cached_data['true_labels']
                self.cal_queries = cached_data.get('cal_queries', None)
                logging.info(f"Successfully loaded cached calibration data: {len(self.cal_scores)} samples")
                return True
            else:
                logging.warning("Cached calibration data validation failed")
                return False
                
        except Exception as e:
            logging.warning(f"Failed to load cached calibration data: {e}")
            return False

    def save_calibration_data_cache(self, config_hash: str, cache_suffix: str = "default") -> None:
        """
        Save calibration data to cache.
        
        Args:
            config_hash: Hash of the configuration
            cache_suffix: Suffix for the cache key
        """
        if not self.is_vector_model or self.load_path is None:
            return
            
        if not self.has_calibration_data():
            logging.warning("No calibration data to cache")
            return
            
        cache_key = self._generate_cache_key(config_hash, cache_suffix)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'cal_scores': self.cal_scores,
                'true_labels': self.true_labels,
                'cal_queries': self.cal_queries,
                'config_hash': config_hash,
                'cache_suffix': cache_suffix,
                'model_name': self.model_name
            }
            
            logging.info(f"Saving calibration data cache to {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logging.info(f"Successfully cached {len(self.cal_scores)} calibration samples")
            
        except Exception as e:
            logging.error(f"Failed to save calibration data cache: {e}")

    def _validate_cached_data(self, cached_data: Dict[str, Any]) -> bool:
        """
        Validate cached calibration data.
        
        Args:
            cached_data: Cached data dictionary
            
        Returns:
            True if data is valid
        """
        required_keys = ['cal_scores', 'true_labels', 'model_name']
        
        for key in required_keys:
            if key not in cached_data:
                logging.warning(f"Missing required key in cached data: {key}")
                return False
        
        if cached_data['model_name'] != self.model_name:
            logging.warning(f"Model name mismatch: cached={cached_data['model_name']}, current={self.model_name}")
            return False
        
        # Validate data shapes for vector models
        if self.is_vector_model:
            cal_scores = cached_data['cal_scores']
            if not isinstance(cal_scores, (list, torch.Tensor)):
                logging.warning("Invalid calibration scores type")
                return False
            
            if isinstance(cal_scores, list) and len(cal_scores) > 0:
                first_score = cal_scores[0]
                
                # Check for new path-aware format (dict with various key patterns)
                if isinstance(first_score, dict):
                    # Support different query types: 3p (hop1/hop2/hop3), 2u (branch1/branch2), 2ip (branch1/branch2/intersection/projection)
                    if "hop1" in first_score:
                        required_keys = ["hop1", "hop2", "hop3"]
                    elif "branch1" in first_score:
                        # Check if it's 2ip (has intersection/projection) or 2u (just branch1/branch2)
                        if "intersection" in first_score or "projection" in first_score:
                            # 2ip format: should have branch1, branch2, intersection, projection
                            required_keys = ["branch1", "branch2", "intersection", "projection"]
                        else:
                            # 2u format: just branch1 and branch2
                            required_keys = ["branch1", "branch2"]
                    else:
                        logging.warning(f"Unrecognized calibration dict keys: {list(first_score.keys())}")
                        return False
                    
                    if not all(key in first_score for key in required_keys):
                        logging.warning(f"Invalid path-aware calibration data: missing required keys {required_keys}, found {list(first_score.keys())}")
                        return False
                    logging.info(f"Validated path-aware calibration cache format with keys: {required_keys}")
                
                # Check for old format (list of tensors with shape [3, num_entities])i
                elif isinstance(first_score, torch.Tensor):
                    if first_score.shape[0] != 3:
                        logging.warning(f"Invalid calibration score shape: {first_score.shape}")
                        return False
                    logging.info(f"Validated legacy tensor calibration cache format")
                
                else:
                    logging.warning(f"Invalid calibration score format: {type(first_score)}")
                    return False
        
        return True

"""
Model configuration and setup utilities.
"""
from typing import Dict, Tuple, Type, Any
from .auxiliary_models import (
    UltraCalibProbModelHandler,
    UltraCalibProbDatasetVal,
    get_x_ultra,
    DBExecCalibProbModelHandler,
    DBExecCalibProbDatasetVal,
    get_x_dbexec,
    TwoUnionPipelineCalibProbDatasetVal,
    get_x_twounion,
    TwoIntersectProjectPipelineCalibProbDatasetVal,
    get_x_twointersectproject,
)
from .auxiliary_models.threehoppipeline import (
    ThreeHopCalibProbDatasetVal,
    get_x as get_x_threehop,
)


class ModelConfig:
    """Configuration for different model types."""
    
    # Model configuration mapping
    MODEL_CONFIGS: Dict[str, Tuple[Type, Any, Type]] = {
        "ultra": (UltraCalibProbDatasetVal, get_x_ultra, UltraCalibProbModelHandler),
        "dbexecmodel": (DBExecCalibProbDatasetVal, get_x_dbexec, DBExecCalibProbModelHandler),
        "multihoppredictor": (UltraCalibProbDatasetVal, get_x_ultra, DBExecCalibProbModelHandler),
        "threehoppipeline": (ThreeHopCalibProbDatasetVal, get_x_threehop, DBExecCalibProbModelHandler),
        "twounionpipeline": (TwoUnionPipelineCalibProbDatasetVal, get_x_twounion, DBExecCalibProbModelHandler),
        "twointersectprojectpipeline": (TwoIntersectProjectPipelineCalibProbDatasetVal, get_x_twointersectproject, DBExecCalibProbModelHandler),
    }
    
    @classmethod
    def get_model_components(cls, model_name: str) -> Tuple[Type, Any, Type]:
        """
        Get model components for a given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tuple of (dataset_class, get_x_fn, handler_class)
            
        Raises:
            ValueError: If model name is not supported
        """
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not supported")
        
        return cls.MODEL_CONFIGS[model_name]
    
    @classmethod
    def is_vector_model(cls, model_name: str) -> bool:
        """
        Check if a model uses vector conformal risk control.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model uses vector conformal risk control
        """
        # Normalize to be robust to caller casing / separators.
        model_key = model_name.lower().replace("_", "").replace("-", "")
        return model_key in {"threehoppipeline", "twounionpipeline", "twointersectprojectpipeline"}
    
    @classmethod
    def get_supported_models(cls) -> list:
        """
        Get list of supported model names.
        
        Returns:
            List of supported model names
        """
        return list(cls.MODEL_CONFIGS.keys())

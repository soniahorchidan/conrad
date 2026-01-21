"""
Model configuration and setup utilities.
"""


class ModelConfig:
    """Configuration for different model types."""
    
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

import torch
import gc
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

class PatchedUNetSpatioTemporalConditionModel(UNetSpatioTemporalConditionModel):
    """
    Patched version of UNetSpatioTemporalConditionModel that supports device_map="auto"
    by implementing the _no_split_modules attribute.
    """
    # Define modules that should not be split across devices
    _no_split_modules = ["CrossAttnDownBlock3D", "DownBlock3D", "CrossAttnUpBlock3D", "UpBlock3D", "UNetMidBlock3DCrossAttn"]
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        """Load model from pretrained weights"""
        return super().from_pretrained(pretrained_model_path, **kwargs)

class DiffusersUNetSpatioTemporalConditionModelDepthCrafter(PatchedUNetSpatioTemporalConditionModel):
    """
    Patched version of the DepthCrafter UNet model that supports device_map="auto"
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        """Load model from pretrained weights with support for device_map='auto'"""
        # Handle device_map="auto" safely
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        torch_dtype = kwargs.get("torch_dtype", torch.float16)
        
        # Clean up memory before loading
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            # First try to load with device_map directly
            if device_map == "auto":
                logger.info(f"Loading model with device_map={device_map} and max_memory={max_memory}")
                
                # Put device_map back for loading
                kwargs["device_map"] = device_map
                if max_memory is not None:
                    kwargs["max_memory"] = max_memory
                
                # Load with device_map
                model = super().from_pretrained(pretrained_model_path, **kwargs)
                logger.info("Successfully loaded model with device_map")
                return model
                
        except Exception as e:
            logger.warning(f"Failed to load with device_map={device_map}: {e}")
            logger.info("Falling back to manual device mapping")
            
            # Clean up after failed attempt
            gc.collect()
            torch.cuda.empty_cache()
            
            # Remove device_map from kwargs if it was added back
            kwargs.pop("device_map", None)
            kwargs.pop("max_memory", None)
        
        # Load without device_map
        logger.info("Loading model without device_map")
        model = super().from_pretrained(pretrained_model_path, **kwargs)
        
        # Then apply device_map manually if specified
        if device_map == "auto" and max_memory is not None:
            try:
                from accelerate import dispatch_model, infer_auto_device_map
                
                # Infer device map first
                device_map = infer_auto_device_map(
                    model, 
                    max_memory=max_memory,
                    dtype=torch_dtype,
                    no_split_module_classes=cls._no_split_modules
                )
                
                # Apply device map after loading
                model = dispatch_model(model, device_map=device_map)
                logger.info(f"Successfully applied manual device_map to UNet model")
            except Exception as e:
                logger.warning(f"Warning: Could not apply manual device_map: {e}")
                logger.info("Model will be loaded on a single device")
                
                # Move to CUDA if available
                if torch.cuda.is_available():
                    model = model.to("cuda")
        
        return model
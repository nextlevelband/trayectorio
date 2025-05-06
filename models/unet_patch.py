import torch
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

class PatchedUNetSpatioTemporalConditionModel(UNetSpatioTemporalConditionModel):
    """
    Patched version of UNetSpatioTemporalConditionModel that supports device_map="auto"
    by implementing the _no_split_modules attribute.
    """
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
        
        # First load without device_map
        model = super().from_pretrained(pretrained_model_path, **kwargs)
        
        # Then apply device_map if specified
        if device_map == "auto" and max_memory is not None:
            try:
                from accelerate import dispatch_model
                
                # Apply device map after loading
                model = dispatch_model(model, device_map=device_map, max_memory=max_memory)
                print(f"Successfully applied device_map={device_map} to UNet model")
            except Exception as e:
                print(f"Warning: Could not apply device_map={device_map}: {e}")
                print("Model will be loaded on a single device")
        
        return model
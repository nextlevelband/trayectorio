import gc
import os
import numpy as np
import torch
import warnings

# Suppress FutureWarning from xformers
warnings.filterwarnings("ignore", category=FutureWarning)

from diffusers.training_utils import set_seed
# from models.depth_crafter_ppl import DepthCrafterPipeline
# from models.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

# Try to use our patched UNet model first
try:
    from models.unet_patch import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
    print("Using patched UNet model with device_map support")
    from DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
except ImportError:
    # Fallback to original implementation
    print("Using original DepthCrafter implementation")
    from DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
    from DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "sequential",  # Changed default to sequential for maximum memory savings
        device: str = "cuda:0",
    ):
        # Clear memory before loading models
        gc.collect()
        torch.cuda.empty_cache()
        
        # Set device map for efficient memory usage
        device_map = {
            "unet": "auto",
            "vae": "auto",
            "scheduler": "cpu"
        }
        
        # Estimate available memory
        total_mem = torch.cuda.get_device_properties(0).total_memory
        available_mem = total_mem - 2 * 1024 * 1024 * 1024  # Reserve 2GB
        
        # Load UNet with optimizations using our patched model with device_map support
        try:
            # Estimate max memory allocation for RTX 4070 Ti Super
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory
                # Reserve 2GB for system
                available_mem = total_mem - 2 * 1024 * 1024 * 1024
                gpu_mem = min(available_mem, 12 * 1024 * 1024 * 1024)  # Cap at 12GB
                
                # Set memory allocation
                max_memory = {
                    "0": f"{gpu_mem // (1024 * 1024 * 1024)}GiB",  # Convert to GB
                    "cpu": "16GiB"  # Use CPU memory as needed
                }
                print(f"Auto-configured memory allocation: {max_memory}")
            else:
                max_memory = None
            
            # Check if we're using the patched model by checking its module path
            module_path = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.__module__
            using_patched_model = 'unet_patch' in module_path
            
            if using_patched_model:
                print(f"Using patched UNet with device_map support from {module_path}")
                unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                    unet_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    device_map="auto" if cpu_offload == "sequential" else None,
                    max_memory=max_memory if cpu_offload == "sequential" else None,
                )
            else:
                # If we're using the original model, don't use device_map
                print(f"Using original UNet without device_map support from {module_path}")
                unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                    unet_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                )
                
                # Move to appropriate device after loading
                if cpu_offload == "full":
                    unet = unet.to("cpu")
                else:
                    unet = unet.to(device)
            
            # Enable memory optimizations
            if hasattr(unet, "enable_gradient_checkpointing"):
                unet.enable_gradient_checkpointing()
                
        except Exception as e:
            print(f"Error loading UNet with optimizations: {e}")
            print("Falling back to basic loading...")
            
            # Fallback to basic loading without device_map
            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                unet_path,
                torch_dtype=torch.float16,
            ).to(device)
        
        # Enable gradient checkpointing for UNet if available
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
        
        # Memory allocation is now handled in the UNet loading section
            
        # Load pipeline with optimizations
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )

        # Apply memory optimization techniques
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will be slower but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to(device)
            
        # Enable attention slicing with auto slice size
        self.pipe.enable_attention_slicing(slice_size="auto")
        
        # Enable VAE slicing if available
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
            
        # Enable tiled processing if available
        if hasattr(self.pipe, "enable_tiled_processing"):
            self.pipe.enable_tiled_processing()
            
        # Enable xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention for DepthCrafter")
        except Exception as e:
            print(e)
            print("Xformers is not enabled for DepthCrafter")
            
        # Clear memory after setup
        gc.collect()
        torch.cuda.empty_cache()

    def infer(
        self,
        frames,
        near,
        far,
        num_denoising_steps: int,
        guidance_scale: float,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = True,
        batch_size: int = 8,  # Process frames in smaller batches to save memory
    ):
        set_seed(seed)
        
        # Clear memory before processing
        gc.collect()
        torch.cuda.empty_cache()
        
        # Get total number of frames
        total_frames = len(frames)
        
        # Process frames in batches to reduce memory usage
        results = []
        
        # Calculate optimal batch size based on frame count
        # For very large videos, use smaller batches
        if total_frames > 100:
            batch_size = min(batch_size, 4)
        elif total_frames > 50:
            batch_size = min(batch_size, 8)
            
        print(f"Processing {total_frames} frames in batches of {batch_size}")
        
        for i in range(0, total_frames, batch_size):
            # Get current batch
            batch_end = min(i + batch_size, total_frames)
            current_batch = frames[i:batch_end]
            
            print(f"Processing frames {i} to {batch_end-1}")
            
            # Process batch with memory-efficient settings
            with torch.inference_mode():
                # Use a smaller window size for large frames to save memory
                adaptive_window = window_size
                if current_batch.shape[1] * current_batch.shape[2] > 640 * 480:
                    adaptive_window = min(window_size, 64)
                    
                # Use fewer inference steps for intermediate batches to save time and memory
                adaptive_steps = num_denoising_steps
                if i > 0 and batch_end < total_frames:
                    adaptive_steps = max(3, num_denoising_steps // 2)
                
                # Process the batch
                batch_result = self.pipe(
                    current_batch,
                    height=current_batch.shape[1],
                    width=current_batch.shape[2],
                    output_type="np",
                    guidance_scale=guidance_scale,
                    num_inference_steps=adaptive_steps,
                    window_size=adaptive_window,
                    overlap=overlap,
                    track_time=track_time,
                ).frames[0]
                
                results.append(batch_result)
                
            # Clear memory after each batch
            gc.collect()
            torch.cuda.empty_cache()
        
        # Combine results from all batches
        if len(results) > 1:
            res = np.concatenate(results, axis=0)
        else:
            res = results[0]
            
        # Convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        
        # Normalize the depth map to [0, 1] across the whole video
        depths = (res - res.min()) / (res.max() - res.min())
        
        # Convert to tensor and format
        depths = torch.from_numpy(depths).unsqueeze(1)  # 49 576 1024 ->
        depths *= 3900  # compatible with da output
        depths[depths < 1e-5] = 1e-5
        depths = 10000.0 / depths
        depths = depths.clip(near, far)
        
        # Clear memory after processing
        gc.collect()
        torch.cuda.empty_cache()

        return depths

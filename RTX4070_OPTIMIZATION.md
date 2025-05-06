# TrajectoryCrafter Optimizations for RTX 4070 Ti Super

This document outlines the optimizations made to the TrajectoryCrafter codebase to ensure it runs efficiently on an RTX 4070 Ti Super GPU without encountering CUDA out of memory errors.

## Optimizations Implemented

### 1. Memory-Efficient Model Loading
- Enabled sequential CPU offloading to move model components to CPU when not in use
- Implemented device mapping to distribute model components across GPU and CPU
- Added low CPU memory usage options for more efficient model loading

### 2. Attention Optimizations
- Enabled attention slicing to process attention operations in smaller chunks
- Added support for xformers memory-efficient attention when available
- Implemented gradient checkpointing for transformer models

### 3. VAE Optimizations
- Implemented VAE slicing for encoding and decoding operations
- Process frames in smaller batches to reduce peak memory usage
- Added tiled processing for large images

### 4. Batch Processing
- Implemented chunked processing for video frames
- Added adaptive batch sizes based on video length and resolution
- Reduced inference steps for intermediate frames in batch processing

### 5. Memory Management
- Added strategic memory cleanup with `gc.collect()` and `torch.cuda.empty_cache()`
- Implemented tensor offloading to CPU after processing
- Optimized tensor operations to reduce memory footprint

### 6. Resolution and Quality Adjustments
- Reduced default resolution to 384x576 for processing
- Implemented adaptive window sizes for depth estimation based on frame size
- Reduced number of inference steps for faster processing with lower memory usage

## Usage

The optimizations are enabled by default in the updated scripts. The key parameters that control memory usage are:

```
--low_gpu_memory_mode True       # Enable low GPU memory mode
--enable_attention_slicing True  # Enable attention slicing
--enable_vae_slicing True        # Enable VAE slicing
--enable_tiled_processing True   # Enable tiled processing
--cpu_offload 'sequential'       # Use sequential CPU offloading
--sample_size 384 576            # Use reduced resolution
```

These settings can be adjusted in the `run.sh` script or passed as command-line arguments to `inference.py`.

## Performance Impact

These optimizations allow the model to run on GPUs with limited VRAM like the RTX 4070 Ti Super without encountering out-of-memory errors. The trade-offs are:

1. Slightly slower processing due to CPU offloading and chunked processing
2. Slightly reduced output quality due to lower resolution and fewer inference steps
3. Higher CPU memory usage due to model component offloading

For higher quality results (if memory allows), you can increase the resolution and number of inference steps:

```
--sample_size 512 768            # Higher resolution
--diffusion_inference_steps 50   # More inference steps
--depth_inference_steps 5        # More depth inference steps
```

## Additional Notes

- The code will automatically adapt to the available GPU memory
- For videos longer than 50 frames, smaller batch sizes are automatically used
- For very high-resolution videos, adaptive window sizes are used for depth estimation
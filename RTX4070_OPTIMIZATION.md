# TrajectoryCrafter Optimizations for RTX 4070 Ti Super

This document outlines the optimizations made to the TrajectoryCrafter codebase to ensure it runs efficiently on an RTX 4070 Ti Super GPU without encountering CUDA out of memory errors.

## Optimizations Implemented

### 1. Memory-Efficient Model Loading
- Enabled sequential CPU offloading to move model components to CPU when not in use
- Implemented device mapping to distribute model components across GPU and CPU
- Added low CPU memory usage options for more efficient model loading
- Created patched UNet model with proper support for `device_map="auto"`
- Added fallback mechanisms for model loading with error handling

### 2. Attention Optimizations
- Enabled attention slicing to process attention operations in smaller chunks
- Added support for xformers memory-efficient attention when available
- Implemented gradient checkpointing for transformer models
- Added warning suppression for xformers FutureWarning messages

### 3. VAE Optimizations
- Implemented VAE slicing for encoding and decoding operations
- Process frames in smaller batches to reduce peak memory usage
- Added tiled processing for large images
- Enhanced decode_latents method with chunked processing for memory-efficient decoding

### 4. Batch Processing
- Implemented chunked processing for video frames
- Added adaptive batch sizes based on video length and resolution
- Reduced inference steps for intermediate frames in batch processing
- Optimized prepare_latents method with chunked processing for large tensors
- Optimized prepare_mask_latents method with memory-efficient VAE slicing

### 5. Memory Management
- Added strategic memory cleanup with `gc.collect()` and `torch.cuda.empty_cache()`
- Implemented tensor offloading to CPU after processing
- Optimized tensor operations to reduce memory footprint
- Added auto-configuration of memory allocation based on available GPU memory
- Enhanced infer.py with intelligent GPU memory detection and allocation

### 6. Resolution and Quality Adjustments
- Reduced default resolution from 384x672 to 384x576 for processing
- Implemented adaptive window sizes for depth estimation based on frame size
- Reduced default diffusion inference steps from 50 to 25
- Updated run.sh with optimized settings for all camera modes

## Usage

The optimizations are enabled by default in the updated scripts. The key parameters that control memory usage are:

```
--low_gpu_memory_mode True       # Enable low GPU memory mode
--enable_attention_slicing True  # Enable attention slicing
--enable_vae_slicing True        # Enable VAE slicing
--enable_tiled_processing True   # Enable tiled processing
--cpu_offload sequential         # Use sequential CPU offloading
--sample_size 384 576            # Use reduced resolution
--diffusion_inference_steps 25   # Use fewer inference steps
```

These settings can be adjusted in the `run.sh` script or passed as command-line arguments to `inference.py`.

## UNet Patching for device_map Support

One of the key optimizations is the patched UNet model that properly supports `device_map="auto"`. This allows the model to be automatically distributed across GPU and CPU memory, preventing CUDA out of memory errors.

The patched model is implemented in `models/unet_patch.py` and includes:

1. Proper implementation of `_no_split_modules` attribute
2. Enhanced `from_pretrained` method with fallback mechanisms
3. Automatic memory allocation based on available GPU memory
4. Strategic memory cleanup during model loading

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

## Adaptive Processing

The optimized code includes several adaptive features:

1. **Automatic Memory Configuration**: The code detects available GPU memory and configures memory allocation accordingly
2. **Adaptive Batch Sizes**: For longer videos, smaller batch sizes are automatically used
3. **Adaptive Window Sizes**: For high-resolution videos, adaptive window sizes are used for depth estimation
4. **Chunked Processing**: Large tensors are processed in chunks to reduce peak memory usage
5. **Fallback Mechanisms**: If optimized loading fails, the code falls back to simpler methods

## Troubleshooting

If you still encounter CUDA out of memory errors:

1. Further reduce the resolution with `--sample_size 320 512`
2. Reduce inference steps with `--diffusion_inference_steps 20`
3. Process videos in smaller segments
4. Enable full CPU offloading with `--cpu_offload full`
5. Increase available system memory for CPU offloading
# gradual mode with memory optimizations for RTX 4070 Ti Super
python inference.py \
    --video_path './test/videos/p7.mp4' \
    --stride 2 \
    --out_dir experiments \
    --radius_scale 1 \
    --camera 'target' \
    --mode 'gradual' \
    --mask \
    --target_pose 0 -30 0.3 0 0 \
    --traj_txt 'test/trajs/loop2.txt' \
    --low_gpu_memory_mode True \
    --enable_attention_slicing True \
    --enable_vae_slicing True \
    --enable_tiled_processing True \
    --cpu_offload 'sequential' \
    --sample_size 384 576 \
    --diffusion_inference_steps 25 \

# # direct mode with memory optimizations for RTX 4070 Ti Super
# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'direct' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \
#     --low_gpu_memory_mode True \
#     --enable_attention_slicing True \
#     --enable_vae_slicing True \
#     --enable_tiled_processing True \
#     --cpu_offload 'sequential' \
#     --sample_size 384 576 \
#     --diffusion_inference_steps 25 \

# # bullet time with memory optimizations for RTX 4070 Ti Super
# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'bullet' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \
#     --low_gpu_memory_mode True \
#     --enable_attention_slicing True \
#     --enable_vae_slicing True \
#     --enable_tiled_processing True \
#     --cpu_offload 'sequential' \
#     --sample_size 384 576 \
#     --diffusion_inference_steps 25 \

# # dolly-zoom mode with memory optimizations for RTX 4070 Ti Super
# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'zoom' \
#     --mask \
#     --target_pose 0 0 0.5 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \
#     --low_gpu_memory_mode True \
#     --enable_attention_slicing True \
#     --enable_vae_slicing True \
#     --enable_tiled_processing True \
#     --cpu_offload 'sequential' \
#     --sample_size 384 576 \
#     --diffusion_inference_steps 25 \
# LingBot-World Driving Demo — Progress Report

**Date:** February 10, 2026
**Hardware:** 1x NVIDIA H100 80GB HBM3 (Lambda Cloud)
**Model:** lingbot-world-base-cam (Wan2.2 14B DiT, MoE with two 14B experts)

## What We Did

### Setup
1. Cloned the veristudio repo on a Lambda H100 instance
2. Cloned lingbot-world (Wan2.2-based world model with camera pose injection)
3. Downloaded the base-cam model weights (~75GB) from HuggingFace
4. Installed dependencies: PyTorch, flash_attn, diffusers, transformers, scipy, etc.
5. Fixed version conflicts (numpy<2, transformers<=4.51.3, filelock upgrade)

### Input
- **Starting image:** A single dashcam frame from nuScenes — urban intersection with brick buildings, traffic lights, pedestrians, and a black car ahead
- **Prompt:** "Dashcam footage driving through a city intersection during daytime. Buildings, traffic lights, cars on the road."
- **Route:** "short" — 2 chunks of forward motion
- **Camera poses:** Generated programmatically as 4x4 camera-to-world transformation matrices (17 frames per chunk, moving straight forward)

### Generation
- **Resolution:** 480x832
- **Frames per chunk:** 17 (~1 second at 16 fps)
- **Denoising steps:** 30 (UniPC solver)
- **Model offloading:** Enabled (swaps MoE experts between CPU/GPU)
- **Chunk 1:** 118.6 seconds
- **Chunk 2:** 120.0 seconds
- **Total output:** 33 frames, 2.1 seconds at 16 fps
- **Total generation time:** ~4 minutes for 2 seconds of video

### How It Works (Under the Hood)
1. The starting dashcam image is encoded through a 3D Video VAE (stride 4,8,8) into latent space
2. WASD inputs are converted to camera-to-world matrices → Plücker ray embeddings (origin + direction per pixel)
3. Plücker embeddings are injected into every transformer block via Adaptive Layer Normalization (AdaLN):
   `x = (1.0 + cam_scale) * x + cam_shift`
4. The DiT denoises latent video conditioned on the text prompt, starting image, and camera motion
5. MoE expert selection: high-noise expert (t ≥ 0.947) handles coarse structure, low-noise expert (t < 0.947) handles fine details
6. The VAE decodes latents back to pixel-space video

## Observations

### What Worked Well
- The model successfully generated a plausible forward-driving continuation from a single dashcam image
- Scene consistency: buildings, road markings, and sky remain coherent across frames
- The camera motion follows the specified forward trajectory
- Quality is reasonable at 480p — not photorealistic but clearly recognizable as a driving scene
- No training or fine-tuning was required — the base-cam model works out of the box with arbitrary starting images

### Limitations Observed
- **Latency:** ~2 minutes per 1-second chunk makes real-time interaction impossible
- **Autoregressive drift:** Chaining chunks (using last frame as next input) may accumulate artifacts over many steps
- **Single image spawn:** The model hallucinates the 3D world from one photo — it doesn't have ground truth geometry, so parallax and occlusion may be inconsistent
- **Not a persistent world:** Each generation is a plausible future, not a stored 3D environment. Revisiting the same location may look different.

## Architecture Summary

```
Input: 1 dashcam image + text prompt + camera poses
  ↓
T5 Text Encoder (UMT5-XXL, on CPU) → text embeddings
3D Video VAE Encoder (stride 4,8,8) → image latent
Pose → Plücker Embeddings (6D per pixel per frame)
  ↓
DiT Backbone (5120-dim, 40 heads, 40 layers)
  - MoE: high-noise expert + low-noise expert (14B each)
  - Camera injection via AdaLN in every block
  - 30 denoising steps (flow matching, UniPC solver)
  ↓
3D Video VAE Decoder → 17 frames at 480x832
  ↓
Output: 1-second driving video (.mp4)
```

## Next Steps
- [ ] Build Gradio web UI for interactive browser-based steering
- [ ] Run longer routes (city_cruise: 12 chunks with turns)
- [ ] Experiment with different starting images and prompts
- [ ] Explore reducing latency (fewer denoising steps, smaller resolution)
- [ ] LoRA fine-tuning on nuScenes for driving-specific quality improvements

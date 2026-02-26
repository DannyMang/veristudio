"""
create_dummy_data.py -- Generate dummy encoded .pt files for dry-run testing.

Creates fake samples matching the PostTrainingDataset schema so you can test
the training pipeline end-to-end without real data or VAE/T5 encoding.

Each .pt file contains random tensors with the correct shapes and dtypes:
    latents:    bf16 (16, T_lat, H_lat, W_lat)
    context:    bf16 (512, 4096)
    c2ws:       fp32 (T_vid, 4, 4)
    intrinsics: fp32 (T_vid, 4)
    y:          bf16 (20, T_lat, H_lat, W_lat)  # 4-ch mask + 16-ch VAE cond

Usage:
    python create_dummy_data.py --output_dir /path/to/encoded --num_samples 20
"""

import argparse
import math
import os

import torch


def create_dummy_sample(
    num_chunks=4,
    chunk_lat_frames=5,
    height=480,
    width=832,
    vae_stride_t=4,
    vae_stride_h=8,
    vae_stride_w=8,
    latent_channels=16,
    text_len=512,
    text_dim=4096,
):
    """Create one dummy encoded sample.

    Returns:
        dict matching the PostTrainingDataset .pt file schema.
    """
    T_lat = num_chunks * chunk_lat_frames
    # Compute spatial dims matching image2video.py / data_pipeline.py formula
    # (ensures patch-aligned dimensions consistent with model's patch_size)
    patch_h, patch_w = 2, 2  # model patch_size (1, 2, 2)
    aspect_ratio = height / width
    max_area = height * width
    H_lat = round(
        math.sqrt(max_area * aspect_ratio) // vae_stride_h // patch_h * patch_h
    )
    W_lat = round(
        math.sqrt(max_area / aspect_ratio) // vae_stride_w // patch_w * patch_w
    )
    T_vid = T_lat * vae_stride_t

    latents = torch.randn(
        latent_channels, T_lat, H_lat, W_lat, dtype=torch.bfloat16
    )
    context = torch.randn(text_len, text_dim, dtype=torch.bfloat16)

    # Camera poses: identity with small forward translation (simulates driving)
    c2ws = torch.zeros(T_vid, 4, 4, dtype=torch.float32)
    for i in range(T_vid):
        c2ws[i] = torch.eye(4)
        c2ws[i, 2, 3] = i * 0.1  # 0.1 m per frame forward motion

    # Typical dashcam intrinsics
    fx, fy = 800.0, 800.0
    cx, cy = width / 2.0, height / 2.0
    intrinsics = torch.tensor(
        [[fx, fy, cx, cy]] * T_vid, dtype=torch.float32
    )

    # y = mask (4 ch) + VAE-encoded first-frame conditioning (16 ch) = 20 ch
    # The 4-channel mask matches VAE temporal stride: each latent frame
    # corresponds to 4 video frames. First latent frame is all-ones (conditioned),
    # rest are all-zeros.
    mask_channels = 4
    y = torch.randn(
        latent_channels + mask_channels, T_lat, H_lat, W_lat, dtype=torch.bfloat16
    )
    y[:mask_channels] = 0.0
    y[:mask_channels, 0] = 1.0  # mask: first latent frame = 1, rest = 0

    return {
        "latents": latents,
        "context": context,
        "c2ws": c2ws,
        "intrinsics": intrinsics,
        "y": y,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy encoded .pt files for dry-run testing"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for .pt files"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of dummy samples"
    )
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--chunk_lat_frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} dummy samples...")
    print(f"  Resolution: {args.height}x{args.width}")
    print(
        f"  Chunks: {args.num_chunks} x {args.chunk_lat_frames} lat frames "
        f"= {args.num_chunks * args.chunk_lat_frames} total lat frames"
    )

    for i in range(args.num_samples):
        sample = create_dummy_sample(
            num_chunks=args.num_chunks,
            chunk_lat_frames=args.chunk_lat_frames,
            height=args.height,
            width=args.width,
        )
        path = os.path.join(args.output_dir, f"{i:06d}.pt")
        torch.save(sample, path)

    # Print shapes for verification
    sample = torch.load(
        os.path.join(args.output_dir, "000000.pt"), weights_only=False
    )
    print("\nSample tensor shapes:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")

    print(f"\nDone! Saved {args.num_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()

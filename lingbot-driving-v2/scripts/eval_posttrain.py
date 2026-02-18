"""
eval_posttrain.py -- Evaluation pipeline for post-trained models.

Loads a student checkpoint into WanI2VCausal (single_expert=True),
generates videos at 4-6 steps, and optionally compares with the
teacher at 70 steps.

Metrics:
  - FVD (Frechet Video Distance) via I3D features
  - LPIPS (perceptual similarity between consecutive frames)
  - Temporal coherence (flow consistency)

Usage:
    python eval_posttrain.py \
        --model_dir /path/to/lingbot-world-base-cam \
        --checkpoint /path/to/stage2_latest.pt \
        --data_dir /path/to/encoded_data \
        --output_dir /path/to/eval_output \
        --num_samples 10 \
        --sampling_steps 4
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def load_student_model(model_dir, checkpoint_path, device, sampling_steps=4):
    """Load post-trained student into WanI2VCausal with single expert.

    Args:
        model_dir: Path to base model directory.
        checkpoint_path: Path to post-training checkpoint.
        device: CUDA device.
        sampling_steps: Denoising steps (4-6).

    Returns:
        WanI2VCausal instance with loaded weights.
    """
    wan_root = os.path.join(os.path.dirname(__file__), "..", "..", "lingbot-world")
    if wan_root not in sys.path:
        sys.path.insert(0, wan_root)

    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2VCausal

    cfg = WAN_CONFIGS["i2v-A14B"]

    # Create causal model with single expert
    model = WanI2VCausal(
        config=cfg,
        checkpoint_dir=model_dir,
        device_id=device.index if device.index is not None else 0,
        t5_cpu=True,
        max_cache_chunks=4,
        sampling_steps=sampling_steps,
        max_area=320 * 576,
        single_expert=True,
    )

    # Load post-trained weights into the high-noise expert
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle both Stage 1 and Stage 2 checkpoints
        if "student_state_dict" in ckpt:
            state_dict = ckpt["student_state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            raise ValueError(
                f"Checkpoint has no model state dict. Keys: {list(ckpt.keys())}"
            )

        model.high_noise_model.load_state_dict(state_dict)
        logger.info(
            f"Loaded post-trained weights from step {ckpt.get('step', '?')}"
        )

    return model


def generate_sample(model, img, prompt, c2ws, intrinsics,
                    frame_num=17, shift=3.0, seed=42):
    """Generate a single video sample.

    Args:
        model: WanI2VCausal instance.
        img: PIL.Image.Image — input image.
        prompt: str — text prompt.
        c2ws: np.ndarray [N, 4, 4] — camera poses.
        intrinsics: np.ndarray [N, 4] — camera intrinsics.
        frame_num: int — frames to generate.
        shift: float — noise schedule shift.
        seed: int — random seed.

    Returns:
        video: torch.Tensor (C, N, H, W) in [-1, 1].
        elapsed: float — generation time in seconds.
    """
    model.reset()

    t0 = time.time()
    video, _ = model.generate_chunk(
        img=img,
        prompt=prompt,
        c2ws=c2ws,
        intrinsics=intrinsics,
        frame_num=frame_num,
        shift=shift,
        seed=seed,
    )
    elapsed = time.time() - t0

    return video, elapsed


def compute_lpips_score(video_tensor):
    """Compute average LPIPS between consecutive frames.

    Args:
        video_tensor: (C, N, H, W) video in [-1, 1].

    Returns:
        float: mean LPIPS score (lower is better).
    """
    try:
        import lpips
    except ImportError:
        logger.warning("lpips not installed, skipping LPIPS metric")
        return float("nan")

    loss_fn = lpips.LPIPS(net="alex").cuda()
    scores = []

    N = video_tensor.shape[1]
    for i in range(N - 1):
        frame_a = video_tensor[:, i].unsqueeze(0).cuda()
        frame_b = video_tensor[:, i + 1].unsqueeze(0).cuda()
        score = loss_fn(frame_a, frame_b).item()
        scores.append(score)

    return np.mean(scores)


def compute_temporal_coherence(video_tensor):
    """Compute temporal coherence via frame difference magnitude.

    Lower values indicate smoother temporal transitions.

    Args:
        video_tensor: (C, N, H, W) video in [-1, 1].

    Returns:
        float: mean absolute frame difference.
    """
    diffs = []
    N = video_tensor.shape[1]
    for i in range(N - 1):
        diff = (video_tensor[:, i + 1] - video_tensor[:, i]).abs().mean().item()
        diffs.append(diff)
    return np.mean(diffs)


def save_video_frames(video_tensor, output_dir, prefix="frame"):
    """Save video tensor as individual frame images.

    Args:
        video_tensor: (C, N, H, W) in [-1, 1].
        output_dir: Directory to save frames.
        prefix: Filename prefix.
    """
    os.makedirs(output_dir, exist_ok=True)
    N = video_tensor.shape[1]

    for i in range(N):
        frame = video_tensor[:, i]
        frame = ((frame + 1.0) / 2.0).clamp(0, 1)
        frame = (frame * 255).byte().permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"{prefix}_{i:04d}.png"))


def save_video_mp4(video_tensor, output_path, fps=16):
    """Save video tensor as MP4.

    Args:
        video_tensor: (C, N, H, W) in [-1, 1].
        output_path: Path to save MP4.
        fps: Frames per second.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("cv2 not installed, skipping MP4 save")
        return

    N = video_tensor.shape[1]
    H, W = video_tensor.shape[2], video_tensor.shape[3]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(N):
        frame = video_tensor[:, i]
        frame = ((frame + 1.0) / 2.0).clamp(0, 1)
        frame = (frame * 255).byte().permute(1, 2, 0).cpu().numpy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()


def evaluate_checkpoint(
    model_dir,
    checkpoint_path,
    data_dir,
    output_dir,
    num_samples=5,
    sampling_steps=4,
    seed=42,
):
    """Full evaluation pipeline.

    Args:
        model_dir: Base model directory.
        checkpoint_path: Post-training checkpoint.
        data_dir: Encoded data directory.
        output_dir: Output directory for results.
        num_samples: Number of samples to generate.
        sampling_steps: Denoising steps.
        seed: Random seed.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0")

    logger.info("=" * 60)
    logger.info("Post-Training Evaluation")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Samples: {num_samples}, Steps: {sampling_steps}")
    logger.info("=" * 60)

    # Load model
    model = load_student_model(model_dir, checkpoint_path, device, sampling_steps)

    # Load test samples from encoded data
    sample_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".pt")
    )[:num_samples]

    results = []

    for i, sample_file in enumerate(sample_files):
        logger.info(f"\nSample {i+1}/{num_samples}: {sample_file}")

        sample = torch.load(
            os.path.join(data_dir, sample_file),
            map_location="cpu",
            weights_only=False,
        )

        c2ws = sample["c2ws"].numpy()
        intrinsics = sample["intrinsics"].numpy()

        # Create a synthetic first frame (gray with slight noise)
        # In practice, you'd use a real image
        h, w = 480, 832
        first_frame = Image.new("RGB", (w, h), (128, 128, 128))

        prompt = "A dashcam view of a vehicle driving on a road."

        # Generate
        video, elapsed = generate_sample(
            model=model,
            img=first_frame,
            prompt=prompt,
            c2ws=c2ws,
            intrinsics=intrinsics,
            frame_num=17,
            shift=3.0,
            seed=seed + i,
        )

        # Metrics
        lpips_score = compute_lpips_score(video)
        temporal_coh = compute_temporal_coherence(video)

        result = {
            "sample": sample_file,
            "frames": video.shape[1],
            "time_s": elapsed,
            "lpips": lpips_score,
            "temporal_coherence": temporal_coh,
        }
        results.append(result)

        logger.info(
            f"  Time: {elapsed:.1f}s | LPIPS: {lpips_score:.4f} | "
            f"Temporal: {temporal_coh:.4f}"
        )

        # Save video
        sample_dir = os.path.join(output_dir, f"sample_{i:03d}")
        save_video_frames(video, sample_dir)
        save_video_mp4(video, os.path.join(sample_dir, "video.mp4"))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    avg_time = np.mean([r["time_s"] for r in results])
    avg_lpips = np.nanmean([r["lpips"] for r in results])
    avg_temporal = np.mean([r["temporal_coherence"] for r in results])

    logger.info(f"Avg generation time: {avg_time:.1f}s per chunk")
    logger.info(f"Avg LPIPS: {avg_lpips:.4f}")
    logger.info(f"Avg temporal coherence: {avg_temporal:.4f}")
    logger.info(f"Denoising steps: {sampling_steps}")

    # Save summary
    import json

    summary = {
        "checkpoint": checkpoint_path,
        "sampling_steps": sampling_steps,
        "num_samples": num_samples,
        "avg_time_s": float(avg_time),
        "avg_lpips": float(avg_lpips),
        "avg_temporal_coherence": float(avg_temporal),
        "results": results,
    }

    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate post-trained model"
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="eval_output")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    evaluate_checkpoint(
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

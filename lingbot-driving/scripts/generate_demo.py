"""
generate_demo.py — Generate a scripted driving demo video.

Takes a single dashcam image and produces one continuous driving video
by generating chunks along a pre-defined route, then stitching them.

Usage:
    # Quick test — built-in example image from lingbot-world
    python generate_demo.py \
        --ckpt_dir ./lingbot-world-base-cam

    # With your own dashcam image
    python generate_demo.py \
        --ckpt_dir ./lingbot-world-base-cam \
        --init_image dashcam.jpg \
        --prompt "Dashcam footage driving through a city intersection"

    # Faster (fewer steps, lower quality)
    python generate_demo.py \
        --ckpt_dir ./lingbot-world-base-cam \
        --init_image dashcam.jpg \
        --sample_steps 20
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Add lingbot-world to path ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # veristudio/
LINGBOT_WORLD = PROJECT_ROOT / "lingbot-world"
if str(LINGBOT_WORLD) not in sys.path:
    sys.path.insert(0, str(LINGBOT_WORLD))


# ── Pose generation (same math as drive_interactive.py) ─────────────────────

def yaw_to_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def generate_route_poses(route, num_frames=17, speed=0.3, turn_rate_deg=2.0):
    """
    Generate camera poses for a scripted route.

    Args:
        route: list of (action_str, num_chunks) tuples
               e.g. [("w", 3), ("wa", 2), ("w", 2), ("wd", 2)]
        num_frames: frames per chunk (must be 4n+1)
        speed: translation per frame
        turn_rate_deg: degrees per frame when turning

    Returns:
        all_poses: list of np.ndarray, each [num_frames, 4, 4]
        descriptions: list of str
    """
    position = np.array([0.0, 0.0, 0.0])
    yaw = 0.0
    current_speed = 0.0
    current_yaw_rate = 0.0

    all_poses = []
    descriptions = []

    for action, n_chunks in route:
        keys = set(action)
        for _ in range(n_chunks):
            target_speed = 0.0
            target_yaw_rate = 0.0

            if 'w' in keys:
                target_speed = speed
            if 's' in keys:
                target_speed = -speed * 0.5
            if 'a' in keys:
                target_yaw_rate = -np.radians(turn_rate_deg)
            if 'd' in keys:
                target_yaw_rate = np.radians(turn_rate_deg)

            if not keys:
                target_speed = current_speed * 0.8
                target_yaw_rate = 0.0

            poses = []
            for i in range(num_frames):
                t = (i + 1) / num_frames
                current_speed += (target_speed - current_speed) * t * 0.3
                current_yaw_rate += (target_yaw_rate - current_yaw_rate) * t * 0.3

                yaw += current_yaw_rate
                forward = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
                position = position + forward * current_speed

                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = yaw_to_rotation(yaw)
                c2w[:3, 3] = position
                poses.append(c2w)

            all_poses.append(np.array(poses, dtype=np.float32))

            parts = []
            if 'w' in keys: parts.append("forward")
            if 's' in keys: parts.append("reverse")
            if 'a' in keys: parts.append("left")
            if 'd' in keys: parts.append("right")
            descriptions.append(' + '.join(parts) if parts else 'coast')

    return all_poses, descriptions


def get_default_intrinsics(num_frames, width=832, height=480):
    fov_h = 70.0
    fx = width / (2.0 * np.tan(np.radians(fov_h / 2.0)))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.tile(np.array([[fx, fy, cx, cy]], dtype=np.float32),
                   (num_frames, 1))


# ── Pre-defined routes ──────────────────────────────────────────────────────

ROUTES = {
    "city_cruise": [
        # Go straight, gentle left, straight, gentle right, straight
        ("w",  3),   # forward 3 chunks
        ("wa", 2),   # forward + left turn
        ("w",  2),   # straighten out
        ("wd", 2),   # forward + right turn
        ("w",  3),   # cruise forward
    ],
    "intersection_turn": [
        ("w",  4),   # approach intersection
        ("wa", 3),   # turn left
        ("w",  3),   # continue straight
    ],
    "lane_change": [
        ("w",  3),   # forward
        ("wd", 1),   # steer right
        ("w",  2),   # straighten
        ("wa", 1),   # steer left (back to center)
        ("w",  3),   # continue
    ],
    "short": [
        ("w", 2),    # just go forward (quick test)
    ],
}


# ── Main generation ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a scripted driving demo video")
    parser.add_argument("--ckpt_dir", required=True, help="Path to lingbot-world-base-cam weights")
    parser.add_argument("--init_image", default=None, help="Starting dashcam image")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--route", default="city_cruise", choices=list(ROUTES.keys()))
    parser.add_argument("--output_dir", default="demo_output")
    parser.add_argument("--size", default="480*832", choices=["480*832", "720*1280"])
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--sample_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t5_cpu", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    h, w = int(args.size.split("*")[0]), int(args.size.split("*")[1])
    prompt = args.prompt or \
        "Dashcam footage of a vehicle driving on a city road during daytime. Clear weather, urban environment with buildings and traffic."

    # ── Load model ──────────────────────────────────────────────────────────
    import wan
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import save_video
    from PIL import Image
    import torch

    cfg = WAN_CONFIGS["i2v-A14B"]
    logger.info(f"Loading WanI2V from {args.ckpt_dir} ...")
    model = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
    )
    logger.info("Model loaded.")

    # ── Load starting image ─────────────────────────────────────────────────
    if args.init_image and os.path.exists(args.init_image):
        current_image = Image.open(args.init_image).convert("RGB")
        logger.info(f"Starting image: {args.init_image}")
    else:
        # Fall back to lingbot-world example
        example_img = LINGBOT_WORLD / "examples" / "02" / "image.jpg"
        if example_img.exists():
            current_image = Image.open(str(example_img)).convert("RGB")
            logger.info(f"Using example image: {example_img}")
        else:
            logger.error("No starting image. Provide --init_image or ensure lingbot-world/examples/02/image.jpg exists.")
            sys.exit(1)

    # ── Generate route poses ────────────────────────────────────────────────
    route = ROUTES[args.route]
    all_poses, descriptions = generate_route_poses(
        route, num_frames=args.frame_num, speed=0.3, turn_rate_deg=2.0)
    intrinsics = get_default_intrinsics(args.frame_num, width=w, height=h)

    total_chunks = len(all_poses)
    logger.info(f"Route '{args.route}': {total_chunks} chunks, {total_chunks * args.frame_num} total frames")

    # ── Generate chunks ─────────────────────────────────────────────────────
    all_videos = []

    for i, (poses, desc) in enumerate(zip(all_poses, descriptions)):
        logger.info(f"[Chunk {i+1}/{total_chunks}] Action: {desc}")
        t0 = time.time()

        # Save poses to temp dir
        tmp = tempfile.mkdtemp(prefix="lingbot_demo_")
        try:
            np.save(os.path.join(tmp, "poses.npy"), poses)
            np.save(os.path.join(tmp, "intrinsics.npy"), intrinsics)

            video = model.generate(
                input_prompt=prompt,
                img=current_image,
                action_path=tmp,
                max_area=h * w,
                frame_num=args.frame_num,
                shift=3.0 if "480" in args.size else 10.0,
                sample_solver="unipc",
                sampling_steps=args.sample_steps,
                guide_scale=5.0,
                seed=args.seed + i,
                offload_model=True,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        dt = time.time() - t0
        logger.info(f"  Generated in {dt:.1f}s")

        if video is None:
            logger.warning(f"  Chunk {i} failed, skipping.")
            continue

        # Save individual chunk
        chunk_path = str(out / f"chunk_{i:03d}_{desc.replace(' + ', '_')}.mp4")
        save_video(tensor=video[None], save_file=chunk_path,
                   fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
        logger.info(f"  Saved: {chunk_path}")

        all_videos.append(video)

        # Use last frame as next starting image
        import torchvision.transforms.functional as TF
        last = video[:, -1, :, :]
        last = ((last + 1.0) / 2.0).clamp(0, 1).cpu()
        current_image = TF.to_pil_image(last)
        current_image.save(str(out / f"chunk_{i:03d}_last.jpg"))

    # ── Stitch all chunks into one video ────────────────────────────────────
    if all_videos:
        logger.info("Stitching chunks into final video...")
        # For chunks after the first, skip frame 0 (it's the same as previous chunk's last frame)
        frames = [all_videos[0]]
        for v in all_videos[1:]:
            frames.append(v[:, 1:, :, :])  # skip first frame to avoid duplicate
        stitched = torch.cat(frames, dim=1)

        final_path = str(out / f"demo_{args.route}.mp4")
        save_video(tensor=stitched[None], save_file=final_path,
                   fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))

        total_frames = stitched.shape[1]
        duration = total_frames / cfg.sample_fps
        logger.info(f"Final video: {final_path}")
        logger.info(f"  {total_frames} frames, {duration:.1f}s at {cfg.sample_fps} fps")
    else:
        logger.error("No chunks generated successfully.")

    logger.info("Done!")


if __name__ == "__main__":
    main()

"""
modal_demo.py — Test the causal driving model on Modal (A100 80GB).

Usage:
    # First time: set up Modal
    pip install modal
    modal setup

    # Run the demo (downloads model, generates driving video chunks)
    modal run modal_demo.py

    # Generate with custom image
    modal run modal_demo.py --image-path /path/to/dashcam.jpg

    # Interactive WASD mode (type commands, get video chunks back)
    modal run modal_demo.py --interactive
"""

import base64 as _b64
import io
import time

import modal

app = modal.App("lingbot-driving-demo")

# Volume for caching model weights between runs
model_volume = modal.Volume.from_name("lingbot-model-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Bundled pose generator (written into the container so we don't need mounts)
# ---------------------------------------------------------------------------
_POSE_UTILS_CODE = r'''
import numpy as np

class DrivingPoseGenerator:
    def __init__(self, speed=0.3, turn_rate=2.0, num_frames=17):
        self.speed = speed
        self.turn_rate_deg = turn_rate
        self.num_frames = num_frames
        self.position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0

    def _yaw_to_rotation(self, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[ c, 0, s], [ 0, 1, 0], [-s, 0, c]])

    def _make_c2w(self, position, yaw):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = self._yaw_to_rotation(yaw)
        c2w[:3, 3] = position
        return c2w

    def wasd_to_poses(self, keys_held):
        target_speed = 0.0
        target_yaw_rate = 0.0
        if "w" in keys_held:
            target_speed = self.speed
        if "s" in keys_held:
            target_speed = -self.speed * 0.5
        if "a" in keys_held:
            target_yaw_rate = -np.radians(self.turn_rate_deg)
        if "d" in keys_held:
            target_yaw_rate = np.radians(self.turn_rate_deg)
        if not keys_held:
            target_speed = self.current_speed * 0.8
            target_yaw_rate = 0.0

        poses = []
        for i in range(self.num_frames):
            t = (i + 1) / self.num_frames
            self.current_speed += (target_speed - self.current_speed) * t * 0.3
            self.current_yaw_rate += (target_yaw_rate - self.current_yaw_rate) * t * 0.3
            self.yaw += self.current_yaw_rate
            forward = np.array([np.sin(self.yaw), 0.0, np.cos(self.yaw)])
            self.position = self.position + forward * self.current_speed
            poses.append(self._make_c2w(self.position.copy(), self.yaw))

        parts = []
        if "w" in keys_held: parts.append("forward")
        if "s" in keys_held: parts.append("reverse")
        if "a" in keys_held: parts.append("left")
        if "d" in keys_held: parts.append("right")
        desc = " + ".join(parts) if parts else "coast"
        return np.array(poses, dtype=np.float32), desc

    def get_default_intrinsics(self, num_frames, width=832, height=480):
        fov_h = 70.0
        fx = width / (2.0 * np.tan(np.radians(fov_h / 2.0)))
        fy = fx
        cx, cy = width / 2.0, height / 2.0
        return np.tile(np.array([[fx, fy, cx, cy]], dtype=np.float32), (num_frames, 1))

    def reset(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
'''

# Base64-encode the module so we can write it safely (no quote-escaping issues)
_POSE_UTILS_B64 = _b64.b64encode(_POSE_UTILS_CODE.encode()).decode()

# ---------------------------------------------------------------------------
# Container image — fully built BEFORE any @app.function decorators
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "einops",
        "easydict",
        "diffusers>=0.31.0",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "scipy",
        "numpy",
        "Pillow",
        "tqdm",
        "huggingface_hub",
        "ftfy",
        "regex",
        "imageio",
        "imageio-ffmpeg",
        "opencv-python-headless",
    )
    # Install flash-attn from pre-built wheel (no CUDA toolkit needed at build time)
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    # Clone our fork with both-expert MoE routing + KV cache
    .run_commands(
        "git clone -b feat/kv-cache-causal https://github.com/DannyMang/lingbot-world.git /opt/lingbot-world",
        force_build=True,
    )
    # Write the bundled pose-utils module into the cloned repo
    .run_commands(
        f"python3 -c \"import base64; open('/opt/lingbot-world/_pose_utils.py','w').write(base64.b64decode('{_POSE_UTILS_B64}').decode())\""
    )
)

MODEL_ID = "robbyant/lingbot-world-base-cam"
MODEL_DIR = "/data/lingbot-world-base-cam"


# ---------------------------------------------------------------------------
# Remote functions (all use the fully-built image)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/data": model_volume},
)
def download_model():
    """Download model weights to the persistent volume."""
    import os
    required_files = [
        f"{MODEL_DIR}/high_noise_model/config.json",
        f"{MODEL_DIR}/low_noise_model/config.json",
        f"{MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{MODEL_DIR}/Wan2.1_VAE.pth",
        f"{MODEL_DIR}/google/umt5-xxl/tokenizer.json",
    ]
    if all(os.path.exists(f) for f in required_files):
        print(f"Model already cached at {MODEL_DIR}")
        return
    from huggingface_hub import snapshot_download
    # Use HF_TOKEN env var if available (set via: modal secret create hf-token HF_TOKEN=hf_xxx)
    token = os.environ.get("HF_TOKEN")
    print(f"Downloading {MODEL_ID} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        token=token,
    )
    model_volume.commit()
    print("Download complete.")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=600,
    volumes={"/data": model_volume},
)
def generate_fast(image_bytes: bytes, prompt: str, num_chunks: int = 3,
                  frame_num: int = 17, seed: int = 42) -> list[bytes]:
    """
    Generate driving video using the fast single-expert path (no KV cache).
    Returns a list of MP4 bytes, one per chunk.
    """
    import sys
    sys.path.insert(0, "/opt/lingbot-world")

    import numpy as np
    import torch
    from PIL import Image
    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2V
    from _pose_utils import DrivingPoseGenerator

    cfg = WAN_CONFIGS['i2v-A14B']
    print("Loading WanI2V ...")
    t0 = time.time()
    model = WanI2V(
        config=cfg,
        checkpoint_dir=MODEL_DIR,
        device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pose_gen = DrivingPoseGenerator(speed=0.3, turn_rate=2.0, num_frames=frame_num)

    results = []
    actions = ["w", "w", "wd", "w", "wa", "w", "w", "w", "wd", "w"]

    for chunk_idx in range(num_chunks):
        keys = set(actions[chunk_idx % len(actions)])
        poses, desc = pose_gen.wasd_to_poses(keys)
        intrinsics = pose_gen.get_default_intrinsics(len(poses), width=832, height=480)

        import tempfile, os, shutil
        tmp = tempfile.mkdtemp()
        np.save(os.path.join(tmp, "poses.npy"), poses[:frame_num])
        np.save(os.path.join(tmp, "intrinsics.npy"), intrinsics[:frame_num])

        print(f"\n--- Chunk {chunk_idx}: {desc} ---")
        t1 = time.time()
        video = model.generate_fast(
            input_prompt=prompt,
            img=img,
            action_path=tmp,
            max_area=480 * 832,
            frame_num=frame_num,
            shift=3.0,
            sampling_steps=6,
            seed=seed + chunk_idx,
        )
        dt = time.time() - t1
        print(f"Generated in {dt:.1f}s")
        shutil.rmtree(tmp)

        if video is not None:
            import torchvision.transforms.functional as TF
            last = video[:, -1, :, :]
            last = ((last + 1.0) / 2.0).clamp(0, 1).cpu()
            img = TF.to_pil_image(last)

            mp4_bytes = _video_to_mp4(video)
            results.append(mp4_bytes)

    return results


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/data": model_volume},
)
def generate_causal(image_bytes: bytes, prompt: str, num_chunks: int = 5,
                    frame_num: int = 17, seed: int = 42) -> list[bytes]:
    """
    Generate driving video using KV-cached causal inference.
    Returns a list of MP4 bytes, one per chunk.

    First chunk: ~5s, subsequent chunks: ~2-4s (from KV cache reuse).
    """
    import sys
    sys.path.insert(0, "/opt/lingbot-world")

    import numpy as np
    import torch
    from PIL import Image
    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2VCausal
    from _pose_utils import DrivingPoseGenerator

    cfg = WAN_CONFIGS['i2v-A14B']
    print("Loading WanI2VCausal (both experts, KV cache, CFG) ...")
    t0 = time.time()
    model = WanI2VCausal(
        config=cfg,
        checkpoint_dir=MODEL_DIR,
        device_id=0,
        t5_cpu=True,
        max_cache_chunks=4,
        sampling_steps=20,
        max_area=480 * 832,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pose_gen = DrivingPoseGenerator(speed=0.3, turn_rate=2.0, num_frames=frame_num)

    results = []
    timings = []
    actions = ["w", "w", "wd", "w", "wa", "w", "w", "w", "wd", "w"]

    for chunk_idx in range(num_chunks):
        keys = set(actions[chunk_idx % len(actions)])
        poses, desc = pose_gen.wasd_to_poses(keys)
        intrinsics = pose_gen.get_default_intrinsics(len(poses), width=832, height=480)

        print(f"\n--- Chunk {chunk_idx}: {desc} ---")
        t1 = time.time()
        video, last_frame = model.generate_chunk(
            img=img,
            prompt=prompt,
            c2ws=poses[:frame_num],
            intrinsics=intrinsics[:frame_num],
            frame_num=frame_num,
            shift=10.0,
            seed=seed + chunk_idx,
        )
        dt = time.time() - t1
        timings.append(dt)
        print(f"Generated in {dt:.1f}s")

        if video is not None:
            img = last_frame
            mp4_bytes = _video_to_mp4(video)
            results.append(mp4_bytes)

    print(f"\n=== Timing Summary ===")
    print(f"First chunk: {timings[0]:.1f}s")
    if len(timings) > 1:
        avg_rest = sum(timings[1:]) / len(timings[1:])
        print(f"Avg subsequent: {avg_rest:.1f}s")
    print(f"Total: {sum(timings):.1f}s for {num_chunks} chunks")

    model.free()
    return results


def _video_to_mp4(video_tensor) -> bytes:
    """Convert a (C, N, H, W) tensor in [-1,1] to MP4 bytes."""
    import subprocess
    import tempfile
    import os
    import shutil

    import torch
    from PIL import Image

    # video_tensor: (C, N, H, W) in [-1, 1]
    video = ((video_tensor + 1.0) / 2.0).clamp(0, 1)
    video = (video.permute(1, 2, 3, 0).cpu().numpy() * 255).astype("uint8")  # (N, H, W, C)

    tmp_dir = tempfile.mkdtemp()
    for i, frame in enumerate(video):
        Image.fromarray(frame).save(os.path.join(tmp_dir, f"frame_{i:04d}.png"))

    out_path = os.path.join(tmp_dir, "output.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "16",
        "-i", os.path.join(tmp_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", out_path
    ], capture_output=True)

    with open(out_path, "rb") as f:
        mp4_bytes = f.read()

    shutil.rmtree(tmp_dir)
    return mp4_bytes


# ---------------------------------------------------------------------------
# Interactive WASD mode — runs on Modal, streams chunks back
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/data": model_volume},
)
def interactive_session(image_bytes: bytes, prompt: str, actions_list: list[str],
                        frame_num: int = 17, seed: int = 42) -> list[bytes]:
    """
    Generate chunks for a sequence of WASD actions.
    Each entry in actions_list is a string like "w", "wa", "wd", "s", "".
    Returns a list of MP4 bytes, one per action.
    """
    import sys
    sys.path.insert(0, "/opt/lingbot-world")

    import numpy as np
    from PIL import Image
    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2VCausal
    from _pose_utils import DrivingPoseGenerator

    cfg = WAN_CONFIGS['i2v-A14B']
    print("Loading WanI2VCausal ...")
    t0 = time.time()
    model = WanI2VCausal(
        config=cfg,
        checkpoint_dir=MODEL_DIR,
        device_id=0,
        t5_cpu=True,
        max_cache_chunks=4,
        sampling_steps=6,
        max_area=480 * 832,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pose_gen = DrivingPoseGenerator(speed=0.3, turn_rate=2.0, num_frames=frame_num)

    results = []
    for i, action in enumerate(actions_list):
        keys = set(c for c in action.lower() if c in 'wasd')
        poses, desc = pose_gen.wasd_to_poses(keys)
        intrinsics = pose_gen.get_default_intrinsics(len(poses), width=832, height=480)

        print(f"\n--- Action {i}: '{action}' -> {desc} ---")
        t1 = time.time()
        video, last_frame = model.generate_chunk(
            img=img,
            prompt=prompt,
            c2ws=poses[:frame_num],
            intrinsics=intrinsics[:frame_num],
            frame_num=frame_num,
            shift=3.0,
            seed=seed + i,
        )
        dt = time.time() - t1
        print(f"Generated in {dt:.1f}s")

        if video is not None:
            img = last_frame
            results.append(_video_to_mp4(video))

    model.free()
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    image_path: str = "",
    prompt: str = "Dashcam footage of a vehicle driving on a road during daytime. Clear weather, suburban environment.",
    num_chunks: int = 5,
    causal: bool = True,
    interactive: bool = False,
    seed: int = 42,
):
    """Run the driving demo on Modal."""
    from pathlib import Path

    # Ensure model is downloaded
    print("Ensuring model weights are cached ...")
    download_model.remote()

    # Load or create a test image
    if image_path and Path(image_path).exists():
        image_bytes = Path(image_path).read_bytes()
        print(f"Using image: {image_path}")
    else:
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (832, 480), (100, 130, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        print("Using placeholder image (pass --image-path for a real dashcam photo)")

    if interactive:
        # Interactive WASD mode: type actions, get chunks back
        print("\n=== Interactive WASD Mode ===")
        print("Type WASD keys (e.g. 'w', 'wa', 'wd', 's') then Enter.")
        print("Type 'q' to quit.\n")

        actions = []
        while True:
            try:
                action = input("Action> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if action == 'q':
                break
            actions.append(action)

        if actions:
            print(f"\nSending {len(actions)} actions to Modal ...")
            t0 = time.time()
            results = interactive_session.remote(
                image_bytes, prompt, actions, seed=seed)
            total = time.time() - t0

            out_dir = Path("modal_output")
            out_dir.mkdir(exist_ok=True)
            for i, mp4 in enumerate(results):
                path = out_dir / f"interactive_{i:03d}.mp4"
                path.write_bytes(mp4)
                print(f"Saved: {path} ({len(mp4) / 1024:.0f} KB)")
            print(f"\nTotal time: {total:.1f}s for {len(actions)} chunks")
        return

    # Non-interactive batch mode
    mode = "causal (KV-cached)" if causal else "fast (single-expert)"
    print(f"\nGenerating {num_chunks} chunks in {mode} mode ...")
    print(f"Prompt: {prompt}")
    print("=" * 60)

    t0 = time.time()
    if causal:
        results = generate_causal.remote(
            image_bytes, prompt, num_chunks=num_chunks, seed=seed)
    else:
        results = generate_fast.remote(
            image_bytes, prompt, num_chunks=num_chunks, seed=seed)
    total = time.time() - t0

    # Save results locally
    out_dir = Path("modal_output")
    out_dir.mkdir(exist_ok=True)
    for i, mp4 in enumerate(results):
        path = out_dir / f"chunk_{i:03d}.mp4"
        path.write_bytes(mp4)
        print(f"Saved: {path} ({len(mp4) / 1024:.0f} KB)")

    print(f"\nTotal time (including cold start): {total:.1f}s")
    print(f"Output: {out_dir}/")

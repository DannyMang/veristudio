"""
train_posttrain_modal.py -- Modal cloud deployment for post-training pipeline.

Full workflow:
    1. Download model weights from HuggingFace
    2. Download Waymo v2.0.1 Parquet data from GCS (or generate dummy data)
    3. Preprocess Waymo → encoded .pt files via VAE/T5
    4. Stage 1: Causal adaptation (8xH100)
    5. Stage 2: DMD distillation (8xH100)
    6. Evaluate checkpoint

Usage:
    # Setup (run once)
    modal run train_posttrain_modal.py --download-model
    modal run train_posttrain_modal.py --download-waymo
    modal run train_posttrain_modal.py --preprocess-waymo

    # Dry run (with dummy data, no real data needed)
    modal run train_posttrain_modal.py --generate-dummy-data
    modal run train_posttrain_modal.py --stage1 --dry-run

    # Full training
    modal run train_posttrain_modal.py --stage1
    modal run train_posttrain_modal.py --stage2

    # Evaluate
    modal run train_posttrain_modal.py --eval-model --checkpoint stage2_latest.pt
"""

from pathlib import Path

import modal

app = modal.App("lingbot-posttrain")

# Local scripts directory -- mounted into container via image.add_local_dir
SCRIPTS_DIR = str(Path(__file__).parent)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

model_volume = modal.Volume.from_name(
    "lingbot-model-cache", create_if_missing=True
)
data_volume = modal.Volume.from_name(
    "lingbot-posttrain-data", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "lingbot-posttrain-checkpoints", create_if_missing=True
)
waymo_volume = modal.Volume.from_name(
    "lingbot-waymo-raw", create_if_missing=True
)

# ---------------------------------------------------------------------------
# Container image
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
        "opencv-python-headless",
        "fastapi",
        "uvicorn",
        "gradio>=5.34.1,<6.0",
        "wandb",
        "pandas",
        "pyarrow",
        "google-cloud-storage",
    )
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .run_commands(
        "git clone -b feat/kv-cache-causal "
        "https://github.com/DannyMang/lingbot-world.git /opt/lingbot-world",
        force_build=True,
    )
    # Mount local scripts into container (picks up latest code, no git push needed)
    .add_local_dir(SCRIPTS_DIR, "/opt/lingbot-driving-v2/scripts")
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = "/data/lingbot-world-base-cam"
WAYMO_DIR = "/waymo-data/training"
ENCODED_DIR = "/posttrain-data/encoded"
DUMMY_DIR = "/posttrain-data/dummy"
CHECKPOINT_DIR = "/checkpoints"

# GCS bucket for Waymo v2.0.1
WAYMO_GCS_BUCKET = "waymo_open_dataset_v_2_0_1"
WAYMO_COMPONENTS = ["camera_image", "camera_calibration", "vehicle_pose"]


# ============================================================================
# Model download (no GPU needed)
# ============================================================================


@app.function(
    image=image,
    volumes={"/data": model_volume},
    timeout=7200,
    memory=16 * 1024,
)
def download_model(repo_id: str = "robbyant/lingbot-world-base-cam"):
    """Download LingBot-World model weights from HuggingFace."""
    import os

    from huggingface_hub import snapshot_download

    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model already exists at {MODEL_DIR}, skipping download")
        return

    print(f"Downloading {repo_id} to {MODEL_DIR}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    # List downloaded files
    total_size = 0
    for root, dirs, files in os.walk(MODEL_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            total_size += size
    print(f"Downloaded {total_size / 1e9:.1f} GB to {MODEL_DIR}")

    model_volume.commit()


# ============================================================================
# Waymo download from GCS
# ============================================================================


@app.function(
    image=image,
    volumes={"/waymo-data": waymo_volume},
    secrets=[modal.Secret.from_name("gcs-secret")],
    timeout=86400,  # 24 hours (Modal max)
    memory=16 * 1024,
)
def download_waymo(
    split: str = "training",
    max_segments: int = 0,
):
    """Download Waymo v2.0.1 Parquet files from GCS.

    Only downloads camera_image, camera_calibration, and vehicle_pose
    components (skips lidar, labels, segmentation to save storage).

    Prerequisites:
        modal secret create gcs-secret \\
            SERVICE_ACCOUNT_JSON="$(cat /path/to/gcs-service-account-key.json)"
    """
    import json
    import os

    from google.cloud import storage

    creds_json = os.environ.get("SERVICE_ACCOUNT_JSON", "")
    if not creds_json:
        print("ERROR: SERVICE_ACCOUNT_JSON not set in gcs-secret")
        print("Create it with:")
        print('  modal secret create gcs-secret SERVICE_ACCOUNT_JSON="$(cat key.json)"')
        return

    creds = json.loads(creds_json)

    # Handle both service account keys and application default credentials
    if "client_email" in creds:
        # Service account JSON key
        client = storage.Client.from_service_account_info(creds)
    else:
        # Application default credentials (from gcloud auth application-default login)
        import google.oauth2.credentials
        credentials = google.oauth2.credentials.Credentials(
            token=creds.get("access_token"),
            refresh_token=creds.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=creds.get("client_id"),
            client_secret=creds.get("client_secret"),
        )
        client = storage.Client(credentials=credentials, project=creds.get("quota_project_id"))

    bucket = client.bucket(WAYMO_GCS_BUCKET)

    for component in WAYMO_COMPONENTS:
        prefix = f"{split}/{component}/"
        print(f"\nListing {prefix}...")
        blobs = list(bucket.list_blobs(prefix=prefix))

        if max_segments > 0:
            blobs = blobs[:max_segments]

        print(f"  Found {len(blobs)} files for {component}")

        for i, blob in enumerate(blobs):
            # Skip directory markers
            if blob.name.endswith("/"):
                continue

            local_path = f"/waymo-data/{blob.name}"

            # Skip if already downloaded
            if os.path.exists(local_path):
                continue

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)

            if (i + 1) % 50 == 0:
                size_gb = blob.size / 1e9 if blob.size else 0
                print(
                    f"  [{i + 1}/{len(blobs)}] {component} "
                    f"(last file: {size_gb:.2f} GB)"
                )
                waymo_volume.commit()

        waymo_volume.commit()
        print(f"  {component} download complete")

    print("\nWaymo download complete!")

    # Print summary
    total_files = 0
    total_bytes = 0
    for root, dirs, files in os.walk("/waymo-data"):
        for f in files:
            total_files += 1
            total_bytes += os.path.getsize(os.path.join(root, f))
    print(f"Total: {total_files} files, {total_bytes / 1e9:.1f} GB")


# ============================================================================
# Waymo preprocessing: Parquet -> encoded .pt (1x H100)
# ============================================================================


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,  # 24 hours (Modal max)
    volumes={
        "/data": model_volume,
        "/waymo-data": waymo_volume,
        "/posttrain-data": data_volume,
    },

    memory=64 * 1024,
)
def preprocess_waymo(
    num_frames: int = 81,
    resolution: int = 480,
    camera: str = "FRONT",
    max_segments: int = 0,
):
    """Read Waymo Parquet -> encode through VAE/T5 -> save .pt files.

    This reads directly from Parquet files in memory, avoiding the
    inode-heavy step of extracting individual PNG frames to disk.
    """
    import os
    import sys
    import time

    import torch

    sys.path.insert(0, "/opt/lingbot-world")
    sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")

    from data_pipeline import PreprocessPipeline
    from waymo_converter import WaymoParquetReader

    print("=" * 60)
    print("Waymo Parquet -> Encoded .pt")
    print(f"Camera: {camera}, Frames/clip: {num_frames}, Resolution: {resolution}p")
    if max_segments > 0:
        print(f"Max segments: {max_segments}")
    print("=" * 60)

    reader = WaymoParquetReader(WAYMO_DIR)
    pipeline = PreprocessPipeline(MODEL_DIR, device="cuda:0")

    os.makedirs(ENCODED_DIR, exist_ok=True)

    # Count existing encoded files to resume from
    existing = set(os.listdir(ENCODED_DIR))
    clip_idx = len([f for f in existing if f.endswith(".pt")])
    print(f"Resuming from clip {clip_idx} ({len(existing)} existing files)")

    t0 = time.time()
    encoded_count = 0

    for clip in reader.iter_clips(
        frames_per_clip=num_frames,
        camera=camera,
        max_segments=max_segments if max_segments > 0 else None,
    ):
        output_path = os.path.join(ENCODED_DIR, f"{clip_idx:06d}.pt")

        if os.path.exists(output_path):
            clip_idx += 1
            continue

        try:
            pipeline.encode_clip_from_arrays(
                frames=clip["frames"],
                c2ws=clip["c2ws"],
                intrinsics=clip["intrinsics"],
                output_path=output_path,
                caption=clip["caption"],
                num_frames=num_frames,
                resolution=resolution,
            )
            encoded_count += 1
            clip_idx += 1

            if encoded_count % 10 == 0:
                elapsed = time.time() - t0
                rate = encoded_count / elapsed * 60
                print(
                    f"  Encoded {encoded_count} clips "
                    f"({rate:.1f} clips/min, {elapsed/60:.1f} min elapsed)"
                )
                torch.cuda.empty_cache()
                data_volume.commit()

        except Exception as e:
            print(
                f"  Failed clip {clip_idx} "
                f"(segment={clip['segment_name']}, "
                f"clip={clip['clip_idx']}): {e}"
            )
            clip_idx += 1
            continue

    data_volume.commit()
    elapsed = time.time() - t0
    print(f"\nEncoded {encoded_count} clips in {elapsed/60:.1f} min")


# ============================================================================
# Legacy preprocessing: raw_clips -> encoded .pt (1x H100)
# ============================================================================


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={
        "/data": model_volume,
        "/posttrain-data": data_volume,
    },

    memory=64 * 1024,
)
def preprocess(num_frames: int = 17, resolution: int = 480):
    """Pre-encode raw_clips through frozen VAE and T5 (legacy path)."""
    import os
    import sys
    import time

    sys.path.insert(0, "/opt/lingbot-world")
    sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")

    from data_pipeline import PreprocessPipeline

    DATA_ROOT = "/posttrain-data/raw_clips"

    print("=" * 60)
    print("Pre-encoding dataset for post-training (legacy raw_clips path)")
    print(f"Resolution: {resolution}p, Frames: {num_frames}")
    print("=" * 60)

    t0 = time.time()
    pipeline = PreprocessPipeline(MODEL_DIR, device="cuda:0")
    pipeline.encode_dataset(DATA_ROOT, ENCODED_DIR, num_frames, resolution)

    elapsed = time.time() - t0
    print(f"\nPre-encoding complete in {elapsed/60:.1f} min")

    data_volume.commit()


# ============================================================================
# Validate Waymo data (no GPU needed)
# ============================================================================


@app.function(
    image=image,
    volumes={"/waymo-data": waymo_volume},
    timeout=1800,
    memory=16 * 1024,
)
def validate_waymo(
    split: str = "training",
    camera: str = "FRONT",
    max_segments: int = 5,
):
    """Sanity-check Waymo Parquet data: verify c2w trajectories are plausible.

    Loads a few segments, computes camera-to-world matrices, and checks:
      1. c2w matrices are valid (no NaN/Inf, proper rotation, det(R)≈1)
      2. Camera positions trace smooth driving paths (no jumps)
      3. Forward motion direction is consistent
      4. Intrinsics are reasonable (positive focal lengths, centered principal point)

    Run before preprocessing to catch data issues early:
        modal run train_posttrain_modal.py --validate-waymo
    """
    import sys

    import numpy as np

    sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")

    from waymo_converter import WaymoParquetReader

    WAYMO_DIR = f"/waymo-data/{split}"
    reader = WaymoParquetReader(WAYMO_DIR)

    camera_map = {"FRONT": 1, "FRONT_LEFT": 2, "FRONT_RIGHT": 3,
                  "SIDE_LEFT": 4, "SIDE_RIGHT": 5}
    camera_id = camera_map[camera.upper()]

    segments = reader.segment_names[:max_segments]
    print(f"Validating {len(segments)} segments (camera={camera})\n")

    all_passed = True
    num_validated = 0
    num_skipped = 0

    for seg_idx, seg_name in enumerate(segments):
        print(f"--- Segment {seg_idx + 1}/{len(segments)}: {seg_name} ---")
        try:
            data = reader.read_segment(seg_name, camera_id)
        except Exception as e:
            print(f"  SKIP: failed to read segment: {e}\n")
            num_skipped += 1
            all_passed = False
            continue

        num_validated += 1

        c2ws = data["c2ws"]           # (N, 4, 4) float32
        intrinsics = data["intrinsics"]  # (N, 4) float32
        n_frames = len(data["frames"])
        print(f"  Frames: {n_frames}, Image: {data['img_height']}x{data['img_width']}")

        errors = []

        # --- Check 1: NaN / Inf ---
        if np.any(np.isnan(c2ws)) or np.any(np.isinf(c2ws)):
            errors.append("c2w contains NaN or Inf")

        # --- Check 2: Rotation matrix validity (det(R) ≈ 1, R^T R ≈ I) ---
        for i in range(0, n_frames, max(1, n_frames // 5)):  # spot-check 5 frames
            R = c2ws[i, :3, :3]
            det = np.linalg.det(R)
            if abs(det - 1.0) > 0.01:
                errors.append(f"Frame {i}: det(R)={det:.4f} (expected ~1.0)")
            ortho_err = np.max(np.abs(R.T @ R - np.eye(3)))
            if ortho_err > 0.01:
                errors.append(f"Frame {i}: R not orthogonal (max err={ortho_err:.4f})")

        # --- Check 3: Camera positions — smooth trajectory, no jumps ---
        positions = c2ws[:, :3, 3]  # (N, 3) — camera XYZ in world frame
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_dist = np.sum(displacements)
        max_jump = np.max(displacements)
        mean_step = np.mean(displacements)
        median_step = np.median(displacements)

        # At 10 Hz, typical driving speed ~10-30 m/s → ~1-3 m/frame
        # A "jump" is >10x median step (teleportation)
        jump_threshold = max(median_step * 10, 5.0)
        num_jumps = int(np.sum(displacements > jump_threshold))
        if num_jumps > 0:
            errors.append(
                f"{num_jumps} trajectory jumps detected "
                f"(>{jump_threshold:.1f}m, max={max_jump:.1f}m)"
            )

        print(f"  Trajectory: total={total_dist:.1f}m, "
              f"mean_step={mean_step:.2f}m, median_step={median_step:.2f}m, "
              f"max_step={max_jump:.2f}m")

        # --- Check 4: Net displacement direction (should be mostly forward) ---
        start_pos = positions[0]
        end_pos = positions[-1]
        net_disp = np.linalg.norm(end_pos - start_pos)
        print(f"  Net displacement: {net_disp:.1f}m "
              f"(start={start_pos}, end={end_pos})")

        # --- Check 5: Intrinsics sanity ---
        fx, fy, cx, cy = intrinsics[0]
        w, h = data["img_width"], data["img_height"]
        if fx <= 0 or fy <= 0:
            errors.append(f"Non-positive focal length: fx={fx}, fy={fy}")
        if not (0.2 * w < cx < 0.8 * w) or not (0.2 * h < cy < 0.8 * h):
            errors.append(
                f"Principal point far from center: ({cx:.0f},{cy:.0f}) "
                f"vs image ({w}x{h})"
            )
        # Check intrinsics are constant across frames
        if not np.allclose(intrinsics, intrinsics[0:1].repeat(n_frames, axis=0)):
            errors.append("Intrinsics vary across frames (unexpected)")

        print(f"  Intrinsics: fx={fx:.1f}, fy={fy:.1f}, "
              f"cx={cx:.1f}, cy={cy:.1f}")

        if errors:
            all_passed = False
            for err in errors:
                print(f"  ERROR: {err}")
        else:
            print(f"  PASS")
        print()

    # --- Summary ---
    print("=" * 60)
    print(f"Validated: {num_validated}, Skipped: {num_skipped}, "
          f"Total: {len(segments)}")
    if num_validated == 0:
        print("FAILED — could not read any segments. Check column names / data.")
    elif all_passed:
        print("ALL CHECKS PASSED — c2w matrices look correct.")
        print("Safe to proceed with preprocessing.")
    else:
        print("SOME CHECKS FAILED — review errors above before preprocessing.")
    print("=" * 60)


# ============================================================================
# Dummy data generation (no GPU needed)
# ============================================================================


@app.function(
    image=image,
    volumes={"/posttrain-data": data_volume},

    timeout=600,
    memory=8 * 1024,
)
def generate_dummy_data(
    num_samples: int = 20,
    num_chunks: int = 4,
    chunk_lat_frames: int = 5,
):
    """Generate dummy encoded .pt files for dry-run testing.

    Creates fake tensors with correct shapes so you can validate the
    training pipeline without real data or VAE/T5 encoding.
    """
    import os
    import sys

    sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")

    from create_dummy_data import create_dummy_sample

    import torch

    os.makedirs(DUMMY_DIR, exist_ok=True)

    print(f"Generating {num_samples} dummy samples...")
    for i in range(num_samples):
        sample = create_dummy_sample(
            num_chunks=num_chunks,
            chunk_lat_frames=chunk_lat_frames,
        )
        torch.save(sample, os.path.join(DUMMY_DIR, f"{i:06d}.pt"))

    data_volume.commit()

    # Print verification
    sample = torch.load(
        os.path.join(DUMMY_DIR, "000000.pt"), weights_only=False
    )
    print("Sample shapes:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
    print(f"Done! {num_samples} dummy samples in {DUMMY_DIR}")


# ============================================================================
# Stage 1: Causal Adaptation (8x H100)
# ============================================================================


@app.function(
    image=image,
    gpu="H100:8",
    timeout=86400,  # 24h max; auto-retries resume from checkpoint
    retries=10,  # auto-restart up to 10x (~10 days total)
    single_use_containers=True,  # fresh container each retry
    volumes={
        "/data": model_volume,
        "/posttrain-data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    memory=64 * 1024,
)
def train_stage1(
    total_steps: int = 50000,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    learning_rate: float = 2e-5,
    warmup_steps: int = 1000,
    num_chunks: int = 4,
    chunk_lat_frames: int = 5,
    save_every: int = 2000,
    log_every: int = 50,
    seed: int = 42,
    resume_step: int = 0,
    wandb_project: str = "lingbot-posttrain",
    use_wandb: bool = True,
    dry_run: bool = False,
):
    """Run Stage 1 causal adaptation on 8xH100 via torchrun."""
    import os
    import subprocess
    import sys

    import torch

    # Reduce CUDA memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    num_gpus = torch.cuda.device_count()

    # For dry runs, regenerate dummy data with correct dimensions
    data_dir = ENCODED_DIR
    if dry_run:
        data_dir = DUMMY_DIR
        sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")
        from create_dummy_data import create_dummy_sample

        os.makedirs(DUMMY_DIR, exist_ok=True)
        print(f"Regenerating dummy data: {num_chunks} chunks x {chunk_lat_frames} lat frames")
        for i in range(20):
            sample = create_dummy_sample(
                num_chunks=num_chunks,
                chunk_lat_frames=chunk_lat_frames,
            )
            torch.save(sample, os.path.join(DUMMY_DIR, f"{i:06d}.pt"))
        s = create_dummy_sample(num_chunks=num_chunks, chunk_lat_frames=chunk_lat_frames)
        print(f"  latents: {s['latents'].shape}, y: {s['y'].shape}")
        data_volume.commit()

    args = [
        "--model_dir", MODEL_DIR,
        "--data_dir", data_dir,
        "--output_dir", CHECKPOINT_DIR,
        "--total_steps", str(total_steps),
        "--batch_size", str(batch_size),
        "--gradient_accumulation", str(gradient_accumulation),
        "--learning_rate", str(learning_rate),
        "--warmup_steps", str(warmup_steps),
        "--save_every", str(save_every),
        "--log_every", str(log_every),
        "--num_chunks", str(num_chunks),
        "--chunk_lat_frames", str(chunk_lat_frames),
        "--seed", str(seed),
        "--wandb_project", wandb_project,
    ]
    if not use_wandb:
        args.append("--no_wandb")

    # Auto-resume from latest checkpoint (supports retry-based multi-day training)
    latest_ckpt = f"{CHECKPOINT_DIR}/stage1_latest.pt"
    if resume_step > 0:
        resume_path = f"{CHECKPOINT_DIR}/stage1_step{resume_step}.pt"
        if not os.path.exists(resume_path):
            resume_path = latest_ckpt
        args.extend(["--resume", resume_path])
    elif os.path.exists(latest_ckpt):
        print(f"Found existing checkpoint, auto-resuming from {latest_ckpt}")
        args.extend(["--resume", latest_ckpt])

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500",
        "/opt/lingbot-driving-v2/scripts/train_stage1_causal.py",
    ] + args

    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    checkpoint_volume.commit()


# ============================================================================
# Stage 2: DMD Distillation (8x H100)
# ============================================================================


@app.function(
    image=image,
    gpu="H100:8",
    timeout=86400,  # 24h max; auto-retries resume from checkpoint
    retries=10,  # auto-restart up to 10x (~10 days total)
    single_use_containers=True,  # fresh container each retry
    volumes={
        "/data": model_volume,
        "/posttrain-data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    memory=64 * 1024,
)
def train_stage2(
    total_steps: int = 50000,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    student_lr: float = 1e-5,
    fake_score_lr: float = 2e-5,
    warmup_steps: int = 2000,
    num_chunks: int = 4,
    chunk_lat_frames: int = 5,
    rollout_chunks: int = 3,
    save_every: int = 5000,
    log_every: int = 50,
    seed: int = 42,
    resume_step: int = 0,
    stage1_ckpt: str = "",
    wandb_project: str = "lingbot-posttrain",
    use_wandb: bool = True,
):
    """Run Stage 2 DMD distillation on 8xH100 via torchrun."""
    import os
    import subprocess

    import torch

    num_gpus = torch.cuda.device_count()

    if not stage1_ckpt:
        stage1_ckpt = os.path.join(CHECKPOINT_DIR, "stage1_latest.pt")

    args = [
        "--model_dir", MODEL_DIR,
        "--student_ckpt", stage1_ckpt,
        "--data_dir", ENCODED_DIR,
        "--output_dir", CHECKPOINT_DIR,
        "--total_steps", str(total_steps),
        "--batch_size", str(batch_size),
        "--gradient_accumulation", str(gradient_accumulation),
        "--student_lr", str(student_lr),
        "--fake_score_lr", str(fake_score_lr),
        "--warmup_steps", str(warmup_steps),
        "--save_every", str(save_every),
        "--log_every", str(log_every),
        "--num_chunks", str(num_chunks),
        "--chunk_lat_frames", str(chunk_lat_frames),
        "--rollout_chunks", str(rollout_chunks),
        "--seed", str(seed),
        "--wandb_project", wandb_project,
    ]
    if not use_wandb:
        args.append("--no_wandb")

    # Auto-resume from latest checkpoint (supports retry-based multi-day training)
    latest_ckpt = f"{CHECKPOINT_DIR}/stage2_latest.pt"
    if resume_step > 0:
        resume_path = f"{CHECKPOINT_DIR}/stage2_step{resume_step}.pt"
        if not os.path.exists(resume_path):
            resume_path = latest_ckpt
        args.extend(["--resume", resume_path])
    elif os.path.exists(latest_ckpt):
        print(f"Found existing checkpoint, auto-resuming from {latest_ckpt}")
        args.extend(["--resume", latest_ckpt])

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500",
        "/opt/lingbot-driving-v2/scripts/train_stage2_dmd.py",
    ] + args

    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    checkpoint_volume.commit()


# ============================================================================
# Evaluation (1x H100)
# ============================================================================


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={
        "/data": model_volume,
        "/posttrain-data": data_volume,
        "/checkpoints": checkpoint_volume,
    },

    memory=64 * 1024,
)
def evaluate(
    checkpoint: str = "stage2_latest.pt",
    num_samples: int = 5,
    sampling_steps: int = 4,
):
    """Evaluate a post-trained checkpoint."""
    import os
    import sys

    sys.path.insert(0, "/opt/lingbot-world")
    sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")

    from eval_posttrain import evaluate_checkpoint

    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    evaluate_checkpoint(
        model_dir=MODEL_DIR,
        checkpoint_path=ckpt_path,
        data_dir=ENCODED_DIR,
        output_dir=os.path.join(CHECKPOINT_DIR, "eval_output"),
        num_samples=num_samples,
        sampling_steps=sampling_steps,
    )

    checkpoint_volume.commit()


# ============================================================================
# Extract inference-only checkpoint (strips optimizer state)
# ============================================================================


@app.function(
    image=image,
    timeout=1800,
    volumes={"/checkpoints": checkpoint_volume},
    memory=128 * 1024,
)
def extract_inference_ckpt(checkpoint: str = "stage1_step10000.pt"):
    """Extract model weights only from a training checkpoint (111 GB -> ~28 GB)."""
    import os

    import torch

    src = os.path.join(CHECKPOINT_DIR, checkpoint)
    dst = os.path.join(CHECKPOINT_DIR, checkpoint.replace(".pt", "_inference.pt"))

    if os.path.exists(dst):
        print(f"Inference checkpoint already exists: {dst}")
        checkpoint_volume.commit()
        return

    print(f"Loading full checkpoint: {src} ...")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    inference_ckpt = {
        "step": ckpt["step"],
        "model_state_dict": ckpt["model_state_dict"],
    }

    print(f"Saving inference checkpoint: {dst} ...")
    torch.save(inference_ckpt, dst)

    size_gb = os.path.getsize(dst) / 1e9
    print(f"Done! {size_gb:.1f} GB (was {os.path.getsize(src) / 1e9:.1f} GB)")
    checkpoint_volume.commit()


# ============================================================================
# Demo (interactive web UI — plain FastAPI + static HTML)
# ============================================================================


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={
        "/data": model_volume,
        "/posttrain-data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    memory=128 * 1024,
    max_containers=1,
)
@modal.asgi_app()
def demo():
    """Launch a FastAPI demo to try the post-trained model in a browser.

    Plain FastAPI + static HTML — no Gradio, no SSE, no queue.
    Modal handles vanilla FastAPI perfectly (just request/response).
    max_containers=1 ensures single instance (global state is safe).
    """
    import io
    import os
    import sys
    import time

    import torch

    sys.path.insert(0, "/opt/lingbot-world")
    sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")

    import numpy as np
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
    from PIL import Image

    from eval_posttrain import load_student_model, save_video_mp4

    # Configure via env vars or defaults
    checkpoint = os.environ.get("DEMO_CHECKPOINT", "stage1_step10000.pt")
    sampling_steps = int(os.environ.get("DEMO_STEPS", "20"))

    # Prefer inference-only checkpoint (28 GB) over full training checkpoint (111 GB)
    inference_ckpt = checkpoint.replace(".pt", "_inference.pt")
    if os.path.exists(os.path.join(CHECKPOINT_DIR, inference_ckpt)):
        checkpoint = inference_ckpt
    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
    print(f"Loading model from {ckpt_path} with {sampling_steps} steps...")
    # NOTE: compile=False for now — torch.compile interferes with KV cache
    # state mutations (_cache_valid_len is a Python int mutated in-place by
    # append_to_cache, which torch.compile may not track across blocks).
    # Re-enable after verifying chunk-to-chunk quality without compile.
    model = load_student_model(
        MODEL_DIR, ckpt_path, torch.device("cuda:0"), sampling_steps,
        compile=False,
    )
    print("Model loaded!")

    # WASD pose generator (inlined from drive_interactive.py to avoid cross-dir import)
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
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def _make_c2w(self, position, yaw):
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = self._yaw_to_rotation(yaw)
            c2w[:3, 3] = position
            return c2w

        def wasd_to_poses(self, keys_held):
            target_speed = 0.0
            target_yaw_rate = 0.0
            if 'w' in keys_held: target_speed = self.speed
            if 's' in keys_held: target_speed = -self.speed * 0.5
            if 'a' in keys_held: target_yaw_rate = -np.radians(self.turn_rate_deg)
            if 'd' in keys_held: target_yaw_rate = np.radians(self.turn_rate_deg)
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
            if 'w' in keys_held: parts.append("forward")
            if 's' in keys_held: parts.append("reverse")
            if 'a' in keys_held: parts.append("left")
            if 'd' in keys_held: parts.append("right")
            return np.array(poses, dtype=np.float32), ' + '.join(parts) or 'coast'

        def get_default_intrinsics(self, num_frames, width=832, height=480):
            fov_h = 70.0
            fx = width / (2.0 * np.tan(np.radians(fov_h / 2.0)))
            cx, cy = width / 2.0, height / 2.0
            return np.tile(np.array([[fx, fx, cx, cy]], dtype=np.float32), (num_frames, 1))

        def reset(self):
            self.position = np.array([0.0, 0.0, 0.0])
            self.yaw = 0.0
            self.current_speed = 0.0
            self.current_yaw_rate = 0.0

    pose_gen = DrivingPoseGenerator(speed=0.3, turn_rate=2.0, num_frames=17)

    # Persistent state for autoregressive generation
    state = {
        "current_image": None,  # PIL Image of last frame
        "prompt": "",
        "chunk_idx": 0,
        "initialized": False,
    }

    direction_key_map = {
        "forward": {"w"},
        "forward_left": {"w", "a"},
        "forward_right": {"w", "d"},
        "reverse": {"s"},
        "coast": set(),
    }

    web_app = FastAPI()

    # ------------------------------------------------------------------
    # Static HTML UI
    # ------------------------------------------------------------------

    INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LingBot-World Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #111; color: #eee; }
  h1 { text-align: center; padding: 18px 0 6px; font-size: 1.4rem; }
  .subtitle { text-align: center; color: #999; font-size: 0.85rem; margin-bottom: 16px; }
  .container { display: grid; grid-template-columns: 320px 1fr; gap: 20px; max-width: 1200px; margin: 0 auto; padding: 0 20px 40px; }
  .panel { background: #1a1a1a; border-radius: 10px; padding: 20px; }
  label { display: block; font-size: 0.8rem; color: #aaa; margin-bottom: 4px; margin-top: 14px; }
  label:first-child { margin-top: 0; }
  input[type="file"], input[type="text"], input[type="range"] { width: 100%; }
  input[type="text"] { padding: 8px; border: 1px solid #333; border-radius: 6px; background: #222; color: #eee; font-size: 0.9rem; }
  input[type="range"] { accent-color: #4a9eff; }
  .steps-val { font-size: 0.8rem; color: #aaa; text-align: right; }
  .btn { display: inline-flex; align-items: center; justify-content: center; padding: 10px 16px; border: none; border-radius: 6px; font-size: 0.9rem; cursor: pointer; transition: background 0.15s; color: #fff; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: #2563eb; }
  .btn-primary:hover:not(:disabled) { background: #1d4ed8; }
  .btn-dir { background: #333; min-width: 80px; }
  .btn-dir:hover:not(:disabled) { background: #444; }
  .btn-danger { background: #dc2626; }
  .btn-danger:hover:not(:disabled) { background: #b91c1c; }
  .start-row { margin-top: 18px; }
  .drive-label { font-size: 0.85rem; color: #aaa; margin: 18px 0 8px; }
  .dir-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
  .dir-bottom { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; }
  .reset-row { margin-top: 18px; }
  .preview-img { width: 100%; border-radius: 8px; background: #222; min-height: 200px; object-fit: contain; }
  .video-el { width: 100%; border-radius: 8px; background: #000; margin-top: 12px; }
  .status { margin-top: 12px; padding: 10px; background: #222; border-radius: 6px; font-size: 0.85rem; color: #aaa; min-height: 40px; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #555; border-top-color: #4a9eff; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (max-width: 720px) { .container { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>LingBot-World Stage 1 Demo</h1>
<p class="subtitle">Upload a dashcam image, click Start, then drive with direction buttons.</p>

<div class="container">
  <!-- Left panel: controls -->
  <div class="panel">
    <label for="img-upload">Upload First Frame</label>
    <input type="file" id="img-upload" accept="image/*">

    <label for="prompt-input">Prompt</label>
    <input type="text" id="prompt-input" value="A dashcam view of a vehicle driving on a road.">

    <label for="steps-slider">Denoising Steps</label>
    <input type="range" id="steps-slider" min="4" max="70" value="STEPS_PLACEHOLDER" step="1">
    <div class="steps-val" id="steps-val">STEPS_PLACEHOLDER</div>

    <div class="start-row">
      <button class="btn btn-primary" id="btn-start" style="width:100%">Start Session</button>
    </div>

    <div class="drive-label">Drive</div>
    <div class="dir-grid">
      <button class="btn btn-dir" id="btn-left" disabled>A Left</button>
      <button class="btn btn-primary btn-dir" id="btn-fwd" disabled>W Fwd</button>
      <button class="btn btn-dir" id="btn-right" disabled>D Right</button>
    </div>
    <div class="dir-bottom">
      <button class="btn btn-dir" id="btn-rev" disabled>S Rev</button>
      <button class="btn btn-dir" id="btn-coast" disabled>Coast</button>
    </div>

    <div class="reset-row">
      <button class="btn btn-danger" id="btn-reset" style="width:100%">Reset</button>
    </div>
  </div>

  <!-- Right panel: outputs -->
  <div class="panel">
    <img id="preview" class="preview-img" alt="Current frame preview">
    <video id="video" class="video-el" controls autoplay muted></video>
    <div class="status" id="status">Upload an image and click Start.</div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const stepsSlider = $('#steps-slider');
const stepsVal = $('#steps-val');
stepsSlider.addEventListener('input', () => { stepsVal.textContent = stepsSlider.value; });

const dirBtns = ['#btn-fwd','#btn-left','#btn-right','#btn-rev','#btn-coast'].map($);
function setDriveBtns(enabled) { dirBtns.forEach(b => b.disabled = !enabled); }
function setAllBtns(enabled) {
  setDriveBtns(enabled);
  $('#btn-start').disabled = !enabled;
  $('#btn-reset').disabled = !enabled;
}

function setStatus(msg, loading) {
  $('#status').innerHTML = (loading ? '<span class="spinner"></span>' : '') + msg;
}

function refreshPreview() {
  $('#preview').src = '/api/preview?' + Date.now();
}

// Start session
$('#btn-start').addEventListener('click', async () => {
  const fileInput = $('#img-upload');
  if (!fileInput.files.length) { setStatus('Please select an image first.'); return; }
  setAllBtns(false);
  setStatus('Starting session...', true);
  const form = new FormData();
  form.append('image', fileInput.files[0]);
  form.append('prompt', $('#prompt-input').value);
  form.append('steps', stepsSlider.value);
  try {
    const res = await fetch('/api/start', { method: 'POST', body: form });
    const data = await res.json();
    if (res.ok) {
      refreshPreview();
      setStatus(data.status);
      setAllBtns(true);
    } else {
      setStatus('Error: ' + (data.detail || res.statusText));
      setAllBtns(true);
    }
  } catch (e) {
    setStatus('Network error: ' + e.message);
    setAllBtns(true);
  }
});

// Drive buttons
const dirMap = {
  'btn-fwd': 'forward',
  'btn-left': 'forward_left',
  'btn-right': 'forward_right',
  'btn-rev': 'reverse',
  'btn-coast': 'coast',
};
Object.entries(dirMap).forEach(([id, direction]) => {
  $('#' + id).addEventListener('click', async () => {
    setAllBtns(false);
    setStatus('Generating ' + direction + '...', true);
    try {
      const res = await fetch('/api/drive', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ direction, steps: parseInt(stepsSlider.value) }),
      });
      if (!res.ok) {
        const err = await res.json();
        setStatus('Error: ' + (err.detail || res.statusText));
        setAllBtns(true);
        return;
      }
      // Response is MP4 binary — read status from header
      const statusMsg = decodeURIComponent(res.headers.get('X-Status') || '');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const video = $('#video');
      video.src = url;
      video.play();
      refreshPreview();
      setStatus(statusMsg || 'Done');
      setAllBtns(true);
    } catch (e) {
      setStatus('Network error: ' + e.message);
      setAllBtns(true);
    }
  });
});

// Reset
$('#btn-reset').addEventListener('click', async () => {
  setAllBtns(false);
  setStatus('Resetting...', true);
  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    const data = await res.json();
    setStatus(data.status || 'Reset.');
    $('#preview').src = '';
    $('#video').src = '';
    setDriveBtns(false);
    $('#btn-start').disabled = false;
    $('#btn-reset').disabled = false;
  } catch (e) {
    setStatus('Network error: ' + e.message);
    setAllBtns(true);
  }
});
</script>
</body>
</html>
"""

    @web_app.get("/", response_class=HTMLResponse)
    async def index():
        return INDEX_HTML.replace("STEPS_PLACEHOLDER", str(sampling_steps))

    @web_app.post("/api/start")
    async def api_start(
        image: UploadFile = File(...),
        prompt: str = Form("A dashcam view of a vehicle driving on a road."),
        steps: int = Form(30),
    ):
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((832, 480))

        state["current_image"] = pil_img
        state["prompt"] = prompt
        state["chunk_idx"] = 0
        state["initialized"] = True

        model.reset()
        pose_gen.reset()

        if steps != model.sampling_steps:
            model.sampling_steps = steps

        return JSONResponse({"status": "Ready! Click a direction to generate chunk 1."})

    @web_app.post("/api/drive")
    async def api_drive(body: dict):
        if not state["initialized"] or state["current_image"] is None:
            return JSONResponse(
                {"detail": "Upload an image and click Start first."},
                status_code=400,
            )

        direction = body.get("direction", "forward")
        steps = int(body.get("steps", 30))
        keys = direction_key_map.get(direction, {"w"})
        c2ws, desc = pose_gen.wasd_to_poses(keys)
        intrinsics = pose_gen.get_default_intrinsics(17)

        if steps != model.sampling_steps:
            model.sampling_steps = steps

        t0 = time.time()
        video, _ = model.generate_chunk(
            img=state["current_image"],
            prompt=state["prompt"],
            c2ws=c2ws,
            intrinsics=intrinsics,
            frame_num=17,
            shift=3.0,
            seed=42,
        )
        elapsed = time.time() - t0
        state["chunk_idx"] += 1

        # Extract last frame as next input — video is (C, N, H, W) in [-1, 1]
        last_frame = video[:, -1]
        last_frame = ((last_frame + 1) * 127.5).clamp(0, 255).byte()
        last_frame = last_frame.permute(1, 2, 0).cpu().numpy()
        state["current_image"] = Image.fromarray(last_frame)

        # Encode video to MP4 in memory via temp file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        save_video_mp4(video, tmp.name, fps=10)
        with open(tmp.name, "rb") as f:
            mp4_bytes = f.read()
        os.unlink(tmp.name)

        status_msg = f"Chunk {state['chunk_idx']}: {desc} | {elapsed:.1f}s ({steps} steps)"

        return StreamingResponse(
            io.BytesIO(mp4_bytes),
            media_type="video/mp4",
            headers={"X-Status": status_msg},
        )

    @web_app.post("/api/reset")
    async def api_reset():
        state["current_image"] = None
        state["prompt"] = ""
        state["chunk_idx"] = 0
        state["initialized"] = False
        model.reset()
        pose_gen.reset()
        return JSONResponse({"status": "Session reset. Upload a new image."})

    @web_app.get("/api/preview")
    async def api_preview():
        if state["current_image"] is None:
            return Response(status_code=204)
        buf = io.BytesIO()
        state["current_image"].save(buf, format="JPEG", quality=90)
        return Response(content=buf.getvalue(), media_type="image/jpeg")

    return web_app


# ============================================================================
# CLI entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    # Actions
    download_model_weights: bool = False,
    download_waymo_data: bool = False,
    validate_waymo_data: bool = False,
    preprocess_waymo_data: bool = False,
    preprocess_data: bool = False,
    gen_dummy_data: bool = False,
    stage1: bool = False,
    stage2: bool = False,
    eval_model: bool = False,
    # Shared
    checkpoint: str = "stage2_latest.pt",
    batch_size: int = 1,
    grad_accum: int = 4,
    num_chunks: int = 4,
    chunk_lat_frames: int = 5,
    num_frames: int = 81,
    resolution: int = 480,
    seed: int = 42,
    resume_step: int = 0,
    dry_run: bool = False,
    wandb_project: str = "lingbot-posttrain",
    use_wandb: bool = True,
    # Stage 1
    s1_steps: int = 10000,
    s1_lr: float = 2e-5,
    s1_save: int = 1000,
    # Stage 2
    s2_steps: int = 50000,
    s2_student_lr: float = 1e-5,
    s2_save: int = 5000,
    s2_stage1_ckpt: str = "",
    # Waymo
    waymo_split: str = "training",
    waymo_camera: str = "FRONT",
    waymo_max_segments: int = 0,
    # Dummy data
    dummy_samples: int = 20,
    # Eval
    sampling_steps: int = 4,
    num_eval_samples: int = 5,
    # Model
    hf_repo: str = "robbyant/lingbot-world-base-cam",
):
    """
    LingBot-World Post-Training Pipeline on Modal.

    Setup (run once):
        modal run train_posttrain_modal.py --download-model-weights
        modal run train_posttrain_modal.py --download-waymo-data
        modal run train_posttrain_modal.py --validate-waymo-data
        modal run train_posttrain_modal.py --preprocess-waymo-data

    Dry run (no real data needed):
        modal run train_posttrain_modal.py --download-model-weights
        modal run train_posttrain_modal.py --gen-dummy-data
        modal run train_posttrain_modal.py --stage1 --dry-run

    Training:
        modal run train_posttrain_modal.py --stage1
        modal run train_posttrain_modal.py --stage2

    Evaluate:
        modal run train_posttrain_modal.py --eval-model --checkpoint stage2_latest.pt
    """
    # --- Download model weights ---
    if download_model_weights:
        print("=" * 60)
        print(f"Downloading model weights from {hf_repo}")
        print("=" * 60)
        download_model.remote(repo_id=hf_repo)
        print("Model download complete!")
        return

    # --- Validate Waymo data ---
    if validate_waymo_data:
        print("=" * 60)
        print("Validating Waymo data (c2w trajectories, intrinsics)")
        print(f"Camera: {waymo_camera}, Max segments: {waymo_max_segments or 5}")
        print("=" * 60)
        validate_waymo.remote(
            split=waymo_split,
            camera=waymo_camera,
            max_segments=waymo_max_segments or 5,
        )
        return

    # --- Download Waymo data ---
    if download_waymo_data:
        print("=" * 60)
        print(f"Downloading Waymo v2.0.1 ({waymo_split}) from GCS")
        if waymo_max_segments > 0:
            print(f"Max segments: {waymo_max_segments}")
        else:
            print("Downloading ALL segments (this may take hours)")
        print("Components: camera_image, camera_calibration, vehicle_pose")
        print("=" * 60)
        download_waymo.remote(
            split=waymo_split,
            max_segments=waymo_max_segments,
        )
        print("Waymo download complete!")
        return

    # --- Preprocess Waymo Parquet -> encoded .pt ---
    if preprocess_waymo_data:
        print("=" * 60)
        print("Preprocessing Waymo Parquet -> encoded .pt")
        print(f"Camera: {waymo_camera}, Frames: {num_frames}, Res: {resolution}p")
        print("=" * 60)
        preprocess_waymo.remote(
            num_frames=num_frames,
            resolution=resolution,
            camera=waymo_camera,
            max_segments=waymo_max_segments,
        )
        print("Preprocessing complete!")
        return

    # --- Legacy preprocess (raw_clips) ---
    if preprocess_data:
        print("=" * 60)
        print("Pre-encoding dataset (legacy raw_clips path)")
        print(f"Resolution: {resolution}p, Frames: {num_frames}")
        print("=" * 60)
        preprocess.remote(num_frames=num_frames, resolution=resolution)
        print("Pre-encoding complete!")
        return

    # --- Generate dummy data ---
    if gen_dummy_data:
        print("=" * 60)
        print(f"Generating {dummy_samples} dummy samples for dry-run")
        print("=" * 60)
        generate_dummy_data.remote(
            num_samples=dummy_samples,
            num_chunks=num_chunks,
            chunk_lat_frames=chunk_lat_frames,
        )
        print("Dummy data generated!")
        return

    # --- Stage 1 ---
    if stage1:
        if dry_run:
            s1_steps = 10
            s1_save = 5
            num_chunks = 2
            chunk_lat_frames = 2  # minimal sequence for dry run
            use_wandb = False
        print("=" * 60)
        print("Stage 1: Causal Architecture Adaptation")
        print(f"Steps: {s1_steps}, LR: {s1_lr}, Batch: {batch_size}x{grad_accum}")
        print(f"Chunks: {num_chunks}x{chunk_lat_frames} lat frames")
        if dry_run:
            print("*** DRY RUN (10 steps, no W&B, auto-regen dummy data) ***")
        print("=" * 60)
        train_stage1.remote(
            total_steps=s1_steps,
            batch_size=batch_size,
            gradient_accumulation=grad_accum,
            learning_rate=s1_lr,
            num_chunks=num_chunks,
            chunk_lat_frames=chunk_lat_frames,
            save_every=s1_save,
            seed=seed,
            resume_step=resume_step,
            wandb_project=wandb_project,
            use_wandb=use_wandb,
            dry_run=dry_run,
        )
        print("Stage 1 complete!")
        return

    # --- Stage 2 ---
    if stage2:
        if dry_run:
            s2_steps = 10
            s2_save = 5
            use_wandb = False
        print("=" * 60)
        print("Stage 2: DMD + Adversarial Distillation")
        print(f"Steps: {s2_steps}, Student LR: {s2_student_lr}")
        print(f"Chunks: {num_chunks}x{chunk_lat_frames} lat frames")
        if dry_run:
            print("*** DRY RUN (10 steps, no W&B) ***")
        print("=" * 60)
        train_stage2.remote(
            total_steps=s2_steps,
            batch_size=batch_size,
            gradient_accumulation=grad_accum,
            student_lr=s2_student_lr,
            num_chunks=num_chunks,
            chunk_lat_frames=chunk_lat_frames,
            save_every=s2_save,
            seed=seed,
            resume_step=resume_step,
            stage1_ckpt=s2_stage1_ckpt,
            wandb_project=wandb_project,
            use_wandb=use_wandb,
        )
        print("Stage 2 complete!")
        return

    # --- Evaluate ---
    if eval_model:
        print(f"Evaluating checkpoint: {checkpoint} ({sampling_steps} steps)")
        evaluate.remote(
            checkpoint=checkpoint,
            num_samples=num_eval_samples,
            sampling_steps=sampling_steps,
        )
        print("Evaluation complete!")
        return

    print("No action specified. Available actions:")
    print("  --download-model-weights  Download HF model weights")
    print("  --download-waymo-data     Download Waymo v2.0.1 from GCS")
    print("  --validate-waymo-data     Sanity-check c2w matrices & intrinsics")
    print("  --preprocess-waymo-data   Encode Waymo Parquet -> .pt files")
    print("  --preprocess-data         Encode raw_clips -> .pt (legacy)")
    print("  --gen-dummy-data          Generate dummy .pt for dry run")
    print("  --stage1                  Run Stage 1 causal adaptation")
    print("  --stage2                  Run Stage 2 DMD distillation")
    print("  --eval-model              Evaluate a checkpoint")
    print("\nAdd --dry-run to stage1/stage2 for a 10-step test run")

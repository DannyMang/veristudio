"""
train_posttrain_modal.py -- Modal cloud deployment for post-training pipeline.

Deploys both Stage 1 (causal adaptation) and Stage 2 (DMD distillation)
on Modal with 8xH100 GPU clusters.

Usage:
    # Preprocess data
    modal run train_posttrain_modal.py --preprocess

    # Stage 1: Causal adaptation (~3-5 days on 8xH100)
    modal run train_posttrain_modal.py --stage1

    # Stage 2: DMD distillation (~7-14 days on 8xH100)
    modal run train_posttrain_modal.py --stage2

    # Evaluate checkpoint
    modal run train_posttrain_modal.py --evaluate --checkpoint stage1_latest.pt

    # Dry run
    modal run train_posttrain_modal.py --stage1 --dry-run
"""

import modal

app = modal.App("lingbot-posttrain")

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
    .run_commands(
        "git clone "
        "https://github.com/DannyMang/lingbot-driving-v2.git /opt/lingbot-driving-v2 "
        "|| true",
    )
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = "/data/lingbot-world-base-cam"
DATA_ROOT = "/posttrain-data/raw_clips"
ENCODED_DIR = "/posttrain-data/encoded"
CHECKPOINT_DIR = "/checkpoints"


# ============================================================================
# Data preprocessing (1x H100)
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
    """Pre-encode all training clips through frozen VAE and T5."""
    import os
    import sys
    import time

    sys.path.insert(0, "/opt/lingbot-world")
    sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")

    from data_pipeline import PreprocessPipeline

    print("=" * 60)
    print("Pre-encoding dataset for post-training")
    print(f"Resolution: {resolution}p, Frames: {num_frames}")
    print("=" * 60)

    t0 = time.time()
    pipeline = PreprocessPipeline(MODEL_DIR, device="cuda:0")
    pipeline.encode_dataset(DATA_ROOT, ENCODED_DIR, num_frames, resolution)

    elapsed = time.time() - t0
    print(f"\nPre-encoding complete in {elapsed/60:.1f} min")

    data_volume.commit()


# ============================================================================
# Stage 1: Causal Adaptation (8x H100)
# ============================================================================


@app.function(
    image=image,
    gpu="H100:8",
    timeout=86400 * 5,  # 5 days max
    volumes={
        "/data": model_volume,
        "/posttrain-data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
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
):
    """Run Stage 1 causal adaptation on 8xH100."""
    import logging
    import os
    import sys

    sys.path.insert(0, "/opt/lingbot-world")
    sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")

    import torch

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(CHECKPOINT_DIR, "stage1.log")),
        ],
    )

    from train_stage1_causal import CausalPostTrainingConfig, Stage1Trainer

    config = CausalPostTrainingConfig(
        model_dir=MODEL_DIR,
        data_dir=ENCODED_DIR,
        output_dir=CHECKPOINT_DIR,
        total_steps=total_steps,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_every=save_every,
        log_every=log_every,
        num_chunks=num_chunks,
        chunk_lat_frames=chunk_lat_frames,
        seed=seed,
    )

    resume_path = None
    if resume_step > 0:
        resume_path = os.path.join(CHECKPOINT_DIR, f"stage1_step{resume_step}.pt")
        if not os.path.exists(resume_path):
            resume_path = os.path.join(CHECKPOINT_DIR, "stage1_latest.pt")

    trainer = Stage1Trainer(config)
    trainer.train(resume_path=resume_path)

    checkpoint_volume.commit()


# ============================================================================
# Stage 2: DMD Distillation (8x H100)
# ============================================================================


@app.function(
    image=image,
    gpu="H100:8",
    timeout=86400 * 14,  # 14 days max
    volumes={
        "/data": model_volume,
        "/posttrain-data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    memory=64 * 1024,
)
def train_stage2(
    total_steps: int = 100000,
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
):
    """Run Stage 2 DMD distillation on 8xH100."""
    import logging
    import os
    import sys

    sys.path.insert(0, "/opt/lingbot-world")
    sys.path.insert(0, "/opt/lingbot-driving-v2/scripts")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(CHECKPOINT_DIR, "stage2.log")),
        ],
    )

    from train_stage2_dmd import DMDConfig, Stage2Trainer

    # Default: use latest Stage 1 checkpoint
    if not stage1_ckpt:
        stage1_ckpt = os.path.join(CHECKPOINT_DIR, "stage1_latest.pt")

    config = DMDConfig(
        model_dir=MODEL_DIR,
        student_ckpt=stage1_ckpt,
        data_dir=ENCODED_DIR,
        output_dir=CHECKPOINT_DIR,
        total_steps=total_steps,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        student_lr=student_lr,
        fake_score_lr=fake_score_lr,
        warmup_steps=warmup_steps,
        save_every=save_every,
        log_every=log_every,
        num_chunks=num_chunks,
        chunk_lat_frames=chunk_lat_frames,
        rollout_chunks=rollout_chunks,
        seed=seed,
    )

    resume_path = None
    if resume_step > 0:
        resume_path = os.path.join(CHECKPOINT_DIR, f"stage2_step{resume_step}.pt")
        if not os.path.exists(resume_path):
            resume_path = os.path.join(CHECKPOINT_DIR, "stage2_latest.pt")

    trainer = Stage2Trainer(config)
    trainer.train(resume_path=resume_path)

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
# CLI entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    preprocess_data: bool = False,
    stage1: bool = False,
    stage2: bool = False,
    eval_model: bool = False,
    checkpoint: str = "stage2_latest.pt",
    # Stage 1 args
    s1_steps: int = 50000,
    s1_lr: float = 2e-5,
    s1_save: int = 2000,
    # Stage 2 args
    s2_steps: int = 100000,
    s2_student_lr: float = 1e-5,
    s2_save: int = 5000,
    s2_stage1_ckpt: str = "",
    # Common
    batch_size: int = 1,
    grad_accum: int = 4,
    num_chunks: int = 4,
    chunk_lat_frames: int = 5,
    num_frames: int = 17,
    resolution: int = 480,
    seed: int = 42,
    resume_step: int = 0,
    dry_run: bool = False,
):
    """
    Post-training pipeline on Modal.

    Examples:
        # Preprocess
        modal run train_posttrain_modal.py --preprocess-data

        # Stage 1
        modal run train_posttrain_modal.py --stage1

        # Stage 2
        modal run train_posttrain_modal.py --stage2

        # Evaluate
        modal run train_posttrain_modal.py --eval-model --checkpoint stage2_latest.pt

        # Dry run
        modal run train_posttrain_modal.py --stage1 --dry-run
    """
    if preprocess_data:
        print("=" * 60)
        print("Pre-encoding dataset for post-training")
        print(f"Resolution: {resolution}p, Frames: {num_frames}")
        print("=" * 60)
        preprocess.remote(num_frames=num_frames, resolution=resolution)
        print("Pre-encoding complete!")
        return

    if stage1:
        if dry_run:
            s1_steps = 10
            s1_save = 5
        print("=" * 60)
        print("Stage 1: Causal Architecture Adaptation")
        print(f"Steps: {s1_steps}, LR: {s1_lr}, Batch: {batch_size}x{grad_accum}")
        print(f"Chunks: {num_chunks}x{chunk_lat_frames} lat frames")
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
        )
        print("Stage 1 complete!")
        return

    if stage2:
        if dry_run:
            s2_steps = 10
            s2_save = 5
        print("=" * 60)
        print("Stage 2: DMD + Adversarial Distillation")
        print(f"Steps: {s2_steps}, Student LR: {s2_student_lr}")
        print(f"Chunks: {num_chunks}x{chunk_lat_frames} lat frames")
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
        )
        print("Stage 2 complete!")
        return

    if eval_model:
        print(f"Evaluating checkpoint: {checkpoint}")
        evaluate.remote(checkpoint=checkpoint)
        print("Evaluation complete!")
        return

    print("No action specified. Use --preprocess-data, --stage1, --stage2, or --eval-model")

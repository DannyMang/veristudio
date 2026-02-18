"""
train_lora_modal.py -- Stage III post-training on Modal.

Two-phase workflow:
  1. `modal run train_lora_modal.py --preprocess`
     Pre-encodes all training clips through VAE + T5 once (~15-30 min).
  2. `modal run train_lora_modal.py`
     Trains LoRA on pre-encoded latents (no VAE/T5 at train time).

Key design decisions (from the LingBot paper, Section 3.4):
  - Initializes from the **high-noise expert only** (single expert).
  - Trains with **diffusion forcing**: per-chunk random timesteps.
  - LoRA rank-16 on self-attention Q/K/V projections only.
  - Fully trains the DrivingActionAdapter (Plucker + multihot -> AdaLN).
  - Everything else is frozen (backbone, VAE, T5).

Data layout (Modal volumes):
  - waymo-training-data:        /waymo/waymo_processed/
  - lingbot-model-cache:        /data/lingbot-world-base-cam/
  - lingbot-lora-checkpoints:   /checkpoints/

Usage:
    # Step 1: Pre-encode dataset (run once)
    modal run train_lora_modal.py --preprocess

    # Step 2: Train LoRA
    modal run train_lora_modal.py

    # Resume from checkpoint
    modal run train_lora_modal.py --resume-step 500

    # Dry-run (10 steps)
    modal run train_lora_modal.py --dry-run
"""

import math
import modal

# ---------------------------------------------------------------------------
# Modal app + volumes
# ---------------------------------------------------------------------------

app = modal.App("lingbot-lora-training")

model_volume = modal.Volume.from_name("lingbot-model-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("waymo-training-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "lingbot-lora-checkpoints", create_if_missing=True
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
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = "/data/lingbot-world-base-cam"
DATA_ROOT = "/waymo/waymo_processed/validation"
MANIFEST_PATH = "/waymo/waymo_processed/training_manifest.json"
LATENT_CACHE_DIR = "/waymo/waymo_processed/latent_cache"
CHECKPOINT_DIR = "/checkpoints"


# ============================================================================
# LoRA implementation
# ============================================================================


def _make_lora_linear(original_linear, rank=16, alpha=16, dropout=0.05):
    """Create a LoRA-wrapped linear layer."""
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, original, rank, alpha, dropout):
            super().__init__()
            self.original = original
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank

            in_f = original.in_features
            out_f = original.out_features

            self.lora_A = nn.Linear(in_f, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_f, bias=False)
            self.lora_dropout = nn.Dropout(dropout)

            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

            for p in self.original.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.original(x) + self.scaling * self.lora_B(
                self.lora_A(self.lora_dropout(x))
            )

    return LoRALinear(original_linear, rank, alpha, dropout)


def inject_lora(model, rank=16, alpha=16):
    """Inject LoRA adapters into self-attention Q/K/V of every block."""
    import torch.nn as nn

    lora_params = []
    replaced = 0

    for block in model.blocks:
        sa = block.self_attn
        for attr_name in ("q", "k", "v"):
            orig = getattr(sa, attr_name)
            if isinstance(orig, nn.Linear):
                lora_layer = _make_lora_linear(orig, rank=rank, alpha=alpha)
                setattr(sa, attr_name, lora_layer)
                lora_params.append(lora_layer.lora_A.weight)
                lora_params.append(lora_layer.lora_B.weight)
                replaced += 1

    total_lora = sum(p.numel() for p in lora_params)
    print(f"[LoRA] Injected rank-{rank} into {replaced} layers, "
          f"{total_lora:,} trainable params")
    return lora_params


# ============================================================================
# DrivingActionAdapter
# ============================================================================


def _make_action_adapter(hidden_dim=512, output_dim=5120):
    """Build the DrivingActionAdapter module (Plucker + multihot -> AdaLN)."""
    import torch
    import torch.nn as nn

    class DrivingActionAdapter(nn.Module):
        def __init__(self, hidden_dim, output_dim):
            super().__init__()
            self.plucker_encoder = nn.Sequential(
                nn.Linear(6, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.discrete_encoder = nn.Sequential(
                nn.Linear(14, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            )
            self.adaln_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, output_dim * 2),
            )
            nn.init.zeros_(self.adaln_proj[-1].weight)
            nn.init.zeros_(self.adaln_proj[-1].bias)

        def forward(self, plucker, discrete_actions):
            h_p = self.plucker_encoder(plucker)
            h_d = self.discrete_encoder(discrete_actions)
            h = self.fusion(torch.cat([h_p, h_d], dim=-1))
            params = self.adaln_proj(h)
            scale, shift = params.chunk(2, dim=-1)
            return 1.0 + scale, shift

    return DrivingActionAdapter(hidden_dim, output_dim)


# ============================================================================
# Datasets
# ============================================================================


class DrivingWorldDataset:
    """Raw frame dataset for preprocessing. Loads video frames from disk."""

    def __init__(self, manifest_path, data_root, num_frames, resolution):
        import json
        from pathlib import Path

        with open(manifest_path) as f:
            manifest = json.load(f)
        self.samples = manifest["samples"]
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.resolution = resolution
        print(f"[Dataset] Loaded {len(self.samples)} training clips")

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, path):
        import os
        import cv2
        import numpy as np
        import torch

        path = str(path)
        res_map = {320: (320, 576), 480: (480, 854), 720: (720, 1280)}
        target_h, target_w = res_map.get(self.resolution, (480, 854))

        if not os.path.exists(path):
            return torch.zeros(3, target_h, target_w)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h))
        img = img.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        import random
        import numpy as np
        import torch

        sample = self.samples[idx]
        frame_dir = self.data_root / sample["frame_dir"]
        frame_files = sample["frames"]

        if len(frame_files) > self.num_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(len(frame_files)))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])

        frames = [self._load_frame(frame_dir / frame_files[i]) for i in indices]
        video = torch.stack(frames)

        actions = sample.get("actions", [])
        plucker_list, multihot_list = [], []
        for i in indices:
            if i < len(actions):
                p = actions[i].get("plucker", [0] * 6)
                if not (isinstance(p, list) and len(p) == 6):
                    p = [0] * 6
                plucker_list.append(p)
                multihot_list.append(actions[i].get("multihot", [0] * 14))
            else:
                plucker_list.append([0] * 6)
                multihot_list.append([0] * 14)

        plucker = torch.tensor(plucker_list, dtype=torch.float32)
        multihot = torch.tensor(multihot_list, dtype=torch.float32)

        if random.random() < 0.3:
            caption = sample.get("scene_static_caption", "")
        else:
            caption = sample.get("narrative_caption", "")
        if not caption:
            caption = "A dashcam view of a vehicle driving on a road."

        return {
            "video": video, "plucker": plucker, "multihot": multihot,
            "caption": caption, "sample_id": sample.get("sample_id", str(idx)),
        }


class PreEncodedDataset:
    """Loads pre-encoded VAE latents + T5 embeddings from disk."""

    def __init__(self, cache_dir):
        import os
        self.cache_dir = cache_dir
        self.files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
        print(f"[PreEncodedDataset] {len(self.files)} cached samples in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import os
        import torch

        path = os.path.join(self.cache_dir, self.files[idx])
        data = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "latents": data["latents"],    # (C_lat, T_lat, H_lat, W_lat) bf16
            "context": data["context"],    # (text_len, text_dim) bf16
            "plucker": data["plucker"],    # (T_vid, 6) fp32
            "multihot": data["multihot"],  # (T_vid, 14) fp32
        }


# ============================================================================
# Checkpoint save / load
# ============================================================================


def save_checkpoint(model, action_adapter, optimizer, scheduler, step,
                    output_dir, config_dict):
    import os
    import torch

    os.makedirs(output_dir, exist_ok=True)

    lora_state = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
            lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight.data.cpu()

    checkpoint = {
        "step": step,
        "lora_state_dict": lora_state,
        "action_adapter_state_dict": action_adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config_dict,
    }

    path = os.path.join(output_dir, f"checkpoint_step{step}.pt")
    torch.save(checkpoint, path)
    torch.save(checkpoint, os.path.join(output_dir, "checkpoint_latest.pt"))

    size_mb = os.path.getsize(path) / 1e6
    print(f"[Checkpoint] Saved step {step} -> {path} ({size_mb:.1f} MB)")


def load_checkpoint(model, action_adapter, checkpoint_path):
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    lora_state = ckpt["lora_state_dict"]
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            a_key = f"{name}.lora_A.weight"
            b_key = f"{name}.lora_B.weight"
            if a_key in lora_state:
                module.lora_A.weight.data = lora_state[a_key].to(module.lora_A.weight.device)
                module.lora_B.weight.data = lora_state[b_key].to(module.lora_B.weight.device)

    action_adapter.load_state_dict(ckpt["action_adapter_state_dict"])
    print(f"[Checkpoint] Loaded from step {ckpt['step']}")
    return ckpt["step"], ckpt.get("optimizer_state_dict"), ckpt.get("scheduler_state_dict")


# ============================================================================
# Training step (uses pre-encoded latents — no VAE/T5 at train time)
# ============================================================================


def train_step_cached(model, action_adapter, batch, device,
                      num_train_timesteps=1000, chunk_frames=5):
    """
    Training step on pre-encoded data. No VAE or T5 needed.

    Flow matching loss with diffusion forcing (per-chunk random timesteps).
    """
    import torch
    import torch.nn.functional as F

    latents = batch["latents"].to(device)    # (B, C, T_lat, H_lat, W_lat)
    context = batch["context"].to(device)    # (B, text_len, text_dim)
    plucker = batch["plucker"].to(device)    # (B, T_vid, 6)
    multihot = batch["multihot"].to(device)  # (B, T_vid, 14)

    model_raw = model.module if hasattr(model, "module") else model
    B, C_lat, T_lat, H_lat, W_lat = latents.shape

    # Temporal chunking for diffusion forcing
    num_chunks = max(1, T_lat // chunk_frames)
    T_use = num_chunks * chunk_frames
    if T_use > T_lat:
        T_use = T_lat
        num_chunks = 1
        chunk_frames = T_lat
    latents = latents[:, :, :T_use]

    # Per-chunk random timesteps (diffusion forcing)
    chunk_timesteps = torch.randint(
        0, num_train_timesteps, (B, num_chunks), device=device
    ).float()

    # Add noise per chunk (flow matching: x_t = (1-sigma)*x_0 + sigma*eps)
    noise = torch.randn_like(latents)
    noisy_latents = torch.zeros_like(latents)
    for c in range(num_chunks):
        s, e = c * chunk_frames, (c + 1) * chunk_frames
        for b in range(B):
            sigma = chunk_timesteps[b, c] / num_train_timesteps
            noisy_latents[b, :, s:e] = (
                (1.0 - sigma) * latents[b, :, s:e] + sigma * noise[b, :, s:e]
            )

    # Action adapter (subsample to latent temporal resolution)
    T_act = plucker.shape[1]
    act_indices = torch.linspace(0, T_act - 1, T_use, dtype=torch.long)
    action_scale, action_shift = action_adapter(
        plucker[:, act_indices], multihot[:, act_indices]
    )

    # Build i2v first-frame mask
    vae_t_stride = 4
    F_vid = T_use * vae_t_stride
    msk = torch.ones(1, F_vid, H_lat, W_lat, device=device)
    msk[:, 1:] = 0
    msk = torch.cat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
        msk[:, 1:]
    ], dim=1)[:, :F_vid]
    if msk.shape[1] % 4 != 0:
        pad = 4 - (msk.shape[1] % 4)
        msk = torch.cat([msk, torch.zeros(1, pad, H_lat, W_lat, device=device)], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, H_lat, W_lat)
    msk = msk.transpose(1, 2)[0][:, :T_use]

    # Model inputs
    tokens_per_frame = (
        H_lat * W_lat // (model_raw.patch_size[1] * model_raw.patch_size[2])
    )
    seq_len = T_use * tokens_per_frame

    # Per-token timesteps
    frame_ts = chunk_timesteps.repeat_interleave(chunk_frames, dim=1)
    token_ts = frame_ts.repeat_interleave(tokens_per_frame, dim=1)

    x_list = [noisy_latents[b] for b in range(B)]
    y_list = [torch.cat([msk, latents[b, :, :T_use]], dim=0) for b in range(B)]
    ctx = [context[b] for b in range(B)]

    # Forward through DiT
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model(
            x=x_list, t=token_ts, context=ctx,
            seq_len=seq_len, y=y_list,
        )

    # Loss: flow matching velocity (v = eps - x_0)
    loss = 0.0
    for b in range(B):
        target = noise[b, :, :T_use] - latents[b, :, :T_use]
        loss = loss + F.mse_loss(preds[b].float(), target.float())
    return loss / B


# ============================================================================
# Preprocessing: encode all clips through VAE + T5 (run once)
# ============================================================================


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/data": model_volume, "/waymo": data_volume},
    memory=64 * 1024,
)
def preprocess(num_frames: int = 17, resolution: int = 480):
    """Pre-encode all training clips through frozen VAE and T5."""
    import os
    import sys
    import time

    sys.path.insert(0, "/opt/lingbot-world")

    import torch
    from wan.configs import WAN_CONFIGS
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE

    device = torch.device("cuda:0")
    cfg = WAN_CONFIGS["i2v-A14B"]

    # Load T5 on GPU (no DiT during preprocessing, so plenty of VRAM)
    print("Loading T5 text encoder (GPU) ...")
    t0 = time.time()
    t5 = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(MODEL_DIR, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(MODEL_DIR, cfg.t5_tokenizer),
    )
    print(f"T5 loaded in {time.time() - t0:.1f}s")

    # Load VAE on GPU
    print("Loading VAE ...")
    t0 = time.time()
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(MODEL_DIR, cfg.vae_checkpoint),
        device=device,
    )
    print(f"VAE loaded in {time.time() - t0:.1f}s")

    # Load raw dataset
    dataset = DrivingWorldDataset(MANIFEST_PATH, DATA_ROOT, num_frames, resolution)

    # Create cache directory
    os.makedirs(LATENT_CACHE_DIR, exist_ok=True)

    print(f"\nPre-encoding {len(dataset)} clips at {resolution}p, {num_frames} frames ...")
    t_start = time.time()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        video = sample["video"].unsqueeze(0)  # (1, T, C, H, W)

        # VAE encode
        with torch.no_grad():
            video_ct = video.permute(0, 2, 1, 3, 4).to(device)  # (1, C, T, H, W)
            latents = vae.encode([video_ct[0]])[0]  # (C_lat, T_lat, H_lat, W_lat)
            latents = latents.to(torch.bfloat16).cpu()

        # T5 encode (on GPU — fast)
        with torch.no_grad():
            ctx = t5([sample["caption"]], device)[0]  # (text_len, dim)
            ctx = ctx.to(torch.bfloat16).cpu()

        # Save cached sample
        cache = {
            "latents": latents,
            "context": ctx,
            "plucker": sample["plucker"],
            "multihot": sample["multihot"],
            "caption": sample["caption"],
            "sample_id": sample["sample_id"],
        }
        torch.save(cache, os.path.join(LATENT_CACHE_DIR, f"{idx:06d}.pt"))

        if (idx + 1) % 25 == 0 or idx == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (len(dataset) - idx - 1)
            lat_shape = list(latents.shape)
            print(f"  [{idx+1}/{len(dataset)}] latent={lat_shape} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

        if idx % 10 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nPre-encoding complete! {len(dataset)} clips in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Cache saved to {LATENT_CACHE_DIR}")

    # Commit to volume
    data_volume.commit()
    print("Volume committed.")


# ============================================================================
# Training (single GPU, pre-encoded data)
# ============================================================================


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={
        "/data": model_volume,
        "/waymo": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    memory=64 * 1024,
)
def train(
    total_steps: int = 2000,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    chunk_frames: int = 5,
    warmup_steps: int = 200,
    save_every: int = 500,
    log_every: int = 10,
    resume_step: int = 0,
    seed: int = 42,
):
    """
    Train LoRA on pre-encoded latents. Single H100, no VAE/T5 at train time.

    Must run `preprocess` first to populate the latent cache.
    """
    import logging
    import os
    import sys
    import time

    sys.path.insert(0, "/opt/lingbot-world")

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda:0")
    torch.manual_seed(seed)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(CHECKPOINT_DIR, "training.log")),
        ],
    )
    log = logging.getLogger(__name__).info

    log("=" * 60)
    log("LingBot-World Stage III Post-Training (Cached LoRA)")
    log("=" * 60)
    log(f"Steps: {total_steps}, Batch: {batch_size}, Grad accum: {gradient_accumulation}")
    log(f"Effective batch: {batch_size * gradient_accumulation}")
    log(f"LoRA rank: {lora_rank}, LR: {learning_rate}")
    log(f"Chunk frames: {chunk_frames}, Seed: {seed}")

    # ---- Check cache exists ----
    if not os.path.isdir(LATENT_CACHE_DIR):
        raise RuntimeError(
            f"Latent cache not found at {LATENT_CACHE_DIR}. "
            "Run `modal run train_lora_modal.py --preprocess` first."
        )

    # ---- Load ONLY the DiT (no VAE, no T5!) ----
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel

    cfg = WAN_CONFIGS["i2v-A14B"]

    log("Loading high-noise expert (DiT) ...")
    t0 = time.time()
    dit = WanModel.from_pretrained(
        MODEL_DIR,
        subfolder=cfg.high_noise_checkpoint,
        torch_dtype=torch.bfloat16,
    )
    log(f"DiT loaded in {time.time() - t0:.1f}s")

    # Freeze everything
    dit.eval()
    for p in dit.parameters():
        p.requires_grad = False

    # Inject LoRA
    log("Injecting LoRA ...")
    lora_params = inject_lora(dit, rank=lora_rank, alpha=lora_alpha)

    # Gradient checkpointing
    from torch.utils.checkpoint import checkpoint as ckpt_fn
    for block in dit.blocks:
        orig = block.forward
        block._original_forward = orig
        def _make(o):
            def f(*a, **kw):
                return ckpt_fn(o, *a, use_reentrant=False, **kw)
            return f
        block.forward = _make(orig)
    log("Gradient checkpointing enabled")

    dit.to(device)
    dit.train()

    # Action adapter
    action_adapter = _make_action_adapter(
        hidden_dim=512, output_dim=cfg.dim
    ).to(device)

    action_params = list(action_adapter.parameters())
    total_model = sum(p.numel() for p in dit.parameters())
    lora_trainable = sum(p.numel() for p in lora_params)
    action_trainable = sum(p.numel() for p in action_params)
    log(f"DiT: {total_model:,} params | LoRA: {lora_trainable:,} | "
        f"Adapter: {action_trainable:,} | "
        f"Trainable: {lora_trainable + action_trainable:,} "
        f"({100*(lora_trainable+action_trainable)/total_model:.3f}%)")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": learning_rate},
        {"params": action_params, "lr": learning_rate * 2},
    ], betas=(0.9, 0.999), weight_decay=0.01)

    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ---- Resume ----
    start_step = 0
    if resume_step > 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step{resume_step}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt")
        if os.path.exists(ckpt_path):
            start_step, opt_state, sched_state = load_checkpoint(
                dit, action_adapter, ckpt_path
            )
            if opt_state:
                optimizer.load_state_dict(opt_state)
            if sched_state:
                scheduler.load_state_dict(sched_state)
            log(f"Resumed from step {start_step}")

    # ---- Dataset (pre-encoded) ----
    log("Loading pre-encoded dataset ...")
    dataset = PreEncodedDataset(LATENT_CACHE_DIR)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # ---- Training loop ----
    log(f"Starting training from step {start_step} to {total_steps}")

    data_iter = iter(dataloader)
    running_loss = 0.0
    step_times = []
    optimizer.zero_grad()

    for step in range(start_step, total_steps):
        step_t0 = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        loss = train_step_cached(
            model=dit,
            action_adapter=action_adapter,
            batch=batch,
            device=device,
            num_train_timesteps=cfg.num_train_timesteps,
            chunk_frames=chunk_frames,
        )

        scaled_loss = loss / gradient_accumulation
        scaled_loss.backward()
        running_loss += loss.item()

        if (step + 1) % gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(lora_params + action_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step_time = time.time() - step_t0
        step_times.append(step_time)

        if (step + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            avg_time = sum(step_times[-log_every:]) / len(step_times[-log_every:])
            current_lr = optimizer.param_groups[0]["lr"]
            eta_seconds = avg_time * (total_steps - step - 1)
            eta_hours = eta_seconds / 3600

            log(
                f"Step {step+1}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Step: {avg_time:.1f}s | "
                f"ETA: {eta_hours:.1f}h"
            )
            running_loss = 0.0

        if (step + 1) % save_every == 0:
            config_dict = {
                "lora_rank": lora_rank, "lora_alpha": lora_alpha,
                "total_steps": total_steps, "learning_rate": learning_rate,
                "chunk_frames": chunk_frames, "dit_dim": cfg.dim,
            }
            save_checkpoint(
                dit, action_adapter, optimizer, scheduler,
                step + 1, CHECKPOINT_DIR, config_dict,
            )

    # Final checkpoint
    config_dict = {
        "lora_rank": lora_rank, "lora_alpha": lora_alpha,
        "total_steps": total_steps, "learning_rate": learning_rate,
        "chunk_frames": chunk_frames, "dit_dim": cfg.dim,
    }
    save_checkpoint(
        dit, action_adapter, optimizer, scheduler,
        total_steps, CHECKPOINT_DIR, config_dict,
    )

    total_time = sum(step_times)
    log("=" * 60)
    log(f"Training complete! {total_steps} steps in {total_time/3600:.1f}h")
    log(f"Avg step time: {total_time/max(len(step_times),1):.1f}s")
    log(f"Checkpoints at {CHECKPOINT_DIR}")
    log("=" * 60)

    checkpoint_volume.commit()


# ============================================================================
# CLI entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    preprocess_data: bool = False,
    steps: int = 2000,
    batch_size: int = 1,
    grad_accum: int = 4,
    lr: float = 1e-4,
    lora_rank: int = 16,
    chunk_frames: int = 5,
    resolution: int = 480,
    num_frames: int = 17,
    save_every: int = 500,
    log_every: int = 10,
    resume_step: int = 0,
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Launch Stage III LoRA training on Modal.

    Examples:
        # Step 1: Pre-encode dataset (run once, ~15-30 min)
        modal run train_lora_modal.py --preprocess-data

        # Step 2: Train LoRA (~3-6 hours on single H100)
        modal run train_lora_modal.py

        # Dry-run (10 steps)
        modal run train_lora_modal.py --dry-run

        # Resume
        modal run train_lora_modal.py --resume-step 500
    """
    if preprocess_data:
        print("=" * 60)
        print("Pre-encoding dataset (VAE + T5)")
        print(f"Resolution: {resolution}p, Frames: {num_frames}")
        print("=" * 60)
        preprocess.remote(num_frames=num_frames, resolution=resolution)
        print("Pre-encoding complete!")
        return

    if dry_run:
        steps = 10
        save_every = 5
        log_every = 2
        print("DRY RUN: 10 steps")

    print("=" * 60)
    print("LingBot-World Stage III Post-Training")
    print("=" * 60)
    print(f"Steps:          {steps}")
    print(f"Batch size:     {batch_size}")
    print(f"Grad accum:     {grad_accum}")
    print(f"Effective batch:{batch_size * grad_accum}")
    print(f"Learning rate:  {lr}")
    print(f"LoRA rank:      {lora_rank}")
    print(f"Chunk frames:   {chunk_frames}")
    print(f"Resume step:    {resume_step}")
    print("=" * 60)

    train.remote(
        total_steps=steps,
        batch_size=batch_size,
        gradient_accumulation=grad_accum,
        learning_rate=lr,
        lora_rank=lora_rank,
        chunk_frames=chunk_frames,
        warmup_steps=200 if not dry_run else 2,
        save_every=save_every,
        log_every=log_every,
        resume_step=resume_step,
        seed=seed,
    )

    print("Training complete!")

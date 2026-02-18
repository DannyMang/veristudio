"""
train_stage1_causal.py -- Stage 1: Causal Architecture Adaptation.

Converts the bidirectional high-noise expert into a causal model via:
  1. Block causal attention (chunk-sequential with accumulated past KV)
  2. Diffusion forcing (per-chunk independent timesteps)
  3. t=0 cache-fill supervision (clean latent forward for KV conditioning)

This is full-parameter training of the 14B high-noise expert.

Usage:
    # Single node, 8xH100
    torchrun --nproc_per_node=8 train_stage1_causal.py \
        --model_dir /path/to/lingbot-world-base-cam \
        --data_dir /path/to/encoded_data \
        --output_dir /path/to/stage1_checkpoints \
        --total_steps 50000

    # Dry run
    torchrun --nproc_per_node=1 train_stage1_causal.py \
        --model_dir /path/to/lingbot-world-base-cam \
        --data_dir /path/to/encoded_data \
        --output_dir /tmp/stage1_test \
        --total_steps 10
"""

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CausalPostTrainingConfig:
    """All hyperparameters for Stage 1 causal adaptation."""

    # Model
    model_dir: str = ""
    expert: str = "high_noise"  # which expert to init from

    # Data
    data_dir: str = ""
    num_chunks: int = 4
    chunk_lat_frames: int = 5  # latent frames per chunk
    resolution: int = 480

    # Training
    total_steps: int = 50000
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 2e-5
    min_lr: float = 2e-6
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    # Diffusion forcing
    target_timesteps: List[int] = field(
        default_factory=lambda: [0, 50, 200, 400, 600, 800, 950, 999]
    )
    num_train_timesteps: int = 1000

    # Checkpointing
    output_dir: str = ""
    save_every: int = 2000
    log_every: int = 50
    gradient_checkpointing: bool = True


# =============================================================================
# Diffusion forcing utilities
# =============================================================================


def sample_diffusion_forcing_timesteps(
    batch_size, num_chunks, target_timesteps, device
):
    """Sample per-chunk timesteps from the target set.

    Each chunk gets an independently sampled timestep from the predefined
    target set. This teaches the model to handle varying noise levels
    across chunks during autoregressive inference.

    Args:
        batch_size: Batch size.
        num_chunks: Number of temporal chunks.
        target_timesteps: List of allowed timestep values.
        device: Target device.

    Returns:
        (B, num_chunks) tensor of timesteps.
    """
    t_set = torch.tensor(target_timesteps, device=device, dtype=torch.long)
    indices = torch.randint(0, len(t_set), (batch_size, num_chunks), device=device)
    return t_set[indices]


def add_noise_per_chunk(latents, noise, timesteps, chunk_lat_frames, num_train_timesteps):
    """Add flow-matching noise with per-chunk sigma.

    x_t = (1 - sigma) * x_0 + sigma * noise
    where sigma = t / num_train_timesteps

    Args:
        latents: (B, C, T_lat, H_lat, W_lat) clean latents.
        noise: (B, C, T_lat, H_lat, W_lat) noise.
        timesteps: (B, num_chunks) per-chunk timesteps.
        chunk_lat_frames: Latent frames per chunk.
        num_train_timesteps: Total training timesteps (1000).

    Returns:
        (B, C, T_lat, H_lat, W_lat) noisy latents.
    """
    B, C, T_lat, H_lat, W_lat = latents.shape
    num_chunks = timesteps.shape[1]
    noisy = torch.zeros_like(latents)

    for c in range(num_chunks):
        s = c * chunk_lat_frames
        e = min(s + chunk_lat_frames, T_lat)
        sigma = timesteps[:, c].float() / num_train_timesteps  # (B,)
        sigma = sigma.view(B, 1, 1, 1, 1)
        noisy[:, :, s:e] = (1.0 - sigma) * latents[:, :, s:e] + sigma * noise[:, :, s:e]

    return noisy


# =============================================================================
# Stage 1 Trainer
# =============================================================================


class Stage1Trainer:
    """
    Stage 1: Causal architecture adaptation trainer.

    Full-parameter training of the high-noise expert with:
    - Block causal attention
    - Diffusion forcing
    - t=0 cache-fill supervision
    - FSDP2 (via PyTorch fully_shard)
    - Gradient checkpointing
    """

    def __init__(self, config: CausalPostTrainingConfig):
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        if self.world_size > 1:
            dist.init_process_group("nccl")

    def setup_model(self):
        """Load high-noise expert and configure for training."""
        wan_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "lingbot-world"
        )
        if wan_root not in sys.path:
            sys.path.insert(0, wan_root)

        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel

        self.wan_cfg = WAN_CONFIGS["i2v-A14B"]

        # Load high-noise expert
        subfolder = (
            self.wan_cfg.high_noise_checkpoint
            if self.config.expert == "high_noise"
            else self.wan_cfg.low_noise_checkpoint
        )
        logger.info(f"Loading {self.config.expert} expert from {self.config.model_dir}")
        self.model = WanModel.from_pretrained(
            self.config.model_dir,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
        )

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint as ckpt_fn

            for block in self.model.blocks:
                orig = block.forward
                block._original_forward = orig

                def _make(o):
                    def f(*a, **kw):
                        return ckpt_fn(o, *a, use_reentrant=False, **kw)
                    return f

                block.forward = _make(orig)
            logger.info("Gradient checkpointing enabled")

        # Full-parameter training
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        # FSDP wrapping
        if self.world_size > 1:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                ShardingStrategy,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            wan_root_mod = __import__("wan.modules.model", fromlist=["WanAttentionBlock"])
            WanAttentionBlock = wan_root_mod.WanAttentionBlock

            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls={WanAttentionBlock},
            )
            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
            self.model = FSDP(
                self.model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mp,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=self.local_rank,
            )
            logger.info(f"FSDP wrapped with {self.world_size} GPUs")
        else:
            self.model.to(self.device)

        # Compute chunk token size for block causal
        vae_stride = self.wan_cfg.vae_stride
        patch_size = self.wan_cfg.patch_size
        # Compute spatial dimensions at the latent/patch level
        aspect_ratio = 480 / 832  # default dashcam aspect
        max_area = 480 * 832
        lat_h = round(
            math.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1]
            * patch_size[1]
        )
        lat_w = round(
            math.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2]
            * patch_size[2]
        )
        tokens_per_lat_frame = lat_h * lat_w // (patch_size[1] * patch_size[2])
        self.chunk_size_tokens = tokens_per_lat_frame * self.config.chunk_lat_frames
        self.tokens_per_lat_frame = tokens_per_lat_frame
        self.lat_h = lat_h
        self.lat_w = lat_w

        logger.info(
            f"Chunk config: {self.config.chunk_lat_frames} lat frames, "
            f"{self.chunk_size_tokens} tokens/chunk, "
            f"lat_h={lat_h}, lat_w={lat_w}"
        )

    def setup_optimizer(self):
        """Configure AdamW optimizer with cosine LR schedule."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay,
        )

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(self.config.warmup_steps, 1)
            progress = (step - self.config.warmup_steps) / max(
                self.config.total_steps - self.config.warmup_steps, 1
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = self.config.min_lr / self.config.learning_rate
            return min_ratio + (1.0 - min_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def setup_data(self):
        """Set up dataset and dataloader."""
        from data_pipeline import PostTrainingDataset

        self.dataset = PostTrainingDataset(
            self.config.data_dir,
            num_chunks=self.config.num_chunks,
            chunk_lat_frames=self.config.chunk_lat_frames,
        )

        sampler = None
        if self.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, num_replicas=self.world_size, rank=self.rank
            )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def train_step(self, batch):
        """Single training step with diffusion forcing + block causal attention.

        Returns:
            loss: scalar tensor.
        """
        latents = batch["latents"].to(self.device)  # (B, C, T_lat, H, W) bf16
        context = batch["context"].to(self.device)  # (B, 512, 4096) bf16
        y = batch["y"].to(self.device)  # (B, 17, T_lat, H, W) bf16
        plucker_emb = batch["plucker_emb"].to(self.device)  # (B, C_p, T_lat, H, W)

        B = latents.shape[0]
        T_lat = latents.shape[2]
        num_chunks = T_lat // self.config.chunk_lat_frames

        # Sample per-chunk timesteps from target set
        chunk_ts = sample_diffusion_forcing_timesteps(
            B, num_chunks, self.config.target_timesteps, self.device
        )

        # Add per-chunk noise
        noise = torch.randn_like(latents)
        noisy_latents = add_noise_per_chunk(
            latents, noise, chunk_ts,
            self.config.chunk_lat_frames,
            self.config.num_train_timesteps,
        )

        # Build per-token timestep tensor
        # Each latent frame in a chunk gets the same timestep
        tokens_per_chunk = self.chunk_size_tokens
        seq_len = num_chunks * tokens_per_chunk

        # Expand chunk timesteps to per-token
        frame_ts = chunk_ts.repeat_interleave(self.config.chunk_lat_frames, dim=1)
        token_ts = frame_ts.repeat_interleave(self.tokens_per_lat_frame, dim=1)

        # Prepare model inputs
        x_list = [noisy_latents[b] for b in range(B)]
        y_list = [y[b] for b in range(B)]
        ctx_list = [context[b] for b in range(B)]

        # Plucker embedding for camera conditioning
        dit_cond_dict = {
            "c2ws_plucker_emb": [plucker_emb[b:b+1] for b in range(B)],
        }

        # Forward with block causal attention
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = self.model(
                x=x_list,
                t=token_ts,
                context=ctx_list,
                seq_len=seq_len,
                y=y_list,
                dit_cond_dict=dit_cond_dict,
                block_causal=True,
                chunk_size_tokens=self.chunk_size_tokens,
            )

        # Flow matching velocity loss: v_target = noise - x_0
        loss = 0.0
        for b in range(B):
            target = noise[b] - latents[b]
            loss = loss + F.mse_loss(preds[b].float(), target.float())
        loss = loss / B

        return loss

    def cache_fill_step(self, batch):
        """t=0 cache-fill supervision step.

        Forward clean latent at t=0, then denoise next chunk using that cache.
        This bridges the high-noise expert to handle clean encoding.

        Returns:
            loss: scalar tensor (or 0 if not enough chunks).
        """
        latents = batch["latents"].to(self.device)
        context = batch["context"].to(self.device)
        y = batch["y"].to(self.device)
        plucker_emb = batch["plucker_emb"].to(self.device)

        B = latents.shape[0]
        T_lat = latents.shape[2]
        clf = self.config.chunk_lat_frames

        if T_lat < 2 * clf:
            return torch.tensor(0.0, device=self.device)

        # Split into first chunk (clean) and second chunk (noisy)
        first_chunk = latents[:, :, :clf]
        second_chunk = latents[:, :, clf:2*clf]

        # Add noise to second chunk at a random timestep
        t_val = torch.randint(
            100, self.config.num_train_timesteps, (B, 1), device=self.device
        )
        sigma = t_val.float() / self.config.num_train_timesteps
        noise_2 = torch.randn_like(second_chunk)
        noisy_2 = (
            (1.0 - sigma.view(B, 1, 1, 1, 1)) * second_chunk
            + sigma.view(B, 1, 1, 1, 1) * noise_2
        )

        # Concatenate: clean first chunk + noisy second chunk
        combined = torch.cat([first_chunk, noisy_2], dim=2)  # (B, C, 2*clf, H, W)

        # Build timesteps: t=0 for first chunk, t_val for second
        seq_per_chunk = self.chunk_size_tokens
        seq_len = 2 * seq_per_chunk

        t0_tokens = torch.zeros(B, seq_per_chunk, device=self.device, dtype=torch.long)
        t1_tokens = t_val.expand(B, seq_per_chunk)
        token_ts = torch.cat([t0_tokens, t1_tokens], dim=1)

        x_list = [combined[b] for b in range(B)]
        y_list = [y[b, :, :2*clf] for b in range(B)]
        ctx_list = [context[b] for b in range(B)]

        dit_cond_dict = {
            "c2ws_plucker_emb": [plucker_emb[b:b+1, :, :2*clf] for b in range(B)],
        }

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = self.model(
                x=x_list,
                t=token_ts,
                context=ctx_list,
                seq_len=seq_len,
                y=y_list,
                dit_cond_dict=dit_cond_dict,
                block_causal=True,
                chunk_size_tokens=self.chunk_size_tokens,
            )

        # Loss only on the second chunk (the denoised part)
        # We want the model to use the clean first chunk's KV to help denoise
        loss = 0.0
        for b in range(B):
            # Extract second chunk's prediction
            pred_2 = preds[b][:, clf:2*clf]
            target_2 = noise_2[b] - second_chunk[b]
            loss = loss + F.mse_loss(pred_2.float(), target_2.float())
        loss = loss / B

        return loss

    def save_checkpoint(self, step):
        """Save training checkpoint."""
        if self.rank != 0:
            return

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Get model state dict (handles FSDP)
        if self.world_size > 1:
            from torch.distributed.fsdp import (
                FullStateDictConfig,
                StateDictType,
            )

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with self.model.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                state_dict = self.model.state_dict()
        else:
            state_dict = self.model.state_dict()

        checkpoint = {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": vars(self.config),
        }

        path = os.path.join(self.config.output_dir, f"stage1_step{step}.pt")
        torch.save(checkpoint, path)
        torch.save(
            checkpoint,
            os.path.join(self.config.output_dir, "stage1_latest.pt"),
        )

        size_gb = os.path.getsize(path) / 1e9
        logger.info(f"Saved checkpoint step {step} -> {path} ({size_gb:.1f} GB)")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and return starting step."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model = self.model
        if hasattr(model, "module"):
            model = model.module
        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from step {ckpt['step']}")
        return ckpt["step"]

    def train(self, resume_path=None):
        """Main training loop."""
        self.setup_model()
        self.setup_optimizer()
        self.setup_data()

        start_step = 0
        if resume_path and os.path.exists(resume_path):
            start_step = self.load_checkpoint(resume_path)

        data_iter = iter(self.dataloader)
        running_loss = 0.0
        running_cf_loss = 0.0
        step_times = []

        self.optimizer.zero_grad()

        logger.info(
            f"Starting Stage 1 training: steps {start_step}-{self.config.total_steps}, "
            f"batch={self.config.batch_size}, grad_accum={self.config.gradient_accumulation}, "
            f"effective_batch={self.config.batch_size * self.config.gradient_accumulation * self.world_size}"
        )

        for step in range(start_step, self.config.total_steps):
            step_t0 = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # Main diffusion forcing loss
            loss = self.train_step(batch)

            # Cache-fill loss (every 4th step to save compute)
            cf_loss = torch.tensor(0.0, device=self.device)
            if step % 4 == 0:
                cf_loss = self.cache_fill_step(batch)
                loss = loss + 0.1 * cf_loss  # Weight cache-fill loss lower

            scaled_loss = loss / self.config.gradient_accumulation
            scaled_loss.backward()

            running_loss += loss.item()
            running_cf_loss += cf_loss.item()

            if (step + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            step_time = time.time() - step_t0
            step_times.append(step_time)

            # Logging
            if (step + 1) % self.config.log_every == 0 and self.rank == 0:
                avg_loss = running_loss / self.config.log_every
                avg_cf = running_cf_loss / self.config.log_every
                avg_time = sum(step_times[-self.config.log_every:]) / len(
                    step_times[-self.config.log_every:]
                )
                lr = self.optimizer.param_groups[0]["lr"]
                eta_h = avg_time * (self.config.total_steps - step - 1) / 3600

                logger.info(
                    f"Step {step+1}/{self.config.total_steps} | "
                    f"Loss: {avg_loss:.4f} | CF: {avg_cf:.4f} | "
                    f"LR: {lr:.2e} | Step: {avg_time:.1f}s | ETA: {eta_h:.1f}h"
                )
                running_loss = 0.0
                running_cf_loss = 0.0

            # Save
            if (step + 1) % self.config.save_every == 0:
                self.save_checkpoint(step + 1)

        # Final checkpoint
        self.save_checkpoint(self.config.total_steps)

        if self.world_size > 1:
            dist.destroy_process_group()

        logger.info("Stage 1 training complete!")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Causal Architecture Adaptation"
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--chunk_lat_frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    torch.manual_seed(args.seed)

    config = CausalPostTrainingConfig(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        num_chunks=args.num_chunks,
        chunk_lat_frames=args.chunk_lat_frames,
        seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    trainer = Stage1Trainer(config)
    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()

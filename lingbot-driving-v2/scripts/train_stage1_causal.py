"""
train_stage1_causal_patched.py -- Stage 1: Causal Architecture Adaptation (PATCHED).

Changes from original:
  1. cache_fill_step runs EVERY step (not every 4th) with 0.5 weight (not 0.1)
     → cache-fill share of gradient: 33% (was 2.5%)
  2. NEW: multi-depth cache-fill — trains on 2-chunk and 3-chunk cache depths,
     not just 1-chunk, so the model learns to use longer cached histories
  3. NEW: optional light self-conditioning — on 25% of steps after warmup,
     replaces ground-truth past chunks with the model's own predictions
     (detached), exposing it to the distribution it encounters at inference
  4. NEW: per-chunk diagnostic logging — prints per-chunk loss breakdown so
     you can see exactly where quality drops off

Usage: same as original (torchrun --nproc_per_node=8 ...)
"""

import argparse
import logging
import math
import os
import random
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
    expert: str = "high_noise"

    # Data
    data_dir: str = ""
    num_chunks: int = 4
    chunk_lat_frames: int = 5
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

    # --- PATCHED: cache-fill and self-conditioning ---
    cache_fill_weight: float = 0.5        # was 0.1; share of total gradient
    cache_fill_max_depth: int = 3         # train on up to 3 past chunks
    self_cond_prob: float = 0.25          # fraction of steps using self-generated context
    self_cond_start_step: int = 5000      # don't self-condition until warmed up

    # Checkpointing
    output_dir: str = ""
    save_every: int = 2000
    log_every: int = 50
    gradient_checkpointing: bool = True

    # W&B
    wandb_project: str = "lingbot-posttrain"
    wandb_run_name: str = ""
    use_wandb: bool = True


# =============================================================================
# Diffusion forcing utilities
# =============================================================================


def sample_diffusion_forcing_timesteps(
    batch_size, num_chunks, target_timesteps, device
):
    """Sample per-chunk timesteps from the target set."""
    t_set = torch.tensor(target_timesteps, device=device, dtype=torch.long)
    indices = torch.randint(0, len(t_set), (batch_size, num_chunks), device=device)
    return t_set[indices]


def add_noise_per_chunk(latents, noise, timesteps, chunk_lat_frames, num_train_timesteps):
    """Add flow-matching noise with per-chunk sigma."""
    B, C, T_lat, H_lat, W_lat = latents.shape
    num_chunks = timesteps.shape[1]
    noisy = torch.zeros_like(latents)

    for c in range(num_chunks):
        s = c * chunk_lat_frames
        e = min(s + chunk_lat_frames, T_lat)
        sigma = timesteps[:, c].float() / num_train_timesteps
        sigma = sigma.view(B, 1, 1, 1, 1)
        noisy[:, :, s:e] = (1.0 - sigma) * latents[:, :, s:e] + sigma * noise[:, :, s:e]

    return noisy


# =============================================================================
# Stage 1 Trainer (PATCHED)
# =============================================================================


class Stage1Trainer:
    def __init__(self, config: CausalPostTrainingConfig):
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        if self.world_size > 1:
            dist.init_process_group("nccl")

    # -----------------------------------------------------------------
    # Model setup (unchanged from original)
    # -----------------------------------------------------------------

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

        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_gb = total_params * 2 / 1e9
        logger.info(f"Model loaded: {total_params/1e9:.2f}B params ({model_size_gb:.1f} GB in bf16)")

        from model_patches import patch_model_for_block_causal
        patch_model_for_block_causal(self.model)
        logger.info("Applied block causal monkey-patch")

        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        # FSDP wrapping
        if self.world_size > 1:
            from functools import partial

            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                ShardingStrategy,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            wan_root_mod = __import__("wan.modules.model", fromlist=["WanAttentionBlock"])
            WanAttentionBlock = wan_root_mod.WanAttentionBlock

            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
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
                use_orig_params=True,
                limit_all_gathers=True,
                forward_prefetch=False,
            )
            logger.info(f"FSDP wrapped with {self.world_size} GPUs")

            local_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Local param count after FSDP: {local_params/1e9:.2f}B "
                f"(expect ~{local_params * self.world_size / 1e9:.1f}B total)"
            )

            # FSDP-aware activation checkpointing
            if self.config.gradient_checkpointing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                )

                non_reentrant_wrapper = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
                apply_activation_checkpointing(
                    self.model,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=lambda module: isinstance(module, WanAttentionBlock),
                )

                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointWrapper,
                )
                ckpt_count = sum(
                    1 for m in self.model.modules()
                    if isinstance(m, CheckpointWrapper)
                )
                logger.info(f"FSDP-aware activation checkpointing: {ckpt_count} wrapped modules")
        else:
            self.model.to(self.device)

            if self.config.gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint as ckpt_fn

                for block in self.model.blocks:
                    orig = block.forward

                    def _make(o):
                        def f(*a, **kw):
                            return ckpt_fn(o, *a, use_reentrant=False, **kw)
                        return f

                    block.forward = _make(orig)
                logger.info("Gradient checkpointing enabled (single GPU)")

        # Compute chunk token sizes
        vae_stride = self.wan_cfg.vae_stride
        patch_size = self.wan_cfg.patch_size
        aspect_ratio = 480 / 832
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

        if self.rank == 0:
            self._log_gpu_memory("after model setup")

    def _log_gpu_memory(self, label=""):
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_alloc = torch.cuda.max_memory_allocated(self.device) / 1e9
        logger.info(
            f"GPU mem [{label}]: "
            f"alloc={alloc:.2f} GB, reserved={reserved:.2f} GB, "
            f"peak={max_alloc:.2f} GB"
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

    # -----------------------------------------------------------------
    # Core training steps
    # -----------------------------------------------------------------

    def train_step(self, batch):
        """Single training step with diffusion forcing + block causal attention."""
        latents = batch["latents"].to(self.device)
        context = batch["context"].to(self.device)
        y = batch["y"].to(self.device)
        plucker_emb = batch["plucker_emb"].to(self.device)

        B, C, T_lat, H_lat, W_lat = latents.shape
        num_chunks = T_lat // self.config.chunk_lat_frames

        patch_h, patch_w = self.wan_cfg.patch_size[1], self.wan_cfg.patch_size[2]
        tokens_per_lat_frame = (H_lat // patch_h) * (W_lat // patch_w)
        tokens_per_chunk = tokens_per_lat_frame * self.config.chunk_lat_frames
        seq_len = num_chunks * tokens_per_chunk

        chunk_ts = sample_diffusion_forcing_timesteps(
            B, num_chunks, self.config.target_timesteps, self.device
        )

        noise = torch.randn_like(latents)
        noisy_latents = add_noise_per_chunk(
            latents, noise, chunk_ts,
            self.config.chunk_lat_frames,
            self.config.num_train_timesteps,
        )

        frame_ts = chunk_ts.repeat_interleave(self.config.chunk_lat_frames, dim=1)
        token_ts = frame_ts.repeat_interleave(tokens_per_lat_frame, dim=1)

        x_list = [noisy_latents[b] for b in range(B)]
        y_list = [y[b] for b in range(B)]
        ctx_list = [context[b] for b in range(B)]

        dit_cond_dict = {
            "c2ws_plucker_emb": [plucker_emb[b:b+1] for b in range(B)],
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
                chunk_size_tokens=tokens_per_chunk,
            )

        loss = 0.0
        for b in range(B):
            target = noise[b] - latents[b]
            loss = loss + F.mse_loss(preds[b].float(), target.float())
        loss = loss / B

        return loss

    # -----------------------------------------------------------------
    # FIX 2: Multi-depth cache-fill step
    # -----------------------------------------------------------------

    def cache_fill_step(self, batch):
        """Multi-depth t=0 cache-fill supervision (PATCHED).

        Randomly samples a cache depth of 2 or 3 chunks:
          - depth=2: clean chunk 1 -> denoise chunk 2  (50%)
          - depth=3: clean chunks 1-2 -> denoise chunk 3  (50%)

        This teaches the model to use longer cached histories, preventing
        degradation at chunks 3+ which only saw depth-2 training before.

        Returns:
            (loss, depth) tuple.
        """
        latents = batch["latents"].to(self.device)
        context = batch["context"].to(self.device)
        y = batch["y"].to(self.device)
        plucker_emb = batch["plucker_emb"].to(self.device)

        B, C, T_lat, H_lat, W_lat = latents.shape
        clf = self.config.chunk_lat_frames
        max_chunks_available = T_lat // clf

        if max_chunks_available < 2:
            return torch.tensor(0.0, device=self.device), 0

        # Sample cache depth: how many total chunks (clean + noisy)
        # depth=2: 1 clean + 1 noisy; depth=3: 2 clean + 1 noisy
        max_depth = min(self.config.cache_fill_max_depth, max_chunks_available)
        depth = random.randint(2, max_depth)

        patch_h, patch_w = self.wan_cfg.patch_size[1], self.wan_cfg.patch_size[2]
        tokens_per_lat_frame = (H_lat // patch_h) * (W_lat // patch_w)
        seq_per_chunk = tokens_per_lat_frame * clf

        # Split: first (depth-1) chunks are clean, last chunk is noisy
        num_clean = depth - 1
        clean_latents = latents[:, :, :num_clean * clf]
        target_chunk = latents[:, :, num_clean * clf:(num_clean + 1) * clf]

        # Add noise to the target chunk
        t_val = torch.randint(
            1, self.config.num_train_timesteps, (B, 1), device=self.device
        )
        sigma = t_val.float() / self.config.num_train_timesteps
        noise_t = torch.randn_like(target_chunk)
        noisy_target = (
            (1.0 - sigma.view(B, 1, 1, 1, 1)) * target_chunk
            + sigma.view(B, 1, 1, 1, 1) * noise_t
        )

        # Concatenate: clean past chunks + noisy target chunk
        combined = torch.cat([clean_latents, noisy_target], dim=2)
        total_lat = combined.shape[2]
        seq_len = depth * seq_per_chunk

        # Build timesteps: t=0 for all clean chunks, t_val for target chunk
        clean_tokens = num_clean * seq_per_chunk
        target_tokens = seq_per_chunk

        t_clean = torch.zeros(B, clean_tokens, device=self.device, dtype=torch.long)
        t_target = t_val.expand(B, target_tokens)
        token_ts = torch.cat([t_clean, t_target], dim=1)

        x_list = [combined[b] for b in range(B)]
        y_list = [y[b, :, :total_lat] for b in range(B)]
        ctx_list = [context[b] for b in range(B)]

        dit_cond_dict = {
            "c2ws_plucker_emb": [plucker_emb[b:b+1, :, :total_lat] for b in range(B)],
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
                chunk_size_tokens=seq_per_chunk,
            )

        # Loss only on the target chunk (last chunk in the sequence)
        loss = 0.0
        for b in range(B):
            pred_target = preds[b][:, num_clean * clf:(num_clean + 1) * clf]
            velocity_target = noise_t[b] - target_chunk[b]
            loss = loss + F.mse_loss(pred_target.float(), velocity_target.float())
        loss = loss / B

        return loss, depth

    # -----------------------------------------------------------------
    # FIX 3: Self-conditioning step
    # -----------------------------------------------------------------

    def self_cond_train_step(self, batch):
        """Training step where past chunks use model's own predictions (PATCHED).

        Instead of ground-truth clean latents for past context, we:
        1. Run a detached forward pass to get the model's prediction for chunk 1
        2. Use that prediction as context for denoising chunk 2

        This exposes the model to the distribution it encounters at inference
        (its own noisy outputs), partially bridging the train-test gap.

        Returns:
            loss: scalar tensor.
        """
        latents = batch["latents"].to(self.device)
        context = batch["context"].to(self.device)
        y = batch["y"].to(self.device)
        plucker_emb = batch["plucker_emb"].to(self.device)

        B, C, T_lat, H_lat, W_lat = latents.shape
        clf = self.config.chunk_lat_frames

        if T_lat < 2 * clf:
            return self.train_step(batch)

        patch_h, patch_w = self.wan_cfg.patch_size[1], self.wan_cfg.patch_size[2]
        tokens_per_lat_frame = (H_lat // patch_h) * (W_lat // patch_w)
        seq_per_chunk = tokens_per_lat_frame * clf

        # Step 1: Generate model's prediction for chunk 1 (detached)
        chunk1_gt = latents[:, :, :clf]
        t1 = torch.randint(
            1, self.config.num_train_timesteps, (B,), device=self.device
        )
        sigma1 = t1.float() / self.config.num_train_timesteps
        noise1 = torch.randn_like(chunk1_gt)
        noisy1 = (1 - sigma1.view(B, 1, 1, 1, 1)) * chunk1_gt + sigma1.view(B, 1, 1, 1, 1) * noise1

        token_ts_1 = t1.unsqueeze(1).expand(B, seq_per_chunk)
        x1_list = [noisy1[b] for b in range(B)]
        y1_list = [y[b, :, :clf] for b in range(B)]
        ctx_list = [context[b] for b in range(B)]
        dit_cond_1 = {
            "c2ws_plucker_emb": [plucker_emb[b:b+1, :, :clf] for b in range(B)],
        }

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds_1 = self.model(
                x=x1_list, t=token_ts_1, context=ctx_list,
                seq_len=seq_per_chunk, y=y1_list,
                dit_cond_dict=dit_cond_1,
                block_causal=True,
                chunk_size_tokens=seq_per_chunk,
            )

        # Recover x_0 prediction from velocity: x_0 = x_t - sigma * v_pred
        self_chunk1 = torch.stack([
            noisy1[b] - sigma1[b].view(1, 1, 1, 1) * preds_1[b]
            for b in range(B)
        ]).detach()

        # Step 2: Train chunk 2 using self-generated chunk 1 as context
        chunk2_gt = latents[:, :, clf:2*clf]
        t2 = torch.randint(
            1, self.config.num_train_timesteps, (B, 1), device=self.device
        )
        sigma2 = t2.float() / self.config.num_train_timesteps
        noise2 = torch.randn_like(chunk2_gt)
        noisy2 = (1 - sigma2.view(B, 1, 1, 1, 1)) * chunk2_gt + sigma2.view(B, 1, 1, 1, 1) * noise2

        # Use self_chunk1 (model's prediction) instead of chunk1_gt
        combined = torch.cat([self_chunk1, noisy2], dim=2)
        seq_len = 2 * seq_per_chunk

        # t=0 for self-generated chunk 1, t2 for chunk 2
        t_past = torch.zeros(B, seq_per_chunk, device=self.device, dtype=torch.long)
        t_curr = t2.expand(B, seq_per_chunk)
        token_ts = torch.cat([t_past, t_curr], dim=1)

        x_list = [combined[b] for b in range(B)]
        y_list = [y[b, :, :2*clf] for b in range(B)]
        dit_cond = {
            "c2ws_plucker_emb": [plucker_emb[b:b+1, :, :2*clf] for b in range(B)],
        }

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = self.model(
                x=x_list, t=token_ts, context=ctx_list,
                seq_len=seq_len, y=y_list,
                dit_cond_dict=dit_cond,
                block_causal=True,
                chunk_size_tokens=seq_per_chunk,
            )

        # Loss on chunk 2 only
        loss = 0.0
        for b in range(B):
            pred_2 = preds[b][:, clf:2*clf]
            target_2 = noise2[b] - chunk2_gt[b]
            loss = loss + F.mse_loss(pred_2.float(), target_2.float())
        loss = loss / B

        return loss

    # -----------------------------------------------------------------
    # FIX 4: Per-chunk diagnostic logging
    # -----------------------------------------------------------------

    def log_per_chunk_loss(self, batch):
        """Compute and log per-chunk loss breakdown (diagnostic, no grad)."""
        latents = batch["latents"].to(self.device)
        context = batch["context"].to(self.device)
        y = batch["y"].to(self.device)
        plucker_emb = batch["plucker_emb"].to(self.device)

        B, C, T_lat, H_lat, W_lat = latents.shape
        clf = self.config.chunk_lat_frames
        num_chunks = T_lat // clf

        patch_h, patch_w = self.wan_cfg.patch_size[1], self.wan_cfg.patch_size[2]
        tokens_per_lat_frame = (H_lat // patch_h) * (W_lat // patch_w)
        tokens_per_chunk = tokens_per_lat_frame * clf
        seq_len = num_chunks * tokens_per_chunk

        # Fixed mid-range timestep for comparable measurement across chunks
        fixed_t = 500
        chunk_ts = torch.full(
            (B, num_chunks), fixed_t, device=self.device, dtype=torch.long
        )

        noise = torch.randn_like(latents)
        noisy_latents = add_noise_per_chunk(
            latents, noise, chunk_ts, clf, self.config.num_train_timesteps
        )

        frame_ts = chunk_ts.repeat_interleave(clf, dim=1)
        token_ts = frame_ts.repeat_interleave(tokens_per_lat_frame, dim=1)

        x_list = [noisy_latents[b] for b in range(B)]
        y_list = [y[b] for b in range(B)]
        ctx_list = [context[b] for b in range(B)]
        dit_cond_dict = {
            "c2ws_plucker_emb": [plucker_emb[b:b+1] for b in range(B)],
        }

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = self.model(
                x=x_list, t=token_ts, context=ctx_list,
                seq_len=seq_len, y=y_list,
                dit_cond_dict=dit_cond_dict,
                block_causal=True,
                chunk_size_tokens=tokens_per_chunk,
            )

        target = noise - latents
        chunk_losses = []
        for c in range(num_chunks):
            s = c * clf
            e = s + clf
            c_loss = 0.0
            for b in range(B):
                c_loss += F.mse_loss(preds[b][:, s:e].float(), target[b, :, s:e].float()).item()
            c_loss /= B
            chunk_losses.append(c_loss)

        parts = " | ".join(f"C{c}: {l:.4f}" for c, l in enumerate(chunk_losses))
        logger.info(f"  Per-chunk loss (t={fixed_t}): {parts}")

        return chunk_losses

    # -----------------------------------------------------------------
    # Checkpoint save/load (unchanged from original)
    # -----------------------------------------------------------------

    def save_checkpoint(self, step):
        os.makedirs(self.config.output_dir, exist_ok=True)

        if self.world_size > 1:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                FullOptimStateDictConfig,
                FullStateDictConfig,
                StateDictType,
            )
            model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, model_cfg, optim_cfg
            ):
                state_dict = self.model.state_dict()
                optim_sd = FSDP.optim_state_dict(self.model, self.optimizer)
        else:
            state_dict = self.model.state_dict()
            optim_sd = self.optimizer.state_dict()

        if self.rank != 0:
            return

        checkpoint = {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optim_sd,
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
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if self.world_size > 1:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
            )
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                osd = FSDP.optim_state_dict_to_load(
                    self.model, self.optimizer, ckpt["optimizer_state_dict"],
                )
                self.optimizer.load_state_dict(osd)
        else:
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from step {ckpt['step']}")
        return ckpt["step"]

    # -----------------------------------------------------------------
    # Main training loop (PATCHED)
    # -----------------------------------------------------------------

    def train(self, resume_path=None):
        self.setup_model()
        self.setup_optimizer()
        self.setup_data()

        # W&B init
        if self.config.use_wandb and self.rank == 0:
            import wandb
            run_name = self.config.wandb_run_name or f"stage1-patched-{self.config.total_steps}steps"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=vars(self.config),
            )
            self._wandb = wandb
        else:
            self._wandb = None

        start_step = 0
        if resume_path and os.path.exists(resume_path):
            start_step = self.load_checkpoint(resume_path)

        data_iter = iter(self.dataloader)
        running_loss = 0.0
        running_cf_loss = 0.0
        running_sc_loss = 0.0
        step_times = []
        grad_norm = None

        self.optimizer.zero_grad()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        if self.rank == 0:
            self._log_gpu_memory("before training loop")

        logger.info(
            f"Starting Stage 1 PATCHED training: steps {start_step}-{self.config.total_steps}, "
            f"cache_fill_weight={self.config.cache_fill_weight}, "
            f"cache_fill_max_depth={self.config.cache_fill_max_depth}, "
            f"self_cond_prob={self.config.self_cond_prob} (start@{self.config.self_cond_start_step}), "
            f"effective_batch={self.config.batch_size * self.config.gradient_accumulation * self.world_size}"
        )

        for step in range(start_step, self.config.total_steps):
            step_t0 = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            if step == start_step and self.rank == 0:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"  batch[{k}]: {v.shape} {v.dtype}")
                self._log_gpu_memory("before first forward")

            # === Main diffusion forcing loss ===
            # Backward immediately to free graph before cache_fill creates another
            loss = self.train_step(batch)
            scaled_loss = loss / self.config.gradient_accumulation
            scaled_loss.backward()

            # === FIX 1+2: Cache-fill EVERY step with higher weight + multi-depth ===
            cf_loss, cf_depth = self.cache_fill_step(batch)
            scaled_cf = self.config.cache_fill_weight * cf_loss / self.config.gradient_accumulation
            scaled_cf.backward()
            cf_loss_val = cf_loss.item()

            # === FIX 3: Self-conditioning (after warmup, 25% of steps) ===
            sc_loss_val = 0.0
            use_self_cond = (
                step >= self.config.self_cond_start_step
                and random.random() < self.config.self_cond_prob
            )
            if use_self_cond:
                sc_loss = self.self_cond_train_step(batch)
                scaled_sc = 0.3 * sc_loss / self.config.gradient_accumulation
                scaled_sc.backward()
                sc_loss_val = sc_loss.item()

            running_loss += loss.item()
            running_cf_loss += cf_loss_val
            running_sc_loss += sc_loss_val

            # Optimizer step on gradient accumulation boundary
            if (step + 1) % self.config.gradient_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            step_time = time.time() - step_t0
            step_times.append(step_time)

            # Per-step logging for short runs
            if self.rank == 0 and self.config.total_steps <= 20:
                logger.info(
                    f"Step {step+1}/{self.config.total_steps} | "
                    f"Loss: {loss.item():.4f} | CF: {cf_loss_val:.4f} (depth={cf_depth}) | "
                    f"SC: {sc_loss_val:.4f} | Time: {step_time:.2f}s"
                )
                self._log_gpu_memory(f"step {step+1}")

            # Periodic logging
            if (step + 1) % self.config.log_every == 0 and self.rank == 0:
                n = self.config.log_every
                avg_loss = running_loss / n
                avg_cf = running_cf_loss / n
                avg_sc = running_sc_loss / n
                avg_time = sum(step_times[-n:]) / len(step_times[-n:])
                lr = self.optimizer.param_groups[0]["lr"]
                eta_h = avg_time * (self.config.total_steps - step - 1) / 3600

                logger.info(
                    f"Step {step+1}/{self.config.total_steps} | "
                    f"Loss: {avg_loss:.4f} | CF: {avg_cf:.4f} | SC: {avg_sc:.4f} | "
                    f"LR: {lr:.2e} | Step: {avg_time:.1f}s | ETA: {eta_h:.1f}h"
                )

                # FIX 4: Per-chunk loss diagnostic every log_every steps
                self.log_per_chunk_loss(batch)

                if self._wandb is not None:
                    log_dict = {
                        "loss": avg_loss,
                        "cache_fill_loss": avg_cf,
                        "self_cond_loss": avg_sc,
                        "lr": lr,
                        "step_time": avg_time,
                        "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                    }
                    self._wandb.log(log_dict, step=step + 1)

                running_loss = 0.0
                running_cf_loss = 0.0
                running_sc_loss = 0.0

            # Save
            if (step + 1) % self.config.save_every == 0:
                self.save_checkpoint(step + 1)

        # Final checkpoint
        self.save_checkpoint(self.config.total_steps)

        if self._wandb is not None:
            self._wandb.finish()

        if self.world_size > 1:
            dist.destroy_process_group()

        logger.info("Stage 1 PATCHED training complete!")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Causal Architecture Adaptation (PATCHED)"
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
    parser.add_argument("--wandb_project", type=str, default="lingbot-posttrain")
    parser.add_argument("--no_wandb", action="store_true")

    # Patched hyperparameters
    parser.add_argument("--cache_fill_weight", type=float, default=0.5,
                        help="Cache-fill loss weight (default: 0.5, was 0.1)")
    parser.add_argument("--cache_fill_max_depth", type=int, default=3,
                        help="Max chunks in cache-fill training (default: 3)")
    parser.add_argument("--self_cond_prob", type=float, default=0.25,
                        help="Fraction of steps using self-conditioning (default: 0.25)")
    parser.add_argument("--self_cond_start_step", type=int, default=5000,
                        help="Step to begin self-conditioning (default: 5000)")

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
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
        cache_fill_weight=args.cache_fill_weight,
        cache_fill_max_depth=args.cache_fill_max_depth,
        self_cond_prob=args.self_cond_prob,
        self_cond_start_step=args.self_cond_start_step,
    )

    trainer = Stage1Trainer(config)
    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()

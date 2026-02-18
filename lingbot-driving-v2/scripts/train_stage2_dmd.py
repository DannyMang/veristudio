"""
train_stage2_dmd.py -- Stage 2: DMD + Adversarial + Self-Rollout Distillation.

Three models in play:
  - Student (14B, from Stage 1 checkpoint, trainable)
  - Teacher (28B MoE frozen, both experts, CPU-offloaded)
  - Fake Score (14B, initialized from high-noise expert, trainable)

Training loop per step:
  1. Self-rollout: student generates L chunks autoregressively with KV cache
  2. DMD loss: compare student vs teacher-fake difference
  3. Adversarial generator loss: fool discriminator
  4. Update student
  5. Update fake score + discriminator (two-timescale)

Usage:
    torchrun --nproc_per_node=8 train_stage2_dmd.py \
        --model_dir /path/to/lingbot-world-base-cam \
        --student_ckpt /path/to/stage1_latest.pt \
        --data_dir /path/to/encoded_data \
        --output_dir /path/to/stage2_checkpoints
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
class DMDConfig:
    """All hyperparameters for Stage 2 DMD distillation."""

    # Model paths
    model_dir: str = ""
    student_ckpt: str = ""  # Stage 1 checkpoint

    # Data
    data_dir: str = ""
    num_chunks: int = 4
    chunk_lat_frames: int = 5
    resolution: int = 480

    # Training
    total_steps: int = 100000
    batch_size: int = 1
    gradient_accumulation: int = 4
    student_lr: float = 1e-5
    fake_score_lr: float = 2e-5
    disc_lr: float = 2e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    # DMD
    dmd_weight: float = 1.0
    adv_weight: float = 0.1
    disc_update_ratio: int = 2  # update fake_score N times per student update

    # Self-rollout
    rollout_chunks: int = 3  # how many chunks to generate in self-rollout
    rollout_steps: int = 4  # denoising steps per chunk during rollout
    grad_trunc_k: int = 2  # backprop only through last K chunks

    # Inference
    num_train_timesteps: int = 1000
    target_timesteps: List[int] = field(
        default_factory=lambda: [200, 400, 600, 800]
    )

    # Checkpointing
    output_dir: str = ""
    save_every: int = 5000
    log_every: int = 50
    gradient_checkpointing: bool = True


# =============================================================================
# Discriminator head (cross-attention based, Fig. 6b)
# =============================================================================


class DiscriminatorHead(nn.Module):
    """Cross-attention based discriminator head.

    Takes a latent video representation and produces a scalar score
    indicating real (from teacher) vs fake (from student).

    Architecture (from LingBot Fig. 6b):
        - Learnable query tokens
        - Cross-attention to the video latent sequence
        - MLP classification head
    """

    def __init__(self, dim=5120, num_heads=40, num_query_tokens=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_query_tokens = num_query_tokens
        self.head_dim = dim // num_heads

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, dim) * 0.02
        )

        # Cross-attention components
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x_seq):
        """
        Args:
            x_seq: (B, L, dim) -- flattened latent video tokens.

        Returns:
            score: (B,) -- discriminator score per sample.
        """
        B = x_seq.shape[0]

        q = self.q_proj(self.norm_q(self.query_tokens.expand(B, -1, -1)))
        k = self.k_proj(self.norm_k(x_seq))
        v = self.v_proj(x_seq)

        # Reshape for multi-head attention
        q = q.view(B, self.num_query_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, self.num_query_tokens, self.dim)
        attn = self.o_proj(attn)

        # Pool query tokens and classify
        pooled = attn.mean(dim=1)  # (B, dim)
        score = self.head(pooled).squeeze(-1)  # (B,)

        return score


# =============================================================================
# FakeScoreNetwork
# =============================================================================


class FakeScoreNetwork(nn.Module):
    """Wraps WanModel + DiscriminatorHead.

    The fake score network provides the "fake" score function for DMD loss.
    It's initialized from the high-noise expert and trained alongside
    the discriminator.
    """

    def __init__(self, wan_model, dim=5120, num_heads=40):
        super().__init__()
        self.model = wan_model
        self.discriminator = DiscriminatorHead(dim=dim, num_heads=num_heads)

    def forward_score(self, x, t, context, seq_len, y=None,
                      dit_cond_dict=None, block_causal=False,
                      chunk_size_tokens=0):
        """Forward pass through the score network (WanModel)."""
        return self.model(
            x=x, t=t, context=context, seq_len=seq_len,
            y=y, dit_cond_dict=dit_cond_dict,
            block_causal=block_causal,
            chunk_size_tokens=chunk_size_tokens,
        )

    def discriminate(self, x_seq):
        """Discriminator forward pass.

        Args:
            x_seq: (B, L, dim) video latent tokens.

        Returns:
            score: (B,) per-sample score.
        """
        return self.discriminator(x_seq)


# =============================================================================
# Loss functions
# =============================================================================


def dmd_loss(student_output, teacher_output, fake_score_output):
    """Distribution Matching Distillation loss (Eq. 4).

    L = 0.5 * ||x̃ - sg[x̃ - (μ_real - μ_fake)]||^2

    where:
        x̃ = student output (velocity prediction)
        μ_real = teacher score function output
        μ_fake = fake score function output

    Args:
        student_output: Student's velocity prediction.
        teacher_output: Teacher's velocity prediction (detached).
        fake_score_output: Fake score network output (detached).

    Returns:
        Scalar loss.
    """
    # The gradient direction: real_score - fake_score
    gradient_direction = teacher_output.detach() - fake_score_output.detach()

    # DMD target: student output adjusted by gradient direction
    target = (student_output - gradient_direction).detach()

    return 0.5 * F.mse_loss(student_output, target)


def adversarial_generator_loss(disc_score_fake):
    """Adversarial generator loss (Eq. 5) — softplus-based.

    L_G = softplus(-D(G(z)))

    Args:
        disc_score_fake: Discriminator score on student-generated samples.

    Returns:
        Scalar loss.
    """
    return F.softplus(-disc_score_fake).mean()


def adversarial_discriminator_loss(disc_score_real, disc_score_fake):
    """Adversarial discriminator loss (Eq. 6) — softplus-based.

    L_D = softplus(-D(x_real)) + softplus(D(x_fake))

    Args:
        disc_score_real: Discriminator score on teacher samples.
        disc_score_fake: Discriminator score on student samples (detached).

    Returns:
        Scalar loss.
    """
    return (
        F.softplus(-disc_score_real).mean()
        + F.softplus(disc_score_fake.detach()).mean()
    )


# =============================================================================
# Self-rollout with KV cache
# =============================================================================


def self_rollout_with_cache(
    student_model, latents, context, y, plucker_emb,
    num_chunks, chunk_lat_frames, chunk_size_tokens,
    tokens_per_lat_frame, rollout_steps, target_timesteps,
    num_train_timesteps, grad_trunc_k, device,
):
    """Student generates chunks autoregressively with KV-cached rollout.

    Stochastic gradient truncation: only backprop through the most recent
    K chunks. Earlier chunks are detached.

    Args:
        student_model: WanModel (student, trainable).
        latents: (B, C, T_lat, H, W) clean latents (ground truth for first chunk).
        context: list of (512, 4096) text embeddings.
        y: list of (17, T_lat, H, W) conditioning.
        plucker_emb: (B, C_p, T_lat, H, W) Plucker embeddings.
        num_chunks: Total chunks to generate.
        chunk_lat_frames: Latent frames per chunk.
        chunk_size_tokens: Tokens per chunk.
        tokens_per_lat_frame: Tokens per latent frame.
        rollout_steps: Denoising steps per chunk.
        target_timesteps: Allowed timestep set for sampling.
        num_train_timesteps: Total timesteps.
        grad_trunc_k: Backprop through last K chunks only.
        device: CUDA device.

    Returns:
        rollout_latents: (B, C, T_lat, H, W) student-generated latents.
        rollout_preds: list of per-chunk velocity predictions.
    """
    B = latents.shape[0]
    C = latents.shape[1]
    H, W = latents.shape[3], latents.shape[4]
    clf = chunk_lat_frames

    # Use first chunk from ground truth (clean conditioning)
    generated_chunks = [latents[:, :, :clf]]

    rollout_preds = []

    for chunk_idx in range(1, num_chunks):
        # Sample noise for this chunk
        noise = torch.randn(B, C, clf, H, W, device=device, dtype=latents.dtype)

        # Simple single-step denoising for speed during training
        # Sample a target timestep
        t_set = torch.tensor(target_timesteps, device=device)
        t_idx = torch.randint(0, len(t_set), (B,), device=device)
        t_val = t_set[t_idx]

        sigma = t_val.float() / num_train_timesteps
        x_t = (1 - sigma.view(B, 1, 1, 1, 1)) * latents[:, :, chunk_idx*clf:(chunk_idx+1)*clf] \
            + sigma.view(B, 1, 1, 1, 1) * noise

        # Build sequence: past chunks + current noisy chunk
        past_clean = torch.cat(generated_chunks, dim=2)
        combined = torch.cat([past_clean, x_t], dim=2)

        total_lat = combined.shape[2]
        seq_len = total_lat * tokens_per_lat_frame

        # Timestep tensor: 0 for past chunks, t_val for current chunk
        past_tokens = past_clean.shape[2] * tokens_per_lat_frame
        curr_tokens = clf * tokens_per_lat_frame

        t_past = torch.zeros(B, past_tokens, device=device, dtype=torch.long)
        t_curr = t_val.unsqueeze(1).expand(B, curr_tokens)
        token_ts = torch.cat([t_past, t_curr], dim=1)

        x_list = [combined[b] for b in range(B)]
        y_chunk = [y[b][:, :total_lat] if y[b].shape[1] >= total_lat
                   else y[b] for b in range(B)]
        plucker_chunk = plucker_emb[:, :, :total_lat]

        dit_cond = {
            "c2ws_plucker_emb": [plucker_chunk[b:b+1] for b in range(B)],
        }

        # Forward with block causal
        preds = student_model(
            x=x_list, t=token_ts, context=context,
            seq_len=seq_len, y=y_chunk,
            dit_cond_dict=dit_cond,
            block_causal=True,
            chunk_size_tokens=chunk_size_tokens,
        )

        # Extract prediction for current chunk
        for b in range(B):
            pred_chunk = preds[b][:, chunk_idx*clf:(chunk_idx+1)*clf]
            rollout_preds.append(pred_chunk)

        # Denoise: x_0 = x_t - sigma * v_pred
        v_pred = torch.stack([
            preds[b][:, chunk_idx*clf:(chunk_idx+1)*clf]
            for b in range(B)
        ])
        x_0_pred = x_t - sigma.view(B, 1, 1, 1, 1) * v_pred

        # Gradient truncation: detach if chunk is beyond backprop window
        if chunk_idx < num_chunks - grad_trunc_k:
            x_0_pred = x_0_pred.detach()

        generated_chunks.append(x_0_pred)

    # Combine all chunks
    rollout_latents = torch.cat(generated_chunks, dim=2)

    return rollout_latents, rollout_preds


# =============================================================================
# Stage 2 Trainer
# =============================================================================


class Stage2Trainer:
    """
    Stage 2: DMD distillation trainer.

    Two-timescale optimization:
    - Student (slow): DMD loss + adversarial generator loss
    - Fake score + discriminator (fast): score matching + adversarial disc loss
    """

    def __init__(self, config: DMDConfig):
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        if self.world_size > 1:
            dist.init_process_group("nccl")

    def setup_models(self):
        """Load student, teacher, and fake score models."""
        wan_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "lingbot-world"
        )
        if wan_root not in sys.path:
            sys.path.insert(0, wan_root)

        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel

        self.wan_cfg = WAN_CONFIGS["i2v-A14B"]

        # ----- Student (from Stage 1 checkpoint) -----
        logger.info("Loading student model (from Stage 1 checkpoint)...")
        self.student = WanModel.from_pretrained(
            self.config.model_dir,
            subfolder=self.wan_cfg.high_noise_checkpoint,
            torch_dtype=torch.bfloat16,
        )

        # Load Stage 1 weights
        if self.config.student_ckpt and os.path.exists(self.config.student_ckpt):
            ckpt = torch.load(
                self.config.student_ckpt, map_location="cpu", weights_only=False
            )
            self.student.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded Stage 1 checkpoint from step {ckpt['step']}")

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint as ckpt_fn

            for block in self.student.blocks:
                orig = block.forward
                block._original_forward = orig

                def _make(o):
                    def f(*a, **kw):
                        return ckpt_fn(o, *a, use_reentrant=False, **kw)
                    return f

                block.forward = _make(orig)

        self.student.train()
        for p in self.student.parameters():
            p.requires_grad = True
        self.student.to(self.device)

        # ----- Fake Score Network -----
        logger.info("Loading fake score network...")
        fake_model = WanModel.from_pretrained(
            self.config.model_dir,
            subfolder=self.wan_cfg.high_noise_checkpoint,
            torch_dtype=torch.bfloat16,
        )
        self.fake_score = FakeScoreNetwork(
            fake_model, dim=self.wan_cfg.dim, num_heads=self.wan_cfg.num_heads
        )
        self.fake_score.train()
        for p in self.fake_score.parameters():
            p.requires_grad = True
        self.fake_score.to(self.device)

        # ----- Teacher (frozen, CPU-offloaded) -----
        logger.info("Loading teacher (both experts, frozen)...")
        self.teacher_low = WanModel.from_pretrained(
            self.config.model_dir,
            subfolder=self.wan_cfg.low_noise_checkpoint,
            torch_dtype=torch.bfloat16,
        )
        self.teacher_low.eval().requires_grad_(False)
        self.teacher_low.to("cpu")

        self.teacher_high = WanModel.from_pretrained(
            self.config.model_dir,
            subfolder=self.wan_cfg.high_noise_checkpoint,
            torch_dtype=torch.bfloat16,
        )
        self.teacher_high.eval().requires_grad_(False)
        self.teacher_high.to("cpu")

        # Compute spatial dims
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

        total_student = sum(p.numel() for p in self.student.parameters())
        total_fake = sum(p.numel() for p in self.fake_score.parameters())
        logger.info(
            f"Student: {total_student/1e9:.1f}B params | "
            f"Fake score: {total_fake/1e9:.1f}B params"
        )

    def setup_optimizers(self):
        """Set up separate optimizers for student and fake score."""
        # Student optimizer
        self.student_optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.student_lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay,
        )

        # Fake score + discriminator optimizer
        self.fake_optimizer = torch.optim.AdamW(
            self.fake_score.parameters(),
            lr=self.config.fake_score_lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay,
        )

        # Cosine LR schedule for both
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(self.config.warmup_steps, 1)
            progress = (step - self.config.warmup_steps) / max(
                self.config.total_steps - self.config.warmup_steps, 1
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

        self.student_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.student_optimizer, lr_lambda
        )
        self.fake_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.fake_optimizer, lr_lambda
        )

    def setup_data(self):
        """Set up dataset and dataloader."""
        from data_pipeline import PostTrainingDataset

        self.dataset = PostTrainingDataset(
            self.config.data_dir,
            num_chunks=self.config.num_chunks,
            chunk_lat_frames=self.config.chunk_lat_frames,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def _get_teacher_for_timestep(self, t_val):
        """Move the correct teacher expert to GPU and return it.

        Offloads the other expert to CPU to save VRAM.
        """
        boundary = self.wan_cfg.boundary * self.config.num_train_timesteps

        if t_val >= boundary:
            model = self.teacher_high
            offload = self.teacher_low
        else:
            model = self.teacher_low
            offload = self.teacher_high

        if next(offload.parameters()).device.type == "cuda":
            offload.to("cpu")
            torch.cuda.empty_cache()

        if next(model.parameters()).device.type == "cpu":
            model.to(self.device)

        return model

    def _offload_teacher(self):
        """Move both teacher models to CPU."""
        self.teacher_low.to("cpu")
        self.teacher_high.to("cpu")
        torch.cuda.empty_cache()

    def _teacher_forward(self, x_list, token_ts, context, seq_len, y_list,
                          dit_cond_dict, t_val):
        """Run teacher forward with correct expert routing."""
        teacher = self._get_teacher_for_timestep(t_val)

        with torch.no_grad():
            preds = teacher(
                x=x_list, t=token_ts, context=context,
                seq_len=seq_len, y=y_list,
                dit_cond_dict=dit_cond_dict,
            )

        self._offload_teacher()
        return preds

    def train_step(self, batch):
        """Single training step: DMD + adversarial.

        Returns:
            (student_loss, fake_loss, disc_loss) tuple.
        """
        latents = batch["latents"].to(self.device)
        context_t = batch["context"].to(self.device)
        y_t = batch["y"].to(self.device)
        plucker_emb = batch["plucker_emb"].to(self.device)

        B = latents.shape[0]
        T_lat = latents.shape[2]
        clf = self.config.chunk_lat_frames
        num_chunks = T_lat // clf

        context = [context_t[b] for b in range(B)]
        y = [y_t[b] for b in range(B)]

        # === 1. Self-rollout with student ===
        rollout_latents, _ = self_rollout_with_cache(
            student_model=self.student,
            latents=latents,
            context=context,
            y=y,
            plucker_emb=plucker_emb,
            num_chunks=min(num_chunks, self.config.rollout_chunks),
            chunk_lat_frames=clf,
            chunk_size_tokens=self.chunk_size_tokens,
            tokens_per_lat_frame=self.tokens_per_lat_frame,
            rollout_steps=self.config.rollout_steps,
            target_timesteps=self.config.target_timesteps,
            num_train_timesteps=self.config.num_train_timesteps,
            grad_trunc_k=self.config.grad_trunc_k,
            device=self.device,
        )

        # === 2. Student forward on noisy data ===
        # Sample a timestep and add noise
        t_val = torch.randint(
            100, self.config.num_train_timesteps, (B,), device=self.device
        )
        sigma = t_val.float() / self.config.num_train_timesteps
        noise = torch.randn_like(latents[:, :, :clf])
        x_t = (
            (1 - sigma.view(B, 1, 1, 1, 1)) * latents[:, :, :clf]
            + sigma.view(B, 1, 1, 1, 1) * noise
        )

        seq_len = clf * self.tokens_per_lat_frame
        token_ts = t_val.unsqueeze(1).expand(B, seq_len)

        x_list = [x_t[b] for b in range(B)]
        y_list = [y_t[b, :, :clf] for b in range(B)]

        dit_cond = {
            "c2ws_plucker_emb": [plucker_emb[b:b+1, :, :clf] for b in range(B)],
        }

        # Student velocity prediction
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            student_preds = self.student(
                x=x_list, t=token_ts, context=context,
                seq_len=seq_len, y=y_list,
                dit_cond_dict=dit_cond,
                block_causal=True,
                chunk_size_tokens=self.chunk_size_tokens,
            )

        # === 3. Teacher forward (frozen, offloaded) ===
        teacher_preds = self._teacher_forward(
            x_list=x_list, token_ts=token_ts,
            context=context, seq_len=seq_len,
            y_list=y_list, dit_cond_dict=dit_cond,
            t_val=t_val[0].item(),
        )

        # === 4. Fake score forward ===
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            fake_preds = self.fake_score.forward_score(
                x=x_list, t=token_ts, context=context,
                seq_len=seq_len, y=y_list,
                dit_cond_dict=dit_cond,
                block_causal=True,
                chunk_size_tokens=self.chunk_size_tokens,
            )

        # === 5. DMD loss on student ===
        student_v = torch.stack([p.float() for p in student_preds])
        teacher_v = torch.stack([p.float() for p in teacher_preds]).detach()
        fake_v = torch.stack([p.float() for p in fake_preds]).detach()

        loss_dmd = dmd_loss(student_v, teacher_v, fake_v)

        # === 6. Adversarial generator loss ===
        # Flatten student output for discriminator
        student_flat = student_v.reshape(B, -1, self.wan_cfg.dim)
        disc_fake = self.fake_score.discriminate(student_flat)
        loss_adv_g = adversarial_generator_loss(disc_fake)

        # === Student total loss ===
        student_loss = (
            self.config.dmd_weight * loss_dmd
            + self.config.adv_weight * loss_adv_g
        )

        # === 7. Fake score + discriminator update ===
        # Fake score matching: fake_score should approximate score of student distribution
        fake_score_loss = F.mse_loss(
            torch.stack([p.float() for p in fake_preds]),
            student_v.detach(),
        )

        # Discriminator loss
        teacher_flat = teacher_v.reshape(B, -1, self.wan_cfg.dim).detach()
        student_flat_detached = student_flat.detach()

        disc_real = self.fake_score.discriminate(teacher_flat)
        disc_fake_detached = self.fake_score.discriminate(student_flat_detached)
        loss_disc = adversarial_discriminator_loss(disc_real, disc_fake_detached)

        fake_total_loss = fake_score_loss + loss_disc

        return student_loss, fake_total_loss, {
            "dmd": loss_dmd.item(),
            "adv_g": loss_adv_g.item(),
            "fake_score": fake_score_loss.item(),
            "disc": loss_disc.item(),
        }

    def save_checkpoint(self, step):
        """Save student and fake score checkpoints."""
        if self.rank != 0:
            return

        os.makedirs(self.config.output_dir, exist_ok=True)

        checkpoint = {
            "step": step,
            "student_state_dict": self.student.state_dict(),
            "fake_score_state_dict": self.fake_score.state_dict(),
            "student_optimizer": self.student_optimizer.state_dict(),
            "fake_optimizer": self.fake_optimizer.state_dict(),
            "config": vars(self.config),
        }

        path = os.path.join(self.config.output_dir, f"stage2_step{step}.pt")
        torch.save(checkpoint, path)
        torch.save(
            checkpoint,
            os.path.join(self.config.output_dir, "stage2_latest.pt"),
        )

        size_gb = os.path.getsize(path) / 1e9
        logger.info(f"Saved checkpoint step {step} -> {path} ({size_gb:.1f} GB)")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.student.load_state_dict(ckpt["student_state_dict"])
        self.fake_score.load_state_dict(ckpt["fake_score_state_dict"])

        if "student_optimizer" in ckpt:
            self.student_optimizer.load_state_dict(ckpt["student_optimizer"])
        if "fake_optimizer" in ckpt:
            self.fake_optimizer.load_state_dict(ckpt["fake_optimizer"])

        logger.info(f"Loaded checkpoint from step {ckpt['step']}")
        return ckpt["step"]

    def train(self, resume_path=None):
        """Main training loop with two-timescale optimization."""
        self.setup_models()
        self.setup_optimizers()
        self.setup_data()

        start_step = 0
        if resume_path and os.path.exists(resume_path):
            start_step = self.load_checkpoint(resume_path)

        data_iter = iter(self.dataloader)
        running_metrics = {"dmd": 0, "adv_g": 0, "fake_score": 0, "disc": 0}
        step_times = []

        self.student_optimizer.zero_grad()
        self.fake_optimizer.zero_grad()

        logger.info(
            f"Starting Stage 2 DMD training: steps {start_step}-{self.config.total_steps}"
        )

        for step in range(start_step, self.config.total_steps):
            step_t0 = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            student_loss, fake_loss, metrics = self.train_step(batch)

            # --- Student update ---
            (student_loss / self.config.gradient_accumulation).backward()

            if (step + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.config.max_grad_norm
                )
                self.student_optimizer.step()
                self.student_scheduler.step()
                self.student_optimizer.zero_grad()

            # --- Fake score + disc update (two-timescale) ---
            for _ in range(self.config.disc_update_ratio):
                (fake_loss / self.config.gradient_accumulation).backward(
                    retain_graph=True
                )

            if (step + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.fake_score.parameters(), self.config.max_grad_norm
                )
                self.fake_optimizer.step()
                self.fake_scheduler.step()
                self.fake_optimizer.zero_grad()

            step_time = time.time() - step_t0
            step_times.append(step_time)

            for k, v in metrics.items():
                running_metrics[k] += v

            # Logging
            if (step + 1) % self.config.log_every == 0 and self.rank == 0:
                n = self.config.log_every
                avg_time = sum(step_times[-n:]) / len(step_times[-n:])
                eta_h = avg_time * (self.config.total_steps - step - 1) / 3600

                log_str = (
                    f"Step {step+1}/{self.config.total_steps} | "
                    + " | ".join(
                        f"{k}: {v/n:.4f}" for k, v in running_metrics.items()
                    )
                    + f" | Step: {avg_time:.1f}s | ETA: {eta_h:.1f}h"
                )
                logger.info(log_str)
                running_metrics = {k: 0 for k in running_metrics}

            # Save
            if (step + 1) % self.config.save_every == 0:
                self.save_checkpoint(step + 1)

        self.save_checkpoint(self.config.total_steps)

        if self.world_size > 1:
            dist.destroy_process_group()

        logger.info("Stage 2 DMD training complete!")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: DMD + Adversarial Distillation"
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--student_ckpt", required=True, help="Stage 1 checkpoint")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--student_lr", type=float, default=1e-5)
    parser.add_argument("--fake_score_lr", type=float, default=2e-5)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--chunk_lat_frames", type=int, default=5)
    parser.add_argument("--rollout_chunks", type=int, default=3)
    parser.add_argument("--rollout_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    torch.manual_seed(args.seed)

    config = DMDConfig(
        model_dir=args.model_dir,
        student_ckpt=args.student_ckpt,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        student_lr=args.student_lr,
        fake_score_lr=args.fake_score_lr,
        save_every=args.save_every,
        log_every=args.log_every,
        num_chunks=args.num_chunks,
        chunk_lat_frames=args.chunk_lat_frames,
        rollout_chunks=args.rollout_chunks,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
    )

    trainer = Stage2Trainer(config)
    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()

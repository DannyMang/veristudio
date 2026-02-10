"""
train_lora_driving.py — LoRA fine-tuning of LingBot-World for driving simulation.

Core approach:
  LingBot-World already supports WASD → Plücker → video generation.
  We add LoRA adapters to teach it driving-world dynamics while keeping
  the backbone frozen. The action adapter (Plücker encoder + AdaLN)
  gets full training since it's small and needs domain-specific learning.

What gets trained:
  1. LoRA adapters on DiT self-attention Q/K/V (visual domain adaptation)
  2. LoRA adapters on DiT cross-attention Q/K/V (text conditioning adaptation)
  3. Full training of action adapter layers (Plücker encoder + AdaLN)

What stays frozen:
  - DiT backbone weights (14B parameters)
  - Text encoder
  - VAE encoder/decoder
  - MoE router

Total trainable params: ~50-100M (vs 14B+ frozen) = <1% of model

Usage:
    # Single GPU prototyping (5s clips, low-res)
    python train_lora_driving.py \
        --model robbyant/lingbot-world-base-cam \
        --data data/training_ready/training_manifest.json \
        --output checkpoints/driving_lora_v1 \
        --lora_rank 16 \
        --resolution 480 \
        --clip_frames 16 \
        --batch_size 1 \
        --steps 5000

    # Multi-GPU training (720p, longer clips)  
    torchrun --nproc_per_node=8 train_lora_driving.py \
        --model robbyant/lingbot-world-base-cam \
        --data data/training_ready/training_manifest.json \
        --output checkpoints/driving_lora_v1 \
        --lora_rank 32 \
        --resolution 720 \
        --clip_frames 32 \
        --batch_size 1 \
        --gradient_accumulation 4 \
        --steps 20000

Prerequisites:
    pip install torch diffusers transformers peft accelerate wandb
    pip install einops scipy opencv-python
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# LoRA Implementation (compatible with DiT attention layers)
# =============================================================================

class LoRALinear(nn.Module):
    """
    LoRA adapter for linear layers in DiT blocks.
    
    Injects low-rank trainable matrices into frozen attention projections:
        output = frozen_linear(x) + scale * (B @ A @ x)
    
    where A is (in_features, rank) and B is (rank, out_features).
    """
    
    def __init__(self, original_linear, rank=16, alpha=16, dropout=0.05):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return original_out + self.scaling * lora_out


class DrivingActionAdapter(nn.Module):
    """
    Action adapter for driving inputs.
    
    Replaces/extends LingBot's Plücker encoder to handle driving-specific
    action representation (20D = 6D Plücker + 14D driving multi-hot).
    
    Architecture mirrors LingBot Section 3.3.2:
        action_embedding → projection → AdaLN scale/shift
    """
    
    def __init__(self, action_dim=20, hidden_dim=512, output_dim=1024, num_layers=3):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Plücker encoder branch (continuous geometric signal)
        self.plucker_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Discrete action encoder branch (multi-hot driving actions)
        self.discrete_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Fusion (concatenate + project)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        
        # AdaLN projection: output scale (γ) and shift (β)
        self.adaln_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim * 2),  # 2x for scale + shift
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize AdaLN output near identity (scale≈1, shift≈0)
        nn.init.zeros_(self.adaln_proj[-1].weight)
        nn.init.zeros_(self.adaln_proj[-1].bias)
    
    def forward(self, plucker, discrete_actions):
        """
        Args:
            plucker: (B, T, 6) Plücker embeddings per frame
            discrete_actions: (B, T, 14) multi-hot driving actions per frame
            
        Returns:
            scale: (B, T, output_dim) — AdaLN scale factors
            shift: (B, T, output_dim) — AdaLN shift factors
        """
        h_plucker = self.plucker_encoder(plucker)
        h_discrete = self.discrete_encoder(discrete_actions)
        
        h_fused = self.fusion(torch.cat([h_plucker, h_discrete], dim=-1))
        
        adaln_params = self.adaln_proj(h_fused)
        scale, shift = adaln_params.chunk(2, dim=-1)
        
        # Scale centered at 1 (identity transform)
        scale = 1.0 + scale
        
        return scale, shift


# =============================================================================
# Dataset
# =============================================================================

class DrivingWorldDataset(Dataset):
    """
    Dataset for driving world model training.
    
    Loads the training manifest produced by build_training_manifest.py
    and returns (video_frames, actions, captions) tuples.
    """
    
    def __init__(self, manifest_path, data_root, num_frames=16, resolution=480,
                 caption_mode='mixed', scene_static_ratio=0.3):
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        self.samples = manifest['samples']
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.resolution = resolution
        self.caption_mode = caption_mode
        self.scene_static_ratio = scene_static_ratio
        
        logger.info(f"Loaded {len(self.samples)} training clips")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video frames
        frame_dir = self.data_root / sample['frame_dir']
        frame_files = sample['frames']
        
        # Sample frames to target count
        if len(frame_files) > self.num_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(len(frame_files)))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
        
        frames = []
        for i in indices:
            img_path = frame_dir / frame_files[i]
            img = load_and_preprocess_frame(img_path, self.resolution)
            frames.append(img)
        
        video = torch.stack(frames)  # (T, C, H, W)
        
        # Load actions
        actions = sample.get('actions', [])
        plucker_list = []
        multihot_list = []
        
        for i in indices:
            if i < len(actions):
                p = actions[i].get('plucker', [0]*6)
                if isinstance(p, list) and len(p) == 6:
                    plucker_list.append(p)
                else:
                    plucker_list.append([0]*6)
                multihot_list.append(actions[i].get('multihot', [0]*14))
            else:
                plucker_list.append([0]*6)
                multihot_list.append([0]*14)
        
        plucker = torch.tensor(plucker_list, dtype=torch.float32)   # (T, 6)
        multihot = torch.tensor(multihot_list, dtype=torch.float32)  # (T, 14)
        
        # Select caption
        if self.caption_mode == 'scene_static':
            caption = sample.get('scene_static_caption', '')
        elif self.caption_mode == 'narrative':
            caption = sample.get('narrative_caption', '')
        elif self.caption_mode == 'mixed':
            if random.random() < self.scene_static_ratio:
                caption = sample.get('scene_static_caption', '')
            else:
                caption = sample.get('narrative_caption', '')
        else:
            caption = sample.get('narrative_caption', '')
        
        if not caption:
            caption = "A dashcam view of a vehicle driving on a road."
        
        return {
            'video': video,
            'plucker': plucker,
            'multihot': multihot,
            'caption': caption,
            'sample_id': sample['sample_id'],
        }


def load_and_preprocess_frame(path, resolution=480):
    """Load and preprocess a single frame."""
    import cv2
    
    path = str(path)
    if not os.path.exists(path):
        # Return black frame if file missing
        if resolution == 480:
            return torch.zeros(3, 480, 854)
        else:
            return torch.zeros(3, 720, 1280)
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    h, w = img.shape[:2]
    if resolution == 480:
        target_h, target_w = 480, 854
    else:
        target_h, target_w = 720, 1280
    
    img = cv2.resize(img, (target_w, target_h))
    
    # Normalize to [-1, 1]
    img = img.astype(np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)
    
    return img


# =============================================================================
# LoRA Injection into LingBot-World
# =============================================================================

def inject_lora_into_model(model, rank=16, alpha=16, target_modules=None):
    """
    Inject LoRA adapters into LingBot-World's DiT blocks.
    
    Targets:
    - Self-attention Q, K, V projections (visual domain adaptation)
    - Cross-attention Q, K, V projections (text conditioning)
    
    The action adapter gets full training (not LoRA) since it's already small.
    """
    if target_modules is None:
        # Default: attention projections in DiT blocks
        target_modules = [
            'to_q', 'to_k', 'to_v',       # Self-attention
            'to_q_cross', 'to_k_cross', 'to_v_cross',  # Cross-attention
        ]
    
    lora_params = []
    replaced_count = 0
    
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                
                parent = dict(model.named_modules())[parent_name] if parent_name else model
                
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                
                lora_params.extend([lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                replaced_count += 1
    
    logger.info(f"Injected LoRA (rank={rank}) into {replaced_count} layers")
    logger.info(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")
    
    return lora_params


def setup_training(model, action_adapter, lora_rank=16, learning_rate=1e-4):
    """
    Set up optimizer with different learning rates for different components.
    
    - LoRA adapters: lr (learning new visual domain)
    - Action adapter: lr * 2 (needs to learn driving dynamics from scratch)
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Inject LoRA (unfreezes LoRA params)
    lora_params = inject_lora_into_model(model, rank=lora_rank)
    
    # Action adapter is fully trainable
    action_params = list(action_adapter.parameters())
    
    # Different learning rates
    param_groups = [
        {'params': lora_params, 'lr': learning_rate, 'name': 'lora'},
        {'params': action_params, 'lr': learning_rate * 2, 'name': 'action_adapter'},
    ]
    
    total_trainable = sum(p.numel() for group in param_groups for p in group['params'])
    total_model = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Total model params: {total_model:,}")
    logger.info(f"Trainable params: {total_trainable:,} ({100*total_trainable/total_model:.2f}%)")
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01)
    
    return optimizer


# =============================================================================
# Training Loop
# =============================================================================

def train_step(model, action_adapter, vae, text_encoder, batch, noise_scheduler, device):
    """
    Single training step for driving LoRA.
    
    Following LingBot's diffusion training (Section 3.1):
    1. Encode video to latents via VAE
    2. Add noise at random timestep
    3. Predict noise conditioned on: text + action signals
    4. Compute MSE loss
    """
    video = batch['video'].to(device)           # (B, T, C, H, W)
    plucker = batch['plucker'].to(device)       # (B, T, 6)
    multihot = batch['multihot'].to(device)     # (B, T, 14)
    captions = batch['caption']                  # list of strings
    
    B, T = video.shape[:2]
    
    # 1. Encode video frames to latent space
    with torch.no_grad():
        # Reshape for VAE: (B*T, C, H, W)
        video_flat = video.reshape(B * T, *video.shape[2:])
        latents = vae.encode(video_flat).latent_dist.sample()
        latents = latents.reshape(B, T, *latents.shape[1:])
        latents = latents * vae.config.scaling_factor
    
    # 2. Encode text captions
    with torch.no_grad():
        text_embeddings = encode_text(text_encoder, captions, device)
    
    # 3. Compute action conditioning via our driving adapter
    action_scale, action_shift = action_adapter(plucker, multihot)
    
    # 4. Sample noise and timesteps
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (B,), device=device).long()
    
    # 5. Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # 6. Predict noise (with action conditioning injected via AdaLN)
    # The action_scale and action_shift modulate DiT features:
    #   features = action_scale * LayerNorm(features) + action_shift
    noise_pred = model(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_embeddings,
        action_scale=action_scale,
        action_shift=action_shift,
    ).sample
    
    # 7. MSE loss
    loss = F.mse_loss(noise_pred, noise)
    
    return loss


def encode_text(text_encoder, captions, device):
    """Encode text captions using the model's text encoder."""
    # This depends on the specific text encoder LingBot uses
    # Wan2.2 uses a T5-like or CLIP-like encoder
    # Placeholder — adapt to actual LingBot architecture
    from transformers import AutoTokenizer
    
    tokenizer = text_encoder.tokenizer if hasattr(text_encoder, 'tokenizer') else None
    if tokenizer is None:
        # Fallback: return dummy embeddings
        B = len(captions)
        return torch.zeros(B, 77, 1024, device=device)
    
    tokens = tokenizer(captions, padding=True, truncation=True,
                       max_length=77, return_tensors='pt').to(device)
    with torch.no_grad():
        embeddings = text_encoder(**tokens).last_hidden_state
    return embeddings


# =============================================================================
# Saving and Loading LoRA weights
# =============================================================================

def save_lora_checkpoint(model, action_adapter, optimizer, step, output_dir):
    """
    Save only LoRA + action adapter weights (tiny compared to full model).
    
    A LoRA checkpoint for rank=16 on a 14B model is typically ~50-200MB,
    vs ~28GB for the full model. This is what makes LoRA practical.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract LoRA weights
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f'{name}.lora_A.weight'] = module.lora_A.weight.data.cpu()
            lora_state[f'{name}.lora_B.weight'] = module.lora_B.weight.data.cpu()
    
    checkpoint = {
        'step': step,
        'lora_state_dict': lora_state,
        'action_adapter_state_dict': action_adapter.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'lora_rank': next(iter(lora_state.values())).shape[0] if lora_state else 0,
            'action_dim': action_adapter.action_dim,
        }
    }
    
    path = output_dir / f'checkpoint_step{step}.pt'
    torch.save(checkpoint, path)
    
    # Also save latest
    torch.save(checkpoint, output_dir / 'checkpoint_latest.pt')
    
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"Saved checkpoint: {path} ({size_mb:.1f}MB)")


def load_lora_checkpoint(model, action_adapter, checkpoint_path):
    """Load LoRA + action adapter weights onto a base model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load LoRA weights
    lora_state = checkpoint['lora_state_dict']
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f'{name}.lora_A.weight'
            b_key = f'{name}.lora_B.weight'
            if a_key in lora_state:
                module.lora_A.weight.data = lora_state[a_key]
                module.lora_B.weight.data = lora_state[b_key]
    
    # Load action adapter
    action_adapter.load_state_dict(checkpoint['action_adapter_state_dict'])
    
    logger.info(f"Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint['step']


# =============================================================================
# Main Training
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train driving LoRA for LingBot-World')
    parser.add_argument('--model', type=str, default='robbyant/lingbot-world-base-cam')
    parser.add_argument('--data', type=str, required=True, help='Training manifest JSON')
    parser.add_argument('--data_root', type=str, default=None, help='Data root directory')
    parser.add_argument('--output', type=str, default='checkpoints/driving_lora')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=480, choices=[480, 720])
    parser.add_argument('--clip_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.data_root is None:
        args.data_root = str(Path(args.data).parent.parent)
    
    logger.info(f"Device: {device}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Resolution: {args.resolution}p")
    
    # W&B logging
    if args.wandb:
        import wandb
        wandb.init(project='lingbot-driving', config=vars(args))
    
    # =========================================================================
    # Load model components
    # =========================================================================
    logger.info(f"Loading model: {args.model}")
    
    # NOTE: The actual loading depends on LingBot-World's codebase structure.
    # This is a template — adapt based on their GitHub repo's model loading.
    #
    # Expected components:
    #   - DiT backbone (the main transformer)
    #   - VAE (video autoencoder)
    #   - Text encoder (T5 or CLIP variant)
    #   - Noise scheduler (flow matching or DDPM)
    
    try:
        # Try HuggingFace diffusers loading
        from diffusers import AutoencoderKL, DDPMScheduler
        from transformers import AutoModel, AutoTokenizer
        
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae').to(device)
        vae.eval()
        
        logger.info("Loading text encoder...")
        text_encoder = AutoModel.from_pretrained(args.model, subfolder='text_encoder').to(device)
        text_encoder.eval()
        
        logger.info("Loading DiT backbone...")
        # This will depend on LingBot's actual model class
        # Placeholder:
        from diffusers import UNet2DConditionModel
        dit = UNet2DConditionModel.from_pretrained(args.model, subfolder='unet').to(device)
        
        noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder='scheduler')
        
    except Exception as e:
        logger.warning(f"Could not load from HuggingFace: {e}")
        logger.info("Using placeholder model for development. Replace with actual LingBot loading.")
        
        # Placeholder for development
        dit = create_placeholder_dit(device)
        vae = create_placeholder_vae(device)
        text_encoder = None
        noise_scheduler = None
    
    # =========================================================================
    # Setup LoRA + Action Adapter
    # =========================================================================
    
    # Create driving action adapter
    action_adapter = DrivingActionAdapter(
        action_dim=20,
        hidden_dim=512,
        output_dim=1024,  # Match DiT hidden dim
    ).to(device)
    
    # Setup optimizer (injects LoRA and configures param groups)
    optimizer = setup_training(dit, action_adapter, args.lora_rank, args.learning_rate)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step = load_lora_checkpoint(dit, action_adapter, args.resume)
        logger.info(f"Resuming from step {start_step}")
    
    # =========================================================================
    # Dataset
    # =========================================================================
    dataset = DrivingWorldDataset(
        manifest_path=args.data,
        data_root=args.data_root,
        num_frames=args.clip_frames,
        resolution=args.resolution,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    logger.info(f"Starting training: {args.steps} steps")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    
    dit.train()
    action_adapter.train()
    
    data_iter = iter(dataloader)
    running_loss = 0.0
    
    for step in range(start_step, args.steps):
        # Get batch (loop dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Learning rate warmup
        if step < args.warmup_steps:
            lr_scale = (step + 1) / args.warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * lr_scale / max(lr_scale, 1e-8)
        
        # Forward + backward
        loss = train_step(dit, action_adapter, vae, text_encoder,
                         batch, noise_scheduler, device)
        loss = loss / args.gradient_accumulation
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                list(action_adapter.parameters()) + 
                [p for p in dit.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * args.gradient_accumulation
        
        # Logging
        if (step + 1) % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            logger.info(f"Step {step+1}/{args.steps} | Loss: {avg_loss:.4f}")
            if args.wandb:
                import wandb
                wandb.log({'loss': avg_loss, 'step': step+1})
            running_loss = 0.0
        
        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            save_lora_checkpoint(dit, action_adapter, optimizer, step+1, args.output)
    
    # Final save
    save_lora_checkpoint(dit, action_adapter, optimizer, args.steps, args.output)
    logger.info(f"Training complete! Checkpoints saved to {args.output}")


# =============================================================================
# Placeholder models for development/testing without full LingBot weights
# =============================================================================

def create_placeholder_dit(device):
    """Minimal DiT-like model for testing the training pipeline."""
    class PlaceholderDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(1024, 1024)
            self.to_k = nn.Linear(1024, 1024)
            self.to_v = nn.Linear(1024, 1024)
            self.proj = nn.Linear(1024, 4)
        
        def forward(self, x, timesteps, encoder_hidden_states=None,
                    action_scale=None, action_shift=None):
            B = x.shape[0]
            flat = x.reshape(B, -1)[:, :1024]
            h = self.to_q(flat) + self.to_k(flat) + self.to_v(flat)
            if action_scale is not None:
                scale = action_scale[:, 0, :h.shape[-1]]
                shift = action_shift[:, 0, :h.shape[-1]]
                h = scale * h + shift
            out = self.proj(h)
            
            class Output:
                def __init__(self, sample):
                    self.sample = sample
            
            return Output(out.reshape_as(x))
    
    return PlaceholderDiT().to(device)


def create_placeholder_vae(device):
    """Minimal VAE for testing."""
    class PlaceholderVAE(nn.Module):
        class Config:
            scaling_factor = 0.18215
        
        def __init__(self):
            super().__init__()
            self.config = self.Config()
        
        def encode(self, x):
            class Dist:
                def sample(self_inner):
                    return torch.randn(x.shape[0], 4, x.shape[2]//8, x.shape[3]//8,
                                       device=x.device)
            class Output:
                latent_dist = Dist()
            return Output()
    
    return PlaceholderVAE().to(device)


if __name__ == '__main__':
    main()

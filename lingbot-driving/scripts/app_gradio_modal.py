"""
Interactive driving demo on Modal with Gradio.

Loads the LoRA-trained high-noise expert and generates short video clips
from a starting image + text prompt + WASD camera direction.

Usage:
    modal serve app_gradio_modal.py     # dev mode (hot-reload)
    modal deploy app_gradio_modal.py    # production
"""

import modal

app = modal.App("veri_vm_1")

model_volume = modal.Volume.from_name("lingbot-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("lingbot-lora-checkpoints", create_if_missing=True)

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
        "gradio>=4.0",
        "fastapi",
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

MODEL_DIR = "/data/lingbot-world-base-cam"
CHECKPOINT_DIR = "/checkpoints"


# ============================================================================
# Pose generation (WASD -> camera extrinsics)
# ============================================================================


def wasd_to_poses(direction, num_frames=17, speed=0.3, turn_rate_deg=2.0):
    """
    Convert a WASD direction string to camera pose matrices.

    Returns:
        poses: np.ndarray of shape (num_frames, 4, 4) -- camera-to-world
        intrinsics: np.ndarray of shape (num_frames, 4) -- [fx, fy, cx, cy]
    """
    import numpy as np

    direction = direction.lower().strip()
    keys = set(c for c in direction if c in "wasd")

    # Determine velocity targets
    target_speed = 0.0
    target_yaw_rate = 0.0
    if "w" in keys:
        target_speed = speed
    if "s" in keys:
        target_speed = -speed * 0.5
    if "a" in keys:
        target_yaw_rate = -np.radians(turn_rate_deg)
    if "d" in keys:
        target_yaw_rate = np.radians(turn_rate_deg)

    # If no direction, default to gentle forward
    if not keys:
        target_speed = speed * 0.5

    # Generate poses with smooth acceleration
    position = np.array([0.0, 0.0, 0.0])
    yaw = 0.0
    current_speed = 0.0
    current_yaw_rate = 0.0
    smooth = 0.3

    poses = []
    for _ in range(num_frames):
        current_speed += (target_speed - current_speed) * smooth
        current_yaw_rate += (target_yaw_rate - current_yaw_rate) * smooth
        yaw += current_yaw_rate

        # Forward direction (OpenCV: Z is forward, X is right, Y is down)
        forward = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
        position = position + forward * current_speed

        # Rotation matrix from yaw (around Y axis)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = position.astype(np.float32)
        poses.append(c2w)

    poses = np.stack(poses, axis=0)  # (num_frames, 4, 4)

    # Default intrinsics for 480p (70 degree horizontal FOV)
    width, height = 832, 480
    fov_h = np.radians(70.0)
    fx = width / (2.0 * np.tan(fov_h / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    intrinsics = np.tile(
        np.array([[fx, fy, cx, cy]], dtype=np.float32), (num_frames, 1)
    )

    return poses, intrinsics


# ============================================================================
# LoRA injection (copied from train script for self-containment)
# ============================================================================


def inject_lora(model, rank=16, alpha=16):
    """Inject LoRA into self-attention Q/K/V."""
    import math
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, original, rank, alpha):
            super().__init__()
            self.original = original
            self.scaling = alpha / rank
            in_f, out_f = original.in_features, original.out_features
            self.lora_A = nn.Linear(in_f, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_f, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            for p in self.original.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.original(x) + self.scaling * self.lora_B(self.lora_A(x))

    replaced = 0
    for block in model.blocks:
        sa = block.self_attn
        for attr in ("q", "k", "v"):
            orig = getattr(sa, attr)
            if isinstance(orig, nn.Linear):
                setattr(sa, attr, LoRALinear(orig, rank, alpha))
                replaced += 1
    print(f"[LoRA] Injected rank-{rank} into {replaced} layers")


def load_lora_checkpoint(model, checkpoint_path):
    """Load LoRA weights from checkpoint."""
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

    print(f"[LoRA] Loaded checkpoint from step {ckpt['step']}")


# ============================================================================
# Gradio app (served on Modal)
# ============================================================================


@app.function(
    image=image,
    gpu="H100:2",
    timeout=3600,
    volumes={"/data": model_volume, "/checkpoints": checkpoint_volume},
    memory=64 * 1024,
    scaledown_window=300,
    max_containers=1,
)
@modal.asgi_app()
def serve():
    import os
    import sys
    import tempfile
    import time

    # Stable temp dir so Gradio uploads survive until generation runs
    os.makedirs("/tmp/gradio_cache", exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = "/tmp/gradio_cache"
    os.environ["GRADIO_ALLOWED_PATHS"] = "/tmp/gradio_cache"

    sys.path.insert(0, "/opt/lingbot-world")

    import gradio as gr
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from einops import rearrange
    from PIL import Image
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.cam_utils import (
        compute_relative_poses,
        interpolate_camera_poses,
        get_plucker_embeddings,
        get_Ks_transformed,
    )

    dev0 = torch.device("cuda:0")  # low-noise expert + T5 + VAE
    dev1 = torch.device("cuda:1")  # high-noise expert
    cfg = WAN_CONFIGS["i2v-A14B"]

    # ---- Load models ----
    print("Loading T5 (GPU 0) ...")
    t0 = time.time()
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=dev0,
        checkpoint_path=os.path.join(MODEL_DIR, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(MODEL_DIR, cfg.t5_tokenizer),
    )
    print(f"T5 loaded in {time.time()-t0:.1f}s")

    print("Loading VAE (GPU 0) ...")
    t0 = time.time()
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(MODEL_DIR, cfg.vae_checkpoint),
        device=dev0,
    )
    print(f"VAE loaded in {time.time()-t0:.1f}s")

    # Load experts on separate GPUs
    print("Loading DiT low-noise expert (GPU 0) ...")
    t0 = time.time()
    low_noise_model = WanModel.from_pretrained(
        MODEL_DIR,
        subfolder=cfg.low_noise_checkpoint,
        torch_dtype=torch.bfloat16,
    )
    low_noise_model.eval().requires_grad_(False).to(dev0)
    print(f"Low-noise expert loaded in {time.time()-t0:.1f}s  "
          f"GPU0={torch.cuda.memory_allocated(dev0)/1e9:.1f}GB")

    print("Loading DiT high-noise expert (GPU 1) ...")
    t0 = time.time()
    high_noise_model = WanModel.from_pretrained(
        MODEL_DIR,
        subfolder=cfg.high_noise_checkpoint,
        torch_dtype=torch.bfloat16,
    )
    high_noise_model.eval().requires_grad_(False).to(dev1)
    print(f"High-noise expert loaded in {time.time()-t0:.1f}s  "
          f"GPU1={torch.cuda.memory_allocated(dev1)/1e9:.1f}GB")

    # Model config
    boundary = cfg.boundary * cfg.num_train_timesteps  # 0.947 * 1000 = 947
    neg_prompt = cfg.sample_neg_prompt
    vae_stride = (4, 8, 8)
    patch_size = high_noise_model.patch_size  # (1, 2, 2)
    param_dtype = torch.bfloat16

    print(f"All models loaded! boundary={boundary}")
    print(f"  GPU0: {torch.cuda.memory_allocated(dev0)/1e9:.1f}GB  "
          f"GPU1: {torch.cuda.memory_allocated(dev1)/1e9:.1f}GB")
    print("Starting Gradio ...")

    # ---- Inference function ----
    @torch.no_grad()
    def generate_video(
        input_image,
        prompt,
        direction,
        num_steps=20,
        frame_num=17,
        shift=3.0,
        seed=42,
    ):
        """Generate video frames from image + prompt + WASD direction."""
        if input_image is None:
            return None, None, "Please upload an image."

        t_start = time.time()
        # Ensure frame_num is 4n+1 (VAE temporal stride requirement)
        frame_num = int(frame_num)
        frame_num = ((frame_num - 1) // 4) * 4 + 1

        # Convert input image
        if isinstance(input_image, np.ndarray):
            img_pil = Image.fromarray(input_image)
        else:
            img_pil = input_image
        img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev0)

        # Compute latent dimensions for 480p (training resolution)
        max_area = 480 * 832
        aspect_ratio = img.shape[1] / img.shape[2]
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
            patch_size[1] * patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
            patch_size[2] * patch_size[2])
        h = lat_h * vae_stride[1]
        w = lat_w * vae_stride[2]
        lat_f = (frame_num - 1) // vae_stride[0] + 1

        max_seq_len = lat_f * lat_h * lat_w // (patch_size[1] * patch_size[2])

        # Initialize noise
        seed_g = torch.Generator(device=dev0)
        seed_g.manual_seed(seed)
        noise = torch.randn(16, lat_f, lat_h, lat_w,
                            dtype=torch.float32, generator=seed_g,
                            device=dev0)

        # First-frame mask
        msk = torch.ones(1, frame_num, lat_h, lat_w, device=dev0)
        msk[:, 1:] = 0
        msk = torch.cat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # Text encoding (conditional + unconditional for CFG)
        context = text_encoder([prompt], dev0)
        context_null = text_encoder([neg_prompt], dev0)

        # Camera conditioning (Plucker embeddings)
        poses, intrinsics_np = wasd_to_poses(direction, num_frames=frame_num)
        c2ws = torch.from_numpy(poses).float()
        Ks = torch.from_numpy(intrinsics_np).float()

        Ks = get_Ks_transformed(
            Ks, height_org=480, width_org=832,
            height_resize=h, width_resize=w,
            height_final=h, width_final=w,
        )
        Ks = Ks[0]

        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, frame_num - 1, frame_num),
            src_rot_mat=c2ws[:, :3, :3].numpy(),
            src_trans_vec=c2ws[:, :3, 3].numpy(),
            tgt_indices=np.linspace(0, frame_num - 1, lat_f),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(dev0)
        Ks = Ks.to(dev0)
        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h), c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'b (f h w) c -> b c f h w',
            f=lat_f, h=lat_h, w=lat_w,
        ).to(param_dtype)

        dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

        # VAE encode first frame (with padding for remaining frames)
        print(f"[{time.time()-t_start:.1f}s] Starting VAE encode ...")
        img_resized = torch.nn.functional.interpolate(
            img[None], size=(h, w), mode="bicubic"
        ).transpose(0, 1)
        video_input = torch.cat([
            img_resized,
            torch.zeros(3, frame_num - 1, h, w, device=dev0),
        ], dim=1)
        y = vae.encode([video_input])[0]
        y = torch.cat([msk, y])

        # Denoising loop (dual expert + CFG)
        guide_scale = 5.0
        print(f"[{time.time()-t_start:.1f}s] Starting denoising ({num_steps} steps, {max_seq_len} tokens, CFG={guide_scale}) ...")
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(num_steps, device=dev0, shift=shift)

        # Pre-build args for each GPU (tensors move with the expert)
        def make_args(target_dev):
            return (
                {
                    "context": [context[0].to(target_dev)],
                    "seq_len": max_seq_len,
                    "y": [y.to(target_dev)],
                    "dit_cond_dict": {"c2ws_plucker_emb": [
                        p.to(target_dev) for p in dit_cond_dict["c2ws_plucker_emb"]
                    ]},
                },
                {
                    "context": [context_null[0].to(target_dev)],
                    "seq_len": max_seq_len,
                    "y": [y.to(target_dev)],
                    "dit_cond_dict": {"c2ws_plucker_emb": [
                        p.to(target_dev) for p in dit_cond_dict["c2ws_plucker_emb"]
                    ]},
                },
            )

        args_dev0 = make_args(dev0)  # for low-noise expert
        args_dev1 = make_args(dev1)  # for high-noise expert

        latent = noise  # starts on dev0
        for i, t in enumerate(scheduler.timesteps):
            t0_step = time.time()

            # Select expert + device based on timestep
            use_high = t.item() >= boundary
            if use_high:
                model = high_noise_model
                target_dev = dev1
                arg_c, arg_null = args_dev1
                expert_name = "high/GPU1"
            else:
                model = low_noise_model
                target_dev = dev0
                arg_c, arg_null = args_dev0
                expert_name = "low/GPU0"

            latent_on_dev = latent.to(target_dev)
            timestep = torch.stack([t]).to(target_dev)

            # Conditional + unconditional forward passes (CFG)
            with torch.amp.autocast("cuda", dtype=param_dtype):
                noise_pred_cond = model([latent_on_dev], t=timestep, **arg_c)[0]
                noise_pred_uncond = model([latent_on_dev], t=timestep, **arg_null)[0]

            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            # Scheduler step on dev0 (where seed_g lives)
            noise_pred = noise_pred.to(dev0)
            latent = scheduler.step(
                noise_pred.unsqueeze(0), t.to(dev0), latent.to(dev0).unsqueeze(0),
                return_dict=False, generator=seed_g,
            )[0].squeeze(0)
            print(f"  step {i+1}/{num_steps} ({expert_name}): {time.time()-t0_step:.2f}s")

        # VAE decode (on dev0)
        print(f"[{time.time()-t_start:.1f}s] Starting VAE decode ...")
        video = vae.decode([latent.to(dev0)])

        elapsed = time.time() - t_start
        print(f"[{elapsed:.1f}s] Done! Total generation time.")
        torch.cuda.empty_cache()
        with torch.cuda.device(dev1):
            torch.cuda.empty_cache()

        # Convert to numpy frames
        video_np = video[0]  # (C, T, H, W)
        video_np = ((video_np + 1.0) / 2.0).clamp(0, 1)
        video_np = (video_np.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)

        # Save as MP4
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp.name, fourcc, 8, (video_np.shape[2], video_np.shape[1]))
        for frame in video_np:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        # Extract last frame for chaining
        last_frame = video_np[-1]

        status = f"Generated {len(video_np)} frames in {elapsed:.1f}s ({elapsed/len(video_np):.2f}s/frame)"
        return tmp.name, last_frame, status

    # ---- Gradio UI ----
    with gr.Blocks(title="Veri WM 1") as demo:
        gr.Markdown("# Veri WM 1 - Interactive Driving Demo")
        gr.Markdown("Upload a dashcam image, enter a prompt, choose a direction, and generate!")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Starting Image", type="numpy")
                prompt = gr.Textbox(
                    label="Prompt",
                    value="A dashcam view of a vehicle driving on a road in clear weather.",
                    lines=2,
                )
                direction = gr.Radio(
                    choices=["w (forward)", "s (backward)", "a (turn left)",
                             "d (turn right)", "wa (forward+left)", "wd (forward+right)",
                             "(no input, drift forward)"],
                    label="Direction (WASD)",
                    value="w (forward)",
                )
                with gr.Row():
                    num_steps = gr.Slider(4, 40, value=20, step=1, label="Denoising Steps")
                    seed = gr.Number(value=42, label="Seed", precision=0)
                frame_count = gr.Slider(17, 81, value=33, step=4, label="Frames (4n+1)")
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Generated Video")
                last_frame = gr.Image(label="Last Frame (use as next input)")
                status = gr.Textbox(label="Status", interactive=False)

        # Wire up the button
        def run_generation(img, prompt_text, direction_str, steps, seed_val, frames):
            # Extract direction key from radio label
            d = direction_str.split(" ")[0].strip("()")
            return generate_video(img, prompt_text, d, int(steps),
                                  frame_num=int(frames), seed=int(seed_val))

        generate_btn.click(
            fn=run_generation,
            inputs=[input_image, prompt, direction, num_steps, seed, frame_count],
            outputs=[output_video, last_frame, status],
        )

        # Button to copy last frame to input
        use_last_btn = gr.Button("Use Last Frame as Input")
        use_last_btn.click(fn=lambda x: x, inputs=[last_frame], outputs=[input_image])

    demo.queue()
    import fastapi
    web_app = fastapi.FastAPI()
    return gr.mount_gradio_app(web_app, demo, path="/")

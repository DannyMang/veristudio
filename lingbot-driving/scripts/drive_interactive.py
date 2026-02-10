"""
drive_interactive.py — Interactive WASD driving demo using LingBot-World.

No LoRA or fine-tuning needed. Uses the base-cam model's existing camera
pose injection (Plucker embeddings via AdaLN) to convert WASD keyboard
input into driving video.

How it works:
    The released LingBot-World-Base-Cam model accepts:
      1. A starting image (dashcam photo)
      2. A text prompt
      3. Camera poses as [num_frames, 4, 4] transformation matrices
    WASD keys are converted to 4x4 camera-to-world matrices, which are
    fed through the model's existing Plucker embedding pipeline.

Modes:
    terminal:  Type WASD + Enter, videos saved to disk.
    pygame:    Real-time window with keyboard input (requires GPU).

Usage:
    # Dry run — test WASD-to-pose math, no GPU needed
    python drive_interactive.py --mode terminal --dry_run

    # Full inference — needs GPU + downloaded model weights
    python drive_interactive.py --mode terminal \
        --ckpt_dir /path/to/lingbot-world-base-cam \
        --init_image dashcam.jpg \
        --prompt "Dashcam footage driving through a suburban neighborhood"

    # Pygame mode
    python drive_interactive.py --mode pygame \
        --ckpt_dir /path/to/lingbot-world-base-cam \
        --init_image dashcam.jpg

Hardware (inference only, no training):
    - Dry run:               Any CPU
    - Full precision + offload, 480p:   1x A100 80GB (~35-40 GB VRAM)
    - NF4 quantized, 480p:             1x RTX 4090 24GB (~12-15 GB VRAM)
    - Full precision, 720p:            8x A100 with FSDP (as in README)

Latency: ~5-15 seconds per 17-frame chunk on A100 (not real-time).
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# WASD -> Camera Pose Conversion
# =============================================================================

class DrivingPoseGenerator:
    """
    Converts WASD keyboard input into camera-to-world transformation matrices
    in the exact format LingBot-World expects (OpenCV coordinates).

    OpenCV coordinate system:
        X = right, Y = down, Z = forward (into the scene)

    Output format matches lingbot-world/examples/00/:
        poses.npy:      [num_frames, 4, 4] camera-to-world matrices
        intrinsics.npy: [num_frames, 4]    [fx, fy, cx, cy]
    """

    def __init__(self, speed=0.3, turn_rate=2.0, num_frames=17):
        """
        Args:
            speed: translation per frame (arbitrary units, normalized later)
            turn_rate: degrees of yaw per frame when turning
            num_frames: frames per chunk (must be 4n+1 for the model)
        """
        self.speed = speed
        self.turn_rate_deg = turn_rate
        self.num_frames = num_frames

        # Vehicle state
        self.position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0  # radians
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0

    def _yaw_to_rotation(self, yaw):
        """Yaw angle -> 3x3 rotation matrix (around Y axis, OpenCV coords)."""
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c],
        ])

    def _make_c2w(self, position, yaw):
        """Create a 4x4 camera-to-world matrix."""
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = self._yaw_to_rotation(yaw)
        c2w[:3, 3] = position
        return c2w

    def wasd_to_poses(self, keys_held):
        """
        Generate a sequence of camera poses from held WASD keys.

        Args:
            keys_held: set of chars, e.g. {'w'}, {'w', 'a'}, {'s', 'd'}

        Returns:
            poses: np.ndarray [num_frames, 4, 4] — camera-to-world matrices
            description: str — human-readable action summary
        """
        target_speed = 0.0
        target_yaw_rate = 0.0

        if 'w' in keys_held:
            target_speed = self.speed
        if 's' in keys_held:
            target_speed = -self.speed * 0.5
        if 'a' in keys_held:
            target_yaw_rate = -np.radians(self.turn_rate_deg)
        if 'd' in keys_held:
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

        # Description
        parts = []
        if 'w' in keys_held: parts.append("forward")
        if 's' in keys_held: parts.append("reverse")
        if 'a' in keys_held: parts.append("left")
        if 'd' in keys_held: parts.append("right")
        desc = ' + '.join(parts) if parts else 'coast'

        return np.array(poses, dtype=np.float32), desc

    def get_default_intrinsics(self, num_frames, width=832, height=480):
        """
        Default dashcam intrinsics (~70 deg horizontal FOV).
        Returns: np.ndarray [num_frames, 4] = [fx, fy, cx, cy]
        """
        fov_h = 70.0
        fx = width / (2.0 * np.tan(np.radians(fov_h / 2.0)))
        fy = fx
        cx, cy = width / 2.0, height / 2.0
        return np.tile(np.array([[fx, fy, cx, cy]], dtype=np.float32),
                       (num_frames, 1))

    def reset(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0


# =============================================================================
# LingBot-World Model Wrapper
# =============================================================================

class LingBotDriver:
    """
    Wraps the real WanI2V pipeline for autoregressive driving generation.

    On each generate() call:
      1. Saves poses/intrinsics to temp .npy files
      2. Calls WanI2V.generate() with action_path pointing to temp dir
      3. Returns video tensor + last frame as PIL image
    """

    def __init__(self, ckpt_dir, size="480*832", t5_cpu=True):
        self.ckpt_dir = ckpt_dir
        self.size = size
        self.t5_cpu = t5_cpu
        self._model = None

        parts = size.split('*')
        self.height, self.width = int(parts[0]), int(parts[1])

    def _load(self):
        if self._model is not None:
            return

        lingbot_root = Path(__file__).resolve().parent.parent.parent / 'lingbot-world'
        if str(lingbot_root) not in sys.path:
            sys.path.insert(0, str(lingbot_root))

        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS['i2v-A14B']
        logger.info(f"Loading WanI2V from {self.ckpt_dir} ...")
        self._model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=self.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=self.t5_cpu,
        )
        self._config = cfg
        logger.info("Model loaded.")

    def generate(self, image, prompt, poses, intrinsics,
                 frame_num=17, shift=3.0, sampling_steps=30, seed=42):
        """
        Generate a video chunk.

        Args:
            image: PIL.Image.Image — starting frame
            prompt: str
            poses: np.ndarray [N, 4, 4]
            intrinsics: np.ndarray [N, 4]
            frame_num: int (must be 4n+1)
            shift: float (3.0 for 480p, 10.0 for 720p)
            sampling_steps: int (fewer = faster)
            seed: int

        Returns:
            (video_tensor, last_frame_pil) or (None, None) on failure
        """
        self._load()

        tmp = tempfile.mkdtemp(prefix='lingbot_drive_')
        try:
            np.save(os.path.join(tmp, 'poses.npy'), poses[:frame_num])
            np.save(os.path.join(tmp, 'intrinsics.npy'), intrinsics[:frame_num])

            video = self._model.generate(
                input_prompt=prompt,
                img=image,
                action_path=tmp,
                max_area=self.height * self.width,
                frame_num=frame_num,
                shift=shift,
                sample_solver='unipc',
                sampling_steps=sampling_steps,
                guide_scale=5.0,
                seed=seed,
                offload_model=True,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        if video is None:
            return None, None

        import torch
        import torchvision.transforms.functional as TF

        last = video[:, -1, :, :]  # [C, H, W]
        last = ((last + 1.0) / 2.0).clamp(0, 1).cpu()
        last_pil = TF.to_pil_image(last)

        return video, last_pil

    def save_video(self, video_tensor, path, fps=16):
        if video_tensor is None:
            return
        lingbot_root = Path(__file__).resolve().parent.parent.parent / 'lingbot-world'
        if str(lingbot_root) not in sys.path:
            sys.path.insert(0, str(lingbot_root))
        from wan.utils.utils import save_video
        save_video(tensor=video_tensor[None], save_file=path, fps=fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        logger.info(f"Saved: {path}")


# =============================================================================
# Terminal Mode
# =============================================================================

def run_terminal(args):
    """Type WASD + Enter to generate driving video chunks."""
    from PIL import Image

    pose_gen = DrivingPoseGenerator(
        speed=args.speed, turn_rate=args.turn_rate, num_frames=args.frame_num)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.init_image and os.path.exists(args.init_image):
        current_image = Image.open(args.init_image).convert('RGB')
    else:
        current_image = Image.new('RGB', (832, 480), (100, 120, 100))
        logger.info("No init image — using placeholder.")

    prompt = args.prompt or \
        "Dashcam footage of a vehicle driving on a road during daytime. Clear weather, suburban environment."

    driver = None
    if not args.dry_run:
        driver = LingBotDriver(ckpt_dir=args.ckpt_dir, size=args.size,
                                t5_cpu=True)

    chunk = 0
    print("\n=== LingBot-World Interactive Driving Demo ===")
    print("Commands: w/a/s/d (combine: 'wa'), r=reset, q=quit\n")

    while True:
        try:
            inp = input(f"[Chunk {chunk}] WASD> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if inp == 'q':
            break
        if inp == 'r':
            pose_gen.reset()
            print("  Reset to origin.")
            continue

        keys = set(c for c in inp if c in 'wasd')
        poses, desc = pose_gen.wasd_to_poses(keys)
        h, w = int(args.size.split('*')[0]), int(args.size.split('*')[1])
        intrinsics = pose_gen.get_default_intrinsics(
            len(poses), width=w, height=h)

        disp = np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3])
        print(f"  Action: {desc}")
        print(f"  Displacement: {disp:.2f}, Yaw: {np.degrees(pose_gen.yaw):.1f} deg")

        if args.dry_run:
            np.save(str(out / f"chunk_{chunk:04d}_poses.npy"), poses)
            print(f"  [dry run] Saved poses. Start: {poses[0][:3,3]}, End: {poses[-1][:3,3]}")
        else:
            print(f"  Generating {args.frame_num} frames ...")
            t0 = time.time()
            video, last_frame = driver.generate(
                image=current_image, prompt=prompt,
                poses=poses, intrinsics=intrinsics,
                frame_num=args.frame_num,
                shift=3.0 if '480' in args.size else 10.0,
                sampling_steps=args.sample_steps,
                seed=args.seed + chunk)
            dt = time.time() - t0
            print(f"  Done in {dt:.1f}s")

            path = str(out / f"chunk_{chunk:04d}.mp4")
            driver.save_video(video, path)

            if last_frame is not None:
                current_image = last_frame
                current_image.save(str(out / f"chunk_{chunk:04d}_last.jpg"))

        chunk += 1

    print(f"\nOutput: {out}")


# =============================================================================
# Pygame Mode
# =============================================================================

def run_pygame(args):
    """Real-time WASD keyboard input with chunked video generation."""
    try:
        import pygame
    except ImportError:
        print("Install pygame: pip install pygame")
        sys.exit(1)
    from PIL import Image

    pose_gen = DrivingPoseGenerator(
        speed=args.speed, turn_rate=args.turn_rate, num_frames=args.frame_num)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    W_DISP, H_DISP = 832, 480
    pygame.init()
    screen = pygame.display.set_mode((W_DISP, H_DISP + 60))
    pygame.display.set_caption("LingBot-World Driving — WASD to drive, Q to quit")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 16)

    if args.init_image and os.path.exists(args.init_image):
        current_image = Image.open(args.init_image).convert('RGB')
    else:
        current_image = Image.new('RGB', (W_DISP, H_DISP), (100, 120, 100))

    prompt = args.prompt or "Dashcam footage driving on a road during daytime."

    driver = None
    if not args.dry_run:
        driver = LingBotDriver(ckpt_dir=args.ckpt_dir, size=args.size,
                                t5_cpu=True)

    frame_buf = deque()
    chunk_idx = 0
    status = "Ready"

    def pil_to_surface(img):
        img = img.resize((W_DISP, H_DISP))
        return pygame.image.fromstring(img.tobytes(), img.size, img.mode)

    current_surface = pil_to_surface(current_image)

    running = True
    while running:
        keys_held = set()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                running = False

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_w]: keys_held.add('w')
        if pressed[pygame.K_a]: keys_held.add('a')
        if pressed[pygame.K_s]: keys_held.add('s')
        if pressed[pygame.K_d]: keys_held.add('d')

        # Play buffered frames
        if frame_buf:
            current_surface = frame_buf.popleft()
        elif keys_held:
            # Need new chunk
            status = "Generating..."
            pygame.display.set_caption(f"LingBot Driving — {status}")

            poses, desc = pose_gen.wasd_to_poses(keys_held)
            h, w = int(args.size.split('*')[0]), int(args.size.split('*')[1])
            intrinsics = pose_gen.get_default_intrinsics(
                len(poses), width=w, height=h)

            if args.dry_run:
                frame_buf.extend([current_surface] * args.frame_num)
                status = f"Dry: {desc}"
            else:
                video, last_frame = driver.generate(
                    image=current_image, prompt=prompt,
                    poses=poses, intrinsics=intrinsics,
                    frame_num=args.frame_num,
                    shift=3.0 if '480' in args.size else 10.0,
                    sampling_steps=args.sample_steps,
                    seed=args.seed + chunk_idx)

                if video is not None:
                    import torch
                    for f in range(video.shape[1]):
                        ft = video[:, f, :, :]
                        ft = ((ft + 1.0) / 2.0).clamp(0, 1)
                        arr = (ft.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        pil = Image.fromarray(arr)
                        frame_buf.append(pil_to_surface(pil))

                if last_frame is not None:
                    current_image = last_frame

                status = f"Chunk {chunk_idx}: {desc}"

            chunk_idx += 1

        # Draw
        screen.fill((20, 20, 30))
        screen.blit(current_surface, (0, 0))

        # HUD
        lines = [
            f"Keys: {''.join(sorted(keys_held)) or '-'}  "
            f"Yaw: {np.degrees(pose_gen.yaw):.0f}  "
            f"Buf: {len(frame_buf)}  {status}",
        ]
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, (0, 255, 100)),
                        (10, H_DISP + 10 + i * 20))

        pygame.display.flip()
        clock.tick(16)

    pygame.quit()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Interactive WASD driving with LingBot-World')
    parser.add_argument('--mode', default='terminal',
                        choices=['terminal', 'pygame'])
    parser.add_argument('--ckpt_dir', default=None,
                        help='Path to lingbot-world-base-cam weights')
    parser.add_argument('--init_image', default=None,
                        help='Starting dashcam image')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--output', default='demo_output')
    parser.add_argument('--size', default='480*832',
                        choices=['480*832', '720*1280'])
    parser.add_argument('--frame_num', type=int, default=17,
                        help='Frames per chunk (must be 4n+1)')
    parser.add_argument('--sample_steps', type=int, default=30,
                        help='Denoising steps (fewer=faster)')
    parser.add_argument('--speed', type=float, default=0.3)
    parser.add_argument('--turn_rate', type=float, default=2.0,
                        help='Degrees per frame when turning')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true',
                        help='Test WASD-to-pose without loading model')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s')

    if not args.dry_run and args.ckpt_dir is None:
        print("ERROR: --ckpt_dir required (or use --dry_run)")
        print("Download: huggingface-cli download robbyant/lingbot-world-base-cam "
              "--local-dir ./lingbot-world-base-cam")
        sys.exit(1)

    if args.mode == 'terminal':
        run_terminal(args)
    elif args.mode == 'pygame':
        run_pygame(args)


if __name__ == '__main__':
    main()

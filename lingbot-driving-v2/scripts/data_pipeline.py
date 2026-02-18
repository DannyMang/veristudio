"""
data_pipeline.py -- Pre-encoded dataset for LingBot-World post-training.

Two components:
  1. PreprocessPipeline -- encodes raw video+text through frozen VAE/T5, saves .pt files.
  2. PostTrainingDataset -- loads pre-encoded samples, computes Plucker embeddings on-the-fly.

Each .pt file on disk:
    {
        "latents":    bfloat16 (16, T_lat, H_lat, W_lat),
        "context":    bfloat16 (512, 4096),
        "c2ws":       float32  (T_vid, 4, 4),
        "intrinsics": float32  (T_vid, 4),
        "y":          bfloat16 (17, T_lat, H_lat, W_lat),  # first-frame mask + VAE cond
    }

Usage:
    # Pre-encode
    python data_pipeline.py preprocess \
        --data_dir /path/to/raw_clips \
        --model_dir /path/to/lingbot-world-base-cam \
        --output_dir /path/to/encoded_data

    # Test dataset loading
    python data_pipeline.py test \
        --encoded_dir /path/to/encoded_data \
        --num_chunks 4 --chunk_frames 5
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# PostTrainingDataset
# =============================================================================


class PostTrainingDataset(Dataset):
    """
    Dataset that loads pre-encoded .pt samples for post-training.

    Slices latents to chunk_size * num_chunks latent frames.
    Computes Plucker embeddings on-the-fly from stored c2ws + intrinsics.
    """

    def __init__(
        self,
        encoded_dir,
        num_chunks=4,
        chunk_lat_frames=5,
        vae_stride=(4, 8, 8),
        patch_size=(1, 2, 2),
        height=480,
        width=832,
    ):
        self.encoded_dir = encoded_dir
        self.num_chunks = num_chunks
        self.chunk_lat_frames = chunk_lat_frames
        self.total_lat_frames = num_chunks * chunk_lat_frames
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.height = height
        self.width = width

        self.files = sorted(
            f for f in os.listdir(encoded_dir) if f.endswith(".pt")
        )
        logger.info(
            f"PostTrainingDataset: {len(self.files)} samples, "
            f"{num_chunks} chunks x {chunk_lat_frames} lat frames"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.encoded_dir, self.files[idx])
        data = torch.load(path, map_location="cpu", weights_only=False)

        latents = data["latents"]  # (16, T_lat, H_lat, W_lat) bf16
        context = data["context"]  # (512, 4096) bf16
        c2ws = data["c2ws"]  # (T_vid, 4, 4) fp32
        intrinsics = data["intrinsics"]  # (T_vid, 4) fp32
        y = data["y"]  # (17, T_lat, H_lat, W_lat) bf16

        T_lat = latents.shape[1]
        T_use = min(self.total_lat_frames, T_lat)

        # Slice latents to required length
        latents = latents[:, :T_use]
        y = y[:, :T_use]

        # Subsample c2ws/intrinsics to latent temporal resolution
        T_vid = c2ws.shape[0]
        T_vid_use = T_use * self.vae_stride[0]
        if T_vid_use > T_vid:
            T_vid_use = T_vid
        c2ws = c2ws[:T_vid_use]
        intrinsics = intrinsics[:T_vid_use]

        # Compute Plucker embeddings
        plucker_emb = self._compute_plucker(
            c2ws, intrinsics, T_use, latents.shape[2], latents.shape[3]
        )

        return {
            "latents": latents,
            "context": context,
            "y": y,
            "plucker_emb": plucker_emb,
            "num_lat_frames": T_use,
        }

    def _compute_plucker(self, c2ws, intrinsics, lat_f, lat_h, lat_w):
        """Compute Plucker embeddings for a clip, matching WanModel's format.

        Returns: (C, lat_f, lat_h, lat_w) in bfloat16, matching the format
        expected by WanModel.forward() dit_cond_dict["c2ws_plucker_emb"].
        """
        from einops import rearrange

        # Add sys path for cam_utils
        wan_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "lingbot-world"
        )
        if wan_root not in sys.path:
            sys.path.insert(0, wan_root)
        from wan.utils.cam_utils import (
            compute_relative_poses,
            get_Ks_transformed,
            get_plucker_embeddings,
            interpolate_camera_poses,
        )

        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        # Transform intrinsics to match target resolution
        Ks = get_Ks_transformed(
            intrinsics,
            height_org=480,
            width_org=832,
            height_resize=h,
            width_resize=w,
            height_final=h,
            width_final=w,
        )
        Ks = Ks[0]  # (4,)

        c2ws_np = c2ws.numpy() if isinstance(c2ws, torch.Tensor) else c2ws
        len_c2ws = len(c2ws_np)

        # Interpolate to latent frame count
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws_np[:, :3, :3],
            src_trans_vec=c2ws_np[:, :3, 3],
            tgt_indices=np.linspace(
                0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1
            ),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks_rep = Ks.repeat(len(c2ws_infer), 1)

        plucker = get_plucker_embeddings(c2ws_infer, Ks_rep, h, w)
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        plucker = plucker[None, ...]  # (1, f*h*w, C)
        plucker = rearrange(
            plucker,
            "b (f h w) c -> b c f h w",
            f=lat_f,
            h=lat_h,
            w=lat_w,
        )
        return plucker.squeeze(0).to(torch.bfloat16)


# =============================================================================
# PreprocessPipeline
# =============================================================================


class PreprocessPipeline:
    """
    Encodes raw video clips through frozen VAE + T5 and saves .pt files.

    Expected input directory structure per clip:
        clip_dir/
            frames/       -- 0000.png, 0001.png, ...
            poses.npy     -- (N, 4, 4) c2w matrices
            intrinsics.npy -- (N, 4) [fx, fy, cx, cy]
            caption.txt   -- text description
    """

    def __init__(self, model_dir, device="cuda:0"):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.vae = None
        self.t5 = None
        self.cfg = None

    def _load_models(self):
        """Lazy-load VAE and T5 (only needed during preprocessing)."""
        if self.vae is not None:
            return

        wan_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "lingbot-world"
        )
        if wan_root not in sys.path:
            sys.path.insert(0, wan_root)

        from wan.configs import WAN_CONFIGS
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_1 import Wan2_1_VAE

        self.cfg = WAN_CONFIGS["i2v-A14B"]

        logger.info("Loading T5 text encoder...")
        self.t5 = T5EncoderModel(
            text_len=self.cfg.text_len,
            dtype=self.cfg.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(
                self.model_dir, self.cfg.t5_checkpoint
            ),
            tokenizer_path=os.path.join(
                self.model_dir, self.cfg.t5_tokenizer
            ),
        )

        logger.info("Loading VAE...")
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(self.model_dir, self.cfg.vae_checkpoint),
            device=self.device,
        )

    def encode_clip(self, clip_dir, output_path, num_frames=17, resolution=480):
        """Encode a single clip and save to disk.

        Args:
            clip_dir: Path to clip directory containing frames/, poses.npy, etc.
            output_path: Where to save the .pt file.
            num_frames: Number of video frames (must be 4n+1).
            resolution: Target height (480 or 720).
        """
        import cv2
        import torchvision.transforms.functional as TF

        self._load_models()

        # Load frames
        frame_dir = os.path.join(clip_dir, "frames")
        frame_files = sorted(
            f
            for f in os.listdir(frame_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        )

        if len(frame_files) < num_frames:
            logger.warning(
                f"Clip {clip_dir} has {len(frame_files)} frames, "
                f"need {num_frames}. Padding."
            )
            while len(frame_files) < num_frames:
                frame_files.append(frame_files[-1])

        frame_files = frame_files[:num_frames]

        # Resolution
        res_map = {320: (320, 576), 480: (480, 832), 720: (720, 1280)}
        h, w = res_map.get(resolution, (480, 832))

        vae_stride = self.cfg.vae_stride
        patch_size = self.cfg.patch_size

        lat_h = round(
            np.sqrt(h * w * (h / w)) // vae_stride[1] // patch_size[1]
            * patch_size[1]
        )
        lat_w = round(
            np.sqrt(h * w / (h / w)) // vae_stride[2] // patch_size[2]
            * patch_size[2]
        )
        h = lat_h * vae_stride[1]
        w = lat_w * vae_stride[2]

        # Load and preprocess video frames
        frames = []
        for ff in frame_files:
            img = cv2.imread(os.path.join(frame_dir, ff))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h))
            img = img.astype(np.float32) / 127.5 - 1.0
            frames.append(torch.from_numpy(img).permute(2, 0, 1))

        video = torch.stack(frames, dim=1)  # (3, T, H, W)

        # Load poses and intrinsics
        c2ws = torch.from_numpy(
            np.load(os.path.join(clip_dir, "poses.npy"))
        ).float()[:num_frames]
        intrinsics = torch.from_numpy(
            np.load(os.path.join(clip_dir, "intrinsics.npy"))
        ).float()[:num_frames]

        # Load caption
        caption_path = os.path.join(clip_dir, "caption.txt")
        if os.path.exists(caption_path):
            with open(caption_path) as f:
                caption = f.read().strip()
        else:
            caption = "A dashcam view of a vehicle driving on a road."

        # VAE encode
        F_frames = num_frames
        lat_f = (F_frames - 1) // vae_stride[0] + 1

        # First-frame mask
        msk = torch.ones(1, F_frames, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.cat(
            [
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                msk[:, 1:],
            ],
            dim=1,
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        with torch.no_grad():
            # Encode video through VAE
            # First frame gets special treatment (image conditioning)
            first_frame = video[:, 0:1]  # (3, 1, H, W)
            img_cond = torch.cat(
                [first_frame, torch.zeros(3, F_frames - 1, h, w)], dim=1
            ).to(self.device)
            latents = self.vae.encode([img_cond])[0]  # (16, T_lat, H_lat, W_lat)

            # Build y = mask + VAE-encoded first-frame conditioning
            y = torch.cat([msk, latents], dim=0)  # (17, T_lat, H_lat, W_lat)

            # Encode the actual video (training target)
            video_latents = self.vae.encode([video.to(self.device)])[0]

            # T5 encode
            ctx = self.t5([caption], self.device)[0]  # (512, 4096)

        # Save
        sample = {
            "latents": video_latents.to(torch.bfloat16).cpu(),
            "context": ctx.to(torch.bfloat16).cpu(),
            "c2ws": c2ws.cpu(),
            "intrinsics": intrinsics.cpu(),
            "y": y.to(torch.bfloat16).cpu(),
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(sample, output_path)
        return sample

    def encode_dataset(self, data_dir, output_dir, num_frames=17, resolution=480):
        """Encode all clips in a directory.

        Args:
            data_dir: Root directory containing clip subdirectories.
            output_dir: Where to save encoded .pt files.
            num_frames: Frames per clip.
            resolution: Target height.
        """
        clip_dirs = sorted(
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )

        logger.info(
            f"Encoding {len(clip_dirs)} clips at {resolution}p, "
            f"{num_frames} frames"
        )

        os.makedirs(output_dir, exist_ok=True)

        for i, clip_name in enumerate(clip_dirs):
            clip_dir = os.path.join(data_dir, clip_name)
            output_path = os.path.join(output_dir, f"{i:06d}.pt")

            if os.path.exists(output_path):
                logger.info(f"  [{i+1}/{len(clip_dirs)}] Skipping {clip_name} (exists)")
                continue

            try:
                self.encode_clip(clip_dir, output_path, num_frames, resolution)
                if (i + 1) % 10 == 0:
                    logger.info(f"  [{i+1}/{len(clip_dirs)}] Encoded {clip_name}")
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"  [{i+1}/{len(clip_dirs)}] Failed {clip_name}: {e}")
                continue

        logger.info(f"Encoding complete. Saved to {output_dir}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Post-training data pipeline")
    sub = parser.add_subparsers(dest="command")

    # Preprocess command
    pp = sub.add_parser("preprocess", help="Encode raw clips through VAE/T5")
    pp.add_argument("--data_dir", required=True, help="Raw clips directory")
    pp.add_argument("--model_dir", required=True, help="Model checkpoint dir")
    pp.add_argument("--output_dir", required=True, help="Output directory")
    pp.add_argument("--num_frames", type=int, default=17)
    pp.add_argument("--resolution", type=int, default=480)

    # Test command
    tt = sub.add_parser("test", help="Test dataset loading")
    tt.add_argument("--encoded_dir", required=True, help="Encoded data dir")
    tt.add_argument("--num_chunks", type=int, default=4)
    tt.add_argument("--chunk_frames", type=int, default=5)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "preprocess":
        pipeline = PreprocessPipeline(args.model_dir)
        pipeline.encode_dataset(
            args.data_dir, args.output_dir, args.num_frames, args.resolution
        )
    elif args.command == "test":
        ds = PostTrainingDataset(
            args.encoded_dir,
            num_chunks=args.num_chunks,
            chunk_lat_frames=args.chunk_frames,
        )
        sample = ds[0]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")
            else:
                print(f"  {k}: {v}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

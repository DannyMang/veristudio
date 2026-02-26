"""
waymo_converter.py -- Read Waymo Open Dataset v2.0.1 Parquet files and yield training clips.

Reads camera_image, camera_calibration, and vehicle_pose Parquet components,
extracts FRONT camera data, computes camera-to-world matrices, and yields
clips ready for VAE/T5 encoding. No intermediate files written to disk.

Requirements:
    pip install pandas pyarrow Pillow

Usage:
    from waymo_converter import WaymoParquetReader

    reader = WaymoParquetReader("/path/to/waymo_v2/training")
    for clip in reader.iter_clips(frames_per_clip=81):
        # clip["frames"]     -- list of np.ndarray (H, W, 3) uint8
        # clip["c2ws"]       -- np.ndarray (N, 4, 4) float32
        # clip["intrinsics"] -- np.ndarray (N, 4) float32 [fx, fy, cx, cy]
        # clip["segment_name"] -- str
        # clip["clip_idx"]   -- int
"""

import logging
import os
from io import BytesIO
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CAMERA_NAME_MAP = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
}


class WaymoParquetReader:
    """Reads Waymo v2.0.1 Parquet files and yields training clips in memory."""

    def __init__(self, data_root):
        """
        Args:
            data_root: Path to a Waymo v2 split directory, e.g.
                       /waymo-data/training/
                       Must contain camera_image/, camera_calibration/,
                       and vehicle_pose/ subdirectories with .parquet files.
        """
        self.data_root = Path(data_root)
        self.camera_image_dir = self.data_root / "camera_image"
        self.camera_cal_dir = self.data_root / "camera_calibration"
        self.vehicle_pose_dir = self.data_root / "vehicle_pose"

        for d in [self.camera_image_dir, self.camera_cal_dir, self.vehicle_pose_dir]:
            if not d.exists():
                raise FileNotFoundError(
                    f"Missing Waymo component directory: {d}\n"
                    f"Expected structure: {data_root}/camera_image/, "
                    f"camera_calibration/, vehicle_pose/"
                )

        self.segment_names = sorted(
            p.stem for p in self.camera_image_dir.glob("*.parquet")
        )
        logger.info(
            f"WaymoParquetReader: {len(self.segment_names)} segments in {data_root}"
        )

    def read_segment(self, segment_name, camera_id=1):
        """Read all frames + poses for one camera from a single segment.

        Args:
            segment_name: Segment context name (Parquet filename stem).
            camera_id: Waymo camera ID (1=FRONT, 2=FRONT_LEFT, ...).

        Returns:
            dict with keys: frames, c2ws, intrinsics, img_height, img_width
        """
        import pandas as pd
        from PIL import Image

        # -- Camera images --
        img_path = self.camera_image_dir / f"{segment_name}.parquet"
        img_df = pd.read_parquet(img_path)
        img_df = img_df[img_df["key.camera_name"] == camera_id]
        img_df = img_df.sort_values("key.frame_timestamp_micros").reset_index(
            drop=True
        )

        if img_df.empty:
            raise ValueError(
                f"No images for camera {camera_id} in segment {segment_name}"
            )

        # -- Camera calibration (one row per camera, constant across segment) --
        cal_path = self.camera_cal_dir / f"{segment_name}.parquet"
        cal_df = pd.read_parquet(cal_path)
        cal_row = cal_df[cal_df["key.camera_name"] == camera_id].iloc[0]

        # Waymo v2 stores intrinsics as separate columns, not a single array
        fx = float(cal_row["[CameraCalibrationComponent].intrinsic.f_u"])
        fy = float(cal_row["[CameraCalibrationComponent].intrinsic.f_v"])
        cx = float(cal_row["[CameraCalibrationComponent].intrinsic.c_u"])
        cy = float(cal_row["[CameraCalibrationComponent].intrinsic.c_v"])

        extrinsic_flat = list(
            cal_row["[CameraCalibrationComponent].extrinsic.transform"]
        )
        vehicle_from_camera = np.array(extrinsic_flat, dtype=np.float64).reshape(
            4, 4
        )

        img_width = int(cal_row["[CameraCalibrationComponent].width"])
        img_height = int(cal_row["[CameraCalibrationComponent].height"])

        # -- Vehicle poses --
        pose_path = self.vehicle_pose_dir / f"{segment_name}.parquet"
        pose_df = pd.read_parquet(pose_path)
        pose_df = pose_df.sort_values("key.frame_timestamp_micros").reset_index(
            drop=True
        )

        pose_timestamps = pose_df["key.frame_timestamp_micros"].values

        # -- Build per-frame data --
        frames = []
        c2ws = []

        for _, img_row in img_df.iterrows():
            ts = img_row["key.frame_timestamp_micros"]

            # Decode JPEG
            img_bytes = img_row["[CameraImageComponent].image"]
            img = Image.open(BytesIO(img_bytes))
            frames.append(np.array(img))

            # Find closest pose by timestamp
            pose_idx = int(np.argmin(np.abs(pose_timestamps - ts)))
            pose_row = pose_df.iloc[pose_idx]
            wfv_flat = list(
                pose_row[
                    "[VehiclePoseComponent].world_from_vehicle.transform"
                ]
            )
            world_from_vehicle = np.array(wfv_flat, dtype=np.float64).reshape(
                4, 4
            )

            # camera-to-world = world_from_vehicle @ vehicle_from_camera
            c2w = (world_from_vehicle @ vehicle_from_camera).astype(np.float32)
            c2ws.append(c2w)

        intrinsics = np.tile(
            np.array([fx, fy, cx, cy], dtype=np.float32),
            (len(frames), 1),
        )

        return {
            "frames": frames,
            "c2ws": np.array(c2ws, dtype=np.float32),
            "intrinsics": intrinsics,
            "img_height": img_height,
            "img_width": img_width,
        }

    def iter_clips(
        self,
        frames_per_clip=81,
        camera="FRONT",
        max_segments=None,
        caption=None,
    ):
        """Iterate over all segments and yield fixed-length training clips.

        Args:
            frames_per_clip: Frames per clip. Should be 4n+1 for the VAE
                (e.g. 17, 21, 81). 81 frames at 10 Hz = 8.1s, yielding
                21 latent frames (enough for 4 chunks x 5 lat frames).
            camera: Camera name string (FRONT, FRONT_LEFT, ...).
            max_segments: Process at most this many segments (for testing).
            caption: Text caption for all clips. Defaults to a generic
                driving description.

        Yields:
            dict: frames, c2ws, intrinsics, segment_name, clip_idx, caption
        """
        camera_id = CAMERA_NAME_MAP[camera.upper()]
        segments = (
            self.segment_names[:max_segments]
            if max_segments
            else self.segment_names
        )

        if caption is None:
            caption = (
                "A front-facing dashcam view of an autonomous vehicle "
                "driving through urban and suburban environments."
            )

        total_clips = 0

        for seg_idx, segment_name in enumerate(segments):
            try:
                data = self.read_segment(segment_name, camera_id)
            except Exception as e:
                logger.warning(f"Skipping segment {segment_name}: {e}")
                continue

            n_frames = len(data["frames"])
            num_clips = n_frames // frames_per_clip

            for clip_idx in range(num_clips):
                s = clip_idx * frames_per_clip
                e = s + frames_per_clip

                yield {
                    "frames": data["frames"][s:e],
                    "c2ws": data["c2ws"][s:e],
                    "intrinsics": data["intrinsics"][s:e],
                    "segment_name": segment_name,
                    "clip_idx": clip_idx,
                    "caption": caption,
                    "img_height": data["img_height"],
                    "img_width": data["img_width"],
                }
                total_clips += 1

            if (seg_idx + 1) % 50 == 0:
                logger.info(
                    f"  [{seg_idx + 1}/{len(segments)}] segments, "
                    f"{total_clips} clips so far"
                )

        logger.info(
            f"Finished: {len(segments)} segments, {total_clips} total clips"
        )


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Test Waymo v2 Parquet reader"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to Waymo v2 split dir (e.g. /waymo-data/training)",
    )
    parser.add_argument("--camera", default="FRONT")
    parser.add_argument("--frames_per_clip", type=int, default=81)
    parser.add_argument("--max_segments", type=int, default=2)
    args = parser.parse_args()

    reader = WaymoParquetReader(args.data_root)
    for clip in reader.iter_clips(
        frames_per_clip=args.frames_per_clip,
        camera=args.camera,
        max_segments=args.max_segments,
    ):
        print(
            f"Segment: {clip['segment_name']}, Clip: {clip['clip_idx']}, "
            f"Frames: {len(clip['frames'])}, "
            f"Frame shape: {clip['frames'][0].shape}, "
            f"c2ws: {clip['c2ws'].shape}, "
            f"intrinsics: {clip['intrinsics'].shape}"
        )

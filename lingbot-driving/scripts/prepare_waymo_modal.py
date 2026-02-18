"""
prepare_waymo_modal.py — Download + process Waymo Open Dataset v2 on Modal.

Extracts FRONT camera frames, ego-poses, and driving actions from Waymo v2
(Parquet format) and saves them to a Modal Volume for LoRA training.

Setup:
    1. Accept the Waymo Open Dataset license at https://waymo.com/open/
    2. Create a GCP service account with Storage Object Viewer role
    3. Download the JSON key and create a Modal secret:
       modal secret create gcp-credentials GCP_SERVICE_ACCOUNT_JSON=@/path/to/key.json
    4. Run:
       modal run prepare_waymo_modal.py

    # Process fewer segments (faster for testing)
    modal run prepare_waymo_modal.py --max-segments 10

    # Then build the training manifest
    modal run prepare_waymo_modal.py --build-manifest
"""

import modal

app = modal.App("waymo-data-prep")

# Volume for storing processed training data
data_volume = modal.Volume.from_name("waymo-training-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "gcsfs",
        "pyarrow",
        "pandas",
        "numpy",
        "Pillow",
        "tqdm",
        "opencv-python-headless",
    )
)

WAYMO_GCS_BUCKET = "waymo_open_dataset_v_2_0_1"
DATA_DIR = "/data/waymo_processed"


# ---------------------------------------------------------------------------
# Helpers (run inside the container)
# ---------------------------------------------------------------------------

def _rotation_matrix_to_yaw(R):
    """Extract yaw angle from 3x3 rotation matrix."""
    import numpy as np
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _compute_plucker_6d(c2w_prev, c2w_curr):
    """
    Compute 6D Plucker representation of camera motion between two frames.
    Returns [direction(3), moment(3)] = 6D vector.
    """
    import numpy as np
    pos_prev = c2w_prev[:3, 3]
    pos_curr = c2w_curr[:3, 3]
    delta = pos_curr - pos_prev
    dist = np.linalg.norm(delta)
    if dist < 1e-8:
        return [0.0] * 6
    direction = delta / dist
    moment = np.cross(pos_curr, direction)
    return direction.tolist() + moment.tolist()


def _compute_driving_actions(c2ws, timestamps_us):
    """
    Derive driving actions from c2w pose sequence.
    Same binning as prepare_nuscenes.py for compatibility.

    Returns list of action dicts with 'plucker' (6D), 'multihot' (14D), etc.
    """
    import numpy as np

    bins = {
        'speed': [0.0, 0.5, 5.0, 15.0, float('inf')],
        'speed_labels': ['stopped', 'slow', 'cruise', 'fast'],
        'yaw_rate': [-float('inf'), -0.15, -0.02, 0.02, 0.15, float('inf')],
        'yaw_rate_labels': ['hard_left', 'soft_left', 'straight', 'soft_right', 'hard_right'],
        'acceleration': [-float('inf'), -2.0, -0.5, 0.5, 2.0, float('inf')],
        'accel_labels': ['hard_brake', 'soft_brake', 'coast', 'soft_accel', 'hard_accel'],
    }

    actions = []
    velocities = []

    for i in range(len(c2ws)):
        pos_i = c2ws[i][:3, 3]
        yaw_i = _rotation_matrix_to_yaw(c2ws[i][:3, :3])

        # Plucker embedding
        if i > 0:
            plucker = _compute_plucker_6d(c2ws[i - 1], c2ws[i])
        else:
            plucker = [0.0] * 6

        # Speed
        if i > 0:
            pos_prev = c2ws[i - 1][:3, 3]
            dt = (timestamps_us[i] - timestamps_us[i - 1]) / 1e6
            if dt > 0:
                velocity = float(np.linalg.norm(pos_i[:2] - pos_prev[:2]) / dt)
            else:
                velocity = 0.0
        elif len(c2ws) > 1:
            pos_next = c2ws[1][:3, 3]
            dt = (timestamps_us[1] - timestamps_us[0]) / 1e6
            velocity = float(np.linalg.norm(pos_next[:2] - pos_i[:2]) / max(dt, 0.01))
        else:
            velocity = 0.0
        velocities.append(velocity)

        # Yaw rate
        if i > 0:
            yaw_prev = _rotation_matrix_to_yaw(c2ws[i - 1][:3, :3])
            dt = max((timestamps_us[i] - timestamps_us[i - 1]) / 1e6, 0.01)
            dyaw = (yaw_i - yaw_prev + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = float(dyaw / dt)
        else:
            yaw_rate = 0.0

        # Acceleration
        if i >= 2:
            dt = max((timestamps_us[i] - timestamps_us[i - 1]) / 1e6, 0.01)
            acceleration = float((velocities[i] - velocities[i - 1]) / dt)
        else:
            acceleration = 0.0

        # Discretize
        speed_idx = int(np.clip(np.digitize(velocity, bins['speed']) - 1, 0, 3))
        yaw_idx = int(np.clip(np.digitize(yaw_rate, bins['yaw_rate']) - 1, 0, 4))
        accel_idx = int(np.clip(np.digitize(acceleration, bins['acceleration']) - 1, 0, 4))

        multihot = [0.0] * 14
        multihot[speed_idx] = 1.0
        multihot[4 + yaw_idx] = 1.0
        multihot[9 + accel_idx] = 1.0

        actions.append({
            'plucker': plucker,
            'multihot': multihot,
            'velocity': velocity,
            'yaw_rate': yaw_rate,
            'acceleration': acceleration,
            'heading': float(yaw_i),
            'speed_label': bins['speed_labels'][speed_idx],
            'steer_label': bins['yaw_rate_labels'][yaw_idx],
            'accel_label': bins['accel_labels'][accel_idx],
            'translation': c2ws[i][:3, 3].tolist(),
        })

    return actions


def _classify_driving_behavior(actions):
    """Classify segment behavior (matches prepare_nuscenes.py categories)."""
    import numpy as np
    if not actions:
        return 'unknown'
    velocities = [a['velocity'] for a in actions]
    yaw_rates = [abs(a['yaw_rate']) for a in actions]
    avg_speed = float(np.mean(velocities))
    max_yaw = float(np.max(yaw_rates)) if yaw_rates else 0
    stopped_frac = float(np.mean([v < 0.5 for v in velocities]))
    if avg_speed < 0.5:
        return 'stationary'
    elif avg_speed < 3.0:
        return 'slow_exploration'
    elif max_yaw > 0.3:
        return 'sharp_maneuver'
    elif max_yaw > 0.1:
        return 'urban_navigation'
    elif avg_speed > 15.0:
        return 'highway'
    elif stopped_frac > 0.3:
        return 'stop_and_go'
    return 'cruising'


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=7200,  # 2 hours max
    memory=32768,  # 32 GB RAM for large parquet reads
    secrets=[modal.Secret.from_name("gcp-credentials")],
)
def process_waymo(
    split: str = "validation",
    camera_name: int = 1,  # 1 = FRONT in Waymo v2
    max_segments: int = 200,
    target_height: int = 480,
    target_width: int = 832,
):
    """
    Download and process Waymo v2 data from GCS.

    Camera names in Waymo v2:
        1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT
    """
    import json
    import os

    import cv2
    import gcsfs
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from tqdm import tqdm

    # Set up GCS access — try multiple credential sources
    gcp_creds = os.environ.get("GCP_APPLICATION_CREDENTIALS")
    gcp_key = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if gcp_creds:
        # User OAuth credentials (from gcloud auth application-default login)
        key_path = "/tmp/gcp_creds.json"
        with open(key_path, "w") as f:
            f.write(gcp_creds)
        fs = gcsfs.GCSFileSystem(token=key_path)
    elif gcp_key:
        # Service account key
        key_path = "/tmp/gcp_key.json"
        with open(key_path, "w") as f:
            f.write(gcp_key)
        fs = gcsfs.GCSFileSystem(token=key_path)
    else:
        # Try anonymous access (may work for some public datasets)
        fs = gcsfs.GCSFileSystem(token="anon")

    base_path = f"{WAYMO_GCS_BUCKET}/{split}"
    out_root = f"{DATA_DIR}/{split}"
    os.makedirs(out_root, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: List available segments
    # -----------------------------------------------------------------------
    print(f"Listing segments in gs://{base_path}/camera_image/ ...")
    try:
        segment_files = fs.ls(f"{base_path}/camera_image/")
    except Exception as e:
        print(f"Error accessing GCS: {e}")
        print("Make sure you've:")
        print("  1. Accepted the Waymo license at waymo.com/open/")
        print("  2. Created a GCP service account with access")
        print("  3. Set up the Modal secret: modal secret create gcp-credentials ...")
        return

    segment_files = [f for f in segment_files if f.endswith(".parquet")]
    segment_files = segment_files[:max_segments]
    print(f"Found {len(segment_files)} segment files, processing {len(segment_files)}")

    # -----------------------------------------------------------------------
    # Step 2: Load camera calibration (small, load all at once)
    # -----------------------------------------------------------------------
    print("Loading camera calibration ...")
    calib_files = fs.ls(f"{base_path}/camera_calibration/")
    calib_files = [f for f in calib_files if f.endswith(".parquet")]

    # Helper: read an array value that might be a single list column (v2.0.0)
    # or split into individual sub-columns (v2.0.1)
    def _read_array_col(row, df_cols, base_col, sub_fields=None):
        """Read a value that's either a single array column or split sub-columns.

        Args:
            row: DataFrame row
            df_cols: list of all column names
            base_col: e.g. "[CameraCalibrationComponent].intrinsic"
            sub_fields: ordered field names, e.g. ["f_u", "f_v", "c_u", "c_v"]
                        If None, tries numbered sub-cols (.0, .1, .2, ...)
        Returns:
            numpy array
        """
        # Case 1: single column with array/list data (v2.0.0)
        if base_col in df_cols:
            val = row[base_col]
            return np.array(val)

        # Case 2: split into named sub-columns (v2.0.1)
        if sub_fields:
            cols = [f"{base_col}.{f}" for f in sub_fields]
            present = [c for c in cols if c in df_cols]
            if present:
                return np.array([row[c] for c in present])

        # Case 3: split into numbered sub-columns (.0, .1, .2, ...)
        numbered = sorted(
            [c for c in df_cols if c.startswith(base_col + ".")],
            key=lambda c: c.split(".")[-1]
        )
        if numbered:
            return np.array([row[c] for c in numbered])

        return None

    # Build a lookup: segment_context_name -> calibration for FRONT camera
    calibrations = {}
    _printed_cols = False

    # Intrinsic sub-fields in correct order (Waymo v2 CameraCalibration proto)
    INTRINSIC_FIELDS = ["f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3"]

    for cf in tqdm(calib_files, desc="Calibrations"):
        with fs.open(cf, "rb") as fh:
            table = pq.read_table(fh)
            df = table.to_pandas()

        if not _printed_cols:
            print(f"Calibration columns: {list(df.columns)}")
            _printed_cols = True

        # Detect key columns (works for both v2.0.0 and v2.0.1)
        cam_col = next((c for c in df.columns if "camera_name" in c.lower()), None)
        ctx_col = next((c for c in df.columns if "segment_context_name" in c.lower()), None)

        if not cam_col or not ctx_col:
            print(f"WARNING: Missing cam/ctx columns. Available: {list(df.columns)[:10]}")
            continue

        # Filter for the target camera
        cam_df = df[df[cam_col] == camera_name]

        for _, row in cam_df.iterrows():
            ctx = row[ctx_col]

            intrinsic = _read_array_col(
                row, list(df.columns),
                "[CameraCalibrationComponent].intrinsic",
                sub_fields=INTRINSIC_FIELDS,
            )
            extrinsic = _read_array_col(
                row, list(df.columns),
                "[CameraCalibrationComponent].extrinsic.transform",
            )

            if intrinsic is None:
                print(f"WARNING: No intrinsic data for {ctx[:20]}...")
                continue
            if extrinsic is None:
                # Fallback: identity
                extrinsic = np.eye(4).flatten()

            calibrations[ctx] = {
                "intrinsic": intrinsic,  # [fu, fv, cu, cv, k1, k2, p1, p2, k3]
                "extrinsic": extrinsic.reshape(4, 4),  # camera_to_vehicle
            }
    print(f"Loaded calibrations for {len(calibrations)} segments")

    # -----------------------------------------------------------------------
    # Step 3: Process each segment
    # -----------------------------------------------------------------------
    results = []
    total_frames = 0

    for seg_file in tqdm(segment_files, desc="Segments"):
        seg_name = os.path.basename(seg_file).replace(".parquet", "")

        # Check if already processed
        seg_dir = os.path.join(out_root, seg_name)
        manifest_path = os.path.join(seg_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                existing = json.load(f)
            results.append(existing.get("metadata", {}))
            total_frames += existing["metadata"]["num_frames"]
            continue

        if seg_name not in calibrations:
            print(f"  Skipping {seg_name[:20]}... (no calibration)")
            continue

        calib = calibrations[seg_name]

        # Read camera images for this segment
        try:
            with fs.open(seg_file, "rb") as fh:
                table = pq.read_table(fh)
                cam_df = table.to_pandas()
        except Exception as e:
            print(f"  Error reading {seg_name[:20]}...: {e}")
            continue

        # Detect column names (v2.0.0 vs v2.0.1 differ)
        def _find_col(df, *candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            # Fuzzy match
            for c in candidates:
                key = c.split(".")[-1].lower()
                for col in df.columns:
                    if key in col.lower():
                        return col
            return None

        if total_frames == 0:
            print(f"Image columns (first segment): {list(cam_df.columns)}")

        img_cam_col = _find_col(cam_df, "key.camera_name")
        img_ts_col = _find_col(cam_df, "key.frame_timestamp_micros")
        img_data_col = _find_col(cam_df, "[CameraImageComponent].image")

        if not img_cam_col or not img_ts_col or not img_data_col:
            print(f"  Skipping {seg_name[:20]}... (missing image columns: {list(cam_df.columns)[:10]})")
            continue

        # Filter for FRONT camera
        cam_df = cam_df[cam_df[img_cam_col] == camera_name]
        cam_df = cam_df.sort_values(img_ts_col)

        if len(cam_df) < 10:
            continue

        # Load vehicle poses for this segment
        pose_file = f"{base_path}/vehicle_pose/{seg_name}.parquet"
        try:
            with fs.open(pose_file, "rb") as fh:
                pose_df = pq.read_table(fh).to_pandas()
        except Exception:
            print(f"  Skipping {seg_name[:20]}... (no pose data)")
            continue

        if total_frames == 0:
            print(f"Pose columns (first segment): {list(pose_df.columns)}")

        pose_ts_col = _find_col(pose_df, "key.frame_timestamp_micros")
        pose_tf_base = "[VehiclePoseComponent].world_from_vehicle.transform"

        if not pose_ts_col:
            print(f"  Skipping {seg_name[:20]}... (missing pose ts col: {list(pose_df.columns)[:5]})")
            continue

        pose_cols = list(pose_df.columns)
        # Check that we can read transforms (single col or split)
        has_transform = (
            pose_tf_base in pose_cols
            or any(c.startswith(pose_tf_base + ".") for c in pose_cols)
        )
        if not has_transform:
            print(f"  Skipping {seg_name[:20]}... (missing pose transform: {pose_cols[:5]})")
            continue

        # Build pose lookup: timestamp -> world_from_vehicle 4x4
        pose_lookup = {}
        for _, row in pose_df.iterrows():
            ts = row[pose_ts_col]
            transform = _read_array_col(row, pose_cols, pose_tf_base)
            if transform is not None and len(transform) == 16:
                pose_lookup[ts] = transform.reshape(4, 4)

        # Process frames
        frames_dir = os.path.join(seg_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        frame_list = []
        c2ws = []
        timestamps = []

        for idx, (_, row) in enumerate(cam_df.iterrows()):
            ts = row[img_ts_col]

            if ts not in pose_lookup:
                continue

            # Decode image
            img_bytes = row[img_data_col]
            if img_bytes is None:
                continue

            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Resize to target resolution
            img = cv2.resize(img, (target_width, target_height))

            # Save frame
            frame_filename = f"{idx:06d}.jpg"
            cv2.imwrite(
                os.path.join(frames_dir, frame_filename), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            # Compute c2w = world_from_vehicle @ camera_to_vehicle
            world_from_vehicle = pose_lookup[ts]
            camera_to_vehicle = calib["extrinsic"]
            c2w = world_from_vehicle @ camera_to_vehicle

            frame_list.append({
                "index": len(frame_list),
                "filename": frame_filename,
                "timestamp": int(ts),
            })
            c2ws.append(c2w)
            timestamps.append(int(ts))

        if len(frame_list) < 10:
            continue

        c2ws = np.array(c2ws)

        # Compute driving actions
        actions = _compute_driving_actions(c2ws, timestamps)

        # Scene metadata
        velocities = [a["velocity"] for a in actions]
        duration_s = (timestamps[-1] - timestamps[0]) / 1e6
        metadata = {
            "scene_name": seg_name,
            "camera": "FRONT",
            "num_frames": len(frame_list),
            "duration_seconds": duration_s,
            "avg_speed_ms": float(np.mean(velocities)),
            "max_speed_ms": float(np.max(velocities)),
            "has_turns": any(
                a["steer_label"] in ("hard_left", "soft_left", "soft_right", "hard_right")
                for a in actions
            ),
            "behavior_category": _classify_driving_behavior(actions),
        }

        # Intrinsics in our format: (fx, fy, cx, cy)
        intr = calib["intrinsic"]
        cam_intrinsics = {
            "fx": float(intr[0]),
            "fy": float(intr[1]),
            "cx": float(intr[2]),
            "cy": float(intr[3]),
        }

        # Save poses.npy and intrinsics.npy (for inference scripts)
        np.save(os.path.join(seg_dir, "poses.npy"), c2ws.astype(np.float32))
        np.save(
            os.path.join(seg_dir, "intrinsics.npy"),
            np.tile(
                np.array([[intr[0], intr[1], intr[2], intr[3]]], dtype=np.float32),
                (len(frame_list), 1),
            ),
        )

        # Save manifest (matches prepare_nuscenes.py format)
        manifest = {
            "metadata": metadata,
            "frames": frame_list,
            "ego_poses": [
                {"translation": c2w[:3, 3].tolist(), "rotation": c2w[:3, :3].tolist()}
                for c2w in c2ws
            ],
            "timestamps": timestamps,
            "actions": actions,
            "cam_intrinsics": cam_intrinsics,
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        results.append(metadata)
        total_frames += len(frame_list)

        if len(results) % 10 == 0:
            data_volume.commit()
            print(f"  Processed {len(results)} segments, {total_frames} frames")

    # Final commit
    data_volume.commit()

    # Save dataset summary
    summary = {
        "dataset": "waymo_v2",
        "split": split,
        "camera": "FRONT",
        "num_segments": len(results),
        "total_frames": total_frames,
        "total_duration_seconds": sum(r.get("duration_seconds", 0) for r in results),
        "behavior_distribution": {},
        "segments": results,
    }
    for r in results:
        cat = r.get("behavior_category", "unknown")
        summary["behavior_distribution"][cat] = (
            summary["behavior_distribution"].get(cat, 0) + 1
        )

    summary_path = os.path.join(out_root, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    data_volume.commit()

    print(f"\n{'=' * 60}")
    print(f"Done! Processed {len(results)} segments, {total_frames} frames")
    print(f"Total duration: {summary['total_duration_seconds']:.0f}s")
    print(f"Behavior distribution: {summary['behavior_distribution']}")
    print(f"Saved to Modal volume at {out_root}")


# ---------------------------------------------------------------------------
# Build training manifest (runs after processing)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=600,
)
def build_manifest(
    split: str = "validation",
    clip_length_sec: float = 2.0,
    min_frames: int = 8,
):
    """
    Build training manifest from processed Waymo data.
    Chops segments into short clips suitable for LoRA training.
    """
    import json
    import os

    import numpy as np

    data_root = f"{DATA_DIR}/{split}"
    if not os.path.exists(data_root):
        print(f"No processed data at {data_root}. Run process_waymo first.")
        return

    summary_path = os.path.join(data_root, "dataset_summary.json")
    if not os.path.exists(summary_path):
        print("No dataset_summary.json found. Run process_waymo first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    samples = []
    for seg_info in summary["segments"]:
        seg_name = seg_info["scene_name"]
        manifest_path = os.path.join(data_root, seg_name, "manifest.json")
        if not os.path.exists(manifest_path):
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        frames = manifest["frames"]
        actions = manifest["actions"]
        timestamps = manifest["timestamps"]

        if len(frames) < min_frames:
            continue

        # Chop into clips
        fps = len(frames) / max(manifest["metadata"]["duration_seconds"], 0.1)
        frames_per_clip = max(min_frames, int(clip_length_sec * fps))

        for clip_start in range(0, len(frames) - min_frames + 1, frames_per_clip):
            clip_end = min(clip_start + frames_per_clip, len(frames))
            if clip_end - clip_start < min_frames:
                continue

            clip_frames = frames[clip_start:clip_end]
            clip_actions = actions[clip_start:clip_end]

            sample_id = f"waymo_{seg_name[:16]}_{clip_start:04d}"

            # Captions (simple static descriptions based on behavior)
            behavior = seg_info.get("behavior_category", "cruising")
            avg_speed = seg_info.get("avg_speed_ms", 5.0)
            caption_map = {
                "stationary": "Dashcam view of a parked vehicle on a street.",
                "slow_exploration": "Dashcam footage of a vehicle slowly navigating a parking lot or residential area.",
                "sharp_maneuver": "Dashcam footage of a vehicle making a sharp turn at an intersection.",
                "urban_navigation": "Dashcam footage of a vehicle driving through an urban area with turns and intersections.",
                "highway": "Dashcam footage of a vehicle driving on a highway at high speed.",
                "stop_and_go": "Dashcam footage of a vehicle in stop-and-go traffic.",
                "cruising": "Dashcam footage of a vehicle cruising on a road during daytime.",
            }
            scene_caption = caption_map.get(
                behavior, "Dashcam footage of a vehicle driving on a road."
            )

            has_turns = any(
                a["steer_label"] in ("hard_left", "soft_left", "soft_right", "hard_right")
                for a in clip_actions
            )
            if has_turns:
                narrative = f"A vehicle navigating through traffic, making turns. Speed around {avg_speed:.0f} m/s."
            else:
                narrative = f"A vehicle driving straight ahead. Speed around {avg_speed:.0f} m/s."

            samples.append({
                "sample_id": sample_id,
                "scene_name": seg_name,
                "clip_start_idx": clip_start,
                "clip_end_idx": clip_end,
                "num_frames": len(clip_frames),
                "frames": [f["filename"] for f in clip_frames],
                "frame_dir": f"{split}/{seg_name}/frames/",
                "actions": [
                    {"plucker": a["plucker"], "multihot": a["multihot"]}
                    for a in clip_actions
                ],
                "scene_static_caption": scene_caption,
                "narrative_caption": narrative,
                "metadata": {
                    "avg_speed_ms": float(np.mean([a["velocity"] for a in clip_actions])),
                    "has_turns": has_turns,
                    "behavior_category": behavior,
                },
            })

    # Save the training manifest
    manifest = {
        "dataset": "waymo_v2",
        "split": split,
        "num_samples": len(samples),
        "samples": samples,
    }

    manifest_path = os.path.join(DATA_DIR, "training_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    data_volume.commit()

    print(f"Built training manifest with {len(samples)} clips")
    print(f"Saved to {manifest_path}")

    # Print stats
    behaviors = {}
    for s in samples:
        b = s["metadata"]["behavior_category"]
        behaviors[b] = behaviors.get(b, 0) + 1
    print(f"Behavior distribution: {behaviors}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    split: str = "validation",
    max_segments: int = 50,
    build_manifest_flag: bool = False,
    clip_length: float = 2.0,
):
    if build_manifest_flag:
        print("Building training manifest ...")
        build_manifest.remote(split=split, clip_length_sec=clip_length)
    else:
        print(f"Processing Waymo {split} split ({max_segments} segments) ...")
        print("This may take 30-60 minutes depending on segment count.")
        process_waymo.remote(split=split, max_segments=max_segments)
        print("\nNow building training manifest ...")
        build_manifest.remote(split=split, clip_length_sec=clip_length)

    print("\nDone! Data is in the 'waymo-training-data' Modal volume.")
    print("Next: use this data for LoRA training.")

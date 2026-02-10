"""
compute_actions.py — Convert ego-pose sequences into LingBot-World action format.

This is the critical mapping layer. LingBot-World uses:
  action = concat([plücker_embedding, keyboard_multihot], dim=channel)

For driving, we compute:
  action = concat([plücker_embedding_from_ego_pose, driving_action_multihot], dim=channel)

Plücker coordinates represent a 3D line (the camera ray) as a 6D vector (d, m)
where d is direction and m = o × d is the moment. LingBot-World uses these to
encode camera motion between frames.

For driving: the ego-vehicle trajectory IS the camera trajectory, so the math
is identical. We just source the poses from vehicle odometry instead of a game engine.

Usage:
    python compute_actions.py --input data/nuscenes_processed/ --output data/nuscenes_actions/
    
Prerequisites:
    pip install numpy scipy
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_rotation_matrix(quat_wxyz):
    """Convert [w, x, y, z] quaternion to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    # scipy uses [x, y, z, w] format
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    return R


def compute_plucker_embedding(pose_src, pose_dst, H=720, W=1280, fx=None, fy=None):
    """
    Compute Plücker embedding representing the camera motion from pose_src to pose_dst.
    
    This follows the standard Plücker coordinate formulation used in LingBot-World's
    Plücker Encoder (Section 3.3.2, Fig. 5).
    
    Each pixel in the source view defines a ray in 3D space. The Plücker embedding
    encodes these rays relative to the destination camera frame, giving the model
    a geometric understanding of how the viewpoint changed.
    
    Args:
        pose_src: dict with 'translation' [x,y,z] and 'rotation' [w,x,y,z]
        pose_dst: dict with 'translation' [x,y,z] and 'rotation' [w,x,y,z]
        H, W: image dimensions
        fx, fy: focal lengths (if None, estimated from image size)
    
    Returns:
        plucker: np.array of shape (H, W, 6) — per-pixel Plücker coordinates
                 channels 0-2: direction d, channels 3-5: moment m = o × d
    """
    # Default focal length estimate (reasonable for dashcam ~70° FOV)
    if fx is None:
        fx = W / (2 * np.tan(np.radians(35)))  # ~70° horizontal FOV
    if fy is None:
        fy = fx  # square pixels
    
    cx, cy = W / 2, H / 2
    
    # Build camera-to-world transforms
    R_src = quaternion_to_rotation_matrix(pose_src['rotation'])
    t_src = np.array(pose_src['translation'])
    
    R_dst = quaternion_to_rotation_matrix(pose_dst['rotation'])
    t_dst = np.array(pose_dst['translation'])
    
    # Compute relative transform: src → dst
    R_rel = R_dst.T @ R_src
    t_rel = R_dst.T @ (t_src - t_dst)
    
    # Camera origin in destination frame
    o = t_rel  # shape (3,)
    
    # Generate pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # both shape (H, W)
    
    # Unproject pixels to rays in source camera frame
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)
    
    # Stack and normalize to get ray directions in source frame
    rays_src = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    rays_src = rays_src / np.linalg.norm(rays_src, axis=-1, keepdims=True)
    
    # Transform ray directions to destination frame
    d = np.einsum('ij,hwj->hwi', R_rel, rays_src)  # (H, W, 3)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    
    # Compute Plücker moment: m = o × d
    m = np.cross(o[np.newaxis, np.newaxis, :], d)  # (H, W, 3)
    
    # Concatenate direction and moment
    plucker = np.concatenate([d, m], axis=-1)  # (H, W, 6)
    
    return plucker


def compute_plucker_compact(pose_src, pose_dst):
    """
    Compute a compact (non-spatial) Plücker representation for the ego-motion.
    
    Instead of per-pixel embeddings (expensive), this computes a single 6D vector
    representing the camera ray through the image center. Useful for:
    - Quick prototyping
    - Action conditioning without full spatial Plücker maps
    - Compatibility with LingBot-World's action adapter
    
    Returns:
        plucker_compact: np.array of shape (6,) — direction (3) + moment (3)
        motion_summary: dict with interpretable motion values
    """
    R_src = quaternion_to_rotation_matrix(pose_src['rotation'])
    t_src = np.array(pose_src['translation'])
    
    R_dst = quaternion_to_rotation_matrix(pose_dst['rotation'])
    t_dst = np.array(pose_dst['translation'])
    
    # Relative transform
    R_rel = R_dst.T @ R_src
    t_rel = R_dst.T @ (t_src - t_dst)
    
    # Forward direction (z-axis of source camera in destination frame)
    d = R_rel @ np.array([0, 0, 1.0])
    d = d / (np.linalg.norm(d) + 1e-8)
    
    # Moment
    m = np.cross(t_rel, d)
    
    plucker_compact = np.concatenate([d, m])
    
    # Interpretable summary
    motion_summary = {
        'displacement': float(np.linalg.norm(t_rel)),
        'forward_displacement': float(t_rel[2]),  # along viewing direction
        'lateral_displacement': float(t_rel[0]),   # left-right
        'vertical_displacement': float(t_rel[1]),  # up-down
        'yaw_change_deg': float(np.degrees(np.arctan2(R_rel[0, 2], R_rel[2, 2]))),
        'pitch_change_deg': float(np.degrees(np.arcsin(-np.clip(R_rel[1, 2], -1, 1)))),
    }
    
    return plucker_compact, motion_summary


def process_scene(manifest_path, output_dir, use_spatial=False):
    """
    Process a single scene: compute Plücker embeddings for all frame pairs.
    
    Args:
        manifest_path: Path to scene manifest.json (from prepare_nuscenes.py)
        output_dir: Where to save action embeddings
        use_spatial: If True, compute full HxWx6 Plücker maps (expensive).
                     If False, compute compact 6D vectors (fast, good for prototyping).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    scene_name = manifest['metadata']['scene_name']
    ego_poses = manifest['ego_poses']
    timestamps = manifest['timestamps']
    actions = manifest['actions']
    
    scene_out = Path(output_dir) / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)
    
    frame_actions = []
    
    for i in range(len(ego_poses)):
        # For each frame, compute Plücker embedding relative to NEXT frame
        # (this encodes "what motion is about to happen" — the action signal)
        if i < len(ego_poses) - 1:
            pose_src = ego_poses[i]
            pose_dst = ego_poses[i + 1]
        else:
            # Last frame: use same as previous
            pose_src = ego_poses[i]
            pose_dst = ego_poses[i]
        
        if use_spatial:
            plucker = compute_plucker_embedding(pose_src, pose_dst)
            plucker_path = scene_out / f'plucker_{i:06d}.npy'
            np.save(plucker_path, plucker.astype(np.float16))
            plucker_ref = str(plucker_path)
        else:
            plucker_compact, motion_summary = compute_plucker_compact(pose_src, pose_dst)
            plucker_ref = plucker_compact.tolist()
        
        # Combine with discrete driving actions (from prepare_nuscenes.py)
        frame_action = {
            'frame_index': i,
            'timestamp': timestamps[i] if i < len(timestamps) else None,
            # Plücker embedding (continuous geometric action)
            'plucker': plucker_ref,
            # Discrete driving action (multi-hot vector)
            'multihot': actions[i]['multihot'] if i < len(actions) else None,
            # Human-readable labels
            'speed_label': actions[i]['speed_label'] if i < len(actions) else None,
            'steer_label': actions[i]['steer_label'] if i < len(actions) else None,
            'accel_label': actions[i]['accel_label'] if i < len(actions) else None,
            # Continuous values
            'velocity': actions[i]['velocity'] if i < len(actions) else 0,
            'yaw_rate': actions[i]['yaw_rate'] if i < len(actions) else 0,
        }
        
        if not use_spatial:
            frame_action['motion_summary'] = motion_summary
        
        frame_actions.append(frame_action)
    
    # Save action sequence
    output_path = scene_out / 'actions.json'
    with open(output_path, 'w') as f:
        json.dump({
            'scene_name': scene_name,
            'num_frames': len(frame_actions),
            'action_dim': {
                'plucker': 6,
                'multihot': 14,
                'total': 20,  # concat([plucker_6d, multihot_14d]) = 20D action vector
            },
            'frame_actions': frame_actions,
        }, f, indent=2)
    
    return len(frame_actions)


def main():
    parser = argparse.ArgumentParser(
        description='Compute Plücker embeddings + discrete actions for LingBot-World fine-tuning'
    )
    parser.add_argument('--input', type=str, required=True, 
                        help='Directory with scene folders from prepare_nuscenes.py')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--spatial', action='store_true',
                        help='Compute full HxWx6 Plücker maps (expensive, ~3MB/frame)')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input
    
    input_dir = Path(args.input)
    manifests = sorted(input_dir.glob('*/manifest.json'))
    
    if not manifests:
        print(f"No manifest.json files found in {input_dir}/*/")
        return
    
    print(f"Processing {len(manifests)} scenes...")
    print(f"Mode: {'spatial (HxWx6)' if args.spatial else 'compact (6D vector)'}")
    print("=" * 60)
    
    total_frames = 0
    for manifest_path in manifests:
        n = process_scene(manifest_path, args.output, use_spatial=args.spatial)
        total_frames += n
    
    print(f"\nDone! Processed {total_frames} frames across {len(manifests)} scenes")
    print(f"\nAction vector layout:")
    print(f"  [0:6]   — Plücker embedding (continuous camera/vehicle motion)")
    print(f"  [6:10]  — Speed: stopped/slow/cruise/fast")
    print(f"  [10:15] — Steering: hard_left/soft_left/straight/soft_right/hard_right")
    print(f"  [15:20] — Accel: hard_brake/soft_brake/coast/soft_accel/hard_accel")
    print(f"\nThis 20D vector replaces LingBot's [Plücker_6D + WASD_multihot] action input.")


if __name__ == '__main__':
    main()

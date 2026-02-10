"""
prepare_nuscenes.py — Extract ego-pose, video frames, and driving actions from nuScenes.

This mirrors LingBot's game data acquisition (Section 2.1.2) but for driving:
- Game RGB frames        → nuScenes camera images (6 cameras, we use FRONT)
- Game user controls     → Ego-vehicle steering/acceleration derived from pose changes
- Game camera parameters → Ego-vehicle pose (translation + rotation) from IMU/GPS

Usage:
    python prepare_nuscenes.py --dataroot /path/to/nuscenes --split mini --outdir data/nuscenes_processed
    
Prerequisites:
    pip install nuscenes-devkit opencv-python numpy scipy
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def rotation_matrix_to_yaw(R):
    """Extract yaw (heading) angle from rotation matrix."""
    return np.arctan2(R[1, 0], R[0, 0])


def compute_driving_actions(ego_poses, timestamps, bins=None):
    """
    Derive discrete driving actions from ego-pose sequences.
    
    This is analogous to LingBot's keyboard labeling — but instead of WASD,
    we discretize real vehicle dynamics into action categories.
    
    Args:
        ego_poses: List of dicts with 'translation' [x,y,z] and 'rotation' [w,x,y,z]
        timestamps: List of timestamps in microseconds
        bins: Optional dict with custom bin edges for discretization
    
    Returns:
        List of action dicts with both continuous and discrete representations
    """
    if bins is None:
        bins = {
            # Speed bins (m/s): stop, slow, cruise, fast
            'speed': [0.0, 0.5, 5.0, 15.0, float('inf')],
            'speed_labels': ['stopped', 'slow', 'cruise', 'fast'],
            # Yaw rate bins (rad/s): hard_left, soft_left, straight, soft_right, hard_right
            'yaw_rate': [-float('inf'), -0.15, -0.02, 0.02, 0.15, float('inf')],
            'yaw_rate_labels': ['hard_left', 'soft_left', 'straight', 'soft_right', 'hard_right'],
            # Acceleration bins (m/s²): hard_brake, soft_brake, coast, soft_accel, hard_accel
            'acceleration': [-float('inf'), -2.0, -0.5, 0.5, 2.0, float('inf')],
            'accel_labels': ['hard_brake', 'soft_brake', 'coast', 'soft_accel', 'hard_accel'],
        }
    
    actions = []
    
    for i in range(len(ego_poses)):
        pos_i = np.array(ego_poses[i]['translation'])
        R_i = quaternion_to_rotation_matrix(ego_poses[i]['rotation'])
        yaw_i = rotation_matrix_to_yaw(R_i)
        
        if i == 0:
            # First frame: estimate from next frame
            if len(ego_poses) > 1:
                pos_next = np.array(ego_poses[1]['translation'])
                dt = (timestamps[1] - timestamps[0]) / 1e6  # microseconds to seconds
                if dt > 0:
                    velocity = np.linalg.norm(pos_next[:2] - pos_i[:2]) / dt
                else:
                    velocity = 0.0
            else:
                velocity = 0.0
            yaw_rate = 0.0
            acceleration = 0.0
        else:
            pos_prev = np.array(ego_poses[i-1]['translation'])
            R_prev = quaternion_to_rotation_matrix(ego_poses[i-1]['rotation'])
            yaw_prev = rotation_matrix_to_yaw(R_prev)
            
            dt = (timestamps[i] - timestamps[i-1]) / 1e6
            if dt <= 0:
                dt = 0.5  # fallback: nuScenes is ~2Hz
            
            # Speed (magnitude of horizontal velocity)
            displacement = pos_i[:2] - pos_prev[:2]
            velocity = np.linalg.norm(displacement) / dt
            
            # Yaw rate (heading change per second)
            dyaw = yaw_i - yaw_prev
            # Normalize to [-pi, pi]
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = dyaw / dt
            
            # Acceleration (velocity change)
            if i >= 2:
                pos_pp = np.array(ego_poses[i-2]['translation'])
                dt_prev = (timestamps[i-1] - timestamps[i-2]) / 1e6
                if dt_prev > 0:
                    vel_prev = np.linalg.norm(pos_prev[:2] - pos_pp[:2]) / dt_prev
                    acceleration = (velocity - vel_prev) / dt
                else:
                    acceleration = 0.0
            else:
                acceleration = 0.0
        
        # Discretize using bins
        speed_idx = np.digitize(velocity, bins['speed']) - 1
        speed_idx = np.clip(speed_idx, 0, len(bins['speed_labels']) - 1)
        
        yaw_idx = np.digitize(yaw_rate, bins['yaw_rate']) - 1
        yaw_idx = np.clip(yaw_idx, 0, len(bins['yaw_rate_labels']) - 1)
        
        accel_idx = np.digitize(acceleration, bins['acceleration']) - 1
        accel_idx = np.clip(accel_idx, 0, len(bins['accel_labels']) - 1)
        
        # Multi-hot encoding (mirrors LingBot's WASD multi-hot)
        # Layout: [stopped, slow, cruise, fast, hard_left, soft_left, straight,
        #          soft_right, hard_right, hard_brake, soft_brake, coast, soft_accel, hard_accel]
        multihot = np.zeros(14, dtype=np.float32)
        multihot[speed_idx] = 1.0           # speed category (indices 0-3)
        multihot[4 + yaw_idx] = 1.0         # steering category (indices 4-8)
        multihot[9 + accel_idx] = 1.0       # acceleration category (indices 9-13)
        
        actions.append({
            # Continuous values (for Plücker embedding computation)
            'velocity': float(velocity),
            'yaw_rate': float(yaw_rate),
            'acceleration': float(acceleration),
            'heading': float(yaw_i),
            # Discrete labels
            'speed_label': bins['speed_labels'][speed_idx],
            'steer_label': bins['yaw_rate_labels'][yaw_idx],
            'accel_label': bins['accel_labels'][accel_idx],
            # Multi-hot vector (analogous to LingBot's keyboard multi-hot)
            'multihot': multihot.tolist(),
            # Raw pose for Plücker computation
            'translation': ego_poses[i]['translation'],
            'rotation': ego_poses[i]['rotation'],
        })
    
    return actions


def extract_scene(nusc, scene, camera='CAM_FRONT', target_fps=8, outdir='data'):
    """
    Extract frames, ego-poses, and actions from a single nuScenes scene.
    
    Mirrors LingBot's game data extraction: synchronized video + actions + camera params.
    """
    scene_name = scene['name']
    scene_dir = Path(outdir) / scene_name
    frames_dir = scene_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Walk through all samples in the scene
    sample_token = scene['first_sample_token']
    
    frames = []
    ego_poses = []
    timestamps = []
    cam_intrinsics = []
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        # Get camera data
        cam_data = nusc.get('sample_data', sample['data'][camera])
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        
        # Get ego pose
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # Get camera calibration
        calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        # Copy frame
        frame_idx = len(frames)
        frame_filename = f'{frame_idx:06d}.jpg'
        frame_dst = frames_dir / frame_filename
        
        if os.path.exists(img_path):
            # Optionally resize to 720p for consistency with LingBot-World
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                if h != 720:
                    scale = 720 / h
                    new_w = int(w * scale)
                    img = cv2.resize(img, (new_w, 720))
                cv2.imwrite(str(frame_dst), img)
        
        frames.append({
            'index': frame_idx,
            'filename': frame_filename,
            'timestamp': cam_data['timestamp'],
            'original_path': cam_data['filename'],
        })
        
        ego_poses.append({
            'translation': ego_pose['translation'],
            'rotation': ego_pose['rotation'],
            'timestamp': ego_pose['timestamp'],
        })
        
        timestamps.append(cam_data['timestamp'])
        
        cam_intrinsics.append({
            'intrinsic': calib['camera_intrinsic'],
            'translation': calib['translation'],  # camera-to-ego transform
            'rotation': calib['rotation'],
        })
        
        sample_token = sample['next'] if sample['next'] else None
    
    if len(frames) < 2:
        print(f"  Skipping {scene_name}: only {len(frames)} frames")
        return None
    
    # Compute driving actions from ego-pose sequence
    actions = compute_driving_actions(
        [ep for ep in ego_poses],
        timestamps
    )
    
    # Compute scene-level statistics
    velocities = [a['velocity'] for a in actions]
    yaw_rates = [a['yaw_rate'] for a in actions]
    
    scene_metadata = {
        'scene_name': scene_name,
        'camera': camera,
        'num_frames': len(frames),
        'duration_seconds': (timestamps[-1] - timestamps[0]) / 1e6,
        'avg_speed_ms': float(np.mean(velocities)),
        'max_speed_ms': float(np.max(velocities)),
        'avg_abs_yaw_rate': float(np.mean(np.abs(yaw_rates))),
        'has_turns': bool(np.any(np.abs(yaw_rates) > 0.1)),
        # Driving behavior category (mirrors LingBot's navigation/sightseeing/long-tail)
        'behavior_category': classify_driving_behavior(actions),
    }
    
    # Save everything
    manifest = {
        'metadata': scene_metadata,
        'frames': frames,
        'ego_poses': [
            {'translation': ep['translation'], 'rotation': ep['rotation']}
            for ep in ego_poses
        ],
        'timestamps': timestamps,
        'actions': actions,
        'cam_intrinsics': cam_intrinsics[0] if cam_intrinsics else None,
    }
    
    manifest_path = scene_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  {scene_name}: {len(frames)} frames, "
          f"{scene_metadata['duration_seconds']:.1f}s, "
          f"avg {scene_metadata['avg_speed_ms']:.1f} m/s, "
          f"behavior: {scene_metadata['behavior_category']}")
    
    return manifest


def classify_driving_behavior(actions):
    """
    Classify the driving behavior in a scene.
    
    Maps to LingBot's categories:
    - Navigation (free/loop/transition) → cruising, highway, urban_navigation
    - Sightseeing → slow_exploration (parking lots, residential)
    - Long-tail → stationary, reversing, emergency_stop
    - World interaction → intersection, lane_change, u_turn
    """
    velocities = [a['velocity'] for a in actions]
    yaw_rates = [abs(a['yaw_rate']) for a in actions]
    accels = [a['acceleration'] for a in actions]
    
    avg_speed = np.mean(velocities)
    max_yaw = np.max(yaw_rates) if yaw_rates else 0
    stopped_frac = np.mean([v < 0.5 for v in velocities])
    
    if avg_speed < 0.5:
        return 'stationary'
    elif avg_speed < 3.0:
        return 'slow_exploration'
    elif max_yaw > 0.3:
        return 'sharp_maneuver'  # u-turn, tight intersection
    elif max_yaw > 0.1:
        return 'urban_navigation'  # turns at intersections
    elif avg_speed > 15.0:
        return 'highway'
    elif stopped_frac > 0.3:
        return 'stop_and_go'  # traffic
    else:
        return 'cruising'


def main():
    parser = argparse.ArgumentParser(description='Prepare nuScenes data for LingBot-World driving fine-tune')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to nuScenes dataset root')
    parser.add_argument('--version', type=str, default='v1.0-mini', 
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                        help='nuScenes version to use')
    parser.add_argument('--camera', type=str, default='CAM_FRONT',
                        choices=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                        help='Which camera to extract')
    parser.add_argument('--outdir', type=str, default='data/nuscenes_processed')
    args = parser.parse_args()
    
    # Import here so the script can show help without nuscenes installed
    from nuscenes.nuscenes import NuScenes
    
    print(f"Loading nuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    print(f"\nProcessing {len(nusc.scene)} scenes, camera: {args.camera}")
    print("=" * 60)
    
    results = []
    for scene in nusc.scene:
        result = extract_scene(nusc, scene, camera=args.camera, outdir=args.outdir)
        if result:
            results.append(result['metadata'])
    
    # Save dataset summary
    summary_path = Path(args.outdir) / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'version': args.version,
            'camera': args.camera,
            'num_scenes': len(results),
            'total_frames': sum(r['num_frames'] for r in results),
            'total_duration_seconds': sum(r['duration_seconds'] for r in results),
            'behavior_distribution': {
                cat: sum(1 for r in results if r['behavior_category'] == cat)
                for cat in set(r['behavior_category'] for r in results)
            },
            'scenes': results,
        }, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Done! Processed {len(results)} scenes")
    print(f"Total frames: {sum(r['num_frames'] for r in results)}")
    print(f"Total duration: {sum(r['duration_seconds'] for r in results):.0f}s")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()

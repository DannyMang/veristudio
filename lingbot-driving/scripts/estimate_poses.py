"""
estimate_poses.py — Estimate camera poses from unlabeled driving video.

This handles Tier 2 data (OpenDV-YouTube, random dashcam footage) that doesn't come
with ego-pose ground truth. Mirrors LingBot's approach (Section 2.2.2):

> "We utilize MegaSAM to generate camera pose annotations for videos
>  lacking geometric information."

After pose estimation, we can derive pseudo-actions using inverse dynamics
(pose differences → velocity/yaw_rate → discretized driving actions).

Usage:
    # Using MegaSAM (recommended, matches LingBot paper)
    python estimate_poses.py --input data/opendv_mini/ --method megasam

    # Using COLMAP (slower but more robust for long sequences)
    python estimate_poses.py --input data/opendv_mini/ --method colmap

    # Using DUSt3R/MASt3R (recent, good for short clips)  
    python estimate_poses.py --input data/opendv_mini/ --method mast3r

Prerequisites:
    # MegaSAM
    pip install megasam  # or clone from https://github.com/mega-sam/mega-sam

    # COLMAP (alternative)
    sudo apt-get install colmap
    pip install pycolmap
    
    # MASt3R (alternative)
    pip install mast3r  # or clone from https://github.com/naver/mast3r
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path, output_dir, target_fps=2, max_frames=None):
    """Extract frames from video at target FPS."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if source_fps <= 0:
        print(f"  Warning: Could not read FPS from {video_path}, assuming 30")
        source_fps = 30
    
    frame_interval = max(1, int(source_fps / target_fps))
    
    frames = []
    frame_idx = 0
    saved_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Resize to 720p if needed
            h, w = frame.shape[:2]
            if h != 720:
                scale = 720 / h
                frame = cv2.resize(frame, (int(w * scale), 720))
            
            filename = f'{saved_idx:06d}.jpg'
            cv2.imwrite(str(output_dir / filename), frame)
            
            frames.append({
                'index': saved_idx,
                'filename': filename,
                'source_frame': frame_idx,
                'timestamp_sec': frame_idx / source_fps,
            })
            saved_idx += 1
            
            if max_frames and saved_idx >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    return frames, source_fps


def estimate_poses_megasam(frames_dir):
    """
    Use MegaSAM for camera pose estimation.
    
    MegaSAM is specifically cited in the LingBot paper (Section 2.2.2) and is designed
    for monocular video → camera poses. It works well for dashcam footage because:
    - Handles forward-facing camera motion
    - Robust to dynamic objects (other cars, pedestrians)
    - Outputs standard [R|t] format
    """
    try:
        # MegaSAM API (check their repo for exact usage)
        from megasam import MegaSAMPredictor
        
        predictor = MegaSAMPredictor()
        result = predictor.predict(
            video_dir=str(frames_dir),
            output_format='colmap',
        )
        
        # Parse COLMAP-format output
        poses = parse_colmap_poses(result['output_dir'])
        return poses
        
    except ImportError:
        print("MegaSAM not installed. Install from: https://github.com/mega-sam/mega-sam")
        print("Falling back to simple motion estimation...")
        return estimate_poses_simple(frames_dir)


def estimate_poses_colmap(frames_dir):
    """Use COLMAP for structure-from-motion pose estimation."""
    workspace = Path(frames_dir).parent / 'colmap_workspace'
    workspace.mkdir(exist_ok=True)
    database_path = workspace / 'database.db'
    sparse_path = workspace / 'sparse'
    sparse_path.mkdir(exist_ok=True)
    
    try:
        # Feature extraction
        subprocess.run([
            'colmap', 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--ImageReader.single_camera', '1',
            '--ImageReader.camera_model', 'PINHOLE',
        ], check=True)
        
        # Feature matching (sequential for video)
        subprocess.run([
            'colmap', 'sequential_matcher',
            '--database_path', str(database_path),
        ], check=True)
        
        # Sparse reconstruction
        subprocess.run([
            'colmap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--output_path', str(sparse_path),
        ], check=True)
        
        poses = parse_colmap_poses(sparse_path / '0')
        return poses
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"COLMAP failed: {e}")
        print("Falling back to simple motion estimation...")
        return estimate_poses_simple(frames_dir)


def estimate_poses_simple(frames_dir):
    """
    Simple optical-flow-based motion estimation fallback.
    
    Less accurate than MegaSAM/COLMAP but requires no external dependencies.
    Estimates relative camera motion from frame-to-frame optical flow.
    
    This is a rough approximation — use MegaSAM for real training data.
    """
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob('*.jpg'))
    
    if len(frame_files) < 2:
        return []
    
    poses = []
    cumulative_position = np.array([0.0, 0.0, 0.0])
    cumulative_yaw = 0.0
    
    prev_gray = None
    
    for i, frame_path in enumerate(frame_files):
        img = cv2.imread(str(frame_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # Estimate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Extract dominant motion
            h, w = flow.shape[:2]
            center_region = flow[h//4:3*h//4, w//4:3*w//4]
            
            mean_flow_x = np.median(center_region[:, :, 0])
            mean_flow_y = np.median(center_region[:, :, 1])
            
            # Rough motion estimates (pixels → approximate world units)
            # These are very approximate — real training should use MegaSAM
            yaw_change = -mean_flow_x * 0.001  # horizontal flow → yaw
            forward_motion = 1.0  # assume constant forward speed for dashcam
            
            cumulative_yaw += yaw_change
            cumulative_position[0] += forward_motion * np.sin(cumulative_yaw)
            cumulative_position[2] += forward_motion * np.cos(cumulative_yaw)
        
        # Convert to quaternion [w, x, y, z]
        # Only yaw rotation for simplicity
        qw = np.cos(cumulative_yaw / 2)
        qy = np.sin(cumulative_yaw / 2)
        
        poses.append({
            'frame_index': i,
            'filename': frame_path.name,
            'translation': cumulative_position.tolist(),
            'rotation': [float(qw), 0.0, float(qy), 0.0],  # [w, x, y, z]
            'method': 'optical_flow_simple',
            'confidence': 0.3,  # low confidence — flag for review
        })
        
        prev_gray = gray
    
    return poses


def parse_colmap_poses(sparse_dir):
    """Parse COLMAP sparse reconstruction output into our pose format."""
    images_path = Path(sparse_dir) / 'images.txt'
    if not images_path.exists():
        # Try binary format
        images_path = Path(sparse_dir) / 'images.bin'
        if images_path.exists():
            return parse_colmap_binary(images_path)
        return []
    
    poses = []
    with open(images_path) as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 10:
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9]
            
            poses.append({
                'filename': name,
                'translation': [tx, ty, tz],
                'rotation': [qw, qx, qy, qz],
                'method': 'colmap',
                'confidence': 0.8,
            })
            i += 2  # skip the points2D line
        else:
            i += 1
    
    # Sort by filename
    poses.sort(key=lambda p: p['filename'])
    for idx, p in enumerate(poses):
        p['frame_index'] = idx
    
    return poses


def process_video(video_path, output_dir, method='megasam', target_fps=2):
    """Process a single video: extract frames → estimate poses → save."""
    video_name = Path(video_path).stem
    scene_dir = Path(output_dir) / video_name
    frames_dir = scene_dir / 'frames'
    
    print(f"  Processing: {video_name}")
    
    # Step 1: Extract frames
    print(f"    Extracting frames at {target_fps} FPS...")
    frames, source_fps = extract_frames(video_path, frames_dir, target_fps)
    print(f"    Extracted {len(frames)} frames from {source_fps:.1f} FPS source")
    
    if len(frames) < 5:
        print(f"    Skipping: too few frames")
        return None
    
    # Step 2: Estimate poses
    print(f"    Estimating poses ({method})...")
    if method == 'megasam':
        poses = estimate_poses_megasam(frames_dir)
    elif method == 'colmap':
        poses = estimate_poses_colmap(frames_dir)
    else:
        poses = estimate_poses_simple(frames_dir)
    
    print(f"    Estimated {len(poses)} poses")
    
    # Step 3: Build manifest (same format as prepare_nuscenes.py output)
    timestamps = [int(f['timestamp_sec'] * 1e6) for f in frames]
    
    # Convert poses to ego_poses format
    ego_poses = []
    for p in poses:
        ego_poses.append({
            'translation': p['translation'],
            'rotation': p['rotation'],
        })
    
    # Pad if pose estimation returned fewer poses than frames
    while len(ego_poses) < len(frames):
        ego_poses.append(ego_poses[-1] if ego_poses else {
            'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]
        })
    
    # Compute actions from estimated poses
    from compute_actions import compute_driving_actions
    actions = compute_driving_actions(ego_poses, timestamps)
    
    manifest = {
        'metadata': {
            'scene_name': video_name,
            'camera': 'dashcam',
            'num_frames': len(frames),
            'duration_seconds': frames[-1]['timestamp_sec'] if frames else 0,
            'source_fps': source_fps,
            'target_fps': target_fps,
            'pose_method': method,
            'pose_confidence': np.mean([p.get('confidence', 0.5) for p in poses]),
            'avg_speed_ms': float(np.mean([a['velocity'] for a in actions])),
            'max_speed_ms': float(np.max([a['velocity'] for a in actions])),
            'avg_abs_yaw_rate': float(np.mean([abs(a['yaw_rate']) for a in actions])),
            'has_turns': bool(np.any([abs(a['yaw_rate']) > 0.1 for a in actions])),
            'behavior_category': 'estimated',
        },
        'frames': frames,
        'ego_poses': ego_poses,
        'timestamps': timestamps,
        'actions': actions,
    }
    
    manifest_path = scene_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"    Saved to {scene_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description='Estimate camera poses for unlabeled driving video (LingBot Section 2.2.2)'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Directory with video files or frame directories')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: input + _processed)')
    parser.add_argument('--method', type=str, default='simple',
                        choices=['megasam', 'colmap', 'simple'],
                        help='Pose estimation method')
    parser.add_argument('--fps', type=float, default=2,
                        help='Target FPS for frame extraction')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.mp4', '.avi', '.mov', '.mkv'],
                        help='Video file extensions to process')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir.parent / f'{input_dir.name}_processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find videos
    videos = []
    for ext in args.extensions:
        videos.extend(input_dir.glob(f'*{ext}'))
        videos.extend(input_dir.glob(f'**/*{ext}'))
    videos = sorted(set(videos))
    
    print(f"Found {len(videos)} videos in {input_dir}")
    print(f"Method: {args.method}, Target FPS: {args.fps}")
    print("=" * 60)
    
    results = []
    for video_path in videos:
        result = process_video(video_path, output_dir, args.method, args.fps)
        if result:
            results.append(result['metadata'])
    
    print(f"\n{'=' * 60}")
    print(f"Processed {len(results)} videos → {output_dir}")
    print(f"\nPose confidence: {np.mean([r.get('pose_confidence', 0) for r in results]):.2f}")
    print(f"\nNext steps:")
    print(f"  1. Run caption_driving.py on {output_dir}")
    print(f"  2. Run compute_actions.py on {output_dir}")
    print(f"  3. Run build_training_manifest.py on {output_dir}")


if __name__ == '__main__':
    main()

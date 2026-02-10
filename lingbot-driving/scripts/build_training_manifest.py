"""
build_training_manifest.py — Combine all processed data into LingBot-World training format.

This produces the final data structure that mirrors what LingBot-World expects:
each training sample = video clip + action sequence + hierarchical captions + camera poses

The output format is designed to be directly compatible with LingBot-World's
dataloader, requiring only minor config changes to point at driving data.

Usage:
    python build_training_manifest.py \
        --input data/nuscenes_processed/ \
        --output data/training_ready/ \
        --clip_length 5.0 \
        --min_frames 10
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


def create_training_sample(
    scene_name: str,
    frames: List[Dict],
    ego_poses: List[Dict],
    actions: List[Dict],
    captions: Dict,
    timestamps: List[int],
    clip_start: int,
    clip_end: int,
    sample_id: str,
) -> Dict:
    """
    Create a single training sample in LingBot-World compatible format.
    
    LingBot-World training sample structure (from paper Section 3.3):
    {
        "video_path": path to video or frame directory,
        "text": caption for conditioning,
        "actions": per-frame action signals,
        "camera_poses": per-frame camera parameters,
        ...
    }
    
    We adapt this for driving by populating each field from our processed data.
    """
    clip_frames = frames[clip_start:clip_end]
    clip_poses = ego_poses[clip_start:clip_end]
    clip_actions = actions[clip_start:clip_end] if actions else None
    clip_timestamps = timestamps[clip_start:clip_end]
    
    if len(clip_frames) < 2:
        return None
    
    duration = (clip_timestamps[-1] - clip_timestamps[0]) / 1e6
    
    # Build per-frame action sequence
    # This is what gets fed into the Plücker Encoder → AdaLN injection
    frame_actions = []
    for i, action in enumerate(clip_actions or []):
        frame_actions.append({
            # Plücker embedding (6D) — continuous camera/vehicle motion
            'plucker': action.get('plucker', [0]*6),
            # Discrete driving action (14D multi-hot) — replaces WASD
            'multihot': action.get('multihot', [0]*14),
            # Combined 20D action vector (for direct injection)
            'action_vector': (
                (action.get('plucker', [0]*6) if isinstance(action.get('plucker'), list) else [0]*6)
                + action.get('multihot', [0]*14)
            ),
        })
    
    # Select caption based on training task
    # LingBot uses different captions for different training objectives:
    # - Narrative caption: for general world simulation
    # - Scene-static caption: for decoupling motion from scene (important!)
    # - Dense temporal: for temporal alignment training
    
    sample = {
        'sample_id': sample_id,
        'scene_name': scene_name,
        'clip_start_idx': clip_start,
        'clip_end_idx': clip_end,
        'num_frames': len(clip_frames),
        'duration_seconds': duration,
        'fps': len(clip_frames) / max(duration, 0.1),
        
        # Video frames
        'frames': [f['filename'] for f in clip_frames],
        'frame_dir': f'{scene_name}/frames/',
        
        # Captions (hierarchical, following LingBot Section 2.3)
        'narrative_caption': captions.get('narrative_caption', ''),
        'scene_static_caption': captions.get('scene_static_caption', ''),
        'dense_temporal_captions': _clip_temporal_captions(
            captions.get('dense_temporal_captions', []),
            (clip_timestamps[0] - timestamps[0]) / 1e6,
            (clip_timestamps[-1] - timestamps[0]) / 1e6,
        ),
        
        # Action signals (per-frame)
        'actions': frame_actions,
        
        # Camera/ego poses (for Plücker computation if not pre-computed)
        'ego_poses': [
            {'translation': p['translation'], 'rotation': p['rotation']}
            for p in clip_poses
        ],
        
        # Metadata for curriculum training (LingBot Section 3.3.1)
        'metadata': {
            'avg_speed_ms': float(np.mean([
                a.get('velocity', 0) for a in (clip_actions or [])
            ])) if clip_actions else 0,
            'has_turns': any(
                a.get('steer_label', '') in ('soft_left', 'hard_left', 'soft_right', 'hard_right')
                for a in (clip_actions or [])
            ),
            'behavior_category': _classify_clip(clip_actions),
        },
    }
    
    return sample


def _clip_temporal_captions(temporal_captions, clip_start_sec, clip_end_sec):
    """Filter and re-time temporal captions to clip boundaries."""
    clipped = []
    for tc in temporal_captions:
        if not isinstance(tc, dict) or 'start_time' not in tc:
            continue
        if tc['end_time'] < clip_start_sec or tc['start_time'] > clip_end_sec:
            continue
        clipped.append({
            'start_time': round(max(0, tc['start_time'] - clip_start_sec), 1),
            'end_time': round(min(clip_end_sec - clip_start_sec, tc['end_time'] - clip_start_sec), 1),
            'Event': tc.get('Event', ''),
            'caption': tc.get('caption', ''),
        })
    return clipped


def _classify_clip(actions):
    if not actions:
        return 'unknown'
    steer_labels = [a.get('steer_label', 'straight') for a in actions]
    speed_labels = [a.get('speed_label', 'cruise') for a in actions]
    
    has_turn = any(s in ('soft_left', 'hard_left', 'soft_right', 'hard_right') for s in steer_labels)
    mostly_stopped = sum(1 for s in speed_labels if s == 'stopped') > len(speed_labels) * 0.5
    
    if mostly_stopped:
        return 'stationary'
    elif has_turn:
        return 'turning'
    else:
        return 'straight'


def segment_scene(manifest, captions, clip_length_sec=5.0, overlap_sec=1.0, min_frames=10):
    """
    Segment a scene into training clips.
    
    LingBot's progressive curriculum (Section 3.3.1) trains on:
    - 5-second clips initially
    - Then extends to 10s, 30s, 60s
    
    We pre-segment at the base clip length. Longer clips can be
    created by concatenating overlapping segments during training.
    """
    timestamps = manifest['timestamps']
    if len(timestamps) < min_frames:
        return []
    
    t0 = timestamps[0]
    total_duration = (timestamps[-1] - t0) / 1e6
    
    clips = []
    clip_start_sec = 0.0
    
    while clip_start_sec < total_duration - 1.0:
        clip_end_sec = min(clip_start_sec + clip_length_sec, total_duration)
        
        # Find frame indices for this time window
        start_idx = None
        end_idx = None
        for i, t in enumerate(timestamps):
            t_sec = (t - t0) / 1e6
            if start_idx is None and t_sec >= clip_start_sec:
                start_idx = i
            if t_sec <= clip_end_sec:
                end_idx = i + 1
        
        if start_idx is not None and end_idx is not None and (end_idx - start_idx) >= min_frames:
            clips.append((start_idx, end_idx))
        
        clip_start_sec += clip_length_sec - overlap_sec
    
    return clips


def main():
    parser = argparse.ArgumentParser(
        description='Build LingBot-World compatible training manifest for driving'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Directory with processed scenes')
    parser.add_argument('--output', type=str, default='data/training_ready',
                        help='Output directory for training manifest')
    parser.add_argument('--clip_length', type=float, default=5.0,
                        help='Clip length in seconds (start with 5, increase for curriculum)')
    parser.add_argument('--overlap', type=float, default=1.0,
                        help='Overlap between clips in seconds')
    parser.add_argument('--min_frames', type=int, default=10,
                        help='Minimum frames per clip')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / 'manifest.json').exists()
    ])
    
    print(f"Building training manifest from {len(scene_dirs)} scenes")
    print(f"Clip length: {args.clip_length}s, overlap: {args.overlap}s")
    print("=" * 60)
    
    all_samples = []
    stats = {'total_clips': 0, 'total_frames': 0, 'behaviors': {}}
    
    for scene_dir in scene_dirs:
        with open(scene_dir / 'manifest.json') as f:
            manifest = json.load(f)
        
        # Load captions if available
        caption_path = scene_dir / 'captions.json'
        if caption_path.exists():
            with open(caption_path) as f:
                captions = json.load(f)
        else:
            captions = {'narrative_caption': '', 'scene_static_caption': '', 'dense_temporal_captions': []}
        
        # Load action data if separately computed
        action_path = scene_dir / 'actions.json'
        if action_path.exists():
            with open(action_path) as f:
                action_data = json.load(f)
            actions = action_data.get('frame_actions', manifest.get('actions', []))
        else:
            actions = manifest.get('actions', [])
        
        # Segment into clips
        clips = segment_scene(manifest, captions, args.clip_length, args.overlap, args.min_frames)
        
        scene_name = manifest['metadata']['scene_name']
        
        for clip_idx, (start, end) in enumerate(clips):
            sample_id = f"{scene_name}_clip{clip_idx:04d}"
            
            sample = create_training_sample(
                scene_name=scene_name,
                frames=manifest['frames'],
                ego_poses=manifest['ego_poses'],
                actions=actions,
                captions=captions,
                timestamps=manifest['timestamps'],
                clip_start=start,
                clip_end=end,
                sample_id=sample_id,
            )
            
            if sample:
                all_samples.append(sample)
                stats['total_clips'] += 1
                stats['total_frames'] += sample['num_frames']
                beh = sample['metadata']['behavior_category']
                stats['behaviors'][beh] = stats['behaviors'].get(beh, 0) + 1
    
    # Save training manifest
    manifest_path = output_dir / 'training_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({
            'dataset': 'lingbot-driving',
            'num_samples': len(all_samples),
            'clip_length_sec': args.clip_length,
            'action_dim': {
                'plucker': 6,
                'multihot': 14,
                'total': 20,
            },
            'caption_levels': ['narrative', 'scene_static', 'dense_temporal'],
            'samples': all_samples,
        }, f, indent=2)
    
    # Also save a lightweight index for the dataloader
    index_path = output_dir / 'train_index.jsonl'
    with open(index_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps({
                'sample_id': sample['sample_id'],
                'frame_dir': sample['frame_dir'],
                'num_frames': sample['num_frames'],
                'duration': sample['duration_seconds'],
                'narrative': sample['narrative_caption'][:200],
                'scene_static': sample['scene_static_caption'][:200],
                'behavior': sample['metadata']['behavior_category'],
            }) + '\n')
    
    print(f"\n{'=' * 60}")
    print(f"Training manifest: {manifest_path}")
    print(f"Training index:    {index_path}")
    print(f"\nDataset stats:")
    print(f"  Total clips:  {stats['total_clips']}")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Behaviors:    {json.dumps(stats['behaviors'], indent=4)}")
    print(f"\nAction vector: 20D = [Plücker_6D | driving_multihot_14D]")
    print(f"  → Replaces LingBot's [Plücker_6D | WASD_keyboard_multihot]")
    print(f"\nNext step: use this manifest with LingBot-World's training script,")
    print(f"pointing the action adapter at this data while keeping the backbone frozen.")


if __name__ == '__main__':
    main()

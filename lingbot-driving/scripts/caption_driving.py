"""
caption_driving.py — Generate hierarchical captions for driving scenes.

LingBot-World uses 3 caption levels (Section 2.3):
  1. Narrative caption     — full story including camera movement + environment
  2. Scene-static caption  — environment only, NO motion/camera descriptions
  3. Dense temporal caption — time-aligned event descriptions

We replicate this exactly for driving, translating their game/exploration captions
into driving-domain equivalents.

This script supports two modes:
  - VLM mode: Uses Qwen3-VL or similar to auto-generate captions from video frames
  - Template mode: Generates structured captions from metadata (actions, GPS, etc.)

Usage:
    # VLM mode (requires GPU + model)
    python caption_driving.py --input data/nuscenes_processed/ --mode vlm --vlm qwen3-vl-7b

    # Template mode (no GPU needed, uses extracted metadata)
    python caption_driving.py --input data/nuscenes_processed/ --mode template
    
Prerequisites:
    pip install transformers torch  # for VLM mode
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional


# ============================================================================
# PROMPT TEMPLATES — Driving-domain versions of LingBot's captioning prompts
# ============================================================================

NARRATIVE_CAPTION_PROMPT = """You are a driving scene narrator. Describe this dashcam video sequence in detail, 
interweaving the driving environment with the vehicle's trajectory and temporal evolution.

Include:
- The road type and layout (highway, urban street, intersection, etc.)
- Weather, lighting, and time of day
- The ego-vehicle's motion: accelerating, braking, turning, lane changes
- Key objects: other vehicles, pedestrians, traffic signals, signs
- How the scene evolves over time

Write as a single flowing paragraph, similar to a first-person exploration narrative.

Example style: "The video captures a first-person driving perspective along a four-lane 
urban boulevard at dusk. The vehicle approaches a signalized intersection at moderate speed, 
with oncoming traffic visible in the left lanes. As the light turns green, the car accelerates 
gently and executes a smooth left turn onto a quieter residential street lined with parked 
vehicles. A pedestrian crosses ahead, prompting a brief deceleration before the car continues 
past a row of storefronts with warm interior lighting..."

Describe the following driving sequence:"""

SCENE_STATIC_CAPTION_PROMPT = """Describe ONLY the static environment visible in this dashcam footage. 
Do NOT describe any vehicle motion, camera movement, or dynamic actions.

Focus exclusively on:
- Road infrastructure: lane markings, curbs, medians, road surface
- Fixed objects: buildings, trees, signs, traffic lights, fences, bridges
- Weather and lighting conditions
- Scene type: urban, suburban, highway, rural, parking lot
- Any distinctive environmental features

Do NOT mention: turning, accelerating, braking, moving, approaching, driving, 
or any verbs implying motion.

Example style: "A four-lane urban road with faded white lane markings and a raised concrete 
median. Both sides feature two-story commercial buildings with illuminated storefronts. 
Several mature trees line the sidewalks. Traffic signals are mounted on overhead arms at the 
intersection ahead. The sky is overcast with diffused late-afternoon light casting soft shadows."

Describe the static environment:"""

DENSE_TEMPORAL_CAPTION_PROMPT = """Analyze this dashcam video and provide time-aligned descriptions 
of driving events. Segment the video into intervals and describe what happens in each.

For each interval, provide:
- start_time and end_time (in seconds)
- Event: a short label for the driving event
- caption: detailed description of what happens

Focus on driving-relevant events: lane changes, turns, stops, interactions with 
other road users, traffic signal changes, road condition changes.

Return as a JSON array. Example:
[
  {{"start_time": 0.0, "end_time": 3.0,
    "Event": "Approaching intersection",
    "caption": "The vehicle maintains a steady speed along a two-lane road, approaching a four-way intersection with a red traffic light visible ahead."}},
  {{"start_time": 3.0, "end_time": 6.0,
    "Event": "Stopping at red light",
    "caption": "The vehicle decelerates smoothly and comes to a stop behind a white SUV at the intersection. Cross traffic flows from left to right."}},
  {{"start_time": 6.0, "end_time": 9.0,
    "Event": "Proceeding through intersection",
    "caption": "The traffic light turns green. The vehicle accelerates and proceeds straight through the intersection, passing the stopped cross-traffic."}}
]

Provide temporal annotations for the following driving sequence:"""


# ============================================================================
# TEMPLATE-BASED CAPTIONING (no GPU required)
# ============================================================================

def generate_template_captions(manifest):
    """
    Generate structured captions from extracted metadata without a VLM.
    
    Useful for:
    - Quick prototyping before investing in VLM inference
    - Datasets with rich metadata (nuScenes has detailed annotations)
    - Creating initial training data that can be refined later
    """
    metadata = manifest['metadata']
    actions = manifest['actions']
    timestamps = manifest['timestamps']
    
    duration = metadata['duration_seconds']
    behavior = metadata['behavior_category']
    avg_speed = metadata['avg_speed_ms']
    
    # --- Narrative Caption ---
    speed_desc = _speed_description(avg_speed)
    behavior_desc = _behavior_description(behavior)
    motion_narrative = _build_motion_narrative(actions, timestamps)
    
    narrative = (
        f"The dashcam footage shows a {duration:.0f}-second driving sequence "
        f"capturing {behavior_desc}. The vehicle {speed_desc}. {motion_narrative}"
    )
    
    # --- Scene-Static Caption ---
    # Without visual analysis, we can only provide a generic template
    # This should be replaced/enhanced with VLM output
    scene_static = (
        f"A driving scene viewed from a front-facing dashcam perspective. "
        f"The road environment is typical of {_environment_guess(behavior, avg_speed)} driving conditions."
    )
    
    # --- Dense Temporal Captions ---
    temporal_events = _build_temporal_events(actions, timestamps)
    
    return {
        'narrative_caption': narrative,
        'scene_static_caption': scene_static,
        'dense_temporal_captions': temporal_events,
        'caption_method': 'template',
    }


def _speed_description(avg_speed_ms):
    if avg_speed_ms < 1.0:
        return "is nearly stationary"
    elif avg_speed_ms < 5.0:
        return "moves at a slow pace, typical of parking or congested traffic"
    elif avg_speed_ms < 15.0:
        return "travels at moderate urban speeds"
    elif avg_speed_ms < 25.0:
        return "maintains a steady highway-like pace"
    else:
        return "moves at high speed, suggesting highway or expressway driving"


def _behavior_description(behavior):
    descriptions = {
        'stationary': "a stationary vehicle observing the surrounding traffic",
        'slow_exploration': "slow-speed navigation through a local area",
        'urban_navigation': "urban driving with turns and intersections",
        'sharp_maneuver': "dynamic driving involving sharp turns or maneuvers",
        'highway': "highway or freeway driving with sustained speed",
        'stop_and_go': "stop-and-go traffic conditions with frequent speed changes",
        'cruising': "steady cruising along a road without major events",
    }
    return descriptions.get(behavior, "general driving")


def _environment_guess(behavior, avg_speed):
    if behavior in ('highway',) or avg_speed > 20:
        return "highway or expressway"
    elif behavior in ('urban_navigation', 'sharp_maneuver', 'stop_and_go'):
        return "urban or suburban"
    elif behavior in ('slow_exploration', 'stationary'):
        return "low-speed residential or parking"
    else:
        return "mixed urban"


def _build_motion_narrative(actions, timestamps):
    """Build a natural language description of the vehicle's motion over time."""
    if not actions or len(actions) < 2:
        return ""
    
    segments = []
    current_state = actions[0]['steer_label']
    segment_start = 0
    
    for i in range(1, len(actions)):
        new_state = actions[i]['steer_label']
        if new_state != current_state or i == len(actions) - 1:
            t_start = (timestamps[segment_start] - timestamps[0]) / 1e6
            t_end = (timestamps[i] - timestamps[0]) / 1e6
            
            steer = current_state
            avg_vel = sum(a['velocity'] for a in actions[segment_start:i]) / max(1, i - segment_start)
            
            desc = _segment_description(steer, avg_vel, t_end - t_start)
            if desc:
                segments.append(desc)
            
            current_state = new_state
            segment_start = i
    
    if segments:
        return "The trajectory shows: " + ", then ".join(segments[:5]) + "."
    return ""


def _segment_description(steer, avg_vel, duration):
    if duration < 0.3:
        return None
    
    speed_word = "slowly" if avg_vel < 5 else ("steadily" if avg_vel < 15 else "briskly")
    
    if steer == 'straight':
        return f"driving {speed_word} ahead for {duration:.1f}s"
    elif 'left' in steer:
        intensity = "sharply " if 'hard' in steer else ""
        return f"turning {intensity}left for {duration:.1f}s"
    elif 'right' in steer:
        intensity = "sharply " if 'hard' in steer else ""
        return f"turning {intensity}right for {duration:.1f}s"
    return None


def _build_temporal_events(actions, timestamps):
    """Segment the drive into time-aligned events."""
    if not actions or len(actions) < 2:
        return []
    
    events = []
    t0 = timestamps[0]
    
    # Simple state-change segmentation
    current_event = _event_label(actions[0])
    segment_start_time = 0.0
    segment_start_idx = 0
    
    for i in range(1, len(actions)):
        new_event = _event_label(actions[i])
        t = (timestamps[i] - t0) / 1e6
        
        if new_event != current_event or i == len(actions) - 1:
            t_start = segment_start_time
            t_end = t
            
            if t_end - t_start >= 0.5:  # minimum 0.5s segment
                avg_vel = sum(a['velocity'] for a in actions[segment_start_idx:i]) / max(1, i - segment_start_idx)
                
                events.append({
                    'start_time': round(t_start, 1),
                    'end_time': round(t_end, 1),
                    'Event': current_event,
                    'caption': _event_caption(current_event, avg_vel, t_end - t_start),
                })
            
            current_event = new_event
            segment_start_time = t
            segment_start_idx = i
    
    return events


def _event_label(action):
    """Map action labels to driving event names."""
    steer = action['steer_label']
    accel = action['accel_label']
    speed = action['speed_label']
    
    if speed == 'stopped':
        return 'Stopped'
    elif 'brake' in accel:
        return 'Decelerating'
    elif 'accel' in accel and 'left' in steer:
        return 'Accelerating through left turn'
    elif 'accel' in accel and 'right' in steer:
        return 'Accelerating through right turn'
    elif 'accel' in accel:
        return 'Accelerating'
    elif 'left' in steer:
        return 'Turning left'
    elif 'right' in steer:
        return 'Turning right'
    else:
        return 'Cruising straight'


def _event_caption(event, avg_vel, duration):
    speed_desc = f"at approximately {avg_vel * 3.6:.0f} km/h" if avg_vel > 0.5 else "from a near standstill"
    return f"The vehicle is {event.lower()} {speed_desc} for {duration:.1f} seconds."


# ============================================================================
# VLM-BASED CAPTIONING (requires GPU)
# ============================================================================

def load_vlm(model_name='Qwen/Qwen3-VL-7B-Instruct'):
    """Load a VLM for automatic captioning."""
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch
        
        print(f"Loading VLM: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        return model, processor
    except ImportError:
        raise RuntimeError("VLM mode requires: pip install transformers torch qwen-vl-utils")


def generate_vlm_caption(model, processor, frame_paths, prompt, max_frames=8):
    """Generate a caption using VLM on sampled frames."""
    import torch
    
    # Sample frames evenly
    indices = np.linspace(0, len(frame_paths) - 1, min(max_frames, len(frame_paths)), dtype=int)
    selected_frames = [frame_paths[i] for i in indices]
    
    # Build message with images
    content = []
    for fp in selected_frames:
        content.append({"type": "image", "image": f"file://{fp}"})
    content.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[fp for fp in selected_frames],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)
    
    output_text = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]
    
    return output_text.strip()


def generate_vlm_captions(manifest, model, processor, frames_dir):
    """Generate all 3 caption levels using VLM."""
    frame_paths = [
        str(Path(frames_dir) / f['filename'])
        for f in manifest['frames']
        if (Path(frames_dir) / f['filename']).exists()
    ]
    
    if not frame_paths:
        return None
    
    narrative = generate_vlm_caption(
        model, processor, frame_paths, NARRATIVE_CAPTION_PROMPT
    )
    
    scene_static = generate_vlm_caption(
        model, processor, frame_paths, SCENE_STATIC_CAPTION_PROMPT
    )
    
    # For temporal captions, process in chunks
    temporal_raw = generate_vlm_caption(
        model, processor, frame_paths, DENSE_TEMPORAL_CAPTION_PROMPT
    )
    
    # Try to parse temporal captions as JSON
    try:
        temporal_events = json.loads(temporal_raw)
    except json.JSONDecodeError:
        # Fall back to treating as plain text
        temporal_events = [{'raw_text': temporal_raw}]
    
    return {
        'narrative_caption': narrative,
        'scene_static_caption': scene_static,
        'dense_temporal_captions': temporal_events,
        'caption_method': 'vlm',
    }


# ============================================================================
# MAIN
# ============================================================================

def process_scene(scene_dir, mode='template', vlm_model=None, vlm_processor=None):
    """Process a single scene directory."""
    manifest_path = scene_dir / 'manifest.json'
    if not manifest_path.exists():
        return None
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    scene_name = manifest['metadata']['scene_name']
    
    if mode == 'vlm' and vlm_model is not None:
        captions = generate_vlm_captions(
            manifest, vlm_model, vlm_processor,
            scene_dir / 'frames'
        )
        if captions is None:
            print(f"  {scene_name}: No frames found, falling back to template")
            captions = generate_template_captions(manifest)
    else:
        captions = generate_template_captions(manifest)
    
    # Save captions
    caption_path = scene_dir / 'captions.json'
    with open(caption_path, 'w') as f:
        json.dump(captions, f, indent=2)
    
    print(f"  {scene_name}: {mode} captions generated")
    print(f"    Narrative: {captions['narrative_caption'][:100]}...")
    print(f"    Temporal events: {len(captions['dense_temporal_captions'])}")
    
    return captions


def main():
    parser = argparse.ArgumentParser(
        description='Generate hierarchical captions for driving scenes (LingBot-World style)'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Directory with scene folders from prepare_nuscenes.py')
    parser.add_argument('--mode', type=str, default='template', choices=['template', 'vlm'],
                        help='Captioning mode: template (no GPU) or vlm (requires GPU + model)')
    parser.add_argument('--vlm', type=str, default='Qwen/Qwen3-VL-7B-Instruct',
                        help='VLM model name for vlm mode')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    scene_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and (d / 'manifest.json').exists()])
    
    if not scene_dirs:
        print(f"No scene directories found in {input_dir}")
        return
    
    # Load VLM if needed
    vlm_model, vlm_processor = None, None
    if args.mode == 'vlm':
        vlm_model, vlm_processor = load_vlm(args.vlm)
    
    print(f"Generating {args.mode} captions for {len(scene_dirs)} scenes...")
    print("=" * 60)
    
    for scene_dir in scene_dirs:
        process_scene(scene_dir, args.mode, vlm_model, vlm_processor)
    
    print(f"\nDone! Captions saved to each scene's captions.json")
    print(f"\nCaption hierarchy (mirrors LingBot-World Section 2.3):")
    print(f"  1. narrative_caption     — full driving story with motion + environment")
    print(f"  2. scene_static_caption  — environment only, no vehicle motion")
    print(f"  3. dense_temporal_captions — time-aligned driving events")


if __name__ == '__main__':
    import numpy as np
    main()

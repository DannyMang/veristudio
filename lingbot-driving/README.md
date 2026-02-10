# LingBot-World → Driving: Adaptation Guide

## The Core Idea

LingBot-World's paper reveals a clean, reproducible recipe:

```
Raw Video + Game Recordings
        ↓
Data Profiling (pose estimation, filtering, semantic analysis)
        ↓
Hierarchical Captioning (narrative / scene-static / dense temporal)
        ↓
Action Labeling (Plücker camera embeddings + discrete keyboard inputs)
        ↓
Fine-tune Wan2.2 with action injection via AdaLN
```

**Our insight**: Every component maps directly to driving:

| LingBot (Gaming)              | Ours (Driving)                              |
|-------------------------------|---------------------------------------------|
| WASD keyboard inputs          | Steering angle (discretized: left/straight/right) |
| Mouse camera rotation         | Ego-vehicle heading change (yaw rate)       |
| Plücker camera embeddings     | Plücker embeddings from ego-vehicle trajectory |
| Game engine ground-truth poses| GPS/IMU ego-pose from driving datasets      |
| General web video (no labels) | OpenDV-YouTube dashcam video (no labels)    |
| Game recordings (with labels) | nuScenes / Waymo (with ego-pose + actions)  |
| UE synthetic rendering        | CARLA synthetic driving scenes              |
| Narrative caption             | "Driving through downtown at dusk, turning left onto a residential street..." |
| Scene-static caption          | "A four-lane urban road with parked cars, a traffic light, and a pedestrian crossing..." |
| Dense temporal caption        | Time-aligned events: "0-3s: approaching intersection", "3-5s: executing left turn"... |

---

## Architecture: What We're Actually Fine-Tuning

From the paper (Section 3.3.2), the key design choice is:

> "We freeze the main DiT blocks of the pre-trained fundamental world model and
> only finetune the newly added action adapter layers (including the action
> embedding projections and AdaLN parameters)."

This means we DON'T retrain the whole 14B model. We:
1. Keep the frozen Wan2.2 backbone (visual quality)
2. Train only the **action adapter** — the Plücker encoder + AdaLN scale/shift layers
3. These adapters learn: "when the driver turns left, the world should rotate right"

**This is why it's feasible.** The action adapter is a small fraction of the total parameters.

---

## Data Strategy (3 Tiers)

### Tier 1: Labeled Driving Data (action + pose ground truth)
- **nuScenes**: 1,000 scenes, 6 cameras, full ego-pose at 2Hz → interpolate to video framerate
- **Waymo Open**: 1,150 scenes, ego-pose from high-quality IMU/GPS
- **CARLA**: Unlimited synthetic, perfect ground truth for everything

### Tier 2: Unlabeled Driving Video (pose estimated, actions inferred)
- **OpenDV-YouTube**: 1,700 hours of dashcam video worldwide
- **OpenDV-mini**: 28 hours subset for prototyping
- Pipeline: MegaSAM → camera poses → inverse dynamics model → pseudo-actions

### Tier 3: General Video (visual prior only, no driving actions)
- Already covered by Wan2.2 pre-training
- We inherit this for free

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────┐
│                  DATA PREPARATION                    │
│                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ nuScenes │  │ OpenDV-YouTube│  │    CARLA      │  │
│  │(labeled) │  │ (unlabeled)  │  │ (synthetic)   │  │
│  └────┬─────┘  └──────┬───────┘  └──────┬────────┘  │
│       │               │                 │            │
│       ▼               ▼                 ▼            │
│  ┌─────────┐   ┌───────────┐    ┌────────────┐      │
│  │ Extract │   │ MegaSAM   │    │ Export GT  │      │
│  │ego-pose │   │ pose est. │    │ poses+acts │      │
│  └────┬────┘   └─────┬─────┘    └─────┬──────┘      │
│       │              │                │              │
│       ▼              ▼                ▼              │
│  ┌──────────────────────────────────────────┐        │
│  │     Unified Format: video + poses +       │        │
│  │     actions + hierarchical captions       │        │
│  └──────────────────────────────────────────┘        │
│       │                                              │
│       ▼                                              │
│  ┌──────────────────────────────────────────┐        │
│  │  1. Compute Plücker embeddings from poses │        │
│  │  2. Discretize actions (steer/throttle)   │        │
│  │  3. Generate 3-level captions via VLM     │        │
│  └──────────────────────────────────────────┘        │
└─────────────────────────────────────────────┬────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────┐
│                   FINE-TUNING                        │
│                                                      │
│  LingBot-World-Base (Wan2.2 14B, frozen)             │
│       + Action Adapter (Plücker encoder + AdaLN)     │
│       = Driving World Model                          │
│                                                      │
│  Training: action adapter only (~few % of params)    │
│  Data: Tier 1 (labeled) + Tier 2 (pseudo-labeled)   │
└─────────────────────────────────────────────────────┘
```

---

## Action Representation: The Key Mapping

LingBot encodes actions as:
```
action = concat([plücker_embedding, keyboard_multihot], dim=channel)
```

For driving, we define:

```python
# Continuous component (Plücker embeddings from ego trajectory)
# - Represents the 6DoF camera/vehicle motion between frames
# - Computed from ego-pose sequences: (position, orientation) at each timestep
# - This is IDENTICAL to what LingBot does — driving ego-pose IS camera pose

# Discrete component (driving actions as multi-hot vector)
# Instead of WASD, we use:
#   [forward, brake, steer_left, steer_right, idle]
# Or finer:
#   [accel_hard, accel_soft, coast, brake_soft, brake_hard,
#    steer_hard_left, steer_soft_left, straight, steer_soft_right, steer_hard_right]
```

The beautiful thing: **Plücker embeddings from a moving car are exactly the same math as Plücker embeddings from a game camera.** The ego-vehicle trajectory IS a camera trajectory.

---

## Quick Start

```bash
# 1. Prepare data (start with nuScenes mini for prototyping)
python scripts/prepare_nuscenes.py --dataroot /path/to/nuscenes --split mini

# 2. Generate hierarchical captions
python scripts/caption_driving.py --input data/nuscenes_processed/ --vlm qwen3-vl

# 3. Compute Plücker embeddings + discretize actions
python scripts/compute_actions.py --input data/nuscenes_processed/

# 4. For unlabeled video (OpenDV), estimate poses first
python scripts/estimate_poses.py --input data/opendv_mini/ --method megasam

# 5. Fine-tune (see configs/finetune_driving.yaml)
# This requires the LingBot-World codebase + multi-GPU setup
```

---

## File Structure

```
lingbot-driving/
├── README.md                          # This file
├── scripts/
│   ├── prepare_nuscenes.py            # Extract ego-pose + video from nuScenes
│   ├── prepare_carla.py               # CARLA data collection script
│   ├── estimate_poses.py              # MegaSAM pose estimation for unlabeled video
│   ├── compute_actions.py             # Ego-pose → Plücker embeddings + discrete actions
│   ├── caption_driving.py             # Hierarchical captioning via VLM
│   └── build_training_manifest.py     # Combine everything into training format
├── configs/
│   └── finetune_driving.yaml          # Training configuration
└── docs/
    └── action_mapping.md              # Detailed action representation docs
```

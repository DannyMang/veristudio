# Action Mapping: LingBot Gaming → Driving

## How LingBot-World Encodes Actions (Paper Section 3.3.2)

LingBot-World uses a **hybrid action representation**:

```
action = concat([plücker_embedding, keyboard_multihot], dim=channel)
           ↑                          ↑
     continuous 6D               discrete multi-hot
     (camera rotation)           (WASD keys)
```

These are injected into DiT blocks via **adaptive layer normalization (AdaLN)**:
- The fused action embedding is projected through learned layers
- The projection outputs **scale** (γ) and **shift** (β) parameters
- These modulate the normalized DiT features: `output = γ * LayerNorm(x) + β`

This is elegant because it doesn't change the backbone architecture at all —
it just modulates the existing features to respond to actions.

---

## Our Driving Action Representation

### Continuous Component: Plücker Embeddings (6D)

**The math is identical.** A dashcam mounted on a car IS a camera moving through space.

```
LingBot:  game camera moves forward  → Plücker encodes forward translation
Driving:  car drives forward         → Plücker encodes forward translation

LingBot:  mouse rotates camera right → Plücker encodes rightward rotation  
Driving:  steering wheel turns right → Plücker encodes rightward rotation
```

Plücker coordinates for a ray: `(d, m)` where:
- `d` = ray direction (3D unit vector)
- `m` = moment = `origin × d` (3D vector)

For frame-to-frame ego-motion, we compute the relative transform between poses
and encode the central ray's Plücker coordinates. This captures both translation
and rotation in a unified 6D representation.

**Key difference**: Driving has much less pitch/roll variation than gaming.
Most motion is:
- Forward translation (dominant)
- Yaw rotation (steering)
- Lateral translation (lane changes)

This actually makes the learning problem **easier** than gaming, because the
Plücker embedding space is more structured and lower-dimensional in practice.

### Discrete Component: Driving Actions (14D multi-hot)

LingBot uses WASD (4 keys, multi-hot = multiple can be active simultaneously):

```
LingBot WASD:  [W, A, S, D] = [forward, left, backward, right]
               Examples: [1,0,0,0] = forward only
                        [1,1,0,0] = forward + left (diagonal)
```

We extend this to a richer driving action space:

```
Driving (14D multi-hot):

Speed category (indices 0-3, one-hot):
  [stopped, slow, cruise, fast]
  
  stopped:  < 0.5 m/s    (~0 km/h)
  slow:     0.5 - 5 m/s  (2 - 18 km/h)  — parking, congestion
  cruise:   5 - 15 m/s   (18 - 54 km/h) — urban driving
  fast:     > 15 m/s     (> 54 km/h)     — highway

Steering category (indices 4-8, one-hot):
  [hard_left, soft_left, straight, soft_right, hard_right]
  
  hard_left:   yaw rate < -0.15 rad/s  — sharp turn, U-turn
  soft_left:   -0.15 to -0.02 rad/s    — gentle curve, lane change
  straight:    -0.02 to +0.02 rad/s    — going straight
  soft_right:  +0.02 to +0.15 rad/s    — gentle curve, lane change
  hard_right:  yaw rate > +0.15 rad/s  — sharp turn

Acceleration category (indices 9-13, one-hot):
  [hard_brake, soft_brake, coast, soft_accel, hard_accel]
  
  hard_brake:  < -2.0 m/s²  — emergency braking
  soft_brake:  -2.0 to -0.5 — normal deceleration
  coast:       -0.5 to +0.5 — maintaining speed
  soft_accel:  +0.5 to +2.0 — normal acceleration
  hard_accel:  > +2.0 m/s²  — aggressive acceleration
```

**Why 14D instead of 4D?** Driving has more nuance than WASD:
- Speed matters independently of acceleration (cruising at 100 km/h ≠ accelerating to 100 km/h)
- Steering intensity matters (gentle lane change ≠ sharp U-turn)
- The model needs to know whether to generate "highway cruising" vs "stop-and-go traffic"

**Why one-hot within categories instead of pure multi-hot?**
Each category (speed, steer, accel) is mutually exclusive within itself,
but all three categories are active simultaneously. This gives the model
three orthogonal control axes, which maps cleanly to how humans think about driving.

### Combined Action Vector (20D)

```
action_vector = [plucker_6d | speed_4d | steer_5d | accel_5d]
                     ↑            ↑          ↑          ↑
               continuous    one-hot     one-hot     one-hot
               geometric    speed cat.  steer cat.  accel cat.

Total: 6 + 4 + 5 + 5 = 20 dimensions
```

This vector is fed into the same Plücker Encoder → AdaLN pathway that
LingBot uses for gaming actions.

---

## Mapping Table: LingBot → Driving Actions

| Gaming Action      | Driving Equivalent      | Plücker Effect          | Discrete Encoding    |
|-------------------|------------------------|------------------------|---------------------|
| W (forward)       | Accelerate             | Forward translation ↑  | accel = soft/hard   |
| S (backward)      | Brake / reverse        | Forward translation ↓  | accel = soft/hard brake |
| A (strafe left)   | Steer left             | Yaw rotation left      | steer = soft/hard left |
| D (strafe right)  | Steer right            | Yaw rotation right     | steer = soft/hard right |
| W+A (diagonal)    | Accelerate + turn left | Forward + yaw left     | accel + steer combined |
| None (idle)       | Coasting               | Minimal Plücker change | speed=cruise, accel=coast |
| Mouse up          | (not applicable)       | —                      | — |
| Mouse down        | (not applicable)       | —                      | — |
| Mouse left        | Steer left (heading)   | Yaw rotation left      | steer = soft_left |
| Mouse right       | Steer right (heading)  | Yaw rotation right     | steer = soft_right |

---

## Deriving Actions from Different Data Sources

### nuScenes / Waymo (ground truth ego-pose)

```python
# We have: translation [x,y,z] + rotation [w,x,y,z] at each timestamp
# Directly compute:
#   velocity = |pos_t - pos_{t-1}| / dt
#   yaw_rate = (heading_t - heading_{t-1}) / dt  
#   acceleration = (velocity_t - velocity_{t-1}) / dt
# Then discretize into bins
```

### OpenDV-YouTube (no ego-pose)

```python
# Step 1: Estimate camera poses with MegaSAM (as LingBot does, Section 2.2.2)
# Step 2: Use estimated poses as if they were ego-poses
# Step 3: Same velocity/yaw_rate/acceleration computation
# Step 4: Discretize (but with lower confidence — flag as pseudo-labels)
```

### CARLA (synthetic, perfect ground truth)

```python
# CARLA provides: steering_angle, throttle, brake, vehicle_speed, vehicle_transform
# Direct mapping:
#   speed → speed category
#   steering_angle → steer category  
#   throttle - brake → accel category
#   vehicle_transform → Plücker embedding (exact, no estimation needed)
```

---

## Curriculum Strategy for Action Learning

Following LingBot's progressive curriculum (Section 3.3.1):

| Stage | Clip Length | Focus | Data Mix |
|-------|-----------|-------|----------|
| 1     | 5 seconds | Basic action following | 60% nuScenes, 20% CARLA, 20% OpenDV |
| 2     | 10 seconds | Turn consistency | Oversample turning clips (urban_navigation) |
| 3     | 20 seconds | Long-term dynamics | Full mix, emphasize varied behaviors |
| 4     | 60 seconds | Extended horizon | Select long continuous sequences |

**Data balancing**: Driving data is heavily biased toward "going straight."
Oversample turning, stopping, and lane-change clips by 3-5x.

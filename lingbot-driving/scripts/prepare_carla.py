"""
prepare_carla.py — Collect driving data from CARLA simulator.

This is the driving equivalent of LingBot's Unreal Engine synthetic rendering
pipeline (Section 2.1.3). CARLA provides:
  - Perfect camera poses (no estimation needed)
  - Exact vehicle dynamics (steering, throttle, brake, speed)
  - Controllable weather, traffic, and scenarios
  - Unlimited data generation

LingBot's UE pipeline generates "collision-free, randomized yet plausible
camera trajectories, yielding RGB streams aligned with ground-truth camera
intrinsics and extrinsics." CARLA does exactly this for driving.

Prerequisites:
    pip install carla  # CARLA Python API
    # CARLA server must be running: ./CarlaUE4.sh -quality-level=Epic

Usage:
    python prepare_carla.py --output data/carla_collected/ --episodes 100 --duration 30

    # With diverse scenarios
    python prepare_carla.py --output data/carla_collected/ --episodes 100 \
        --weather random --traffic dense --scenarios all
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np


def collect_episode(
    client, world, episode_id, output_dir, duration_sec=30, fps=10,
    weather='random', traffic_density='medium', scenario='free_navigation'
):
    """
    Collect one driving episode from CARLA.
    
    Mirrors LingBot's game data categories (Section 2.1.2):
    - Navigation (free/loop/transition) → free_navigation, route_following
    - Sightseeing → slow_exploration  
    - Long-tail → stationary, reversing, emergency
    - World interaction → intersection, lane_change
    """
    import carla
    
    episode_dir = Path(output_dir) / f'episode_{episode_id:06d}'
    frames_dir = episode_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Set weather
    if weather == 'random':
        weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.SoftRainNoon,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.WetSunset,
            carla.WeatherParameters.ClearNight,
        ]
        world.set_weather(random.choice(weather_presets))
    
    # Spawn ego vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        print(f"    Failed to spawn vehicle for episode {episode_id}")
        return None
    
    # Attach front camera (dashcam perspective)
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '70')
    camera_bp.set_attribute('sensor_tick', str(1.0 / fps))
    
    # Mount position: typical dashcam location
    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=1.8),  # front of car, roof height
        carla.Rotation(pitch=-5)         # slight downward tilt
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    # Set autopilot based on scenario
    if scenario == 'free_navigation':
        vehicle.set_autopilot(True)
    elif scenario == 'stationary':
        vehicle.set_autopilot(False)
        vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
    
    # Spawn traffic
    traffic_counts = {'sparse': 20, 'medium': 50, 'dense': 100}
    num_traffic = traffic_counts.get(traffic_density, 50)
    traffic_vehicles = []
    for _ in range(num_traffic):
        try:
            bp = random.choice(blueprint_library.filter('vehicle.*'))
            sp = random.choice(spawn_points)
            tv = world.try_spawn_actor(bp, sp)
            if tv:
                tv.set_autopilot(True)
                traffic_vehicles.append(tv)
        except:
            pass
    
    # Data collection
    frames = []
    ego_poses = []
    vehicle_controls = []
    timestamps = []
    
    frame_buffer = []
    
    def camera_callback(image):
        frame_buffer.append(image)
    
    camera.listen(camera_callback)
    
    print(f"    Collecting episode {episode_id}: {scenario}, {duration_sec}s at {fps} FPS...")
    
    start_time = time.time()
    frame_idx = 0
    
    try:
        while time.time() - start_time < duration_sec:
            world.tick()
            
            # Process any captured frames
            while frame_buffer:
                image = frame_buffer.pop(0)
                
                # Save frame
                filename = f'{frame_idx:06d}.jpg'
                image.save_to_disk(str(frames_dir / filename))
                
                # Record ego pose
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                control = vehicle.get_control()
                
                # Convert CARLA transform to our format
                location = transform.location
                rotation = transform.rotation
                
                # CARLA uses Euler angles; convert to quaternion [w, x, y, z]
                quat = euler_to_quaternion(
                    np.radians(rotation.roll),
                    np.radians(rotation.pitch),
                    np.radians(rotation.yaw)
                )
                
                speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                frames.append({
                    'index': frame_idx,
                    'filename': filename,
                    'timestamp': image.timestamp,
                })
                
                ego_poses.append({
                    'translation': [location.x, location.y, location.z],
                    'rotation': quat.tolist(),
                })
                
                vehicle_controls.append({
                    'throttle': control.throttle,
                    'steer': control.steer,
                    'brake': control.brake,
                    'reverse': control.reverse,
                    'speed_ms': speed,
                    'hand_brake': control.hand_brake,
                })
                
                timestamps.append(int(image.timestamp * 1e6))
                
                frame_idx += 1
    
    finally:
        # Cleanup
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        for tv in traffic_vehicles:
            tv.destroy()
    
    if len(frames) < 10:
        print(f"    Too few frames ({len(frames)}), skipping")
        return None
    
    # Compute driving actions from CARLA control signals
    # This is more accurate than pose-based estimation because we have
    # direct access to steering, throttle, and brake
    actions = compute_carla_actions(vehicle_controls, ego_poses, timestamps)
    
    # Build manifest
    duration_actual = (timestamps[-1] - timestamps[0]) / 1e6 if timestamps else 0
    manifest = {
        'metadata': {
            'scene_name': f'episode_{episode_id:06d}',
            'camera': 'CAM_FRONT',
            'num_frames': len(frames),
            'duration_seconds': duration_actual,
            'source': 'carla',
            'scenario': scenario,
            'weather': str(world.get_weather()),
            'traffic_density': traffic_density,
            'avg_speed_ms': float(np.mean([c['speed_ms'] for c in vehicle_controls])),
            'max_speed_ms': float(np.max([c['speed_ms'] for c in vehicle_controls])),
            'behavior_category': scenario,
        },
        'frames': frames,
        'ego_poses': ego_poses,
        'timestamps': timestamps,
        'actions': actions,
        'vehicle_controls': vehicle_controls,  # raw CARLA controls (bonus data)
    }
    
    with open(episode_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"    Episode {episode_id}: {len(frames)} frames, {duration_actual:.1f}s")
    return manifest


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion [w, x, y, z]."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def compute_carla_actions(vehicle_controls, ego_poses, timestamps):
    """
    Compute driving actions from CARLA's direct control signals.
    
    CARLA gives us exact steering and throttle/brake — much more accurate
    than estimating from pose differences. This is the equivalent of
    LingBot having exact WASD key presses from game recordings.
    """
    actions = []
    
    for i, ctrl in enumerate(vehicle_controls):
        speed = ctrl['speed_ms']
        steer = ctrl['steer']       # -1 (full left) to +1 (full right)
        throttle = ctrl['throttle']  # 0 to 1
        brake = ctrl['brake']        # 0 to 1
        
        # Speed category
        if speed < 0.5:
            speed_label = 'stopped'
        elif speed < 5.0:
            speed_label = 'slow'
        elif speed < 15.0:
            speed_label = 'cruise'
        else:
            speed_label = 'fast'
        
        # Steering category (from CARLA's normalized steering angle)
        if steer < -0.3:
            steer_label = 'hard_left'
        elif steer < -0.05:
            steer_label = 'soft_left'
        elif steer > 0.3:
            steer_label = 'hard_right'
        elif steer > 0.05:
            steer_label = 'soft_right'
        else:
            steer_label = 'straight'
        
        # Acceleration category
        net_accel = throttle - brake
        if net_accel < -0.5:
            accel_label = 'hard_brake'
        elif net_accel < -0.1:
            accel_label = 'soft_brake'
        elif net_accel > 0.5:
            accel_label = 'hard_accel'
        elif net_accel > 0.1:
            accel_label = 'soft_accel'
        else:
            accel_label = 'coast'
        
        # Multi-hot encoding (same 14D layout as nuScenes pipeline)
        multihot = [0.0] * 14
        speed_idx = ['stopped', 'slow', 'cruise', 'fast'].index(speed_label)
        steer_idx = ['hard_left', 'soft_left', 'straight', 'soft_right', 'hard_right'].index(steer_label)
        accel_idx = ['hard_brake', 'soft_brake', 'coast', 'soft_accel', 'hard_accel'].index(accel_label)
        
        multihot[speed_idx] = 1.0
        multihot[4 + steer_idx] = 1.0
        multihot[9 + accel_idx] = 1.0
        
        actions.append({
            'velocity': speed,
            'yaw_rate': steer * 0.5,  # approximate yaw rate from steering
            'acceleration': net_accel * 3.0,  # approximate m/s²
            'speed_label': speed_label,
            'steer_label': steer_label,
            'accel_label': accel_label,
            'multihot': multihot,
            'translation': ego_poses[i]['translation'] if i < len(ego_poses) else [0,0,0],
            'rotation': ego_poses[i]['rotation'] if i < len(ego_poses) else [1,0,0,0],
            # CARLA bonus: raw control signals
            'carla_steer': steer,
            'carla_throttle': throttle,
            'carla_brake': brake,
        })
    
    return actions


def main():
    parser = argparse.ArgumentParser(
        description='Collect driving data from CARLA (LingBot UE pipeline equivalent)'
    )
    parser.add_argument('--output', type=str, default='data/carla_collected/')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--duration', type=float, default=30, help='Episode duration (seconds)')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--weather', type=str, default='random', choices=['random', 'clear', 'rain'])
    parser.add_argument('--traffic', type=str, default='medium', choices=['sparse', 'medium', 'dense'])
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--scenarios', type=str, default='mixed',
                        choices=['free_navigation', 'stationary', 'mixed', 'all'])
    args = parser.parse_args()
    
    import carla
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Set synchronous mode for deterministic data collection
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)
    
    # Define scenario distribution (mirrors LingBot's data categories)
    if args.scenarios == 'mixed':
        scenario_weights = {
            'free_navigation': 0.6,
            'stationary': 0.1,
            # Add more scenarios as needed
        }
    elif args.scenarios == 'all':
        scenario_weights = {
            'free_navigation': 0.4,
            'stationary': 0.1,
        }
    else:
        scenario_weights = {args.scenarios: 1.0}
    
    scenarios = list(scenario_weights.keys())
    weights = list(scenario_weights.values())
    
    print(f"Collecting {args.episodes} episodes from CARLA")
    print(f"Duration: {args.duration}s, FPS: {args.fps}")
    print(f"Scenarios: {scenario_weights}")
    print("=" * 60)
    
    results = []
    for i in range(args.episodes):
        scenario = random.choices(scenarios, weights=weights, k=1)[0]
        result = collect_episode(
            client, world, i, args.output,
            duration_sec=args.duration, fps=args.fps,
            weather=args.weather, traffic_density=args.traffic,
            scenario=scenario,
        )
        if result:
            results.append(result['metadata'])
    
    # Reset world settings
    settings.synchronous_mode = False
    world.apply_settings(settings)
    
    print(f"\n{'=' * 60}")
    print(f"Collected {len(results)} episodes")
    print(f"Total frames: {sum(r['num_frames'] for r in results)}")
    print(f"Total duration: {sum(r['duration_seconds'] for r in results):.0f}s")


if __name__ == '__main__':
    main()

"""
Diagnostic: Measure peg tilt at different gripper X positions.
Each position tested with a fresh env.reset() to avoid state corruption.

This tells us whether moving the hole/table closer or farther from the
robot reduces the arm's natural tilt.
"""

import numpy as np
import gymnasium as gym
import gymnasium_robotics
from gymnasium_robotics.utils.rotations import quat2euler

gym.register_envs(gymnasium_robotics)

POS_SCALE = 0.05
HOLE_Y = 0.825
TEST_Z = 0.60  # approach height


def grip_pos(env):
    return env.unwrapped._utils.get_site_xpos(
        env.unwrapped.model, env.unwrapped.data, "robot0:grip"
    ).copy()


def peg_tilt(env):
    peg_quat = env.unwrapped._utils.get_joint_qpos(
        env.unwrapped.model, env.unwrapped.data, "object0:joint"
    )[3:7]
    euler = quat2euler(peg_quat)
    total = np.sqrt(euler[0]**2 + euler[1]**2)
    return np.degrees(total), np.degrees(euler[0]), np.degrees(euler[1])


def move_to(env, target, max_steps=300, tol=0.003, gain=6.0, max_action=0.3):
    target = np.asarray(target, dtype=np.float64)
    for _ in range(max_steps):
        g = grip_pos(env)
        if np.linalg.norm(g - target) < tol:
            break
        raw = gain * (target - g) / POS_SCALE
        action = np.zeros(4)
        action[:3] = np.clip(raw, -max_action, max_action)
        env.step(action)


def settle(env, steps=60):
    for _ in range(steps):
        env.step(np.zeros(4))


env = gym.make("FetchPegInHolePreHeldDense-v1", max_episode_steps=3000, render_mode=None)

print("=" * 70)
print("TILT vs GRIPPER X POSITION (at Y=0.825, Z=0.60)")
print("=" * 70)
print(f"  Robot base ≈ X=0.40")
print(f"  Current hole X = 1.475")
print()

# Test X positions from 1.15 to 1.45
x_positions = np.arange(1.10, 1.46, 0.025)

print(f"  {'X pos':>7s}  {'Total':>6s}  {'tilt_X':>7s}  {'tilt_Y':>7s}  {'Grip actual':>20s}  Note")
print(f"  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*20}  {'-'*15}")

for x in x_positions:
    env.reset()
    move_to(env, [x, HOLE_Y, TEST_Z], max_steps=400, tol=0.003)
    settle(env, steps=80)
    g = grip_pos(env)
    total, tx, ty = peg_tilt(env)

    # Check if grip actually reached target
    reached = np.linalg.norm(g[:2] - np.array([x, HOLE_Y])) < 0.01
    note = ""
    if not reached:
        note = f"MISS (at {g[0]:.3f},{g[1]:.3f})"
    elif abs(x - 1.475) < 0.001:
        note = "<-- current"
    elif total < 5.74:
        note = "< 5.74° OK!"

    print(f"  x={x:.3f}  {total:5.1f}°  tx={tx:+5.1f}°  ty={ty:+5.1f}°  grip=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f})  {note}")

# Also test different Y positions at current X=1.30
print()
print("=" * 70)
print("TILT vs GRIPPER Y POSITION (at X=1.475, Z=0.60)")
print("=" * 70)

y_positions = np.arange(0.75, 1.01, 0.025)

print(f"  {'Y pos':>7s}  {'Total':>6s}  {'tilt_X':>7s}  {'tilt_Y':>7s}  {'Grip actual':>20s}  Note")
print(f"  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*20}  {'-'*15}")

for y in y_positions:
    env.reset()
    move_to(env, [1.475, y, TEST_Z], max_steps=400, tol=0.003)
    settle(env, steps=80)
    g = grip_pos(env)
    total, tx, ty = peg_tilt(env)

    reached = np.linalg.norm(g[:2] - np.array([1.475, y])) < 0.01
    note = ""
    if not reached:
        note = f"MISS (at {g[0]:.3f},{g[1]:.3f})"
    elif abs(y - 0.825) < 0.001:
        note = "<-- current"
    elif total < 5.74:
        note = "< 5.74° OK!"

    print(f"  y={y:.3f}  {total:5.1f}°  tx={tx:+5.1f}°  ty={ty:+5.1f}°  grip=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f})  {note}")

# Also test different Z (height) positions at current X=1.30, Y=0.95
print()
print("=" * 70)
print("TILT vs GRIPPER Z HEIGHT (at X=1.475, Y=0.825)")
print("=" * 70)

z_positions = np.arange(0.48, 0.71, 0.02)

print(f"  {'Z pos':>7s}  {'Total':>6s}  {'tilt_X':>7s}  {'tilt_Y':>7s}  {'Grip actual':>20s}  Note")
print(f"  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*20}  {'-'*15}")

for z in z_positions:
    env.reset()
    move_to(env, [1.475, HOLE_Y, z], max_steps=400, tol=0.003)
    settle(env, steps=80)
    g = grip_pos(env)
    total, tx, ty = peg_tilt(env)

    reached = abs(g[2] - z) < 0.01
    note = ""
    if not reached:
        note = f"MISS (at z={g[2]:.3f})"
    elif abs(z - 0.60) < 0.001:
        note = "<-- approach"
    elif total < 5.74:
        note = "< 5.74° OK!"

    print(f"  z={z:.3f}  {total:5.1f}°  tx={tx:+5.1f}°  ty={ty:+5.1f}°  grip=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f})  {note}")

env.close()
print()
print("5.74° = max tilt for 4mm clearance with 4cm peg half-height")

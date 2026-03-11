"""
Step-by-step peg insertion test with active correction.

Strategy:
  1. Move peg center directly above hole center
  2. Ensure peg is vertical
  3. Descend step-by-step (2mm per step)
  4. At each step, re-check and correct:
     a) Peg center XY alignment with hole center
     b) Peg verticality (tilt) via grip XY shift

Usage:
    poetry run python gr_fetch/peg_in_hole_pre_held/test_insertion.py
    poetry run python gr_fetch/peg_in_hole_pre_held/test_insertion.py --no-render
"""

import argparse
import time
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from gymnasium_robotics.utils.rotations import quat2euler

gym.register_envs(gymnasium_robotics)

POS_SCALE = 0.05   # env internal position action scaling
WALL_TOP_Z = 0.505  # hole opening Z in world frame


def get_grip_pos(env):
    return env.unwrapped._utils.get_site_xpos(
        env.unwrapped.model, env.unwrapped.data, "robot0:grip"
    ).copy()


def get_peg_center(env):
    return env.unwrapped._utils.get_joint_qpos(
        env.unwrapped.model, env.unwrapped.data, "object0:joint"
    )[:3].copy()


def get_peg_tilt(env):
    """Peg tilt as (euler_x, euler_y).
    euler_x = rotation around X = lean in Y direction
    euler_y = rotation around Y = lean in X direction
    """
    quat = env.unwrapped._utils.get_joint_qpos(
        env.unwrapped.model, env.unwrapped.data, "object0:joint"
    )[3:7]
    e = quat2euler(quat)
    return e[0], e[1]


def tilt_magnitude(env):
    tx, ty = get_peg_tilt(env)
    return np.sqrt(tx**2 + ty**2)


def move_to(env, target, max_steps=200, tol=0.002):
    """P-control to move gripper to target."""
    target = np.asarray(target, dtype=np.float64)
    obs = None
    for _ in range(max_steps):
        g = get_grip_pos(env)
        if np.linalg.norm(g - target) < tol:
            break
        raw = 6.0 * (target - g) / POS_SCALE
        action = np.zeros(4)
        action[:3] = np.clip(raw, -0.3, 0.3)
        obs, *_ = env.step(action)
    return obs


def settle(env, steps=30):
    obs = None
    for _ in range(steps):
        obs, *_ = env.step(np.zeros(4))
    return obs


def correct_and_move(env, hole_xy, target_z, tilt_gain=0.03):
    """Measure peg state, compute corrections, move gripper.

    Correction 1 — Centering:
        peg center XY should equal hole center XY.
        Shift grip by -(peg_xy - hole_xy).

    Correction 2 — Tilt:
        euler_y (lean in X) → shift grip in -X
        euler_x (lean in Y) → shift grip in -Y
    """
    peg = get_peg_center(env)
    grip = get_grip_pos(env)

    # Centering correction
    xy_err = peg[:2] - hole_xy
    corrected_xy = grip[:2] - xy_err

    # Tilt correction
    tx, ty = get_peg_tilt(env)
    corrected_xy[0] -= ty * tilt_gain
    corrected_xy[1] -= tx * tilt_gain

    # Clamp to safe range around hole
    corrected_xy[0] = np.clip(corrected_xy[0], hole_xy[0] - 0.05, hole_xy[0] + 0.05)
    corrected_xy[1] = np.clip(corrected_xy[1], hole_xy[1] - 0.05, hole_xy[1] + 0.05)

    target = np.array([corrected_xy[0], corrected_xy[1], target_z])
    obs = move_to(env, target, max_steps=80)
    obs = settle(env, 15)
    return obs


def run_test(render=True):
    env = gym.make(
        "FetchPegInHolePreHeldDense-v1",
        max_episode_steps=10000,
        render_mode="human" if render else None,
    )

    obs, info = env.reset()
    goal = obs["desired_goal"]        # peg tip target [x, y, z]
    hole_xy = goal[:2].copy()         # hole center XY

    print("=" * 60)
    print("PEG INSERTION — Step-by-Step with Active Correction")
    print("=" * 60)
    print(f"Goal (peg tip): {goal.round(4)}")
    print(f"Hole center XY: ({hole_xy[0]:.4f}, {hole_xy[1]:.4f})")
    print()

    # ── Phase 1: Position above hole ──────────────────────────────
    # peg_tip_z = grip_z - 0.06  →  grip_z = 0.62 gives tip at 0.56
    approach_z = 0.62

    print("Phase 1: Moving above hole center...")

    # Initial move: compensate grip-to-peg offset
    grip = get_grip_pos(env)
    peg = get_peg_center(env)
    offset_xy = grip[:2] - peg[:2]
    target = np.array([hole_xy[0] + offset_xy[0],
                       hole_xy[1] + offset_xy[1],
                       approach_z])
    move_to(env, target, max_steps=400)
    settle(env, 60)

    # Iterative centering (3 rounds)
    for _ in range(3):
        correct_and_move(env, hole_xy, approach_z, tilt_gain=0.03)
        settle(env, 20)

    peg = get_peg_center(env)
    grip = get_grip_pos(env)
    tx, ty = get_peg_tilt(env)
    xy_err = peg[:2] - hole_xy
    print(f"  Peg XY error:  ({xy_err[0]*1000:+.2f}, {xy_err[1]*1000:+.2f}) mm")
    print(f"  Tilt:          ({np.degrees(tx):+.3f}, {np.degrees(ty):+.3f})  "
          f"total={np.degrees(tilt_magnitude(env)):.3f} deg")
    print(f"  Grip pos:      {grip.round(4)}")

    # ── Phase 2: Step-by-step descent ─────────────────────────────
    print(f"\nPhase 2: Descending (2mm/step)...")

    # Final grip Z: peg_tip at goal_z → grip = goal_z + 0.06
    final_grip_z = goal[2] + 0.06
    current_z = grip[2]
    z_step = 0.002

    step_num = 0
    while current_z > final_grip_z - 0.005:
        current_z -= z_step
        step_num += 1

        # Correct centering + tilt, then move to new Z
        obs = correct_and_move(env, hole_xy, current_z, tilt_gain=0.03)

        # Measure state after correction
        peg = get_peg_center(env)
        grip = get_grip_pos(env)
        peg_tip_z = peg[2] - 0.04
        xy_err = peg[:2] - hole_xy
        tilt = tilt_magnitude(env)
        dist = np.linalg.norm(obs["achieved_goal"] - goal) if obs is not None else 999

        # Print: every 5 steps, first step, or when near/inside hole
        if step_num % 5 == 0 or step_num == 1 or peg_tip_z < WALL_TOP_Z + 0.005:
            print(f"  [{step_num:3d}] grip_z={grip[2]:.4f}  tip_z={peg_tip_z:.4f}  "
                  f"xy=({xy_err[0]*1000:+.2f},{xy_err[1]*1000:+.2f})mm  "
                  f"tilt={np.degrees(tilt):.2f} deg  dist={dist:.4f}")

        # Stuck check
        if abs(grip[2] - current_z) > 0.008:
            print(f"  !! Stuck: grip_z={grip[2]:.4f} target={current_z:.4f}")
            move_to(env, np.array([grip[0], grip[1], current_z]), max_steps=200)
            settle(env, 40)
            grip = get_grip_pos(env)
            if abs(grip[2] - current_z) > 0.008:
                print(f"  !! Still stuck. Stopping.")
                break

        # Success check
        if dist < 0.02 and tilt < 0.1:
            print(f"\n  >>> SUCCESS at step {step_num}!  "
                  f"dist={dist:.4f}  tilt={np.degrees(tilt):.2f} deg")
            break

    # ── Final report ──────────────────────────────────────────────
    peg = get_peg_center(env)
    peg_tip_z = peg[2] - 0.04
    tilt = tilt_magnitude(env)
    if obs is not None:
        dist = np.linalg.norm(obs["achieved_goal"] - goal)
    else:
        dist = 999

    entered = peg_tip_z < WALL_TOP_Z
    depth_cm = max(0, (WALL_TOP_Z - peg_tip_z) * 100)
    target_depth_cm = (WALL_TOP_Z - goal[2]) * 100

    print(f"\n{'=' * 60}")
    print("RESULT")
    print(f"{'=' * 60}")
    print(f"  Peg tip:   ({peg[0]:.4f}, {peg[1]:.4f}, {peg_tip_z:.4f})")
    print(f"  Goal:      ({goal[0]:.4f}, {goal[1]:.4f}, {goal[2]:.4f})")
    print(f"  Distance:  {dist:.4f} m")
    print(f"  Tilt:      {np.degrees(tilt):.3f} deg")
    print(f"  Entered:   {'YES' if entered else 'NO'}  "
          f"({depth_cm:.1f} cm / {target_depth_cm:.1f} cm)")
    print(f"  Success:   {'YES' if dist < 0.02 and tilt < 0.1 else 'NO'}")

    if render:
        print("\nWindow open — Ctrl+C to exit")
        try:
            while True:
                env.step(np.zeros(4))
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    run_test(render=not args.no_render)

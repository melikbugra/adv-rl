"""
Hard-coded deterministic stir policy for FetchStirDense-v1.

Phase 1 — Approach: proportional control to bowl center
Phase 2 — Stir:    circular sinusoidal trajectory

Usage:
    poetry run python scripts/stir_hardcoded.py           # headless
    poetry run python scripts/stir_hardcoded.py --render  # MuJoCo viewer
"""

import argparse
import math
import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENV_ID            = "FetchStirDense-v1"
MAX_EPISODE_STEPS = 600

BOWL_XY    = np.array([1.525, 0.825])
STIR_Z     = 0.57       # inside bowl, between ball Z range 0.455–0.772
ABOVE_Z    = 0.88       # above bowl wall top (~0.800) — safe entry height
APPROACH_Z = 0.57       # final depth inside bowl

APPROACH_THRESH = 0.015  # m — phase transition distance

STIR_RADIUS = 0.07       # m  (well inside inner radius ~0.110 m)
OMEGA       = 2 * math.pi / 60  # rad/step → 1 lap in 60 steps
N_LAPS      = 5          # total stirring laps → ~300 steps

ACTION_SCALE = 0.05      # env scales delta by 0.05 m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(render_mode=None):
    return gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS,
                    render_mode=render_mode)


def get_grip_pos(obs) -> np.ndarray:
    """Extract gripper XYZ from observation dict."""
    return obs["observation"][:3].copy()


def clip_action(delta: np.ndarray) -> np.ndarray:
    """Convert a 3-D positional delta into a clipped [-1,1] action vector."""
    act = np.zeros(4, dtype=np.float32)
    act[:3] = np.clip(delta / ACTION_SCALE, -1.0, 1.0)
    act[3]  = 0.0
    return act


def extract_info_metrics(info: dict):
    """Return (bowl_tilt, spoon_dist, mixing) from info dict (with defaults)."""
    bowl_tilt  = info.get("bowl_tilt",   0.0)
    spoon_dist = info.get("spoon_dist",  float("nan"))
    mixing     = info.get("mixing_score", info.get("mixing", 0.0))
    return bowl_tilt, spoon_dist, mixing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(render: bool = False):
    render_mode = "human" if render else None
    env = make_env(render_mode)

    obs, _ = env.reset()
    grip   = get_grip_pos(obs)

    total_reward = 0.0
    step         = 0

    # Tracking for summary
    bowl_tilt_max      = 0.0
    spoon_dist_min     = float("inf")
    mixing_score_final = 0.0

    # -----------------------------------------------------------------------
    # Phase 1a — Rise above bowl (XY + high Z simultaneously)
    # -----------------------------------------------------------------------
    print("[Phase 1a] Moving above bowl...")
    target_above = np.array([BOWL_XY[0], BOWL_XY[1], ABOVE_Z])

    while True:
        delta  = target_above - grip
        dist   = float(np.linalg.norm(delta))
        action = clip_action(delta)

        obs, reward, terminated, truncated, info = env.step(action)
        grip   = get_grip_pos(obs)
        total_reward += reward
        step         += 1

        bowl_tilt, spoon_dist, mixing = extract_info_metrics(info)
        bowl_tilt_max  = max(bowl_tilt_max, bowl_tilt)
        if not math.isnan(spoon_dist):
            spoon_dist_min = min(spoon_dist_min, spoon_dist)
        mixing_score_final = mixing

        if step % 30 == 0:
            print(f"  step={step:04d} | reward={reward:+.3f} | "
                  f"bowl_tilt={bowl_tilt:.3f} | dist_to_target={dist:.3f}")

        if dist < APPROACH_THRESH:
            print(f"  [Phase 1a done] Above bowl in {step} steps.")
            break

        if terminated or truncated:
            print("  Episode ended during phase 1a!")
            _print_summary(step, total_reward, bowl_tilt_max,
                           spoon_dist_min, mixing_score_final)
            env.close()
            return

    # -----------------------------------------------------------------------
    # Phase 1b — Descend into bowl
    # -----------------------------------------------------------------------
    print("[Phase 1b] Descending into bowl...")
    target_approach = np.array([BOWL_XY[0], BOWL_XY[1], APPROACH_Z])

    while True:
        delta  = target_approach - grip
        dist   = float(np.linalg.norm(delta))
        action = clip_action(delta)

        obs, reward, terminated, truncated, info = env.step(action)
        grip   = get_grip_pos(obs)
        total_reward += reward
        step         += 1

        bowl_tilt, spoon_dist, mixing = extract_info_metrics(info)
        bowl_tilt_max  = max(bowl_tilt_max, bowl_tilt)
        if not math.isnan(spoon_dist):
            spoon_dist_min = min(spoon_dist_min, spoon_dist)
        mixing_score_final = mixing

        if step % 30 == 0:
            print(f"  step={step:04d} | reward={reward:+.3f} | "
                  f"bowl_tilt={bowl_tilt:.3f} | dist_to_target={dist:.3f}")

        if dist < APPROACH_THRESH:
            print(f"  [Phase 1b done] Inside bowl at step {step}.\n")
            break

        if terminated or truncated:
            print("  Episode ended during descent!")
            _print_summary(step, total_reward, bowl_tilt_max,
                           spoon_dist_min, mixing_score_final)
            env.close()
            return

    # -----------------------------------------------------------------------
    # Phase 2 — Stir
    # -----------------------------------------------------------------------
    total_stir_steps = int(N_LAPS * (2 * math.pi / OMEGA))
    print(f"[Phase 2] Stirring for {N_LAPS} laps (~{total_stir_steps} steps)...")

    angle = 0.0

    for lap in range(1, N_LAPS + 1):
        steps_this_lap = int(2 * math.pi / OMEGA)
        print(f"  Lap {lap}/{N_LAPS}...")

        for _ in range(steps_this_lap):
            angle += OMEGA
            tx = BOWL_XY[0] + STIR_RADIUS * math.cos(angle)
            ty = BOWL_XY[1] + STIR_RADIUS * math.sin(angle)
            target_stir = np.array([tx, ty, STIR_Z])

            delta  = target_stir - grip
            action = clip_action(delta)

            obs, reward, terminated, truncated, info = env.step(action)
            grip   = get_grip_pos(obs)
            total_reward += reward
            step         += 1

            bowl_tilt, spoon_dist, mixing = extract_info_metrics(info)
            bowl_tilt_max  = max(bowl_tilt_max, bowl_tilt)
            if not math.isnan(spoon_dist):
                spoon_dist_min = min(spoon_dist_min, spoon_dist)
            mixing_score_final = mixing

            if step % 30 == 0:
                print(f"  step={step:04d} | reward={reward:+.3f} | "
                      f"bowl_tilt={bowl_tilt:.3f} | spoon_dist={spoon_dist:.3f} | "
                      f"mixing={mixing:.3f}")

            if terminated or truncated:
                print(f"  Episode ended at step {step}.")
                _print_summary(step, total_reward, bowl_tilt_max,
                               spoon_dist_min, mixing_score_final)
                env.close()
                return

    env.close()
    _print_summary(step, total_reward, bowl_tilt_max,
                   spoon_dist_min, mixing_score_final)


def _print_summary(step, total_reward, bowl_tilt_max, spoon_dist_min,
                   mixing_score_final):
    print("\n--- SUMMARY ---")
    print(f"total_steps:          {step}")
    print(f"total_reward:         {total_reward:.3f}")
    print(f"bowl_tilt_max:        {bowl_tilt_max:.4f} rad  "
          f"{'(PASS)' if bowl_tilt_max < 0.05 else '(FAIL — bowl tipped)'}")
    if math.isinf(spoon_dist_min):
        print("spoon_dist_min:       n/a (not in info)")
    else:
        print(f"spoon_dist_min:       {spoon_dist_min:.4f} m   "
              f"{'(PASS)' if spoon_dist_min < 0.08 else '(FAIL — spoon left bowl)'}")
    print(f"mixing_score_final:   {mixing_score_final:.4f}     "
          f"{'(PASS)' if mixing_score_final > 0.3 else '(low — check env reward)'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hard-coded stir policy")
    parser.add_argument("--render", action="store_true",
                        help="Open MuJoCo viewer")
    args = parser.parse_args()
    run(render=args.render)

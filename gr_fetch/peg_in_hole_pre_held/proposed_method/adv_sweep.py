"""Checkpoint sweep for FetchPegInHolePreHeldDense-v1 adversarial training.

Scans a range of checkpoint indices, runs N episodes per checkpoint, prints a
per-checkpoint summary row immediately, and prints an aggregate table at the end.

Usage:
    python adv_sweep.py 0 150
    python adv_sweep.py 100 210 --episodes 3 --render --step-delay 0.05
    python adv_sweep.py 0 50 --episodes 1 --verbose
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation

from rl_baselines.policy_based.sac.sac import SAC
from adv_eval import (
    ENV_ID,
    MAX_EPISODE_STEPS,
    _HDR,
    _SEP,
    _fmt_row,
    _to_action,
    run_episode,
)

gym.register_envs(gymnasium_robotics)

# ---------------------------------------------------------------------------
# Sweep table headers
# ---------------------------------------------------------------------------
_SWEEP_HDR = (
    f"{'ckpt':>5}  {'adv_score':>9}  {'prt_score':>9}  "
    f"{'success':>7}  {'adv_d_xy':>8}  {'adv_tilt':>8}  "
    f"{'prt_d_xy':>8}  {'prt_d_z':>7}"
)
_SWEEP_SEP = "-" * len(_SWEEP_HDR)


def _fmt_sweep_row(
    ckpt: int,
    adv_score: float,
    prt_score: float,
    successes: int,
    n: int,
    adv_d_xy: float,
    adv_tilt_deg: float,
    prt_d_xy: float,
    prt_d_z: float,
) -> str:
    return (
        f"{ckpt:>5}  {adv_score:>+9.3f}  {prt_score:>+9.3f}  "
        f"{successes:>3}/{n:<3}  {adv_d_xy:>8.4f}  {adv_tilt_deg:>7.1f}°  "
        f"{prt_d_xy:>8.4f}  {prt_d_z:>7.4f}"
    )


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(models_dir: str, start: int, end: int) -> list[int]:
    """Return sorted list of indices i in [start, end] where both
    adversary_{i}.ckpt and protagonist_{i}.ckpt exist."""
    valid: list[int] = []
    for i in range(start, end + 1):
        adv_path = os.path.join(models_dir, f"adversary_{i}.ckpt")
        prt_path = os.path.join(models_dir, f"protagonist_{i}.ckpt")
        if os.path.isfile(adv_path) and os.path.isfile(prt_path):
            valid.append(i)
    return sorted(valid)


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation (reuses an existing env)
# ---------------------------------------------------------------------------

def eval_checkpoint(
    ckpt: int,
    models_dir: str,
    eval_env: gym.Env,
    dummy_env: gym.Env,
    episodes: int,
    adversary_horizon: int,
    protagonist_horizon: int,
    device: str,
    verbose: bool,
    step_delay: float,
) -> dict:
    """Load one checkpoint pair, run *episodes* episodes, return aggregate stats."""
    adversary_path = os.path.join(models_dir, f"adversary_{ckpt}")
    protagonist_path = os.path.join(models_dir, f"protagonist_{ckpt}")

    action_dim = dummy_env.action_space.shape[0]
    target_entropy = -float(action_dim)

    adversary = SAC(
        env=dummy_env,
        network_type="mlp",
        network_arch=[256, 256],
        device=device,
        target_entropy=target_entropy,
        num_q_heads=2,
    )
    adversary.load(model_path=adversary_path)

    protagonist = SAC(
        env=dummy_env,
        network_type="mlp",
        network_arch=[512, 512, 512],
        device=device,
        target_entropy=target_entropy,
        num_q_heads=5,
    )
    protagonist.load(model_path=protagonist_path)

    summaries: list[dict] = []

    for ep in range(episodes):
        if verbose:
            print(f"{'=' * len(_HDR)}")
            print(f"  ckpt={ckpt}  Episode {ep + 1}/{episodes}")
            print(f"{'=' * len(_HDR)}")

        s = run_episode(
            env=eval_env,
            adversary=adversary,
            protagonist=protagonist,
            adversary_horizon=adversary_horizon,
            protagonist_horizon=protagonist_horizon,
            verbose=verbose,
            step_delay=step_delay,
        )
        summaries.append(s)

        if verbose:
            print()

    n = len(summaries)
    agg = dict(
        ckpt=ckpt,
        n=n,
        mean_adv_score=float(np.mean([s["adv_score"] for s in summaries])),
        mean_prt_score=float(np.mean([s["prt_score"] for s in summaries])),
        successes=sum(s["prt_success"] for s in summaries),
        mean_adv_d_xy=float(np.nanmean([s["adv_final_d_xy"] for s in summaries])),
        mean_adv_tilt_deg=float(np.nanmean([s["adv_final_tilt_deg"] for s in summaries])),
        mean_prt_d_xy=float(np.nanmean([s["prt_final_d_xy"] for s in summaries])),
        mean_prt_d_z=float(np.nanmean([s["prt_final_d_z"] for s in summaries])),
    )
    return agg


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(
    start: int,
    end: int,
    models_dir: str,
    episodes: int,
    adversary_horizon: int,
    protagonist_horizon: int,
    device: str,
    render: bool,
    verbose: bool,
    step_delay: float,
) -> None:
    checkpoints = discover_checkpoints(models_dir, start, end)

    if not checkpoints:
        print(f"No complete checkpoint pairs found in '{models_dir}' for range [{start}, {end}].")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) in [{start}, {end}]: {checkpoints}")
    print()

    render_mode = "human" if render else None
    eval_env = FlattenObservation(
        gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS, render_mode=render_mode)
    )
    dummy_env = FlattenObservation(
        gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
    )

    # Print sweep table header once
    print(_SWEEP_HDR)
    print(_SWEEP_SEP)

    all_agg: list[dict] = []

    try:
        for ckpt in checkpoints:
            agg = eval_checkpoint(
                ckpt=ckpt,
                models_dir=models_dir,
                eval_env=eval_env,
                dummy_env=dummy_env,
                episodes=episodes,
                adversary_horizon=adversary_horizon,
                protagonist_horizon=protagonist_horizon,
                device=device,
                verbose=verbose,
                step_delay=step_delay,
            )
            all_agg.append(agg)

            row = _fmt_sweep_row(
                ckpt=agg["ckpt"],
                adv_score=agg["mean_adv_score"],
                prt_score=agg["mean_prt_score"],
                successes=agg["successes"],
                n=agg["n"],
                adv_d_xy=agg["mean_adv_d_xy"],
                adv_tilt_deg=agg["mean_adv_tilt_deg"],
                prt_d_xy=agg["mean_prt_d_xy"],
                prt_d_z=agg["mean_prt_d_z"],
            )
            print(row)
    finally:
        eval_env.close()
        dummy_env.close()

    # ------------------------------------------------------------------
    # Aggregate sweep summary
    # ------------------------------------------------------------------
    if not all_agg:
        return

    print()
    print(
        f"=== SWEEP SUMMARY  (checkpoints {checkpoints[0]} → {checkpoints[-1]}, "
        f"{episodes} episode{'s' if episodes != 1 else ''} each) ==="
    )

    best_prt = max(all_agg, key=lambda a: a["mean_prt_score"])
    best_success = max(all_agg, key=lambda a: a["successes"])
    hardest_adv = max(all_agg, key=lambda a: a["mean_adv_d_xy"])

    print(
        f"  Best prt_score:              ckpt={best_prt['ckpt']:>5}  "
        f"mean={best_prt['mean_prt_score']:+.3f}"
    )
    print(
        f"  Best success%:               ckpt={best_success['ckpt']:>5}  "
        f"{best_success['successes']}/{best_success['n']}  "
        f"({100 * best_success['successes'] / best_success['n']:.0f}%)"
    )
    print(
        f"  Highest adv difficulty d_xy: ckpt={hardest_adv['ckpt']:>5}  "
        f"mean={hardest_adv['mean_adv_d_xy']:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep checkpoints for FetchPegInHolePreHeldDense-v1 adversarial training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("start", type=int, help="Range start (inclusive)")
    parser.add_argument("end", type=int, help="Range end (inclusive)")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models_adv",
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes per checkpoint",
    )
    parser.add_argument(
        "--adversary-horizon",
        type=int,
        default=100,
        help="Steps for adversary phase Ha",
    )
    parser.add_argument(
        "--protagonist-horizon",
        type=int,
        default=200,
        help="Steps for protagonist phase Hp",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between steps (e.g. 0.05 for slow rendering)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open MuJoCo viewer",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step table for every episode",
    )

    args = parser.parse_args()

    sweep(
        start=args.start,
        end=args.end,
        models_dir=args.models_dir,
        episodes=args.episodes,
        adversary_horizon=args.adversary_horizon,
        protagonist_horizon=args.protagonist_horizon,
        device=args.device,
        render=args.render,
        verbose=args.verbose,
        step_delay=args.step_delay,
    )

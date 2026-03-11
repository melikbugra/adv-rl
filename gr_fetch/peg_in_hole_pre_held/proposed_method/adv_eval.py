"""Adversarial evaluation for FetchPegInHolePreHeldDense-v1.

Loads adversary + protagonist checkpoints and replays N episodes in the
two-phase format used during adv_train.py:

  Phase A (Ha steps)  — adversary acts, trying to push the peg to a hard state.
  Phase P (Hp steps)  — protagonist acts, trying to insert from that hard state.

Per-step diagnostics are printed to the terminal:
  step | phase | d_xy  | d_z   | tilt  | r_lat | r_dep | r_aln | reward

Usage:
    # load iteration 100 (adversary_100.ckpt + protagonist_100.ckpt)
    python adv_eval.py 100

    # with rendering and slow playback
    python adv_eval.py 150 --render --step-delay 0.05 --episodes 3

    # per-step table
    python adv_eval.py 150 --verbose

    # different checkpoint directory
    python adv_eval.py 100 --models-dir /path/to/other/models_adv
"""

from __future__ import annotations

import argparse
import os

import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation

from rl_baselines.policy_based.sac.sac import SAC

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchPegInHolePreHeldDense-v1"
MAX_EPISODE_STEPS = 300

# Column widths for the step table
_HDR = (
    f"{'step':>4}  {'phase':>5}  {'d_xy':>7}  {'d_z':>7}  "
    f"{'tilt°':>6}  {'r_lat':>6}  {'r_dep':>6}  {'r_aln':>6}  {'reward':>8}"
)
_SEP = "-" * len(_HDR)


def _fmt_row(
    step: int,
    phase: str,
    d_xy: float,
    d_z: float,
    tilt_deg: float,
    r_lat: float,
    r_dep: float,
    r_aln: float,
    reward: float,
) -> str:
    return (
        f"{step:>4}  {phase:>5}  {d_xy:>7.4f}  {d_z:>7.4f}  "
        f"{tilt_deg:>6.2f}  {r_lat:>6.3f}  {r_dep:>6.3f}  {r_aln:>6.3f}  {reward:>8.4f}"
    )


def _to_action(agent: SAC, action_tensor: torch.Tensor) -> np.ndarray:
    """Greedy action → numpy array (env requires shape (4,))."""
    return action_tensor.detach().cpu().numpy().flatten()


def run_episode(
    env: gym.Env,
    adversary: SAC,
    protagonist: SAC,
    adversary_horizon: int,
    protagonist_horizon: int,
    verbose: bool = False,
    step_delay: float = 0.0,
) -> dict:
    """Execute one full adversarial episode and return summary stats.

    Parameters
    ----------
    env : gym.Env
        Flattened FetchPegInHolePreHeldDense-v1 environment.
    adversary : SAC
        Loaded adversary policy.
    protagonist : SAC
        Loaded protagonist policy.
    adversary_horizon : int
        Maximum steps for the adversary phase.
    protagonist_horizon : int
        Maximum steps for the protagonist phase.
    verbose : bool
        Print per-step table if True.

    Returns
    -------
    dict
        Summary statistics for the episode.
    """
    obs, _ = env.reset()

    adv_score = 0.0
    prt_score = 0.0
    adv_steps_taken = 0
    prt_steps_taken = 0
    prt_success = False

    # Per-phase accumulators for averages
    adv_d_xy_list, adv_d_z_list, adv_tilt_list = [], [], []
    prt_d_xy_list, prt_d_z_list, prt_tilt_list = [], [], []

    if verbose:
        print(_HDR)
        print(_SEP)

    # ------------------------------------------------------------------
    # Phase A — adversary (greedy)
    # ------------------------------------------------------------------
    state = adversary.state_to_torch(obs)
    step = 0
    episode_done = False

    for _ in range(adversary_horizon):
        step += 1
        action_tensor = adversary.agent.select_greedy_action(state, eval=True)
        env_action = _to_action(adversary, action_tensor)
        obs, reward, terminated, truncated, info = env.step(env_action)
        if step_delay > 0.0:
            time.sleep(step_delay)

        d_xy   = info.get("d_xy", float("nan"))
        d_z    = info.get("d_z", float("nan"))
        tilt   = info.get("orientation_error", float("nan"))
        r_lat  = info.get("reward_lateral", float("nan"))
        r_dep  = info.get("reward_depth", float("nan"))
        r_aln  = info.get("reward_alignment", float("nan"))

        adv_score += reward
        adv_d_xy_list.append(d_xy)
        adv_d_z_list.append(d_z)
        adv_tilt_list.append(tilt)
        adv_steps_taken += 1

        if verbose:
            print(
                _fmt_row(step, "ADV", d_xy, d_z, np.degrees(tilt), r_lat, r_dep, r_aln, reward)
            )

        episode_done = terminated or truncated
        if episode_done:
            break
        state = adversary.state_to_torch(obs)

    adv_terminal_obs = obs.copy()

    if verbose:
        print(_SEP)
        print(
            f"  [Adversary terminal]  d_xy={adv_d_xy_list[-1]:.4f}  "
            f"d_z={adv_d_z_list[-1]:.4f}  "
            f"tilt={np.degrees(adv_tilt_list[-1]):.2f}°  "
            f"adv_score={adv_score:.4f}"
        )
        print(_SEP)

    # ------------------------------------------------------------------
    # Phase P — protagonist (greedy)
    # ------------------------------------------------------------------
    if not episode_done:
        state = protagonist.state_to_torch(adv_terminal_obs)

        for _ in range(protagonist_horizon):
            step += 1
            action_tensor = protagonist.agent.select_greedy_action(state, eval=True)
            env_action = _to_action(protagonist, action_tensor)
            obs, reward, terminated, truncated, info = env.step(env_action)
            if step_delay > 0.0:
                time.sleep(step_delay)

            d_xy   = info.get("d_xy", float("nan"))
            d_z    = info.get("d_z", float("nan"))
            tilt   = info.get("orientation_error", float("nan"))
            r_lat  = info.get("reward_lateral", float("nan"))
            r_dep  = info.get("reward_depth", float("nan"))
            r_aln  = info.get("reward_alignment", float("nan"))

            prt_score += reward
            prt_d_xy_list.append(d_xy)
            prt_d_z_list.append(d_z)
            prt_tilt_list.append(tilt)
            prt_steps_taken += 1

            if terminated:
                prt_success = True

            if verbose:
                marker = " ✓" if prt_success and terminated else ""
                print(
                    _fmt_row(step, "PRT", d_xy, d_z, np.degrees(tilt), r_lat, r_dep, r_aln, reward)
                    + marker
                )

            if terminated or truncated:
                break
            state = protagonist.state_to_torch(obs)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = dict(
        adv_score=adv_score,
        prt_score=prt_score,
        prt_success=prt_success,
        adv_steps=adv_steps_taken,
        prt_steps=prt_steps_taken,
        # Adversary phase final state
        adv_final_d_xy=adv_d_xy_list[-1] if adv_d_xy_list else float("nan"),
        adv_final_d_z=adv_d_z_list[-1] if adv_d_z_list else float("nan"),
        adv_final_tilt_deg=np.degrees(adv_tilt_list[-1]) if adv_tilt_list else float("nan"),
        adv_mean_d_xy=float(np.mean(adv_d_xy_list)) if adv_d_xy_list else float("nan"),
        adv_mean_d_z=float(np.mean(adv_d_z_list)) if adv_d_z_list else float("nan"),
        # Protagonist phase final state
        prt_final_d_xy=prt_d_xy_list[-1] if prt_d_xy_list else float("nan"),
        prt_final_d_z=prt_d_z_list[-1] if prt_d_z_list else float("nan"),
        prt_final_tilt_deg=np.degrees(prt_tilt_list[-1]) if prt_tilt_list else float("nan"),
        prt_min_d_xy=float(np.min(prt_d_xy_list)) if prt_d_xy_list else float("nan"),
        prt_min_d_z=float(np.min(prt_d_z_list)) if prt_d_z_list else float("nan"),
    )

    if verbose:
        print(_SEP)
        print(
            f"  [Protagonist final]   d_xy={summary['prt_final_d_xy']:.4f}  "
            f"d_z={summary['prt_final_d_z']:.4f}  "
            f"tilt={summary['prt_final_tilt_deg']:.2f}°  "
            f"prt_score={prt_score:.4f}  "
            f"success={'YES' if prt_success else 'no'}"
        )

    return summary


def evaluate(
    checkpoint: int,
    adversary_horizon: int,
    protagonist_horizon: int,
    episodes: int,
    render: bool,
    device: str,
    verbose: bool = False,
    step_delay: float = 0.0,
    models_dir: str = "models_adv",
) -> None:
    """Load adversary + protagonist from a shared checkpoint index and evaluate.

    Parameters
    ----------
    checkpoint : int
        Iteration number shared by both checkpoints.
        Loads ``{models_dir}/adversary_{checkpoint}.ckpt`` and
        ``{models_dir}/protagonist_{checkpoint}.ckpt``.
    models_dir : str
        Directory that contains the checkpoint files (default: ``models_adv``).
    """
    adversary_path = os.path.join(models_dir, f"adversary_{checkpoint}")
    protagonist_path = os.path.join(models_dir, f"protagonist_{checkpoint}")

    # Build environments
    render_mode = "human" if render else None
    eval_env = FlattenObservation(
        gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS, render_mode=render_mode)
    )
    # Dummy env for SAC constructor (no training, just shape info)
    dummy_env = FlattenObservation(
        gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
    )

    action_dim = dummy_env.action_space.shape[0]
    target_entropy = -float(action_dim)

    # Adversary — same arch as adv_train.py
    adversary = SAC(
        env=dummy_env,
        network_type="mlp",
        network_arch=[256, 256],
        device=device,
        target_entropy=target_entropy,
        num_q_heads=2,
    )
    adversary.load(model_path=adversary_path)
    print(f"[Adversary]   loaded  {adversary_path}.ckpt")

    # Protagonist — same arch as sac_train.py / adv_train.py
    protagonist = SAC(
        env=dummy_env,
        network_type="mlp",
        network_arch=[512, 512, 512],
        device=device,
        target_entropy=target_entropy,
        num_q_heads=5,
    )
    protagonist.load(model_path=protagonist_path)
    print(f"[Protagonist] loaded  {protagonist_path}.ckpt")
    print()

    all_summaries: list[dict] = []

    for ep in range(episodes):
        print(f"{'=' * len(_HDR)}")
        print(f"  Episode {ep + 1} / {episodes}")
        print(f"{'=' * len(_HDR)}")

        summary = run_episode(
            env=eval_env,
            adversary=adversary,
            protagonist=protagonist,
            adversary_horizon=adversary_horizon,
            protagonist_horizon=protagonist_horizon,
            verbose=verbose,
            step_delay=step_delay,
        )
        all_summaries.append(summary)
        print()

    # ------------------------------------------------------------------
    # Aggregate across all episodes
    # ------------------------------------------------------------------
    n = len(all_summaries)
    if n > 1:
        adv_scores   = [s["adv_score"]           for s in all_summaries]
        prt_scores   = [s["prt_score"]           for s in all_summaries]
        successes    = [s["prt_success"]         for s in all_summaries]
        adv_d_xy     = [s["adv_final_d_xy"]      for s in all_summaries]
        adv_d_z      = [s["adv_final_d_z"]       for s in all_summaries]
        adv_tilt     = [s["adv_final_tilt_deg"]  for s in all_summaries]
        prt_d_xy     = [s["prt_final_d_xy"]      for s in all_summaries]
        prt_d_z      = [s["prt_final_d_z"]       for s in all_summaries]

        print("=" * len(_HDR))
        print(f"  SUMMARY  ({n} episodes)")
        print("=" * len(_HDR))
        print(f"  Adversary score      mean={np.mean(adv_scores):+.3f}  std={np.std(adv_scores):.3f}")
        print(f"  Protagonist score    mean={np.mean(prt_scores):+.3f}  std={np.std(prt_scores):.3f}")
        print(f"  Protagonist success  {sum(successes)}/{n}  ({100*np.mean(successes):.0f}%)")
        print()
        print(f"  Adv terminal state:")
        print(f"    d_xy  mean={np.mean(adv_d_xy):.4f}  (higher = harder for protagonist)")
        print(f"    d_z   mean={np.mean(adv_d_z):.4f}")
        print(f"    tilt  mean={np.mean(adv_tilt):.2f}°")
        print()
        print(f"  Prt final state:")
        print(f"    d_xy  mean={np.mean(prt_d_xy):.4f}")
        print(f"    d_z   mean={np.mean(prt_d_z):.4f}")
        print("=" * len(_HDR))

    dummy_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adversarial evaluation for FetchPegInHolePreHeldDense-v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=int,
        help="Checkpoint iteration to load (e.g. 100 loads adversary_100.ckpt + protagonist_100.ckpt)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models_adv",
        help="Directory containing the checkpoint files",
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
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open MuJoCo viewer (human render mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step table (default: off, only summary is shown)",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between steps, e.g. 0.05 to slow down rendering (default: 0.0)",
    )

    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        adversary_horizon=args.adversary_horizon,
        protagonist_horizon=args.protagonist_horizon,
        episodes=args.episodes,
        render=args.render,
        device=args.device,
        verbose=args.verbose,
        step_delay=args.step_delay,
        models_dir=args.models_dir,
    )

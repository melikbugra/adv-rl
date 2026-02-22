"""
SAC evaluation script for robosuite Lift.

Loads a trained checkpoint and runs visual evaluation episodes.
"""

import argparse

import numpy as np
import robosuite as suite
from gymnasium.envs.registration import EnvSpec
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.wrappers import GymWrapper
from rl_baselines.policy_based.sac import SAC

# =============================================================================
# CONFIGURATION (must match training)
# =============================================================================

ENV_NAME = "Lift"
ROBOT = "Panda"
HORIZON = 500
NETWORK_ARCH = [256, 256, 256]
DEVICE = "cuda:0"
FIXED_INIT = True  # Must match training setting


def make_env(render: bool = False):
    """Create robosuite Lift env wrapped for Gymnasium."""
    kwargs = dict(
        robots=ROBOT,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        reward_scale=0.2,
        horizon=HORIZON,
    )
    if FIXED_INIT:
        kwargs["initialization_noise"] = None
        kwargs["placement_initializer"] = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=None,
            x_range=[0, 0],
            y_range=[0, 0],
            rotation=0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=np.array([0, 0, 0.8]),
            z_offset=0.02,
        )
    env = suite.make(ENV_NAME, **kwargs)
    gym_env = GymWrapper(env, keys=["robot0_proprio-state", "object-state"])
    spec = EnvSpec(f"{ENV_NAME}-{ROBOT}-v0", max_episode_steps=HORIZON)
    gym_env.spec = spec
    gym_env.unwrapped.spec = spec
    return gym_env


def evaluate(device: str, checkpoint: str, episodes: int):
    """Load and evaluate a trained SAC model."""
    env = make_env(render=False)
    eval_env = make_env(render=True)

    model = SAC(
        env=env,
        eval_env=eval_env,
        experience_replay_type="er",
        network_type="mlp",
        network_arch=NETWORK_ARCH,
        device=device,
        env_seed=42,
    )

    model.load(folder="models", checkpoint=checkpoint)
    model.evaluate(
        episodes=episodes, eval_env=eval_env, render=True, print_episode_score=True
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC evaluation for robosuite Lift",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to use (default: {DEVICE})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_avg",
        help="Checkpoint name to load (default: best_avg)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    args = parser.parse_args()
    evaluate(device=args.device, checkpoint=args.checkpoint, episodes=args.episodes)

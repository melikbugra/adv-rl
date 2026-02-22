"""
SAC training script for robosuite NutAssemblySquare with dense reward.

Uses GymWrapper to convert robosuite env to standard Gymnasium interface.
Observation: 64-dim flat (robot proprio 50 + object state 14)
Action: 7-dim continuous [-1, 1] (OSC position + orientation + gripper)
"""

import argparse

import numpy as np
import robosuite as suite
from gymnasium.envs.registration import EnvSpec
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.wrappers import GymWrapper
from rl_baselines.policy_based.sac import SAC

# =============================================================================
# CONFIGURATION
# =============================================================================

ENV_NAME = "NutAssemblySquare"
ROBOT = "Panda"
HORIZON = 500
TOTAL_TIMESTEPS = 5_000_000
NETWORK_ARCH = [512, 512, 512]
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
TAU = 0.002
GAMMA = 0.95
REPLAY_SIZE = 2_000_000
LEARNING_STARTS = 5_000
GRADIENT_STEPS = 1
NUM_Q_HEADS = 5
GRADIENT_CLIPPING = 1.0
DEVICE = "cuda:0"
CHECKPOINT_NAME = "latest"
FIXED_INIT = True  # True: sabit başlangıç (hızlı öğrenme), False: random (robust)


def make_env(render: bool = False):
    """Create robosuite NutAssemblySquare env wrapped for Gymnasium."""
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


def train(timesteps: int, device: str, render_eval: bool):
    """Run SAC training on NutAssemblySquare."""
    env = make_env(render=False)
    eval_env = make_env(render=render_eval)

    action_dim = env.action_space.shape[0]

    model = SAC(
        env=env,
        eval_env=eval_env,
        experience_replay_type="er",
        time_steps=timesteps,
        learning_rate=LEARNING_RATE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gradient_steps=GRADIENT_STEPS,
        network_type="mlp",
        network_arch=NETWORK_ARCH,
        tau=TAU,
        gamma=GAMMA,
        target_entropy=-float(action_dim),
        experience_replay_size=REPLAY_SIZE,
        device=device,
        plot_train_sores=True,
        writing_period=5_000,
        gradient_clipping_max_norm=GRADIENT_CLIPPING,
        env_seed=42,
        evaluation=True,
        eval_episodes=10,
        render=render_eval,
        num_q_heads=NUM_Q_HEADS,
        mlflow_tracking_uri="https://mlflow.melikbugraozcelik.com/",
    )

    model.train()

    save_path = model.save(
        checkpoint=CHECKPOINT_NAME,
        save_optimizer=True,
        save_replay_buffer=False,
    )
    print(f"Model saved: {save_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC training for robosuite NutAssemblySquare",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to use (default: {DEVICE})",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation environment during training",
    )
    args = parser.parse_args()
    train(timesteps=args.timesteps, device=args.device, render_eval=args.render)

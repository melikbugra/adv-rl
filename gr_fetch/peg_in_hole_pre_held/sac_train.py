"""
Direct training script for FetchPegInHolePreHeldDense-v1 with SAC + HER.

PreHeld variant: peg is already grasped, only insertion needs to be learned.
Dense reward: continuous gradient signal (-distance - 0.5 * orientation_error).
"""

import argparse

import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation
from rl_baselines.policy_based.sac import SAC

gym.register_envs(gymnasium_robotics)

# =============================================================================
# CONFIGURATION
# =============================================================================

ENV_ID = "FetchPegInHolePreHeldDense-v1"
MAX_EPISODE_STEPS = 100
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
N_SAMPLED_GOAL = 8
GOAL_STRATEGY = "future"
GRADIENT_CLIPPING = 1.0
DEVICE = "cuda:0"
CHECKPOINT_NAME = "latest"


def make_env(env_id: str, max_episode_steps: int, render_mode: str = None, flatten: bool = False):
    """Create environment with optional rendering and observation flattening."""
    env = gym.make(
        env_id, max_episode_steps=max_episode_steps, render_mode=render_mode
    )
    if flatten:
        env = FlattenObservation(env)
    return env


def train(timesteps: int, device: str, render_eval: bool):
    """Run SAC + HER training on FetchPegInHolePreHeldDense-v1."""
    # Training env: no rendering for speed, flatten Dict obs for regular ER
    env = make_env(ENV_ID, MAX_EPISODE_STEPS, render_mode=None, flatten=True)
    # Eval env: optionally render
    eval_render = "human" if render_eval else None
    eval_env = make_env(ENV_ID, MAX_EPISODE_STEPS, render_mode=eval_render, flatten=True)

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
        # n_sampled_goal=N_SAMPLED_GOAL,
        # goal_selection_strategy=GOAL_STRATEGY,
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


def eval_only(device: str):
    """Load and evaluate a trained model."""
    env = make_env(ENV_ID, MAX_EPISODE_STEPS)
    eval_env = make_env(ENV_ID, MAX_EPISODE_STEPS, render_mode="human")

    model = SAC(
        env=env,
        eval_env=eval_env,
        experience_replay_type="her",
        network_type="mlp",
        network_arch=NETWORK_ARCH,
        device=device,
        n_sampled_goal=N_SAMPLED_GOAL,
        goal_selection_strategy=GOAL_STRATEGY,
    )

    model.load(folder="models", checkpoint="best_avg")
    model.evaluate(
        episodes=10, eval_env=eval_env, render=True, print_episode_score=True
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC + HER training for FetchPegInHolePreHeldDense-v1",
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
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only evaluate best_avg checkpoint",
    )

    args = parser.parse_args()

    if args.eval_only:
        eval_only(device=args.device)
    else:
        train(timesteps=args.timesteps, device=args.device, render_eval=args.render)

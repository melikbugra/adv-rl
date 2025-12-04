"""
Training script for FetchAssembly environment with SAC.
Uses reward normalization wrapper for better learning.
"""

import gymnasium_robotics
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, TransformReward, NormalizeReward
import numpy as np
from rl_baselines.policy_based.sac import SAC


# Custom wrapper to scale rewards
class RewardScaler(gym.RewardWrapper):
    """Scale rewards by a constant factor."""

    def __init__(self, env, scale=10.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


def make_env(max_episode_steps=100, reward_scale=10.0):
    """Create and wrap the environment."""
    env = gym.make("FetchAssemblyDense-v1", max_episode_steps=max_episode_steps)
    env = FlattenObservation(env)
    # Scale rewards: dense reward is ~[-0.5, 0], scale to ~[-5, 0]
    env = RewardScaler(env, scale=reward_scale)
    return env


if __name__ == "__main__":
    # Create environments
    env = make_env(max_episode_steps=100, reward_scale=10.0)
    eval_env = make_env(max_episode_steps=100, reward_scale=10.0)

    action_dim = env.action_space.shape[0]

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Action dim: {action_dim}")

    model = SAC(
        env=env,
        time_steps=3_000_000,
        # === Learning ===
        learning_rate=3e-4,
        learning_starts=10_000,  # More random exploration at start
        batch_size=256,
        gradient_steps=1,
        # === Network - smaller is better for this task ===
        network_type="mlp",
        network_arch=[256, 256],  # Simpler network
        # === SAC Specific ===
        tau=0.005,
        gamma=0.98,  # Slightly lower for 100-step episodes
        target_entropy=-float(action_dim),  # Standard: -dim(action)
        # === Buffer ===
        experience_replay_size=1_000_000,
        # === Misc ===
        device="cuda:0",
        plot_train_sores=True,
        writing_period=10_000,
        gradient_clipping_max_norm=1.0,
        env_seed=42,
        # === Evaluation ===
        evaluation=True,
        eval_episodes=5,
        eval_env=eval_env,
        num_q_heads=2,
    )

    print("\nStarting training...")
    print("=" * 50)
    model.train()

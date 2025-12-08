"""
Evaluation script for FetchAssembly with SAC + HER
Must match the training configuration from sac_train.py
"""

import gymnasium_robotics
import gymnasium as gym
from rl_baselines.policy_based.sac import SAC

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)


# =============================================================================
# CONFIGURATION (must match training)
# =============================================================================

ENV_ID = "FetchAssembly-v1"
MAX_EPISODE_STEPS = 100
DEVICE = "cuda:0"
CHECKPOINT = "curriculum_5_assembly"  # or "last"

# Network architecture (must match training)
NETWORK_ARCH = [512, 1024, 512]


def make_env(env_id: str, max_episode_steps: int = 100, render_mode: str = None):
    """Create environment."""
    return gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)


if __name__ == "__main__":
    # Create environments
    env = make_env(ENV_ID, MAX_EPISODE_STEPS)
    eval_env = make_env(ENV_ID, MAX_EPISODE_STEPS, render_mode="human")

    action_dim = env.action_space.shape[0]

    # Create model with same config as training
    model = SAC(
        env=env,
        eval_env=eval_env,
        experience_replay_type="her",
        network_type="mlp",
        network_arch=NETWORK_ARCH,
        device=DEVICE,
        env_seed=42,
        n_sampled_goal=8,
        goal_selection_strategy="future",
    )

    # Load trained weights
    model.load(folder="models", checkpoint=CHECKPOINT)

    # Evaluate
    model.evaluate(episodes=10, eval_env=eval_env, render=True, print_episode_score=True)

    env.close()
    eval_env.close()

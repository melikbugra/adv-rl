"""
Evaluation script for FetchPegInHolePreHeldDense with SAC + ER (FlattenObservation).
Must match the training configuration from sac_train.py
"""

import gymnasium_robotics
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from rl_baselines.policy_based.sac import SAC

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)


# =============================================================================
# CONFIGURATION (must match training)
# =============================================================================

ENV_ID = "FetchPegInHolePreHeldDense-v1"
MAX_EPISODE_STEPS = 100
NETWORK_ARCH = [512, 512, 512]
NUM_Q_HEADS = 5
DEVICE = "cuda:0"
CHECKPOINT = "best_avg"  # or "best_avg"


def make_env(env_id: str, max_episode_steps: int = 100, render_mode: str = None):
    """Create environment matching training config (flatten=True)."""
    env = gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
    env = FlattenObservation(env)
    return env


if __name__ == "__main__":
    # Create environments
    env = make_env(ENV_ID, MAX_EPISODE_STEPS)
    eval_env = make_env(ENV_ID, MAX_EPISODE_STEPS, render_mode="human")

    action_dim = env.action_space.shape[0]

    # Create model with same config as training
    model = SAC(
        env=env,
        eval_env=eval_env,
        experience_replay_type="er",
        network_type="mlp",
        network_arch=NETWORK_ARCH,
        device=DEVICE,
        env_seed=42,
        num_q_heads=NUM_Q_HEADS,
    )

    # Load trained weights
    model.load(folder="models", checkpoint=CHECKPOINT)

    # Evaluate
    model.evaluate(
        episodes=10, eval_env=eval_env, render=True, print_episode_score=True
    )

    env.close()
    eval_env.close()

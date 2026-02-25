"""
Evaluation script for FetchStirDense-v1 with SAC.
Must match training configuration from sac_train.py.
"""

import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation
from rl_baselines.policy_based.sac import SAC

gym.register_envs(gymnasium_robotics)

# =============================================================================
# CONFIGURATION (must match training)
# =============================================================================

ENV_ID            = "FetchStirDense-v1"
MAX_EPISODE_STEPS = 200
NETWORK_ARCH      = [512, 512, 512]
NUM_Q_HEADS       = 5
DEVICE            = "cuda:0"
CHECKPOINT        = "best_avg"


def make_env(env_id: str, max_episode_steps: int = 200, render_mode: str = None):
    env = gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
    env = FlattenObservation(env)
    return env


if __name__ == "__main__":
    env = make_env(ENV_ID, MAX_EPISODE_STEPS)
    eval_env = make_env(ENV_ID, MAX_EPISODE_STEPS, render_mode="human")

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

    model.load(folder="models", checkpoint=CHECKPOINT)
    model.evaluate(episodes=10, eval_env=eval_env, render=True, print_episode_score=True)

    env.close()
    eval_env.close()

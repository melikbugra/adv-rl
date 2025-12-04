import gymnasium_robotics
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from rl_baselines.policy_based.sac import SAC


class RewardScaler(gym.RewardWrapper):
    """Scale rewards by a constant factor."""

    def __init__(self, env, scale=10.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


env = gym.make("FetchAssemblyDense-v1", max_episode_steps=100)
env = FlattenObservation(env)
action_dim = env.action_space.shape[0]
env = RewardScaler(env, scale=10.0)
eval_env = gym.make("FetchAssemblyDense-v1", max_episode_steps=100, render_mode="human")
eval_env = FlattenObservation(eval_env)
eval_env = RewardScaler(eval_env, scale=10.0)

model = SAC(
    env=env,
    eval_env=eval_env,
    device="cuda:0",
    env_seed=42,
)

model.load(folder="models", checkpoint="last")

model.evaluate(episodes=10, eval_env=eval_env, render=True, print_episode_score=True)

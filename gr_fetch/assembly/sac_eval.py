import gymnasium_robotics
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from rl_baselines.policy_based.sac import SAC

env = gym.make("FetchAssemblyDense-v1", max_episode_steps=100)
env = FlattenObservation(env)
action_dim = env.action_space.shape[0]

eval_env = gym.make("FetchAssemblyDense-v1", max_episode_steps=100, render_mode="human")
eval_env = FlattenObservation(eval_env)

model = SAC(
    env=env,
    eval_env=eval_env,
    device="cuda:0",
)

model.load(folder="models", checkpoint="last")

model.evaluate(episodes=10, eval_env=eval_env, render=True)

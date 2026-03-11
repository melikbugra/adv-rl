"""Adversarial SAC training for FetchPegInHolePreHeldDense-v1 (Baseline).

Adapts the proposed method to use the baseline adversary reward:
    r_A = -scale * V_P(s')

This is the "value-only" baseline that does NOT use the Q-ensemble
uncertainty term (sigma_Q). Two-phase adversarial training loop:

  - Adversary phase (Ha steps): adversary SAC controls the robot arm, moving the peg
    to a hard configuration (away from hole, tilted, etc.)
  - Protagonist phase (Hp steps): protagonist SAC attempts insertion from the hard
    state left by the adversary.

Adversary reward: r = -scale * V̂_protagonist(s')
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import tqdm
from gymnasium.wrappers import FlattenObservation

from rl_baselines.policy_based.sac.sac import SAC
from rl_baselines.utils.base_classes.base_experience_replay import Transition
from rl_baselines.utils import MLFlowLogger

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchPegInHolePreHeldDense-v1"
MAX_EPISODE_STEPS = 300  # adv training env
PRETRAIN_EPISODE_STEPS = 100  # protagonist pretrain env


# ---------------------------------------------------------------------------
# Value estimation (baseline: value only, no uncertainty)
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_protagonist_soft_value(
    state_tensor: torch.Tensor,
    protagonist: SAC,
    sample_count: int = 8,
    use_target: bool = True,
) -> torch.Tensor:
    """Estimate the protagonist's soft value for a batch of observations.

    Parameters
    ----------
    state_tensor : torch.Tensor
        Batch of observations (``[B, obs_dim]``).
    protagonist : SAC
        The protagonist agent that provides actor/critic networks.
    sample_count : int, optional
        Number of action reparameterisation samples, by default ``8``.
    use_target : bool, optional
        Use target critics when available, by default ``True``.

    Returns
    -------
    torch.Tensor
        Estimated soft value ``V_hat`` shaped ``[B, 1]``.
    """
    state_tensor = state_tensor.to(protagonist.device)
    batch_size = state_tensor.shape[0]

    actor_outputs, *_ = protagonist.agent.net(state=state_tensor, actor_pass=True)
    mean, std = actor_outputs[0]
    std = torch.clamp(std, min=1e-8)

    action_dim = mean.shape[-1]
    eps = torch.randn(
        (sample_count, batch_size, action_dim), device=mean.device, dtype=mean.dtype
    )
    z_samples = mean.unsqueeze(0) + std.unsqueeze(0) * eps
    tanh_actions = torch.tanh(z_samples)

    max_action = protagonist.agent.max_action
    if not torch.is_tensor(max_action):
        max_action = torch.tensor(max_action, device=mean.device, dtype=mean.dtype)
    actions = tanh_actions * max_action

    log_prob_gauss = -0.5 * (
        ((z_samples - mean.unsqueeze(0)) / std.unsqueeze(0)) ** 2
        + 2.0 * torch.log(std.unsqueeze(0))
        + np.log(2.0 * np.pi)
    ).sum(-1, keepdim=True)
    log_det = torch.log(1.0 - tanh_actions.pow(2) + 1e-6).sum(-1, keepdim=True)
    log_prob = log_prob_gauss - log_det

    repeated_states = (
        state_tensor.unsqueeze(0)
        .expand(sample_count, batch_size, -1)
        .reshape(sample_count * batch_size, -1)
    )
    repeated_actions = actions.reshape(sample_count * batch_size, -1)

    try:
        if use_target:
            _, _, _, _, _, target_mean, _ = protagonist.agent.net(
                state=repeated_states, action=repeated_actions, target_pass=True
            )
            q_mean = target_mean.reshape(sample_count, batch_size, 1)
        else:
            _, _, online_mean, _, _, _, _ = protagonist.agent.net(
                state=repeated_states, action=repeated_actions, critic_pass=True
            )
            q_mean = online_mean.reshape(sample_count, batch_size, 1)
    except Exception:
        if use_target:
            _, _, _, tgt_q1, tgt_q2 = protagonist.agent.net(
                state=repeated_states, action=repeated_actions, target_pass=True
            )
            q_stack = torch.cat([tgt_q1, tgt_q2], dim=1).reshape(
                sample_count, batch_size, 2
            )
        else:
            _, q1, q2, _, _ = protagonist.agent.net(
                state=repeated_states, action=repeated_actions, critic_pass=True
            )
            q_stack = torch.cat([q1, q2], dim=1).reshape(sample_count, batch_size, 2)
        q_mean = q_stack.mean(dim=2, keepdim=True)

    entropy_coeff = torch.exp(protagonist.agent.log_alpha.detach())
    soft_values = q_mean - entropy_coeff * log_prob
    return soft_values.mean(0)


@torch.no_grad()
def adversary_reward_from_value(
    next_state_np: np.ndarray,
    protagonist: SAC,
    sample_count: int = 8,
    scale: float = 1.0,
) -> float:
    """Return the baseline adversary reward ``r_A = -scale * V_P(s')``.

    Parameters
    ----------
    next_state_np : numpy.ndarray
        Next observation encountered by the environment.
    protagonist : SAC
        Protagonist policy that supplies critic ensembles.
    sample_count : int, optional
        Number of action samples for the soft value amortisation.
    scale : float, optional
        Global multiplier applied to the negative value term.

    Returns
    -------
    float
        The adversary reward scalar.
    """
    state_tensor = torch.tensor(
        next_state_np, dtype=torch.float32, device=protagonist.device
    ).unsqueeze(0)

    v_hat = estimate_protagonist_soft_value(
        state_tensor, protagonist, sample_count=sample_count, use_target=False
    )
    protagonist_value = float(v_hat.item())
    return -scale * protagonist_value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyper-parameters for baseline adversarial peg-in-hole training.

    Same architecture and training parameters as the proposed method,
    but the adversary reward is value-only (no uncertainty term).
    """

    outer_iterations: int = 1000
    adversary_episodes_per_iter: int = 10
    protagonist_episodes_per_iter: int = 10
    adversary_horizon: int = 100  # Ha: adversary step budget
    protagonist_horizon: int = 200  # Hp: protagonist step budget
    learning_rate: float = 1e-4
    network_arch: tuple = (512, 512, 512)  # protagonist architecture (matches proposed)
    num_q_heads: int = 2  # protagonist Q-heads (baseline uses 2)
    batch_size: int = 256
    device: str = "cuda:0"
    tau: float = 0.002
    gamma: float = 0.95
    pretrained_protagonist_path: str | None = None  # load existing checkpoint
    checkpoint_period: int = 10
    models_dir: str = "models_adv"
    mlflow_tracking_uri: str = "https://mlflow.melikbugraozcelik.com/"
    # Baseline reward config
    num_samples: int = 8
    value_reward_scale: float = 1.0


# ---------------------------------------------------------------------------
# Episode outcome container
# ---------------------------------------------------------------------------


@dataclass
class EpisodeOutcome:
    """Structured container for per-episode logging."""

    adversary_score: float
    protagonist_score: float
    protagonist_success: bool
    reward_value_component: float
    last_observation: np.ndarray
    # Peg-in-hole specific (last step of protagonist phase)
    reward_lateral: float = 0.0
    reward_depth: float = 0.0
    reward_alignment: float = 0.0
    d_xy: float = 0.0
    d_z: float = 0.0


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------


class AdvSacTrainer:
    """Coordinate the two-phase adversarial SAC training for peg-in-hole (baseline)."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.mlflow_logger = MLFlowLogger(
            mlflow_tracking_uri=config.mlflow_tracking_uri
        )
        self.total_steps = 0
        self._ensure_model_dirs()
        self._build_envs()
        self._build_agents()
        self._setup_mlflow_run()

    def _ensure_model_dirs(self) -> None:
        os.makedirs(self.config.models_dir, exist_ok=True)

    def _build_envs(self) -> None:
        self.env = FlattenObservation(
            gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
        )
        self.eval_env = FlattenObservation(
            gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
        )

    def _build_agents(self) -> None:
        action_dim = self.env.action_space.shape[0]
        target_entropy = -float(action_dim)

        # Adversary: lightweight, 2 Q-heads, [256,256]
        self.adversary = SAC(
            env=self.env,
            learning_rate=self.config.learning_rate,
            network_type="mlp",
            device=self.config.device,
            batch_size=self.config.batch_size,
            network_arch=[256, 256],
            gradient_steps=1,
            target_entropy=target_entropy,
            experience_replay_size=1_000_000,
            learning_starts=5_000,
            gradient_clipping_max_norm=1.0,
            eval_env=self.eval_env,
            tau=self.config.tau,
            gamma=self.config.gamma,
            num_q_heads=2,
        )

        # Protagonist: same architecture as sac_train.py (512x3, 5 Q-heads)
        protagonist_kwargs = dict(
            env=self.env,
            learning_rate=self.config.learning_rate,
            network_type="mlp",
            device=self.config.device,
            batch_size=self.config.batch_size,
            network_arch=list(self.config.network_arch),
            gradient_steps=1,
            target_entropy=target_entropy,
            experience_replay_size=2_000_000,
            learning_starts=5_000,
            gradient_clipping_max_norm=1.0,
            eval_env=self.eval_env,
            tau=self.config.tau,
            gamma=self.config.gamma,
            num_q_heads=self.config.num_q_heads,
        )

        if self.config.pretrained_protagonist_path is not None:
            self.protagonist = SAC(**protagonist_kwargs)
            self.protagonist.load(
                folder=self.config.pretrained_protagonist_path,
                checkpoint="latest",
            )
            tqdm.tqdm.write(
                f"[Protagonist] Loaded checkpoint from "
                f"'{self.config.pretrained_protagonist_path}'"
            )
        else:
            # Short pretrain so the adversary reward signal is meaningful.
            # Use temporary 100-step envs so the protagonist learns basic
            # insertion before being exposed to the longer adversarial episodes.
            pretrain_env = FlattenObservation(
                gym.make(ENV_ID, max_episode_steps=PRETRAIN_EPISODE_STEPS)
            )
            pretrain_eval_env = FlattenObservation(
                gym.make(ENV_ID, max_episode_steps=PRETRAIN_EPISODE_STEPS)
            )
            pretrain_kwargs = {
                **protagonist_kwargs,
                "env": pretrain_env,
                "eval_env": pretrain_eval_env,
            }
            self.protagonist = SAC(
                **pretrain_kwargs,
                time_steps=300_000,
                plot_train_sores=False,
                writing_period=10_000,
                evaluation=True,
                eval_episodes=5,
                mlflow_tracking_uri=self.config.mlflow_tracking_uri,
            )
            tqdm.tqdm.write("[Protagonist] Starting 300K-step pretrain …")
            self.protagonist.train()
            pretrain_env.close()
            pretrain_eval_env.close()
            tqdm.tqdm.write("[Protagonist] Pretrain complete.")

    def _setup_mlflow_run(self) -> None:
        self.mlflow_logger.define_experiment_and_run(
            params_to_log={
                "env_id": ENV_ID,
                "max_episode_steps": MAX_EPISODE_STEPS,
                "N": self.config.outer_iterations,
                "Ka": self.config.adversary_episodes_per_iter,
                "Kp": self.config.protagonist_episodes_per_iter,
                "Ha": self.config.adversary_horizon,
                "Hp": self.config.protagonist_horizon,
                "device": self.config.device,
                "learning_rate": self.config.learning_rate,
                "network_arch": list(self.config.network_arch),
                "num_q_heads": self.config.num_q_heads,
                "num_samples_K": self.config.num_samples,
                "tau": self.config.tau,
                "gamma": self.config.gamma,
                "value_reward_scale": self.config.value_reward_scale,
                "pretrained_protagonist_path": str(
                    self.config.pretrained_protagonist_path
                ),
                "target_entropy": -float(self.env.action_space.shape[0]),
            },
            env=self.env,
            algo_name="Adv-SAC (Baseline, value-only reward, peg-in-hole)",
            run_name_prefix="peg_in_hole_baseline",
        )

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self) -> None:
        """Run the full adversary / protagonist training schedule."""
        iteration_bar = tqdm.tqdm(
            range(self.config.outer_iterations),
            desc="Adversarial Training (Baseline)",
            position=0,
            bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} [ETA: {remaining}, {rate_fmt}]",
            colour="green",
        )
        for iteration_idx in iteration_bar:
            adv_scores, prt_scores = self._train_adversary_iteration(iteration_idx)
            tqdm.tqdm.write(
                f"[Iter {iteration_idx}] ADV mean score: {np.mean(adv_scores):.3f} | "
                f"PRT greedy mean: {np.mean(prt_scores):.3f}"
            )

            prt_adv_scores, prt_train_scores = self._train_protagonist_iteration(
                iteration_idx
            )
            tqdm.tqdm.write(
                f"[Iter {iteration_idx}] PRT mean env-reward: {np.mean(prt_train_scores):.3f} | "
                f"ADV (greedy) mean: {np.mean(prt_adv_scores):.3f}"
            )

            self.mlflow_logger.log_metric(
                "iteration", float(iteration_idx), step=self.total_steps
            )

            if iteration_idx > 0 and iteration_idx % self.config.checkpoint_period == 0:
                adv_path = os.path.join(
                    self.config.models_dir, f"adversary_{iteration_idx}"
                )
                prt_path = os.path.join(
                    self.config.models_dir, f"protagonist_{iteration_idx}"
                )
                self.adversary.save(save_path=adv_path)
                self.protagonist.save(save_path=prt_path)
                tqdm.tqdm.write(
                    f"[Iter {iteration_idx}] Checkpoint saved → {self.config.models_dir}/"
                )

    # -----------------------------------------------------------------------
    # Per-iteration helpers
    # -----------------------------------------------------------------------

    def _train_adversary_iteration(
        self, iteration_idx: int
    ) -> Tuple[list[float], list[float]]:
        """Execute the adversary-focused phase for one outer iteration."""
        adv_scores: list[float] = []
        protagonist_scores: list[float] = []
        progress_bar = tqdm.tqdm(
            range(self.config.adversary_episodes_per_iter),
            desc=f"Training Adversary (Iter {iteration_idx})",
            leave=False,
            position=1,
            bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
            colour="red",
        )
        for _ in progress_bar:
            outcome = self._run_adversary_episode(iteration_idx)
            adv_scores.append(outcome.adversary_score)
            protagonist_scores.append(outcome.protagonist_score)
            self._log_phase_metrics("adv_train", outcome)
        return adv_scores, protagonist_scores

    def _train_protagonist_iteration(
        self, iteration_idx: int
    ) -> Tuple[list[float], list[float]]:
        """Execute the protagonist-focused phase for one outer iteration."""
        adv_scores: list[float] = []
        protagonist_scores: list[float] = []
        progress_bar = tqdm.tqdm(
            range(self.config.protagonist_episodes_per_iter),
            desc=f"Training Protagonist (Iter {iteration_idx})",
            leave=False,
            position=1,
            bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
            colour="blue",
        )
        for _ in progress_bar:
            outcome = self._run_protagonist_episode(iteration_idx)
            adv_scores.append(outcome.adversary_score)
            protagonist_scores.append(outcome.protagonist_score)
            self._log_phase_metrics("prt_train", outcome)
        return adv_scores, protagonist_scores

    # -----------------------------------------------------------------------
    # Episode runners
    # -----------------------------------------------------------------------

    def _run_adversary_episode(self, iteration_idx: int) -> EpisodeOutcome:
        """Collect one adversary-training episode.

        Adversary acts for Ha steps (learning), then protagonist rolls out greedily
        from the adversary's terminal state to measure difficulty.
        """
        observation, _ = self.env.reset()
        state_tensor = self.adversary.state_to_torch(observation)
        last_obs = observation.copy()

        adv_score = 0.0
        value_component = 0.0
        episode_done = False
        last_info: dict = {}

        for _ in range(self.config.adversary_horizon):
            self.total_steps += 1

            action_tensor = self.adversary.agent.select_action(state_tensor)
            env_action = self._to_env_action(self.adversary, action_tensor)
            next_obs, _, terminated, truncated, info = self.env.step(env_action)

            r_value = adversary_reward_from_value(
                next_state_np=next_obs,
                protagonist=self.protagonist,
                sample_count=self.config.num_samples,
                scale=self.config.value_reward_scale,
            )
            value_component += r_value

            shaped_reward = r_value
            if terminated:
                shaped_reward -= 10.0

            adv_score += shaped_reward
            reward_tensor = torch.tensor([shaped_reward], device=self.adversary.device)
            bootstrap_done = terminated
            episode_done = terminated or truncated
            next_state_tensor = (
                None if bootstrap_done else self.adversary.state_to_torch(next_obs)
            )

            transition = Transition(
                state=state_tensor,
                action=action_tensor,
                next_state=next_state_tensor,
                reward=reward_tensor,
                done=bootstrap_done,
            )
            self.adversary.agent.experience_replay.push(transition=transition)
            if (
                len(self.adversary.agent.experience_replay)
                > self.adversary.agent.learning_starts
            ):
                self.adversary.agent.optimize_model(time_step=iteration_idx)

            state_tensor = next_state_tensor if not episode_done else state_tensor
            last_obs = next_obs.copy()
            last_info = info
            if episode_done:
                break

        # Protagonist greedy rollout from adversary's terminal state
        protagonist_score = 0.0
        protagonist_success = False
        if not episode_done:
            protagonist_score, protagonist_success, prt_info = (
                self._rollout_protagonist_greedy(last_obs)
            )
            if prt_info:
                last_info = prt_info

        return EpisodeOutcome(
            adversary_score=adv_score,
            protagonist_score=protagonist_score,
            protagonist_success=protagonist_success,
            reward_value_component=value_component,
            last_observation=last_obs.copy(),
            reward_lateral=last_info.get("reward_lateral", 0.0),
            reward_depth=last_info.get("reward_depth", 0.0),
            reward_alignment=last_info.get("reward_alignment", 0.0),
            d_xy=last_info.get("d_xy", 0.0),
            d_z=last_info.get("d_z", 0.0),
        )

    def _run_protagonist_episode(self, iteration_idx: int) -> EpisodeOutcome:
        """Run one protagonist-training episode with adversary-in-the-loop.

        Adversary plays greedily for Ha steps (no learning), then protagonist
        learns for Hp steps from the resulting hard state.
        """
        observation, _ = self.env.reset()
        adversary_state = self.adversary.state_to_torch(observation)
        last_obs = observation.copy()

        adv_score = 0.0
        value_component = 0.0
        episode_done = False
        last_info: dict = {}

        # Adversary plays greedily for Ha steps to set the hard starting condition.
        for _ in range(self.config.adversary_horizon):
            self.total_steps += 1

            action_tensor = self.adversary.agent.select_greedy_action(
                adversary_state, eval=True
            )
            env_action = self._to_env_action(self.adversary, action_tensor)
            next_obs, _, terminated, truncated, info = self.env.step(env_action)

            r_value = adversary_reward_from_value(
                next_state_np=next_obs,
                protagonist=self.protagonist,
                sample_count=self.config.num_samples,
                scale=self.config.value_reward_scale,
            )
            value_component += r_value

            shaped_reward = r_value
            if terminated:
                shaped_reward -= 10.0
            adv_score += shaped_reward

            episode_done = terminated or truncated
            adversary_state = (
                self.adversary.state_to_torch(next_obs)
                if not episode_done
                else adversary_state
            )
            last_obs = next_obs.copy()
            last_info = info
            if episode_done:
                break

        adversary_terminal_obs = last_obs.copy()

        # Protagonist learns from the adversary's terminal state.
        protagonist_score = 0.0
        protagonist_success = False
        protagonist_state = self.protagonist.state_to_torch(last_obs)
        for _ in range(self.config.protagonist_horizon):
            self.total_steps += 1
            action_tensor = self.protagonist.agent.select_action(protagonist_state)
            env_action = self._to_env_action(self.protagonist, action_tensor)
            next_obs, env_reward, terminated, truncated, info = self.env.step(
                env_action
            )

            protagonist_score += env_reward
            if terminated:
                protagonist_success = True

            reward_tensor = torch.tensor([env_reward], device=self.protagonist.device)
            bootstrap_done = terminated
            episode_done = terminated or truncated
            next_state_tensor = (
                None if bootstrap_done else self.protagonist.state_to_torch(next_obs)
            )

            transition = Transition(
                state=protagonist_state,
                action=action_tensor,
                next_state=next_state_tensor,
                reward=reward_tensor,
                done=bootstrap_done,
            )
            self.protagonist.agent.experience_replay.push(transition=transition)
            if (
                len(self.protagonist.agent.experience_replay)
                > self.protagonist.agent.learning_starts
            ):
                self.protagonist.agent.optimize_model(time_step=iteration_idx)

            protagonist_state = (
                next_state_tensor if not episode_done else protagonist_state
            )
            last_obs = next_obs.copy()
            last_info = info
            if episode_done:
                break

        return EpisodeOutcome(
            adversary_score=adv_score,
            protagonist_score=protagonist_score,
            protagonist_success=protagonist_success,
            reward_value_component=value_component,
            last_observation=adversary_terminal_obs,
            reward_lateral=last_info.get("reward_lateral", 0.0),
            reward_depth=last_info.get("reward_depth", 0.0),
            reward_alignment=last_info.get("reward_alignment", 0.0),
            d_xy=last_info.get("d_xy", 0.0),
            d_z=last_info.get("d_z", 0.0),
        )

    def _rollout_protagonist_greedy(
        self, start_observation: np.ndarray
    ) -> Tuple[float, bool, dict]:
        """Let the protagonist continue greedily from the adversary's terminal state.

        Returns
        -------
        tuple(float, bool, dict)
            ``(score, success, last_info)``
        """
        score = 0.0
        success = False
        last_info: dict = {}
        state_tensor = self.protagonist.state_to_torch(start_observation)
        for _ in range(self.config.protagonist_horizon):
            self.total_steps += 1
            action_tensor = self.protagonist.agent.select_greedy_action(
                state_tensor, eval=True
            )
            env_action = self._to_env_action(self.protagonist, action_tensor)
            next_obs, env_reward, terminated, truncated, info = self.env.step(
                env_action
            )
            score += env_reward
            last_info = info
            if terminated:
                success = True
            if terminated or truncated:
                break
            state_tensor = self.protagonist.state_to_torch(next_obs)
        return score, success, last_info

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log_phase_metrics(self, phase_prefix: str, outcome: EpisodeOutcome) -> None:
        """Send scalar metrics to MLflow."""
        self.mlflow_logger.log_metric(
            f"{phase_prefix}_adv_score",
            outcome.adversary_score,
            step=self.total_steps,
        )
        self.mlflow_logger.log_metric(
            f"{phase_prefix}_prt_score",
            outcome.protagonist_score,
            step=self.total_steps,
        )
        self.mlflow_logger.log_metric(
            f"{phase_prefix}_adv_score_r_v_hat",
            outcome.reward_value_component,
            step=self.total_steps,
        )
        # Peg-in-hole specific decomposed reward metrics (from info dict)
        self.mlflow_logger.log_metric(
            f"{phase_prefix}_reward_lateral",
            outcome.reward_lateral,
            step=self.total_steps,
        )
        self.mlflow_logger.log_metric(
            f"{phase_prefix}_reward_depth",
            outcome.reward_depth,
            step=self.total_steps,
        )
        self.mlflow_logger.log_metric(
            f"{phase_prefix}_reward_alignment",
            outcome.reward_alignment,
            step=self.total_steps,
        )
        self.mlflow_logger.log_metric(
            f"{phase_prefix}_d_xy", outcome.d_xy, step=self.total_steps
        )
        self.mlflow_logger.log_metric(
            f"{phase_prefix}_d_z", outcome.d_z, step=self.total_steps
        )

        if phase_prefix == "adv_train":
            self.mlflow_logger.log_metric(
                "adv_train_adv_actor_loss",
                self.adversary.agent.writer.avg_actor_loss,
                step=self.total_steps,
            )
            self.mlflow_logger.log_metric(
                "adv_train_adv_critic_loss",
                self.adversary.agent.writer.avg_critic_loss,
                step=self.total_steps,
            )
        elif phase_prefix == "prt_train":
            self.mlflow_logger.log_metric(
                "prt_train_prt_actor_loss",
                self.protagonist.agent.writer.avg_actor_loss,
                step=self.total_steps,
            )
            self.mlflow_logger.log_metric(
                "prt_train_prt_critic_loss",
                self.protagonist.agent.writer.avg_critic_loss,
                step=self.total_steps,
            )

    # -----------------------------------------------------------------------
    # Action conversion
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_env_action(agent: SAC, action_tensor: torch.Tensor) -> np.ndarray:
        """Convert policy output to a numpy array for env.step().

        The peg-in-hole env's _set_action() asserts action.shape == (4,),
        so we return a numpy array rather than a list.
        """
        if agent.agent.action_type == "discrete":
            return action_tensor.item()
        return action_tensor.detach().cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def train(config: TrainingConfig | None = None) -> None:
    """Instantiate :class:`AdvSacTrainer` and start baseline adversarial training."""
    trainer = AdvSacTrainer(config or TrainingConfig())
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline Adversarial SAC training for FetchPegInHolePreHeldDense-v1"
    )
    parser.add_argument(
        "--outer-iterations",
        type=int,
        default=1000,
        help="Number of outer adversarial iterations (default: 1000)",
    )
    parser.add_argument(
        "--adversary-horizon",
        type=int,
        default=100,
        help="Adversary step budget Ha per episode (default: 100)",
    )
    parser.add_argument(
        "--protagonist-horizon",
        type=int,
        default=200,
        help="Protagonist step budget Hp per episode (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device (default: cuda:0)",
    )
    parser.add_argument(
        "--pretrained-protagonist-path",
        type=str,
        default=None,
        help="Folder of an existing protagonist checkpoint to skip pretraining",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models_adv",
        help="Directory for saving adversarial checkpoints (default: models_adv)",
    )
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        default=10,
        help="Save checkpoints every N iterations (default: 10)",
    )
    args = parser.parse_args()

    cfg = TrainingConfig(
        outer_iterations=args.outer_iterations,
        adversary_horizon=args.adversary_horizon,
        protagonist_horizon=args.protagonist_horizon,
        device=args.device,
        pretrained_protagonist_path=args.pretrained_protagonist_path,
        models_dir=args.models_dir,
        checkpoint_period=args.checkpoint_period,
    )
    train(cfg)

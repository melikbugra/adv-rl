"""Cleaned Adv-SAC training script.

This module mirrors the original ``adv.py`` logic but introduces
well-documented, clean-code friendly abstractions. It keeps behaviour
compatible while providing Sphinx-style docstrings and clearer naming.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import tqdm

from rl_baselines.policy_based.sac.sac import SAC
from rl_baselines.utils.base_classes.base_experience_replay import Transition
from rl_baselines.utils import MLFlowLogger
import continuous_maze_env  # Needed to register the maze environments


CSV_HEADER = [
    "iter",
    "phase",
    "episode",
    "adv_last_x",
    "adv_last_y",
    "protagonist_success",
]


class EpisodeCSVLogger:
    """Append adversary endpoints to a CSV file safely.

    Parameters
    ----------
    csv_path : str
            Relative or absolute path of the CSV file.
    reset_on_start : bool, optional
            Truncate the CSV the moment logging starts, by default ``True``.
    """

    def __init__(self, csv_path: str, reset_on_start: bool = True):
        self.csv_path = csv_path
        directory = os.path.dirname(csv_path) or "."
        os.makedirs(directory, exist_ok=True)
        if reset_on_start:
            self._write_header()
        else:
            self._ensure_header()

    def _ensure_header(self) -> None:
        """Write the header once if the file is empty or absent."""

        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            self._write_header()

    def _write_header(self) -> None:
        """Overwrite the file and seed it with the CSV header."""

        with open(self.csv_path, mode="w", newline="") as file:
            csv.writer(file).writerow(CSV_HEADER)

    def log(
        self,
        iter_idx: int,
        phase: str,
        episode_idx: int,
        last_obs: np.ndarray,
        protagonist_success: bool,
    ) -> None:
        """Write a single endpoint row.

        Parameters
        ----------
        iter_idx : int
                Outer loop iteration identifier.
        phase : str
                Either ``"adv"`` or ``"prt"`` to denote the training block.
        episode_idx : int
                Episode index within the phase.
        last_obs : numpy.ndarray
                Observation whose first two entries encode the adversary position.
        protagonist_success : bool
                Whether the protagonist completed the maze successfully.
        """

        try:
            x_val = (
                float(last_obs[0])
                if last_obs is not None and len(last_obs) > 0
                else float("nan")
            )
            y_val = (
                float(last_obs[1])
                if last_obs is not None and len(last_obs) > 1
                else float("nan")
            )
            with open(self.csv_path, mode="a", newline="") as file:
                csv.writer(file).writerow(
                    [
                        int(iter_idx),
                        phase,
                        int(episode_idx),
                        x_val,
                        y_val,
                        bool(protagonist_success),
                    ]
                )
        except Exception as exc:  # pragma: no cover - logging best-effort only
            tqdm.tqdm.write(f"[CSV] log error: {exc}")


class RunningNorm:
    """Compute a streaming z-score with exponential moving averages.

    Parameters
    ----------
    decay : float, optional
            Decay factor for the exponential moving averages, by default ``0.99``.
    eps : float, optional
            Numerical guard added to the denominator, by default ``1e-6``.
    """

    def __init__(self, decay: float = 0.99, eps: float = 1e-6):
        self.m = 0.0
        self.v = 1.0
        self.ready = False
        self.decay = decay
        self.eps = eps

    def update(self, value: float) -> None:
        """Update the running mean/variance with a new measurement."""

        value = float(value)
        if not self.ready:
            self.m = value
            self.v = 1.0
            self.ready = True
            return
        self.m = self.decay * self.m + (1.0 - self.decay) * value
        self.v = self.decay * self.v + (1.0 - self.decay) * (value - self.m) ** 2

    def z(self, value: float) -> float:
        """Return the z-score of ``value`` under the current statistics."""

        return (float(value) - self.m) / (self.v**0.5 + self.eps)


VALUE_STATS = RunningNorm(decay=0.99)
SIGMA_STATS = RunningNorm(decay=0.995)


@torch.no_grad()
def estimate_soft_value_and_uncertainty(
    state_tensor: torch.Tensor,
    protagonist: SAC,
    sample_count: int = 8,
    use_target: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate soft value and critic ensemble dispersion for a batch.

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
    tuple(torch.Tensor, torch.Tensor, torch.Tensor)
            ``(V_hat, sigma_Q, mean_Q)`` each shaped ``[B, 1]``.
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
            _, _, _, _, _, target_mean, target_std = protagonist.agent.net(
                state=repeated_states, action=repeated_actions, target_pass=True
            )
            q_mean = target_mean.reshape(sample_count, batch_size, 1)
            q_std = (
                target_std if target_std is not None else torch.zeros_like(target_mean)
            ).reshape(sample_count, batch_size, 1)
        else:
            _, _, online_mean, online_std, _, _, _ = protagonist.agent.net(
                state=repeated_states, action=repeated_actions, critic_pass=True
            )
            q_mean = online_mean.reshape(sample_count, batch_size, 1)
            q_std = (
                online_std if online_std is not None else torch.zeros_like(online_mean)
            ).reshape(sample_count, batch_size, 1)
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
        q_std = q_stack.std(dim=2, keepdim=True, unbiased=False)

    entropy_coeff = torch.exp(protagonist.agent.log_alpha.detach())
    soft_values = q_mean - entropy_coeff * log_prob
    v_hat = soft_values.mean(0)
    sigma_q = q_std.mean(0)
    q_mean_raw = q_mean.mean(0)
    return v_hat, sigma_q, q_mean_raw


@torch.no_grad()
def adversary_reward_qensemble(
    next_state_np: np.ndarray,
    protagonist: SAC,
    sample_count: int = 8,
    lambda_value: float = 0.5,
    lambda_sigma: float = 1.0,
) -> Tuple[float, float]:
    """Compute the adversary reward using value suppression and uncertainty boosts.

    Parameters
    ----------
    next_state_np : numpy.ndarray
            Next observation encountered by the environment.
    protagonist : SAC
            Protagonist policy that supplies critic ensembles.
    sample_count : int, optional
            Number of Monte-Carlo samples for ``estimate_soft_value_and_uncertainty``.
    lambda_value : float, optional
            Scaling weight for the (negative) value term.
    lambda_sigma : float, optional
            Scaling weight for the positive uncertainty term.

    Returns
    -------
    tuple(float, float)
            ``(r_value, r_sigma)`` components before summation.
    """

    device = protagonist.device
    state_tensor = torch.tensor(
        next_state_np, dtype=torch.float32, device=device
    ).unsqueeze(0)

    v_hat, sigma_q, mean_q = estimate_soft_value_and_uncertainty(
        state_tensor, protagonist, sample_count=sample_count, use_target=True
    )

    v_value = float(v_hat.item())
    sigma_value = float(sigma_q.item())
    mean_value = float(torch.abs(mean_q).item())

    cv_denom_constant = 5.0
    denom = (mean_value * mean_value + cv_denom_constant * cv_denom_constant) ** 0.5
    coefficient_of_variation = sigma_value / (denom + 1e-8)

    VALUE_STATS.update(v_value)
    SIGMA_STATS.update(coefficient_of_variation)

    z_value = VALUE_STATS.z(v_value)

    sigma_std = (SIGMA_STATS.v**0.5) + SIGMA_STATS.eps
    sigma_margin = 0.3
    sigma_clip = 2.5
    positive_z = max(
        0.0, (coefficient_of_variation - SIGMA_STATS.m) / sigma_std - sigma_margin
    )
    positive_z = min(positive_z, sigma_clip)

    reward_value = -lambda_value * z_value
    reward_sigma = lambda_sigma * positive_z
    return reward_value, reward_sigma


@dataclass
class TrainingConfig:
    """Container for hyper-parameters required by the training loop."""

    level_name: str = "level_three"
    max_steps: int = 400
    outer_iterations: int = 2006
    adversary_episodes_per_iter: int = 10
    protagonist_episodes_per_iter: int = 10
    learning_rate: float = 3e-4
    network_arch: Tuple[int, int] = (256, 512, 256)
    device: str = "cuda:1"
    batch_size: int = 256
    network_type: str = "mlp"
    env_seed: int | None = None
    tau: float = 0.005
    target_entropy_scale: float = 1.0
    num_q_heads: int = 6
    sigma_samples: int = 8
    base_lambda_v: float = 1.0
    base_lambda_sigma: float = 0.35
    value_ramp_fraction: float = 1.0
    sigma_warmup_fraction: float = 0.5
    sigma_decay_fraction: float = 0.40
    sigma_min_fraction: float = 0.1   
    csv_path: str = "heatmap/adv_prt_endpoints.csv"
    reset_csv_on_start: bool = True
    mlflow_tracking_uri: str = "https://mlflow.melikbugraozcelik.com/"
    checkpoint_period: int = 5


class RewardScheduler:
    """Time-dependent weighting for the value and uncertainty components."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        total_steps_per_iter = (
            config.adversary_episodes_per_iter + config.protagonist_episodes_per_iter
        ) * config.max_steps
        self.total_nominal_steps = max(
            1, config.outer_iterations * total_steps_per_iter
        )
        self.value_ramp_steps = max(
            1, int(self.total_nominal_steps * config.value_ramp_fraction)
        )
        self.sigma_warmup_steps = max(
            1, int(self.total_nominal_steps * config.sigma_warmup_fraction)
        )
        self.sigma_decay_steps = max(
            1, int(self.total_nominal_steps * config.sigma_decay_fraction)
        )

    def value_weight(self, total_steps: int) -> float:
        """Smoothly ramp the negative value multiplier."""

        progress = 1.0 - np.exp(-float(total_steps) / float(self.value_ramp_steps))
        return float(self.config.base_lambda_v * progress)

    def sigma_weight(self, total_steps: int, use_min_floor: bool) -> float:
        """Schedule the uncertainty weight with optional floor protection."""

        if total_steps < self.sigma_warmup_steps:
            return float(self.config.base_lambda_sigma)

        elapsed = float(total_steps - self.sigma_warmup_steps)
        decay = np.exp(-elapsed / float(self.sigma_decay_steps))
        if use_min_floor:
            min_frac = self.config.sigma_min_fraction
            return float(
                self.config.base_lambda_sigma * (min_frac + (1.0 - min_frac) * decay)
            )
        return float(self.config.base_lambda_sigma * decay)


@dataclass
class EpisodeOutcome:
    """Structured container for per-episode logging."""

    adversary_score: float
    protagonist_score: float
    protagonist_success: bool
    reward_value_component: float
    reward_sigma_component: float
    lambda_value: float
    lambda_sigma: float
    last_observation: np.ndarray


class AdvSacTrainer:
    """Coordinate the two-phase adversarial SAC training procedure."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scheduler = RewardScheduler(config)
        self.csv_logger = EpisodeCSVLogger(
            config.csv_path, reset_on_start=config.reset_csv_on_start
        )
        self.mlflow_logger = MLFlowLogger(
            mlflow_tracking_uri=config.mlflow_tracking_uri
        )
        self._ensure_model_dirs()
        self.adversary_horizon = config.max_steps // 2
        self.protagonist_horizon = self.adversary_horizon
        self.total_steps = 0
        self._build_envs()
        self._build_agents()
        self._setup_mlflow_run()

    @staticmethod
    def _ensure_model_dirs() -> None:
        """Create model output folders ahead of checkpointing."""

        for directory in ("models", "level_three_models"):
            os.makedirs(directory, exist_ok=True)

    def _build_envs(self) -> None:
        """Instantiate the training/evaluation environments."""

        env_kwargs = dict(
            level=self.config.level_name,
            max_steps=self.config.max_steps,
            random_start=False,
            render_mode=None,
            dense_reward=True,
        )
        self.env = gym.make("ContinuousMaze-v0", **env_kwargs)
        self.eval_env = gym.make("ContinuousMaze-v0", **env_kwargs)
        self.pretrain_env = gym.make("ContinuousMaze-v0", **env_kwargs)

    def _build_agents(self) -> None:
        """Create SAC agents for adversary and protagonist roles."""

        action_dim = self.env.action_space.shape[0]
        target_entropy = -float(action_dim) * self.config.target_entropy_scale

        self.adversary = SAC(
            env=self.env,
            learning_rate=self.config.learning_rate,
            network_type=self.config.network_type,
            device=self.config.device,
            batch_size=self.config.batch_size,
            network_arch=list(self.config.network_arch),
            env_seed=self.config.env_seed,
            gradient_steps=1,
            target_entropy=target_entropy,
            experience_replay_size=1_000_000,
            learning_starts=5_000,
            gradient_clipping_max_norm=1.0,
            eval_env=self.eval_env,
            tau=self.config.tau,
            num_q_heads=2,
        )

        self.protagonist = SAC(
            env=self.pretrain_env,
            time_steps=50_000,
            learning_rate=self.config.learning_rate,
            network_type=self.config.network_type,
            device=self.config.device,
            batch_size=self.config.batch_size,
            network_arch=list(self.config.network_arch),
            plot_train_sores=False,
            writing_period=10_000,
            tau=self.config.tau,
            evaluation=True,
            gradient_steps=1,
            target_entropy=target_entropy,
            experience_replay_size=1_000_000,
            learning_starts=5_000,
            gradient_clipping_max_norm=1.0,
            env_seed=self.config.env_seed,
            eval_episodes=1,
            eval_env=self.eval_env,
            num_q_heads=self.config.num_q_heads,
        )
        self.protagonist.train()

    def _setup_mlflow_run(self) -> None:
        """Initialize the MLflow run with static metadata."""

        self.mlflow_logger.define_experiment_and_run(
            params_to_log={
                "max_steps": self.config.max_steps,
                "N": self.config.outer_iterations,
                "Ka": self.config.adversary_episodes_per_iter,
                "Kp": self.config.protagonist_episodes_per_iter,
                "Ha": self.adversary_horizon,
                "Hp": self.protagonist_horizon,
                "device": self.config.device,
                "learning_rate": self.config.learning_rate,
                "network_type": self.config.network_type,
                "network_arch": list(self.config.network_arch),
                "num_q_heads": self.config.num_q_heads,
                "sigma_samples_K": self.config.sigma_samples,
                "target_entropy": -float(self.env.action_space.shape[0])
                * self.config.target_entropy_scale,
            },
            env=self.env,
            algo_name="Adv-SAC (Q-ensemble reward)",
            run_name_prefix=self.config.level_name,
        )

    def train(self) -> None:
        """Run the full adversary/protagonist training schedule."""

        iteration_bar = tqdm.tqdm(
            range(self.config.outer_iterations),
            desc="Training Progress",
            position=0,
            bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} [ETA: {remaining}, {rate_fmt}]",
            colour="green",
        )
        for iteration_idx in iteration_bar:
            adv_scores, prt_scores = self._train_adversary_iteration(iteration_idx)
            tqdm.tqdm.write(
                f"[Iter {iteration_idx}] ADV mean score: {np.mean(adv_scores):.3f} | PRT test mean: {np.mean(prt_scores):.3f}"
            )

            prt_adv_scores, prt_train_scores = self._train_protagonist_iteration(
                iteration_idx
            )
            tqdm.tqdm.write(
                f"[Iter {iteration_idx}] PRT mean env-reward: {np.mean(prt_train_scores):.3f} | ADV (greedy) mean: {np.mean(prt_adv_scores):.3f}"
            )

            self.mlflow_logger.log_metric(
                "iteration", iteration_idx / 100.0, step=self.total_steps
            )

            if iteration_idx % self.config.checkpoint_period == 0:
                self.adversary.save(
                    save_path=f"level_three_models/adversary_sac_{iteration_idx}"
                )
                self.protagonist.save(
                    save_path=f"level_three_models/protagonist_sac_{iteration_idx}"
                )

    def _train_adversary_iteration(
        self, iteration_idx: int
    ) -> Tuple[list[float], list[float]]:
        """Execute the adversary-focused phase for one iteration."""

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
        for episode_idx in progress_bar:
            outcome = self._run_adversary_episode(iteration_idx)
            adv_scores.append(outcome.adversary_score)
            protagonist_scores.append(outcome.protagonist_score)
            self._log_episode_endpoint(iteration_idx, "adv", episode_idx, outcome)
            self._log_phase_metrics("adv_train", outcome)
        return adv_scores, protagonist_scores

    def _train_protagonist_iteration(
        self, iteration_idx: int
    ) -> Tuple[list[float], list[float]]:
        """Execute the protagonist-focused phase for one iteration."""

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
        for episode_idx in progress_bar:
            outcome = self._run_protagonist_episode(iteration_idx)
            adv_scores.append(outcome.adversary_score)
            protagonist_scores.append(outcome.protagonist_score)
            self._log_episode_endpoint(iteration_idx, "prt", episode_idx, outcome)
            self._log_phase_metrics("prt_train", outcome)
        return adv_scores, protagonist_scores

    def _run_adversary_episode(self, iteration_idx: int) -> EpisodeOutcome:
        """Collect one adversary-training episode with the new reward."""

        observation, _ = self.env.reset(seed=self.config.env_seed)
        state_tensor = self.adversary.state_to_torch(observation)
        last_obs = observation.copy()

        adv_score = 0.0
        value_component = 0.0
        sigma_component = 0.0
        lambda_value = 0.0
        lambda_sigma = 0.0
        episode_done = False

        for _ in range(self.adversary_horizon):
            self.total_steps += 1
            lambda_sigma = self.scheduler.sigma_weight(
                self.total_steps, use_min_floor=True
            )
            lambda_value = self.scheduler.value_weight(self.total_steps)

            action_tensor = self.adversary.agent.select_action(state_tensor)
            env_action = self._to_env_action(self.adversary, action_tensor)
            next_obs, _, terminated, truncated, _ = self.env.step(env_action)

            r_value, r_sigma = adversary_reward_qensemble(
                next_state_np=next_obs,
                protagonist=self.protagonist,
                sample_count=self.config.sigma_samples,
                lambda_value=lambda_value,
                lambda_sigma=lambda_sigma,
            )
            value_component += r_value
            sigma_component += r_sigma

            shaped_reward = r_value + r_sigma
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
            if episode_done:
                break

        protagonist_score = 0.0
        protagonist_success = False
        if not episode_done:
            protagonist_score, protagonist_success = self._rollout_protagonist_greedy(
                last_obs
            )

        return EpisodeOutcome(
            adversary_score=adv_score,
            protagonist_score=protagonist_score,
            protagonist_success=protagonist_success,
            reward_value_component=value_component,
            reward_sigma_component=sigma_component,
            lambda_value=lambda_value,
            lambda_sigma=lambda_sigma,
            last_observation=last_obs.copy(),
        )

    def _run_protagonist_episode(self, iteration_idx: int) -> EpisodeOutcome:
        """Run one protagonist-training episode with adversary-in-the-loop."""

        observation, _ = self.env.reset(seed=self.config.env_seed)
        adversary_state = self.adversary.state_to_torch(observation)
        last_obs = observation.copy()

        adv_score = 0.0
        value_component = 0.0
        sigma_component = 0.0
        lambda_value = 0.0
        lambda_sigma = 0.0
        episode_done = False

        # Adversary plays greedily for Ha steps to set the starting condition.
        for _ in range(self.adversary_horizon):
            self.total_steps += 1
            lambda_sigma = self.scheduler.sigma_weight(
                self.total_steps, use_min_floor=False
            )
            lambda_value = self.scheduler.value_weight(self.total_steps)

            action_tensor = self.adversary.agent.select_greedy_action(
                adversary_state, eval=True
            )
            env_action = self._to_env_action(self.adversary, action_tensor)
            next_obs, _, terminated, truncated, _ = self.env.step(env_action)

            r_value, r_sigma = adversary_reward_qensemble(
                next_state_np=next_obs,
                protagonist=self.protagonist,
                sample_count=self.config.sigma_samples,
                lambda_value=lambda_value,
                lambda_sigma=lambda_sigma,
            )
            value_component += r_value
            sigma_component += r_sigma

            shaped_reward = r_value + r_sigma
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
            if episode_done:
                break

        adversary_terminal_obs = last_obs.copy()

        protagonist_score = 0.0
        protagonist_success = False
        protagonist_state = self.protagonist.state_to_torch(last_obs)
        for _ in range(self.protagonist_horizon):
            self.total_steps += 1
            action_tensor = self.protagonist.agent.select_action(protagonist_state)
            env_action = self._to_env_action(self.protagonist, action_tensor)
            next_obs, env_reward, terminated, truncated, _ = self.env.step(env_action)

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
            if episode_done:
                break

        return EpisodeOutcome(
            adversary_score=adv_score,
            protagonist_score=protagonist_score,
            protagonist_success=protagonist_success,
            reward_value_component=value_component,
            reward_sigma_component=sigma_component,
            lambda_value=lambda_value,
            lambda_sigma=lambda_sigma,
            last_observation=adversary_terminal_obs,
        )

    def _rollout_protagonist_greedy(
        self, start_observation: np.ndarray
    ) -> Tuple[float, bool]:
        """Let the protagonist continue greedily from the adversary's state."""

        score = 0.0
        success = False
        state_tensor = self.protagonist.state_to_torch(start_observation)
        for _ in range(self.protagonist_horizon):
            self.total_steps += 1
            action_tensor = self.protagonist.agent.select_greedy_action(
                state_tensor, eval=True
            )
            env_action = self._to_env_action(self.protagonist, action_tensor)
            next_obs, env_reward, terminated, truncated, _ = self.env.step(env_action)
            score += env_reward
            if terminated:
                success = True
            if terminated or truncated:
                break
            state_tensor = self.protagonist.state_to_torch(next_obs)
        return score, success

    def _log_episode_endpoint(
        self, iteration_idx: int, phase: str, episode_idx: int, outcome: EpisodeOutcome
    ) -> None:
        """Persist adversary endpoints for later heat-map visualisation."""

        self.csv_logger.log(
            iter_idx=iteration_idx,
            phase=phase,
            episode_idx=episode_idx,
            last_obs=outcome.last_observation,
            protagonist_success=outcome.protagonist_success,
        )

    def _log_phase_metrics(self, phase_prefix: str, outcome: EpisodeOutcome) -> None:
        """Send scalar metrics to MLflow."""

        self.mlflow_logger.log_metric(
            f"{phase_prefix}_adv_score", outcome.adversary_score, step=self.total_steps
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
        self.mlflow_logger.log_metric(
            f"{phase_prefix}_adv_score_r_sigmaQ",
            outcome.reward_sigma_component,
            step=self.total_steps,
        )
        self.mlflow_logger.log_metric(
            "lambda_v_eff", outcome.lambda_value, step=self.total_steps
        )
        self.mlflow_logger.log_metric(
            "lambda_sigma_eff", outcome.lambda_sigma, step=self.total_steps
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

    @staticmethod
    def _to_env_action(agent: SAC, action_tensor: torch.Tensor):
        """Convert policy outputs to Gym-compatible actions."""

        if agent.agent.action_type == "discrete":
            return action_tensor.item()
        return action_tensor.detach().cpu().numpy().flatten().tolist()


def train(config: TrainingConfig | None = None) -> None:
    """Entrypoint that instantiates :class:`AdvSacTrainer` and starts training.

    Parameters
    ----------
    config : TrainingConfig | None, optional
            Custom configuration override. Defaults to :class:`TrainingConfig`.
    """

    trainer = AdvSacTrainer(config or TrainingConfig())
    trainer.train()


if __name__ == "__main__":
    train()

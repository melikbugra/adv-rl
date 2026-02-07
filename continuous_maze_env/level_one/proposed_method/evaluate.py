"""Evaluation script for trained Adversary-Protagonist models.

This script loads trained adversary and protagonist checkpoints and evaluates
the protagonist's performance when starting from adversary-chosen positions.

Usage:
    python evaluate.py --adv-checkpoint 100 --prt-checkpoint 100 --episodes 50
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import tqdm

from rl_baselines.policy_based.sac.sac import SAC
import continuous_maze_env  # Register maze environments

# Import TrainingConfig to get Ha, Hp, max_steps
from train import TrainingConfig


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""

    adv_checkpoint: int
    prt_checkpoint: int
    episodes: int = 100
    device: str = "cuda:0"
    render: bool = False
    seed: int | None = None
    output_csv: str | None = None


@dataclass
class EpisodeResult:
    """Results from a single evaluation episode."""

    protagonist_score: float
    protagonist_success: bool
    adversary_endpoint_x: float
    adversary_endpoint_y: float
    adversary_steps: int
    protagonist_steps: int


class Evaluator:
    """Evaluate trained adversary-protagonist model pairs."""

    def __init__(self, config: EvalConfig, train_config: TrainingConfig | None = None):
        self.config = config
        self.train_config = train_config or TrainingConfig()

        # Get Ha, Hp from train config
        self.adversary_horizon = self.train_config.max_steps // 2
        self.protagonist_horizon = self.adversary_horizon

        self._build_env()
        self._load_agents()

    def _build_env(self) -> None:
        """Create the evaluation environment."""
        render_mode = "human" if self.config.render else None
        self.env = gym.make(
            "ContinuousMaze-v0",
            level=self.train_config.level_name,
            max_steps=self.train_config.max_steps,
            random_start=False,
            render_mode=render_mode,
            dense_reward=True,
        )

    def _load_agents(self) -> None:
        """Load adversary and protagonist from checkpoints."""
        action_dim = self.env.action_space.shape[0]
        target_entropy = -float(action_dim) * self.train_config.target_entropy_scale

        # Get script directory for relative model paths
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load adversary (2 Q-heads as in training)
        adv_path = os.path.join(
            script_dir,
            f"{self.train_config.level_name}_models/adversary_sac_{self.config.adv_checkpoint}.ckpt",
        )
        self.adversary = SAC(
            env=self.env,
            learning_rate=self.train_config.learning_rate,
            network_type=self.train_config.network_type,
            device=self.config.device,
            batch_size=self.train_config.batch_size,
            network_arch=list(self.train_config.network_arch),
            target_entropy=target_entropy,
            num_q_heads=2,
        )
        self._load_checkpoint(self.adversary, adv_path)
        print(f"Loaded adversary from: {adv_path}")

        # Load protagonist (6 Q-heads as in training)
        prt_path = os.path.join(
            script_dir,
            f"{self.train_config.level_name}_models/protagonist_sac_{self.config.prt_checkpoint}.ckpt",
        )
        self.protagonist = SAC(
            env=self.env,
            learning_rate=self.train_config.learning_rate,
            network_type=self.train_config.network_type,
            device=self.config.device,
            batch_size=self.train_config.batch_size,
            network_arch=list(self.train_config.network_arch),
            target_entropy=target_entropy,
            num_q_heads=self.train_config.num_q_heads,
        )
        self._load_checkpoint(self.protagonist, prt_path)
        print(f"Loaded protagonist from: {prt_path}")

    def _load_checkpoint(self, agent: SAC, ckpt_path: str) -> None:
        """Load a checkpoint file directly into the agent."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        agent.load(model_path=ckpt_path)

    def _to_env_action(self, agent: SAC, action_tensor: torch.Tensor):
        """Convert policy outputs to Gym-compatible actions."""
        if agent.agent.action_type == "discrete":
            return action_tensor.item()
        return action_tensor.detach().cpu().numpy().flatten().tolist()

    def run_episode(self) -> EpisodeResult:
        """Run a single evaluation episode."""
        observation, _ = self.env.reset(seed=self.config.seed)
        if self.config.render:
            self.env.render()
        last_obs = observation.copy()
        episode_done = False
        adv_steps = 0

        # Phase 1: Adversary acts greedily for Ha steps
        adversary_state = self.adversary.state_to_torch(observation)
        for _ in range(self.adversary_horizon):
            adv_steps += 1
            action_tensor = self.adversary.agent.select_greedy_action(
                adversary_state, eval=True
            )
            env_action = self._to_env_action(self.adversary, action_tensor)
            next_obs, _, terminated, truncated, _ = self.env.step(env_action)

            if self.config.render:
                self.env.render()

            episode_done = terminated or truncated
            last_obs = next_obs.copy()

            if episode_done:
                break

            adversary_state = self.adversary.state_to_torch(next_obs)

        # Record adversary endpoint
        adv_endpoint_x = float(last_obs[0]) if len(last_obs) > 0 else float("nan")
        adv_endpoint_y = float(last_obs[1]) if len(last_obs) > 1 else float("nan")

        # Phase 2: Protagonist acts greedily for Hp steps
        protagonist_score = 0.0
        protagonist_success = False
        prt_steps = 0

        if not episode_done:
            protagonist_state = self.protagonist.state_to_torch(last_obs)
            for _ in range(self.protagonist_horizon):
                prt_steps += 1
                action_tensor = self.protagonist.agent.select_greedy_action(
                    protagonist_state, eval=True
                )
                env_action = self._to_env_action(self.protagonist, action_tensor)
                next_obs, env_reward, terminated, truncated, _ = self.env.step(
                    env_action
                )

                if self.config.render:
                    self.env.render()

                protagonist_score += env_reward

                if terminated:
                    protagonist_success = True

                if terminated or truncated:
                    break

                protagonist_state = self.protagonist.state_to_torch(next_obs)

        return EpisodeResult(
            protagonist_score=protagonist_score,
            protagonist_success=protagonist_success,
            adversary_endpoint_x=adv_endpoint_x,
            adversary_endpoint_y=adv_endpoint_y,
            adversary_steps=adv_steps,
            protagonist_steps=prt_steps,
        )

    def evaluate(self) -> list[EpisodeResult]:
        """Run all evaluation episodes and collect results."""
        results: list[EpisodeResult] = []

        progress_bar = tqdm.tqdm(
            range(self.config.episodes),
            desc="Evaluating",
            bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="green",
        )

        for episode_idx in progress_bar:
            result = self.run_episode()
            results.append(result)

            # Update progress bar with running stats
            success_rate = sum(r.protagonist_success for r in results) / len(results)
            avg_score = np.mean([r.protagonist_score for r in results])
            progress_bar.set_postfix(
                {"success": f"{success_rate:.1%}", "avg_score": f"{avg_score:.2f}"}
            )

        return results

    def print_summary(self, results: list[EpisodeResult]) -> None:
        """Print evaluation summary statistics."""
        n = len(results)
        successes = sum(r.protagonist_success for r in results)
        success_rate = successes / n if n > 0 else 0.0

        scores = [r.protagonist_score for r in results]
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)

        adv_x = [r.adversary_endpoint_x for r in results]
        adv_y = [r.adversary_endpoint_y for r in results]

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Adversary Checkpoint:  {self.config.adv_checkpoint}")
        print(f"Protagonist Checkpoint: {self.config.prt_checkpoint}")
        print(f"Episodes:              {n}")
        print(f"Ha (adversary horizon): {self.adversary_horizon}")
        print(f"Hp (protagonist horizon): {self.protagonist_horizon}")
        print("-" * 60)
        print(f"Success Rate:          {success_rate:.1%} ({successes}/{n})")
        print(f"Average Score:         {avg_score:.4f} +/- {std_score:.4f}")
        print(f"Score Range:           [{min_score:.4f}, {max_score:.4f}]")
        print("-" * 60)
        print("Adversary Endpoint Distribution:")
        print(f"  X: mean={np.mean(adv_x):.4f}, std={np.std(adv_x):.4f}")
        print(f"  Y: mean={np.mean(adv_y):.4f}, std={np.std(adv_y):.4f}")
        print("=" * 60)

    def save_results_csv(self, results: list[EpisodeResult], path: str) -> None:
        """Save evaluation results to CSV file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "protagonist_score",
                    "protagonist_success",
                    "adv_endpoint_x",
                    "adv_endpoint_y",
                    "adv_steps",
                    "prt_steps",
                ]
            )
            for idx, r in enumerate(results):
                writer.writerow(
                    [
                        idx,
                        r.protagonist_score,
                        r.protagonist_success,
                        r.adversary_endpoint_x,
                        r.adversary_endpoint_y,
                        r.adversary_steps,
                        r.protagonist_steps,
                    ]
                )
        print(f"Results saved to: {path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Adversary-Protagonist models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--adv-checkpoint",
        type=int,
        required=True,
        help="Adversary checkpoint number (e.g., 100 for adversary_sac_100)",
    )
    parser.add_argument(
        "--prt-checkpoint",
        type=int,
        required=True,
        help="Protagonist checkpoint number (e.g., 100 for protagonist_sac_100)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes (opens visualization window)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for environment resets",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save results CSV (optional)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for evaluation."""
    args = parse_args()

    config = EvalConfig(
        adv_checkpoint=args.adv_checkpoint,
        prt_checkpoint=args.prt_checkpoint,
        episodes=args.episodes,
        device=args.device,
        render=args.render,
        seed=args.seed,
        output_csv=args.output_csv,
    )

    print(
        f"Evaluating with Ha={TrainingConfig().max_steps // 2}, Hp={TrainingConfig().max_steps // 2}"
    )

    evaluator = Evaluator(config)
    results = evaluator.evaluate()
    evaluator.print_summary(results)

    if config.output_csv:
        evaluator.save_results_csv(results, config.output_csv)


if __name__ == "__main__":
    main()

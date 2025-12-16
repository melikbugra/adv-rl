"""
Transfer Learning script with Curriculum Learning
Curriculum: FetchReach → FetchPush → FetchPickAndPlace → FetchSlide → FetchAssembly
"""

import gymnasium_robotics
import gymnasium as gym
from pathlib import Path
from rl_baselines.policy_based.sac import SAC

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)


# =============================================================================
# CURRICULUM CONFIGURATION
# =============================================================================

# Görev sıralaması: Kolaydan zora
# NOT: Önceki eğitimde Push/PickPlace platoya ulaştı, Slide/Assembly öğrenemedi
# Çözüm: Slide çıkarıldı, daha fazla adım, daha yüksek n_sampled_goal
CURRICULUM = [
    {
        "env_id": "FetchReach-v4",
        "max_episode_steps": 50,
        "time_steps": 100_000,  # ✅ Bu iyi çalıştı
        "learning_rate": 3e-4,
        "checkpoint": "curriculum_1_reach",
        "description": "Sadece hedefe ulaş (en kolay)",
    },
    {
        "env_id": "FetchPush-v4",
        "max_episode_steps": 50,
        "time_steps": 1_000_000,  # 300K → 500K (platoya ulaşmıştı)
        "learning_rate": 3e-4,
        "checkpoint": "curriculum_2_push",
        "description": "Objeyi it",
    },
    {
        "env_id": "FetchPickAndPlace-v4",
        "max_episode_steps": 50,
        "time_steps": 1_000_000,  # 500K → 1M (daha fazla süre lazım)
        "learning_rate": 3e-4,
        "checkpoint": "curriculum_3_pickplace",
        "description": "Objeyi al ve bırak",
    },
    {
        "env_id": "FetchPegInHolePreHeld-v1",  # Sparse reward + HER
        "max_episode_steps": 50,
        "time_steps": 5_000_000,  # Daha uzun eğitim
        "learning_rate": 3e-4,  
        "checkpoint": "curriculum_4_assembly",
        "description": "Montaj görevi (en zor)",
    },
]

COMMON_CONFIG = {
    "network_arch": [512, 1024, 1024, 512],
    "batch_size": 256,  # Çalışan değer
    "tau": 0.005,  # Çalışan değer
    "gamma": 0.99,  # Çalışan değer
    "experience_replay_size": 2_000_000,
    "device": "cuda:1",
    "n_sampled_goal": 8,
    "goal_selection_strategy": "future",
    "gradient_steps": 1,
}


def make_env(env_id: str, max_episode_steps: int = 100):
    """Create environment."""
    return gym.make(env_id, max_episode_steps=max_episode_steps)


def get_checkpoint_path(task_config: dict) -> Path:
    """Generate checkpoint path for a task."""
    device_str = COMMON_CONFIG["device"].replace(":", "_")
    return Path(
        f"models/{task_config['env_id']}_SAC_{device_str}_{task_config['checkpoint']}.ckpt"
    )


def train_task(
    task_config: dict,
    previous_checkpoint: Path = None,
    skip_if_exists: bool = True,
) -> Path:
    """
    Train a single task in the curriculum.

    Args:
        task_config: Task configuration dict
        previous_checkpoint: Path to previous task's checkpoint (for transfer)
        skip_if_exists: Skip training if checkpoint already exists

    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = get_checkpoint_path(task_config)

    if skip_if_exists and checkpoint_path.exists():
        print(f"✓ Checkpoint exists, skipping: {task_config['env_id']}")
        return checkpoint_path

    print("\n" + "=" * 60)
    print(f"🎯 Training: {task_config['env_id']}")
    print(f"   {task_config['description']}")
    print(f"   Steps: {task_config['time_steps']:,}")
    if previous_checkpoint:
        print(f"   Transfer from: {previous_checkpoint.name}")
    print("=" * 60)

    env = make_env(task_config["env_id"], task_config["max_episode_steps"])
    eval_env = make_env(task_config["env_id"], task_config["max_episode_steps"])

    action_dim = env.action_space.shape[0]

    model = SAC(
        env=env,
        eval_env=eval_env,
        experience_replay_type="her",
        time_steps=task_config["time_steps"],
        learning_rate=task_config["learning_rate"],
        learning_starts=5000,  # 1000 → 5000 (daha fazla exploration)
        batch_size=COMMON_CONFIG["batch_size"],
        gradient_steps=COMMON_CONFIG["gradient_steps"],
        network_type="mlp",
        network_arch=COMMON_CONFIG["network_arch"],
        tau=COMMON_CONFIG["tau"],
        gamma=COMMON_CONFIG["gamma"],
        target_entropy=-float(action_dim),
        experience_replay_size=COMMON_CONFIG["experience_replay_size"],
        device=COMMON_CONFIG["device"],
        plot_train_sores=True,
        writing_period=5_000,
        gradient_clipping_max_norm=1.0,
        env_seed=None,
        evaluation=True,
        eval_episodes=10,
        num_q_heads=2,
        n_sampled_goal=COMMON_CONFIG["n_sampled_goal"],
        goal_selection_strategy=COMMON_CONFIG["goal_selection_strategy"],
        mlflow_tracking_uri="https://mlflow.melikbugraozcelik.com/",
    )

    # Transfer learning from previous task
    if previous_checkpoint and previous_checkpoint.exists():
        print(f"\n📥 Loading weights from: {previous_checkpoint.name}")
        load_info = model.load(
            model_path=str(previous_checkpoint),
            transfer_learning=True,
            strict=False,
            load_optimizer=False,
            load_hyperparams=False,
        )
        print(f"   Loaded: {load_info.get('loaded', [])}")
        if load_info.get("warnings"):
            print(f"   Warnings: {load_info['warnings']}")

    # Train
    model.train()

    # Save checkpoint
    save_path = model.save(
        checkpoint=task_config["checkpoint"],
        save_optimizer=True,
        save_replay_buffer=False,
    )

    env.close()
    eval_env.close()

    print(f"✓ Saved: {save_path}")
    return checkpoint_path


def run_curriculum_learning(
    start_from: int = 0,
    end_at: int = None,
    skip_existing: bool = True,
):
    """
    Run full curriculum learning pipeline.

    Args:
        start_from: Index of task to start from (0-indexed)
        end_at: Index of task to end at (inclusive, None = all tasks)
        skip_existing: Skip tasks that already have checkpoints
    """
    if end_at is None:
        end_at = len(CURRICULUM) - 1

    print("\n" + "=" * 70)
    print("🚀 CURRICULUM LEARNING")
    print("=" * 70)
    print("\nGörev Sıralaması:")
    for i, task in enumerate(CURRICULUM):
        marker = "→" if start_from <= i <= end_at else " "
        status = "✓" if get_checkpoint_path(task).exists() else "○"
        print(f"  {status} {i + 1}. {task['env_id']:<25} - {task['description']}")
    print("=" * 70 + "\n")

    previous_checkpoint = None

    # If starting from middle, find previous checkpoint
    if start_from > 0:
        previous_checkpoint = get_checkpoint_path(CURRICULUM[start_from - 1])
        if not previous_checkpoint.exists():
            print(f"⚠️  Warning: Previous checkpoint not found: {previous_checkpoint}")
            print("   Starting without transfer learning.")
            previous_checkpoint = None

    # Train each task in curriculum
    for i in range(start_from, end_at + 1):
        task = CURRICULUM[i]

        checkpoint_path = train_task(
            task_config=task,
            previous_checkpoint=previous_checkpoint,
            skip_if_exists=skip_existing,
        )

        previous_checkpoint = checkpoint_path

    print("\n" + "=" * 70)
    print("✅ CURRICULUM LEARNING COMPLETED!")
    print("=" * 70)
    print("\nFinal checkpoints:")
    for i, task in enumerate(CURRICULUM[start_from : end_at + 1], start=start_from):
        path = get_checkpoint_path(task)
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {task['env_id']}: {path}")
    print()


def train_single_task(task_index: int, use_transfer: bool = True):
    """
    Train a single task from the curriculum.

    Args:
        task_index: Index of task in CURRICULUM (0-indexed)
        use_transfer: Whether to use transfer learning from previous task
    """
    if task_index < 0 or task_index >= len(CURRICULUM):
        raise ValueError(
            f"Invalid task index: {task_index}. Must be 0-{len(CURRICULUM) - 1}"
        )

    task = CURRICULUM[task_index]

    previous_checkpoint = None
    if use_transfer and task_index > 0:
        previous_checkpoint = get_checkpoint_path(CURRICULUM[task_index - 1])

    train_task(
        task_config=task,
        previous_checkpoint=previous_checkpoint,
        skip_if_exists=False,
    )


def train_from_scratch(task_index: int = -1):
    """
    Train a task from scratch without transfer learning.
    Default: last task (FetchAssembly)
    """
    if task_index == -1:
        task_index = len(CURRICULUM) - 1

    task = CURRICULUM[task_index]

    print("\n" + "=" * 60)
    print(f"🎯 Training from scratch: {task['env_id']}")
    print(f"   (No transfer learning - baseline comparison)")
    print("=" * 60)

    env = make_env(task["env_id"], task["max_episode_steps"])
    eval_env = make_env(task["env_id"], task["max_episode_steps"])

    action_dim = env.action_space.shape[0]

    model = SAC(
        env=env,
        eval_env=eval_env,
        experience_replay_type="her",
        time_steps=task["time_steps"],
        learning_rate=3e-4,
        learning_starts=1000,
        batch_size=COMMON_CONFIG["batch_size"],
        gradient_steps=COMMON_CONFIG["gradient_steps"],
        network_type="mlp",
        network_arch=COMMON_CONFIG["network_arch"],
        tau=COMMON_CONFIG["tau"],
        gamma=COMMON_CONFIG["gamma"],
        target_entropy=-float(action_dim),
        experience_replay_size=COMMON_CONFIG["experience_replay_size"],
        device=COMMON_CONFIG["device"],
        plot_train_sores=True,
        writing_period=5_000,
        gradient_clipping_max_norm=1.0,
        env_seed=None,
        evaluation=True,
        eval_episodes=10,
        num_q_heads=2,
        n_sampled_goal=COMMON_CONFIG["n_sampled_goal"],
        goal_selection_strategy=COMMON_CONFIG["goal_selection_strategy"],
        mlflow_tracking_uri="https://mlflow.melikbugraozcelik.com/",
    )

    model.train()
    model.save(checkpoint=f"from_scratch_{task['env_id'].split('-')[0].lower()}")

    env.close()
    eval_env.close()


def list_tasks():
    """Print all tasks in curriculum with their status."""
    print("\n" + "=" * 70)
    print("📋 CURRICULUM TASKS")
    print("=" * 70)
    print(f"{'#':<3} {'Environment':<25} {'Steps':<12} {'Status':<10} {'Description'}")
    print("-" * 70)
    for i, task in enumerate(CURRICULUM):
        path = get_checkpoint_path(task)
        status = "✓ Done" if path.exists() else "○ Pending"
        print(
            f"{i:<3} {task['env_id']:<25} {task['time_steps']:<12,} {status:<10} {task['description']}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Curriculum Learning for Fetch environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transfer.py                      # Run full curriculum
  python transfer.py --list               # List all tasks
  python transfer.py --start 2            # Start from task 2 (PickAndPlace)
  python transfer.py --start 2 --end 3    # Train tasks 2-3 only
  python transfer.py --task 4             # Train only task 4 (Assembly)
  python transfer.py --scratch            # Train Assembly from scratch (baseline)
  python transfer.py --scratch --task 2   # Train PickAndPlace from scratch
        """,
    )
    parser.add_argument("--list", action="store_true", help="List all tasks and exit")
    parser.add_argument(
        "--start", type=int, default=0, help="Start from task index (0-indexed)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End at task index (inclusive)"
    )
    parser.add_argument("--task", type=int, default=None, help="Train only this task")
    parser.add_argument(
        "--scratch", action="store_true", help="Train from scratch (no transfer)"
    )
    parser.add_argument(
        "--no-skip", action="store_true", help="Don't skip existing checkpoints"
    )

    args = parser.parse_args()

    if args.list:
        list_tasks()
    elif args.scratch:
        task_idx = args.task if args.task is not None else -1
        train_from_scratch(task_idx)
    elif args.task is not None:
        train_single_task(args.task, use_transfer=True)
    else:
        run_curriculum_learning(
            start_from=args.start,
            end_at=args.end,
            skip_existing=not args.no_skip,
        )

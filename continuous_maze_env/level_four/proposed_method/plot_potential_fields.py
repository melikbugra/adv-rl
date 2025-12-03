"""Plot protagonist and adversary potential fields over the maze blueprint.

This script mirrors the denormalization logic used by
``generate_phase_heatmaps.py`` so arrow placements align with the scatter
plots. Provide checkpoint identifiers (either numeric steps or explicit
filenames) for both agents and the script will render two PNGs next to the
base `empty_level.png` blueprint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
import torch

import continuous_maze_env  # noqa: F401 - registers the gym env
from continuous_maze_env.game.utils.constants import (
    PLAYER_SIZE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from rl_baselines.policy_based.sac.sac import SAC

CANVAS_WIDTH = WINDOW_WIDTH
CANVAS_HEIGHT = WINDOW_HEIGHT
BASE_IMAGE_RELATIVE = Path("heatmap/empty_level.png")
MODELS_RELATIVE = Path("level_four_models")
ENV_ID = "ContinuousMaze-v0"
ENV_KWARGS = dict(
    level="level_four",
    max_steps=2500,
    random_start=False,
    render_mode=None,
    dense_reward=False,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render potential-field arrows for selected adversary and protagonist checkpoints."
    )
    parser.add_argument(
        "--adv",
        required=True,
        help="Identifier for the adversary checkpoint (number, stem, or path).",
    )
    parser.add_argument(
        "--prt",
        required=True,
        help="Identifier for the protagonist checkpoint (number, stem, or path).",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=50,
        help="Number of samples per axis over the [0, 1] playfield.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=12.0,
        help="Pixel length of unit-norm arrows.",
    )
    return parser.parse_args()


def resolve_model_path(prefix: str, identifier: str, models_dir: Path) -> Path:
    candidate = Path(identifier)
    if candidate.is_file():
        return candidate
    if not candidate.suffix:
        as_ckpt = candidate.with_suffix(".ckpt")
        if as_ckpt.is_file():
            return as_ckpt
    local = models_dir / identifier
    if local.is_file():
        return local
    if not local.suffix:
        local_ckpt = local.with_suffix(".ckpt")
        if local_ckpt.is_file():
            return local_ckpt
    try:
        step = int(identifier)
        templ = models_dir / f"{prefix}_{step}.ckpt"
        if templ.is_file():
            return templ
    except ValueError:
        pass
    direct_ckpt = models_dir / f"{identifier}"
    if direct_ckpt.is_file():
        return direct_ckpt
    raise FileNotFoundError(
        f"Could not resolve checkpoint '{identifier}'. Place it inside {models_dir} or provide a full path."
    )


def sanitize_label(path: Path) -> str:
    return path.stem.replace(" ", "_")


def denormalize_to_canvas(x_norm: float, y_norm: float) -> Tuple[float, float]:
    x = x_norm * CANVAS_WIDTH + PLAYER_SIZE / 2.0
    y_world = y_norm * CANVAS_HEIGHT + PLAYER_SIZE / 2.0
    y = CANVAS_HEIGHT - y_world
    x = min(max(x, 0.0), CANVAS_WIDTH - 1.0)
    y = min(max(y, 0.0), CANVAS_HEIGHT - 1.0)
    return x, y


def build_playable_mask(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    return (arr[:, :, 0] > 250) & (arr[:, :, 1] > 250) & (arr[:, :, 2] > 250)


def make_agent(env, device: str) -> SAC:
    agent = SAC(env=env, eval_env=env, env_seed=None, device=device)
    agent.agent.net.eval()
    agent.agent.net.training = False
    return agent


def select_action_with_confidence(
    agent: SAC, state_np: np.ndarray
) -> Tuple[Tuple[float, float], float]:
    """Select greedy action and return confidence based on policy std.

    Returns
    -------
    tuple
        ((action_x, action_y), confidence) where confidence is in [0, 1].
        Lower std means higher confidence.
    """
    state_tensor = agent.state_to_torch(state_np)
    with torch.no_grad():
        outs, *_ = agent.agent.net(state=state_tensor, actor_pass=True)

    if agent.agent.action_type == "discrete":
        action = agent.agent.select_greedy_action(state_tensor, eval=True)
        arr = np.array([action.item()], dtype=np.float32)
        confidence = 1.0  # Discrete actions don't have std
    else:
        mean, std = outs[0]
        # Greedy action is tanh(mean) * max_action
        action = torch.tanh(mean) * agent.agent.max_action
        arr = action.detach().cpu().numpy().reshape(-1)

        # Compute confidence from std: lower std = higher confidence
        # std is typically in range [exp(-20), exp(2)] ≈ [2e-9, 7.4]
        # We use mean std across action dimensions
        mean_std = std.mean().item()
        # Map std to confidence: std=0 -> conf=1, std=2 -> conf~0.1
        # Using exponential decay: confidence = exp(-std * scale)
        confidence = float(np.exp(-mean_std * 1.5))
        confidence = max(0.15, min(1.0, confidence))  # Clamp to [0.15, 1.0]

    if arr.size == 1:
        return (float(arr[0]), 0.0), confidence
    return (float(arr[0]), float(arr[1])), confidence


def select_action(agent: SAC, state_np: np.ndarray) -> Tuple[float, float]:
    """Legacy wrapper for backward compatibility."""
    action, _ = select_action_with_confidence(agent, state_np)
    return action


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    origin: Tuple[float, float],
    vector: Tuple[float, float],
    color: Tuple[int, int, int, int],
    scale: float,
    line_width: int = 2,
) -> None:
    ox, oy = origin
    vx, vy = vector
    norm = float(np.hypot(vx, vy))
    if norm < 1e-6:
        return
    dx = (vx / norm) * scale
    dy = -(vy / norm) * scale
    ex = ox + dx
    ey = oy + dy
    draw.line((ox, oy, ex, ey), fill=color, width=line_width)
    # Scale head size proportionally to line width
    width_scale = line_width / 2.0
    head_len = scale * 0.35 * width_scale
    head_w = scale * 0.25 * width_scale
    back_norm = np.hypot(dx, dy) or 1.0
    bx = ex - dx / back_norm * head_len
    by = ey - dy / back_norm * head_len
    px = -dy
    py = dx
    perp_norm = np.hypot(px, py) or 1.0
    px /= perp_norm
    py /= perp_norm
    lx = bx + px * head_w
    ly = by + py * head_w
    rx = bx - px * head_w
    ry = by - py * head_w
    draw.polygon([(ex, ey), (lx, ly), (rx, ry)], fill=color)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    base_image_path = script_dir / BASE_IMAGE_RELATIVE
    models_dir = script_dir / MODELS_RELATIVE

    if not base_image_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    base_image = Image.open(base_image_path).convert("RGBA")
    playable_mask = build_playable_mask(base_image)
    width, height = base_image.size

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    adv_env = gym.make(ENV_ID, **ENV_KWARGS)
    prt_env = gym.make(ENV_ID, **ENV_KWARGS)
    try:
        adv_agent = make_agent(adv_env, device)
        prt_agent = make_agent(prt_env, device)

        adv_path = resolve_model_path("adversary_sac", args.adv, models_dir)
        prt_path = resolve_model_path("protagonist_sac", args.prt, models_dir)
        if not adv_path.exists():
            raise FileNotFoundError(f"Adversary checkpoint not found: {adv_path}")
        if not prt_path.exists():
            raise FileNotFoundError(f"Protagonist checkpoint not found: {prt_path}")

        adv_agent.load(model_path=str(adv_path), eval_mode=True)
        prt_agent.load(model_path=str(prt_path), eval_mode=True)

        grid = np.linspace(0.0, 1.0, args.grid_resolution)
        adv_img = base_image.copy()
        prt_img = base_image.copy()
        adv_draw = ImageDraw.Draw(adv_img, "RGBA")
        prt_draw = ImageDraw.Draw(prt_img, "RGBA")

        for x_norm in grid:
            for y_norm in grid:
                x_pix, y_pix = denormalize_to_canvas(x_norm, y_norm)
                px = int(round(x_pix))
                py = int(round(y_pix))
                if not (0 <= px < width and 0 <= py < height):
                    continue
                if not playable_mask[py, px]:
                    continue
                state = np.array([x_norm, y_norm], dtype=np.float32)
                adv_vec, adv_conf = select_action_with_confidence(adv_agent, state)
                prt_vec, prt_conf = select_action_with_confidence(prt_agent, state)
                # Map confidence to line width: [0.15, 1.0] -> [1, 5]
                adv_width = max(1, int(1 + adv_conf * 4))
                prt_width = max(1, int(1 + prt_conf * 4))
                draw_arrow(
                    adv_draw,
                    (x_pix, y_pix),
                    adv_vec,
                    (255, 0, 0, 255),
                    args.arrow_scale,
                    line_width=adv_width,
                )
                draw_arrow(
                    prt_draw,
                    (x_pix, y_pix),
                    prt_vec,
                    (0, 102, 255, 255),
                    args.arrow_scale,
                    line_width=prt_width,
                )

        adv_label = sanitize_label(adv_path)
        prt_label = sanitize_label(prt_path)
        adv_out = base_image_path.with_name(f"potential_field_adv_{adv_label}.png")
        prt_out = base_image_path.with_name(f"potential_field_prt_{prt_label}.png")
        adv_img.save(adv_out)
        prt_img.save(prt_out)
        print(f"Saved adversary potential field to {adv_out}")
        print(f"Saved protagonist potential field to {prt_out}")
    finally:
        adv_env.close()
        prt_env.close()


if __name__ == "__main__":
    main()

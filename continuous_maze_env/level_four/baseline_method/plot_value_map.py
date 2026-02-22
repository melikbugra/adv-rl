"""Render a protagonist value heatmap on top of the maze blueprint.

Given a protagonist SAC checkpoint, this utility samples a dense grid over the
normalized playfield, queries the critic value at each point, and paints the
results onto `heatmap/empty_level.png` using the viridis colormap. The
world-to-pixel mapping matches `generate_phase_heatmaps.py` so dots, arrows,
and value tiles stay aligned.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from matplotlib import cm

import continuous_maze_env  # noqa: F401 - registers envs
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
DEFAULT_ALPHA = 208
LEGEND_WIDTH = 80
LEGEND_MARGIN = 16
LEGEND_RIGHT_OFFSET = 40
LEGEND_TEXT_COLOR = (0, 0, 0, 255)
LEGEND_OUTLINE_COLOR = (0, 0, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a protagonist value heatmap using the viridis colormap."
    )
    parser.add_argument(
        "--prt",
        required=True,
        help="Identifier for the protagonist checkpoint (number, stem, or path).",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=60,
        help="Number of samples per axis over the normalized playfield.",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=DEFAULT_ALPHA,
        help="Overlay alpha (0-255) applied to each tile.",
    )
    return parser.parse_args()


def resolve_model_path(identifier: str, models_dir: Path) -> Path:
    candidate = Path(identifier)
    if candidate.is_file():
        return candidate
    if not candidate.suffix:
        ckpt_candidate = candidate.with_suffix(".ckpt")
        if ckpt_candidate.is_file():
            return ckpt_candidate
    local = models_dir / identifier
    if local.is_file():
        return local
    if not local.suffix:
        local_ckpt = local.with_suffix(".ckpt")
        if local_ckpt.is_file():
            return local_ckpt
    try:
        step = int(identifier)
        templ = models_dir / f"protagonist_sac_{step}.ckpt"
        if templ.is_file():
            return templ
    except ValueError:
        pass
    raise FileNotFoundError(
        f"Could not resolve checkpoint '{identifier}'. Store it under {models_dir} or pass an absolute path."
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


def evaluate_value(agent: SAC, state_np: np.ndarray) -> float:
    state_tensor = agent.state_to_torch(state_np).float()
    with torch.no_grad():
        action_tensor = agent.agent.select_greedy_action(state_tensor, eval=True)
        action_tensor = action_tensor.float()
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        sa = torch.cat([state_tensor, action_tensor], dim=-1)
        q_values: List[torch.Tensor] = []
        for critic in agent.agent.net.critics:
            q_head = critic(sa)[0]
            q_values.append(q_head)
        q_tensor = torch.stack(q_values, dim=0).mean()
    return float(q_tensor.item())


def build_overlay_from_grid(
    values_grid: np.ndarray,
    valid_mask: np.ndarray,
    alpha: int,
    playable_mask: np.ndarray,
    vmin: float,
    vmax: float,
) -> Image.Image:
    cmap = cm.get_cmap("viridis")
    norm_grid = np.zeros_like(values_grid, dtype=np.float32)
    if vmax - vmin < 1e-9:
        norm_grid[valid_mask] = 0.5
    else:
        norm_grid[valid_mask] = (values_grid[valid_mask] - vmin) / (vmax - vmin)
    rgba = (cmap(norm_grid) * 255).astype(np.uint8)
    rgba[..., 3] = 0
    rgba[..., 3][valid_mask] = np.clip(alpha, 0, 255)
    small_overlay = Image.fromarray(rgba, mode="RGBA")
    overlay = small_overlay.resize(
        (CANVAS_WIDTH, CANVAS_HEIGHT), Image.Resampling.BILINEAR
    )
    overlay_arr = np.array(overlay)
    playable_bool = playable_mask.astype(bool)
    overlay_arr[~playable_bool] = 0
    return Image.fromarray(overlay_arr, mode="RGBA")


def fill_missing_with_nearest(
    values_grid: np.ndarray, valid_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    filled = values_grid.copy()
    filled_mask = valid_mask.copy()
    queue = deque(map(tuple, np.argwhere(valid_mask)))
    if not queue:
        return filled, filled_mask, 0.0, 0.0
    rows, cols = values_grid.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and not filled_mask[nr, nc]:
                filled[nr, nc] = filled[r, c]
                filled_mask[nr, nc] = True
                queue.append((nr, nc))
    valid_values = filled[filled_mask]
    return filled, filled_mask, float(valid_values.min()), float(valid_values.max())


def draw_legend(
    image: Image.Image,
    vmin: float,
    vmax: float,
    cmap_name: str,
) -> None:
    legend_height = image.height - 2 * LEGEND_MARGIN
    if legend_height <= 0:
        return
    bar_width = max(10, LEGEND_WIDTH // 2)
    slot_left = image.width - LEGEND_WIDTH - LEGEND_RIGHT_OFFSET
    bar_x = slot_left + (LEGEND_WIDTH - bar_width) // 2
    bar_y = LEGEND_MARGIN

    gradient = np.linspace(1.0, 0.0, legend_height).reshape(-1, 1)
    cmap = cm.get_cmap(cmap_name)
    legend_rgba = (cmap(gradient) * 255).astype(np.uint8)
    legend_rgba[:, :, 3] = 255
    legend_img = Image.fromarray(legend_rgba, mode="RGBA").resize(
        (bar_width, legend_height), Image.Resampling.BILINEAR
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay.paste(legend_img, (bar_x, bar_y))
    image.alpha_composite(overlay)

    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [bar_x - 1, bar_y - 1, bar_x + bar_width, bar_y + legend_height],
        outline=LEGEND_OUTLINE_COLOR,
        width=1,
    )

    font = ImageFont.load_default()

    def _measure(text: str) -> Tuple[int, int]:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _draw_bold_text(position: Tuple[float, float], text: str) -> None:
        x, y = position
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for dx, dy in offsets:
            draw.text((x + dx, y + dy), text, fill=LEGEND_TEXT_COLOR, font=font)

    text_x = bar_x + bar_width + 8
    tick_values = np.linspace(vmax, vmin, 5)
    for idx, val in enumerate(tick_values):
        text = f"{val:.2f}"
        _, text_h = _measure(text)
        y = bar_y + (legend_height) * (idx / (len(tick_values) - 1)) - text_h / 2
        _draw_bold_text((text_x, y), text)


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
    env = gym.make(ENV_ID, **ENV_KWARGS)
    try:
        protagonist = make_agent(env, device)
        prt_path = resolve_model_path(args.prt, models_dir)
        if not prt_path.exists():
            raise FileNotFoundError(f"Protagonist checkpoint not found: {prt_path}")
        protagonist.load(model_path=str(prt_path), eval_mode=True)

        grid = np.linspace(0.0, 1.0, args.grid_resolution)
        values_grid = np.full(
            (args.grid_resolution, args.grid_resolution), np.nan, dtype=np.float32
        )
        valid_mask = np.zeros_like(values_grid, dtype=bool)

        for ix, x_norm in enumerate(grid):
            for iy, y_norm in enumerate(grid):
                x_pix, y_pix = denormalize_to_canvas(x_norm, y_norm)
                px = int(round(x_pix))
                py = int(round(y_pix))
                if not (0 <= px < width and 0 <= py < height):
                    continue
                if not playable_mask[py, px]:
                    continue
                state = np.array([x_norm, y_norm], dtype=np.float32)
                value = evaluate_value(protagonist, state)
                row = args.grid_resolution - 1 - iy
                values_grid[row, ix] = value
                valid_mask[row, ix] = True

        if not valid_mask.any():
            print("No playable samples found; value map not generated.")
            return

        filled_values, filled_mask, vmin, vmax = fill_missing_with_nearest(
            values_grid, valid_mask
        )
        overlay = build_overlay_from_grid(
            filled_values, filled_mask, args.alpha, playable_mask, vmin, vmax
        )
        value_map = Image.alpha_composite(base_image, overlay)
        draw_legend(value_map, vmin, vmax, cmap_name="viridis")

        prt_label = sanitize_label(prt_path)
        # Create value_map directory if it doesn't exist
        output_dir = script_dir / "heatmap" / "value_map"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"value_map_prt_{prt_label}.png"
        value_map.save(out_path)
        print(f"Saved protagonist value map to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List
from continuous_maze_env.game.utils.constants import (
    PLAYER_SIZE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


from PIL import Image, ImageDraw

# Canvas dimensions used by the environment normalization step
CANVAS_WIDTH = WINDOW_WIDTH
CANVAS_HEIGHT = WINDOW_HEIGHT

# Rendering constants
BATCH_SIZE = 50
POINT_DIAMETER = 5
BORDER_THICKNESS = 0.5
FAIL_COLOR = (255, 0, 0)  # red when protagonist fails
SUCCESS_COLOR = (0, 102, 255)  # blue when protagonist succeeds
BORDER_COLOR = (255, 255, 255)

CSV_FILENAME = "heatmap/adv_prt_endpoints.csv"
BASE_IMAGE_FILENAME = "heatmap/empty_level.png"
OUTPUT_DIRNAME = "heatmap/batches"


def load_rows(csv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw_row in reader:
            rows.append(
                {
                    "iter": int(raw_row["iter"]),
                    "phase": raw_row["phase"].strip().lower(),
                    "x": float(raw_row["adv_last_x"]),
                    "y": float(raw_row["adv_last_y"]),
                    "success": raw_row["protagonist_success"].strip().lower() == "true",
                }
            )
    return rows


def denormalize(point: Dict[str, object]) -> tuple[float, float]:
    # Convert normalized lower-left coordinates into the rendered canvas.
    x = float(point["x"]) * CANVAS_WIDTH + PLAYER_SIZE / 2
    # Flip the vertical axis (env origin bottom-left, image origin top-left)
    y_world = float(point["y"]) * CANVAS_HEIGHT + PLAYER_SIZE / 2
    y = CANVAS_HEIGHT - y_world
    # Clamp to image bounds to avoid drawing outside the canvas
    x = min(max(x, 0.0), CANVAS_WIDTH - 1)
    y = min(max(y, 0.0), CANVAS_HEIGHT - 1)
    return x, y


def draw_point(
    draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int]
) -> None:
    radius = POINT_DIAMETER / 2.0
    border_radius = radius + BORDER_THICKNESS
    border_box = [
        x - border_radius,
        y - border_radius,
        x + border_radius,
        y + border_radius,
    ]
    fill_box = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(border_box, fill=BORDER_COLOR)
    draw.ellipse(fill_box, fill=color)


def render_points(
    base_image: Image.Image,
    points: Iterable[Dict[str, object]],
) -> Image.Image:
    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    for point in points:
        x, y = denormalize(point)
        color = SUCCESS_COLOR if point["success"] else FAIL_COLOR
        draw_point(draw, x, y, color)
    return image


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def group_points(
    rows: List[Dict[str, object]], start_iter: int, end_iter: int, phase: str
) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if start_iter <= int(row["iter"]) <= end_iter and row["phase"] == phase
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    csv_path = repo_root / CSV_FILENAME
    base_image_path = repo_root / BASE_IMAGE_FILENAME
    output_dir = repo_root / OUTPUT_DIRNAME

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not base_image_path.exists():
        raise FileNotFoundError(f"Base image not found: {base_image_path}")

    rows = load_rows(csv_path)
    if not rows:
        print("CSV is empty; nothing to render.")
        return

    ensure_output_dir(output_dir)
    base_image = Image.open(base_image_path).convert("RGBA")

    max_iter = max(row["iter"] for row in rows)
    batch_counter = 0
    for start_iter in range(0, max_iter + 1, BATCH_SIZE):
        end_iter = start_iter + BATCH_SIZE - 1
        for phase in ("adv", "prt"):
            phase_points = group_points(rows, start_iter, end_iter, phase)
            image = render_points(base_image, phase_points)
            out_name = f"{phase}_iter_{start_iter:04d}_{end_iter:04d}.png"
            image.save(output_dir / out_name)
        batch_counter += 1

    print(f"Generated heatmaps for {batch_counter} batches of {BATCH_SIZE} iterations.")


if __name__ == "__main__":
    main()

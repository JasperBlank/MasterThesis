"""
Generate clean AprilGrid textures for systematic degradation experiments.

The original `april_grid.png` is useful, but it is screenshot-like and includes
UI/margins. These generated textures are clean targets with known tag size,
spacing, and visible-tag count.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender")
TEXTURE_DIR = ROOT / "degradation_matrix" / "textures"
TEXTURE_CONFIG = TEXTURE_DIR / "texture_config.json"

BOARD_SIZE_MM = 609.6  # 24 in
TEXTURE_SIZE_PX = 2400
ROWS = 6
COLS = 6


def center_ranked_ids() -> list[int]:
    center_r = (ROWS - 1) / 2.0
    center_c = (COLS - 1) / 2.0
    positions = []
    for row in range(ROWS):
        for col in range(COLS):
            tag_id = row * COLS + col
            d2 = (row - center_r) ** 2 + (col - center_c) ** 2
            positions.append((d2, row, col, tag_id))
    return [tag_id for _, _, _, tag_id in sorted(positions)]


def tag_world_metadata(
    tag_id: int,
    row: int,
    col: int,
    tag_size_mm: float,
    spacing_mm: float,
    margin_mm: float,
) -> dict[str, object]:
    left_mm = margin_mm + col * (tag_size_mm + spacing_mm)
    top_mm = margin_mm + row * (tag_size_mm + spacing_mm)
    right_mm = left_mm + tag_size_mm
    bottom_mm = top_mm + tag_size_mm

    def to_world(x_mm: float, y_from_top_mm: float) -> list[float]:
        x = (x_mm - BOARD_SIZE_MM / 2.0) / 1000.0
        y = (BOARD_SIZE_MM / 2.0 - y_from_top_mm) / 1000.0
        return [x, y, 0.0]

    return {
        "id": tag_id,
        "row": row,
        "col": col,
        "corners_board_m": [
            to_world(left_mm, top_mm),
            to_world(right_mm, top_mm),
            to_world(right_mm, bottom_mm),
            to_world(left_mm, bottom_mm),
        ],
    }


def create_texture(
    name: str,
    tag_size_mm: float,
    spacing_mm: float,
    visible_count: int = 36,
) -> dict[str, object]:
    px_per_mm = TEXTURE_SIZE_PX / BOARD_SIZE_MM
    marker_px = int(round(tag_size_mm * px_per_mm))
    spacing_px = int(round(spacing_mm * px_per_mm))
    grid_w_px = COLS * marker_px + (COLS - 1) * spacing_px
    grid_h_px = ROWS * marker_px + (ROWS - 1) * spacing_px

    if grid_w_px > TEXTURE_SIZE_PX or grid_h_px > TEXTURE_SIZE_PX:
        raise ValueError(f"{name} does not fit within the 24 in board")

    margin_x_px = (TEXTURE_SIZE_PX - grid_w_px) // 2
    margin_y_px = (TEXTURE_SIZE_PX - grid_h_px) // 2
    margin_mm = (BOARD_SIZE_MM - (COLS * tag_size_mm + (COLS - 1) * spacing_mm)) / 2.0

    image = np.full((TEXTURE_SIZE_PX, TEXTURE_SIZE_PX), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    visible_ids = set(center_ranked_ids()[:visible_count])
    tags = []

    for row in range(ROWS):
        for col in range(COLS):
            tag_id = row * COLS + col
            if tag_id not in visible_ids:
                continue

            marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_px)
            x0 = margin_x_px + col * (marker_px + spacing_px)
            y0 = margin_y_px + row * (marker_px + spacing_px)
            image[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker
            tags.append(tag_world_metadata(tag_id, row, col, tag_size_mm, spacing_mm, margin_mm))

    texture_path = TEXTURE_DIR / f"{name}.png"
    cv2.imwrite(str(texture_path), image)

    return {
        "name": name,
        "path": str(texture_path),
        "board_size_mm": BOARD_SIZE_MM,
        "texture_size_px": TEXTURE_SIZE_PX,
        "rows": ROWS,
        "cols": COLS,
        "tag_size_mm": tag_size_mm,
        "spacing_mm": spacing_mm,
        "visible_count": visible_count,
        "visible_ids": sorted(visible_ids),
        "tags": tags,
    }


def main() -> None:
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, float, float, int]] = [
        ("baseline_t60_s25_v36", 60.0, 25.0, 36),
    ]

    for count in [36, 24, 18, 12, 6, 3, 1]:
        specs.append((f"visible_{count:02d}", 60.0, 25.0, count))

    for tag_size in [30.0, 45.0, 60.0, 75.0]:
        specs.append((f"tag_size_{int(tag_size):02d}mm", tag_size, 25.0, 36))

    for spacing in [0.0, 10.0, 25.0, 40.0]:
        specs.append((f"spacing_{int(spacing):02d}mm", 60.0, spacing, 36))

    configs: dict[str, dict[str, object]] = {}
    for name, tag_size_mm, spacing_mm, visible_count in specs:
        configs[name] = create_texture(name, tag_size_mm, spacing_mm, visible_count)

    with TEXTURE_CONFIG.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "board_size_mm": BOARD_SIZE_MM,
                "texture_size_px": TEXTURE_SIZE_PX,
                "rows": ROWS,
                "cols": COLS,
                "textures": configs,
            },
            f,
            indent=2,
        )

    print(f"Generated {len(configs)} texture configs in {TEXTURE_DIR}")


if __name__ == "__main__":
    main()

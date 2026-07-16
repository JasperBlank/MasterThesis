"""
Add post-processed degradation cases to the rendered matrix.

This script starts from the baseline rendered image and creates controlled
resolution, motion-blur, and lighting/exposure variants.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender")
MATRIX_DIR = ROOT / "degradation_matrix"
IMAGE_DIR = MATRIX_DIR / "images"
RENDERED_CASES_CSV = MATRIX_DIR / "cases_rendered.csv"
ALL_CASES_CSV = MATRIX_DIR / "cases_all.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def motion_blur(image: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return image.copy()
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel)


def exposure(image: np.ndarray, multiplier: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * multiplier, 0, 255).astype(np.uint8)


def clone_case(
    baseline: dict[str, str],
    index: int,
    factor: str,
    level: str,
    image_name: str,
    postprocess: str,
    width: int,
    height: int,
) -> dict[str, str]:
    row = dict(baseline)
    row.update(
        {
            "case_id": f"post_{index:03d}_{factor}_{level}".replace(".", "p"),
            "source_case_id": baseline["case_id"],
            "image": image_name,
            "case_origin": "postprocess",
            "factor": factor,
            "level": level,
            "render_width_px": str(width),
            "render_height_px": str(height),
            "postprocess": postprocess,
        }
    )
    return row


def main() -> None:
    rows = read_rows(RENDERED_CASES_CSV)
    baseline = next(row for row in rows if row["factor"] == "baseline")
    baseline_image = cv2.imread(str(IMAGE_DIR / baseline["image"]), cv2.IMREAD_COLOR)
    if baseline_image is None:
        raise SystemExit(f"Could not read baseline image {baseline['image']}")

    post_rows: list[dict[str, str]] = []
    index = 0
    base_h, base_w = baseline_image.shape[:2]

    for width in [960, 400, 300, 200, 150, 100]:
        scale = width / base_w
        height = int(round(base_h * scale))
        resized = cv2.resize(baseline_image, (width, height), interpolation=cv2.INTER_AREA)
        image_name = f"post_{index:03d}_resolution_{width}px.png"
        cv2.imwrite(str(IMAGE_DIR / image_name), resized)
        post_rows.append(
            clone_case(
                baseline,
                index,
                "resolution",
                f"{width}px",
                image_name,
                f"resize_width={width}_preserve_aspect",
                width,
                height,
            )
        )
        index += 1

    for length in [0, 3, 7, 11, 15, 21, 31]:
        blurred = motion_blur(baseline_image, length)
        image_name = f"post_{index:03d}_motion_blur_{length}px.png"
        cv2.imwrite(str(IMAGE_DIR / image_name), blurred)
        post_rows.append(
            clone_case(
                baseline,
                index,
                "motion_blur",
                f"{length}px",
                image_name,
                f"horizontal_linear_blur_kernel={length}",
                base_w,
                base_h,
            )
        )
        index += 1

    for multiplier in [0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]:
        lit = exposure(baseline_image, multiplier)
        level_safe = f"{multiplier:.2f}".replace(".", "p")
        image_name = f"post_{index:03d}_lighting_{level_safe}x.png"
        cv2.imwrite(str(IMAGE_DIR / image_name), lit)
        post_rows.append(
            clone_case(
                baseline,
                index,
                "lighting",
                f"{multiplier:.2f}x",
                image_name,
                f"exposure_multiplier={multiplier:.2f}",
                base_w,
                base_h,
            )
        )
        index += 1

    all_rows = rows + post_rows
    write_rows(ALL_CASES_CSV, all_rows, list(all_rows[0].keys()))
    print(f"Wrote {len(post_rows)} post-processed cases")
    print(f"Wrote unified cases file: {ALL_CASES_CSV}")


if __name__ == "__main__":
    main()

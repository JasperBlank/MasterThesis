"""
Separate AprilGrid coverage from AprilTag readability.

Coverage: how many complete tags are geometrically inside the camera image.
Readability: how many of those visible tags are detected by the AprilTag detector.

This removes close-range board cropping from the pixels-per-tag threshold analysis.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender")
MATRIX_DIR = ROOT / "degradation_matrix"
TEXTURE_CONFIG = MATRIX_DIR / "textures" / "texture_config.json"
RATIO_DIR = MATRIX_DIR / "ratio_scale"
CASES_CSV = RATIO_DIR / "ratio_scale_cases.csv"
BLUR_CSV = RATIO_DIR / "pixels_per_tag_threshold_blur.csv"
OUT_CSV = RATIO_DIR / "visibility_readability_split.csv"
OUT_JSON = RATIO_DIR / "visibility_readability_split.json"
READABILITY_PNG = RATIO_DIR / "readability_vs_pixels_per_tag.png"
COVERAGE_PNG = RATIO_DIR / "coverage_vs_distance_tag_ratio.png"
HEATMAP_PNG = RATIO_DIR / "readability_heatmap_blur_pixel_bins.png"


PIXEL_BINS = [
    (0, 10),
    (10, 15),
    (15, 20),
    (20, 25),
    (25, 30),
    (30, 35),
    (35, 45),
    (45, 60),
    (60, 80),
    (80, 120),
    (120, 200),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_base_tag_corners() -> list[dict[str, object]]:
    with TEXTURE_CONFIG.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config["textures"]["baseline_t60_s25_v36"]["tags"]


def project_tag_corners(
    corners_board_m: list[list[float]],
    scale_factor: float,
    distance_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    pts = np.array(corners_board_m, dtype=float) * scale_factor
    x = pts[:, 0]
    y = pts[:, 1]
    z = distance_m
    u = cx + fx * x / z
    v = cy - fy * y / z
    return np.column_stack([u, v])


def count_visible_tags(case: dict[str, str], tags: list[dict[str, object]]) -> dict[str, object]:
    scale = float(case["scale_factor"])
    distance = float(case["distance_m"])
    fx = float(case["fx_px"])
    fy = float(case["fy_px"])
    cx = float(case["cx_px"])
    cy = float(case["cy_px"])
    width = float(case["render_width_px"])
    height = float(case["render_height_px"])

    fully_visible_ids: list[int] = []
    partially_visible_ids: list[int] = []
    for tag in tags:
        pixels = project_tag_corners(
            tag["corners_board_m"],
            scale,
            distance,
            fx,
            fy,
            cx,
            cy,
        )
        xs = pixels[:, 0]
        ys = pixels[:, 1]
        fully_visible = bool(np.all((0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)))
        bbox_intersects = bool(xs.max() >= 0 and xs.min() < width and ys.max() >= 0 and ys.min() < height)
        if fully_visible:
            fully_visible_ids.append(int(tag["id"]))
        if bbox_intersects:
            partially_visible_ids.append(int(tag["id"]))

    return {
        "fully_visible_count": len(fully_visible_ids),
        "partially_visible_count": len(partially_visible_ids),
        "fully_visible_ids": fully_visible_ids,
        "partially_visible_ids": partially_visible_ids,
    }


def case_key(scale_factor: float, ratio: float) -> tuple[float, float]:
    return (round(scale_factor, 6), round(ratio, 6))


def main() -> None:
    cases = read_csv(CASES_CSV)
    blur_rows = read_csv(BLUR_CSV)
    tags = load_base_tag_corners()

    visibility_by_case: dict[tuple[float, float], dict[str, object]] = {}
    for case in cases:
        key = case_key(float(case["scale_factor"]), float(case["distance_over_tag_size"]))
        visibility_by_case[key] = count_visible_tags(case, tags)

    rows: list[dict[str, object]] = []
    for row in blur_rows:
        key = case_key(float(row["scale_factor"]), float(row["distance_over_tag_size"]))
        visibility = visibility_by_case[key]
        fully_visible = int(visibility["fully_visible_count"])
        partially_visible = int(visibility["partially_visible_count"])
        detected = int(row["detected_count"])
        readability = detected / fully_visible if fully_visible else 0.0
        rows.append(
            {
                **row,
                "fully_visible_count": fully_visible,
                "partially_visible_count": partially_visible,
                "detected_out_of_36_rate": detected / 36.0,
                "readability_rate_among_fully_visible": min(readability, 1.0),
                "unreadable_visible_count": max(fully_visible - detected, 0),
            }
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    blur_levels = sorted({int(row["blur_kernel_px"]) for row in rows})
    ratios = sorted({round(float(row["distance_over_tag_size"]), 6) for row in rows})

    # Coverage by ratio is independent of blur and resize.
    coverage_summary = []
    for ratio in ratios:
        matches = [
            visibility_by_case[key]
            for key in visibility_by_case
            if round(key[1], 6) == ratio
        ]
        full_counts = [int(m["fully_visible_count"]) for m in matches]
        partial_counts = [int(m["partially_visible_count"]) for m in matches]
        coverage_summary.append(
            {
                "distance_over_tag_size": ratio,
                "fully_visible_count": int(np.mean(full_counts)),
                "partially_visible_count": int(np.mean(partial_counts)),
            }
        )

    # Thresholds now use readability among fully visible tags, not raw count out of 36.
    thresholds = {}
    for blur in blur_levels:
        blur_rows_for_threshold = [r for r in rows if int(r["blur_kernel_px"]) == blur]
        thresholds[str(blur)] = {}
        for target_rate in [0.25, 0.50, 0.75, 1.00]:
            values = [
                float(r["projected_tag_side_px"])
                for r in blur_rows_for_threshold
                if int(r["fully_visible_count"]) > 0
                and float(r["readability_rate_among_fully_visible"]) >= target_rate
            ]
            thresholds[str(blur)][f"first_readability_{target_rate:.2f}_px"] = min(values) if values else None

    heatmap = np.zeros((len(blur_levels), len(PIXEL_BINS)), dtype=float)
    for y, blur in enumerate(blur_levels):
        for x, (lo, hi) in enumerate(PIXEL_BINS):
            values = [
                float(r["readability_rate_among_fully_visible"])
                for r in rows
                if int(r["blur_kernel_px"]) == blur
                and int(r["fully_visible_count"]) > 0
                and lo <= float(r["projected_tag_side_px"]) < hi
            ]
            heatmap[y, x] = float(np.mean(values)) if values else np.nan

    summary = {
        "coverage_by_distance_over_tag_size": coverage_summary,
        "readability_thresholds_px": thresholds,
        "pixel_bins": [{"min": lo, "max": hi} for lo, hi in PIXEL_BINS],
        "mean_readability_by_blur_and_pixel_bin": np.nan_to_num(heatmap, nan=-1).tolist(),
        "definition": {
            "coverage": "complete AprilTag markers with all four tag corners inside the image frame",
            "readability": "detected tags divided by complete visible tags",
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Readability curves.
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    colors = {0: "#1f77b4", 3: "#2ca02c", 7: "#ff7f0e", 11: "#d62728", 15: "#9467bd"}
    for blur in blur_levels:
        grouped: dict[float, list[float]] = defaultdict(list)
        for r in rows:
            if int(r["blur_kernel_px"]) != blur or int(r["fully_visible_count"]) == 0:
                continue
            grouped[round(float(r["projected_tag_side_px"]), 6)].append(
                float(r["readability_rate_among_fully_visible"])
            )
        xs = sorted(grouped)
        ys = [float(np.mean(grouped[x])) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors.get(blur), label=f"{blur}px blur")
    ax.set_xlabel("Projected AprilTag side length (pixels)")
    ax.set_ylabel("Readability among fully visible tags")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("AprilTag Readability Separated from Grid Coverage")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(READABILITY_PNG)
    plt.close(fig)

    # Coverage curve.
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    ratio_values = [r["distance_over_tag_size"] for r in coverage_summary]
    full_values = [r["fully_visible_count"] for r in coverage_summary]
    partial_values = [r["partially_visible_count"] for r in coverage_summary]
    ax.plot(ratio_values, full_values, marker="o", label="fully visible tags")
    ax.plot(ratio_values, partial_values, marker="o", linestyle="--", label="partially visible tags")
    ax.axhline(36, color="0.7", linestyle=":", label="full grid")
    ax.set_xlabel("Camera distance / AprilTag side length")
    ax.set_ylabel("Tags in camera frame")
    ax.set_title("AprilGrid Coverage vs Distance/Tag-Size Ratio")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(COVERAGE_PNG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    im = ax.imshow(heatmap, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(PIXEL_BINS)))
    ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in PIXEL_BINS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(blur_levels)))
    ax.set_yticklabels([str(b) for b in blur_levels])
    ax.set_xlabel("Projected tag side length bin (px)")
    ax.set_ylabel("Motion blur kernel (px)")
    ax.set_title("Mean Readability Among Fully Visible Tags")
    for y in range(len(blur_levels)):
        for x in range(len(PIXEL_BINS)):
            if not np.isnan(heatmap[y, x]):
                ax.text(x, y, f"{heatmap[y, x]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Readability")
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"CSV: {OUT_CSV}")
    print(f"Readability plot: {READABILITY_PNG}")
    print(f"Coverage plot: {COVERAGE_PNG}")
    print(f"Heatmap: {HEATMAP_PNG}")


if __name__ == "__main__":
    main()

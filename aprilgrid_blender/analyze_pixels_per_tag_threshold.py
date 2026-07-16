"""
Derive a pixels-per-tag-side threshold from the ratio-scale experiment.
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
OUT_DIR = ROOT / "degradation_matrix" / "ratio_scale"
INPUT_CSV = OUT_DIR / "ratio_scale_detection_summary.csv"
PIXEL_SUMMARY_CSV = OUT_DIR / "pixels_per_tag_threshold.csv"
PIXEL_SUMMARY_JSON = OUT_DIR / "pixels_per_tag_threshold.json"
PIXEL_PLOT = OUT_DIR / "pixels_per_tag_threshold.png"


def read_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = read_rows()
    enriched: list[dict[str, object]] = []

    for row in rows:
        fx = float(row["fx_px"])
        ratio = float(row["distance_over_tag_size"])
        width = float(row["width_px"])
        render_width = float(row["render_width_px"])
        projected_tag_px = fx / ratio * (width / render_width)
        detected = int(row["detected_count"])
        enriched.append(
            {
                "projected_tag_side_px": projected_tag_px,
                "detected_count": detected,
                "width_px": int(float(row["width_px"])),
                "distance_over_tag_size": ratio,
                "scale_factor": float(row["scale_factor"]),
            }
        )

    bins = [
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
    by_bin: list[dict[str, object]] = []
    for lo, hi in bins:
        values = [r["detected_count"] for r in enriched if lo <= float(r["projected_tag_side_px"]) < hi]
        if not values:
            continue
        by_bin.append(
            {
                "tag_side_px_min": lo,
                "tag_side_px_max": hi,
                "samples": len(values),
                "mean_detected_count": float(np.mean(values)),
                "min_detected_count": int(np.min(values)),
                "max_detected_count": int(np.max(values)),
                "mean_detection_rate": float(np.mean(values) / 36.0),
            }
        )

    with PIXEL_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_bin[0].keys()))
        writer.writeheader()
        writer.writerows(by_bin)

    thresholds = {
        "first_any_detection_px": min(
            float(r["projected_tag_side_px"]) for r in enriched if int(r["detected_count"]) > 0
        ),
        "first_at_least_half_tags_px": min(
            float(r["projected_tag_side_px"]) for r in enriched if int(r["detected_count"]) >= 18
        ),
        "first_all_visible_36_tags_px": min(
            float(r["projected_tag_side_px"]) for r in enriched if int(r["detected_count"]) >= 36
        ),
        "bins": by_bin,
    }
    PIXEL_SUMMARY_JSON.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    xs = [float(r["projected_tag_side_px"]) for r in enriched]
    ys = [int(r["detected_count"]) for r in enriched]
    widths = [int(r["width_px"]) for r in enriched]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    scatter = ax.scatter(xs, ys, c=widths, cmap="viridis", alpha=0.85, edgecolors="none")
    ax.axvline(thresholds["first_any_detection_px"], color="0.45", linestyle=":", label="first any")
    ax.axvline(thresholds["first_at_least_half_tags_px"], color="#ff7f0e", linestyle="--", label=">=18 tags")
    ax.axvline(thresholds["first_all_visible_36_tags_px"], color="#2ca02c", linestyle="-.", label="36 tags")
    ax.set_xlabel("Projected AprilTag side length (pixels)")
    ax.set_ylabel("Detected tags out of 36")
    ax.set_title("AprilTag Readability vs Pixels per Tag Side")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="resized image width (px)")
    fig.tight_layout()
    fig.savefig(PIXEL_PLOT)
    plt.close(fig)

    print(json.dumps(thresholds, indent=2))
    print(f"CSV: {PIXEL_SUMMARY_CSV}")
    print(f"Plot: {PIXEL_PLOT}")


if __name__ == "__main__":
    main()

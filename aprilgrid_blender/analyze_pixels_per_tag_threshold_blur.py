"""
Derive pixels-per-tag-side thresholds under controlled motion blur.

This reuses the ratio-scale renders, resizes each image, applies horizontal
linear blur, detects AprilTags, and estimates thresholds per blur level.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender")
OUT_DIR = ROOT / "degradation_matrix" / "ratio_scale"
IMAGE_DIR = OUT_DIR / "images"
CASES_CSV = OUT_DIR / "ratio_scale_cases.csv"
SUMMARY_CSV = OUT_DIR / "pixels_per_tag_threshold_blur.csv"
SUMMARY_JSON = OUT_DIR / "pixels_per_tag_threshold_blur.json"
PLOT_PNG = OUT_DIR / "pixels_per_tag_threshold_blur.png"
HEATMAP_PNG = OUT_DIR / "pixels_per_tag_threshold_blur_heatmap.png"

WIDTHS = [960, 400, 300, 200, 150, 100]
BLUR_KERNELS = [0, 3, 7, 11, 15]


def make_detector() -> cv2.aruco.ArucoDetector:
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    params.minMarkerPerimeterRate = 0.005
    params.maxMarkerPerimeterRate = 4.0
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    return aruco.ArucoDetector(dictionary, params)


def read_cases() -> list[dict[str, str]]:
    with CASES_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def motion_blur(image: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return image.copy()
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel)


def detect_count(detector: cv2.aruco.ArucoDetector, image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None:
        return 0
    return len(set(int(i) for i in ids.flatten()))


def threshold_for(rows: list[dict[str, object]], min_count: int) -> float | None:
    values = [
        float(row["projected_tag_side_px"])
        for row in rows
        if int(row["detected_count"]) >= min_count
    ]
    return min(values) if values else None


def main() -> None:
    detector = make_detector()
    cases = read_cases()
    rows: list[dict[str, object]] = []

    for case in cases:
        source = cv2.imread(str(IMAGE_DIR / case["image"]), cv2.IMREAD_COLOR)
        if source is None:
            raise SystemExit(f"Could not read {case['image']}")
        h0, w0 = source.shape[:2]
        fx = float(case["fx_px"])
        ratio = float(case["distance_over_tag_size"])

        for width in WIDTHS:
            height = int(round(h0 * width / w0))
            resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
            projected_tag_px = fx / ratio * (width / float(case["render_width_px"]))

            for blur in BLUR_KERNELS:
                degraded = motion_blur(resized, blur)
                detected = detect_count(detector, degraded)
                rows.append(
                    {
                        "blur_kernel_px": blur,
                        "projected_tag_side_px": projected_tag_px,
                        "detected_count": detected,
                        "width_px": width,
                        "height_px": height,
                        "distance_over_tag_size": ratio,
                        "scale_factor": float(case["scale_factor"]),
                        "image": case["image"],
                    }
                )

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_blur: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_blur[int(row["blur_kernel_px"])].append(row)

    thresholds = {}
    for blur in BLUR_KERNELS:
        blur_rows = by_blur[blur]
        thresholds[str(blur)] = {
            "first_any_detection_px": threshold_for(blur_rows, 1),
            "first_at_least_half_tags_px": threshold_for(blur_rows, 18),
            "first_all_visible_36_tags_px": threshold_for(blur_rows, 36),
            "max_detected_count": max(int(row["detected_count"]) for row in blur_rows),
            "mean_detected_count": float(np.mean([int(row["detected_count"]) for row in blur_rows])),
        }

    bins = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 45), (45, 60), (60, 80), (80, 120), (120, 200)]
    heatmap = np.zeros((len(BLUR_KERNELS), len(bins)), dtype=float)
    for b_idx, blur in enumerate(BLUR_KERNELS):
        blur_rows = by_blur[blur]
        for bin_idx, (lo, hi) in enumerate(bins):
            values = [
                int(row["detected_count"])
                for row in blur_rows
                if lo <= float(row["projected_tag_side_px"]) < hi
            ]
            heatmap[b_idx, bin_idx] = float(np.mean(values)) if values else np.nan

    output = {
        "blur_kernels_px": BLUR_KERNELS,
        "thresholds": thresholds,
        "bins_tag_side_px": [{"min": lo, "max": hi} for lo, hi in bins],
        "mean_detected_count_by_blur_and_bin": np.nan_to_num(heatmap, nan=-1).tolist(),
    }
    SUMMARY_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    colors = {0: "#1f77b4", 3: "#2ca02c", 7: "#ff7f0e", 11: "#d62728", 15: "#9467bd"}
    for blur in BLUR_KERNELS:
        blur_rows = by_blur[blur]
        # Group by projected tag pixels; average duplicate scale-factor samples.
        grouped: dict[float, list[int]] = defaultdict(list)
        for row in blur_rows:
            grouped[round(float(row["projected_tag_side_px"]), 6)].append(int(row["detected_count"]))
        xs = sorted(grouped)
        ys = [float(np.mean(grouped[x])) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors[blur], label=f"{blur}px blur")
        full = thresholds[str(blur)]["first_all_visible_36_tags_px"]
        if full is not None:
            ax.axvline(full, color=colors[blur], linestyle=":", alpha=0.65)
    ax.set_xlabel("Projected AprilTag side length (pixels)")
    ax.set_ylabel("Mean detected tags out of 36")
    ax.set_title("AprilTag Readability vs Pixels per Tag Side With Motion Blur")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    im = ax.imshow(heatmap, cmap="viridis", vmin=0, vmax=36, aspect="auto")
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in bins], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(BLUR_KERNELS)))
    ax.set_yticklabels([str(b) for b in BLUR_KERNELS])
    ax.set_xlabel("Projected tag side length bin (px)")
    ax.set_ylabel("Motion blur kernel (px)")
    ax.set_title("Mean Detected Tags by Pixel Size and Blur")
    for y in range(len(BLUR_KERNELS)):
        for x in range(len(bins)):
            if not np.isnan(heatmap[y, x]):
                ax.text(x, y, f"{heatmap[y, x]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean detected tags")
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG)
    plt.close(fig)

    print(json.dumps(output, indent=2))
    print(f"CSV: {SUMMARY_CSV}")
    print(f"Plot: {PLOT_PNG}")
    print(f"Heatmap: {HEATMAP_PNG}")


if __name__ == "__main__":
    main()

"""
Analyze distance/tag-size ratio behavior for rendered AprilGrid images.

This script resizes each ratio-scale render to several widths, detects AprilTags,
checks scale invariance across target sizes, and plots heatmaps using
distance/tag-size ratio on the Y axis.
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
ANNOTATED_DIR = OUT_DIR / "annotated"
CASES_CSV = OUT_DIR / "ratio_scale_cases.csv"
SUMMARY_CSV = OUT_DIR / "ratio_scale_detection_summary.csv"
SUMMARY_JSON = OUT_DIR / "ratio_scale_summary.json"
HEATMAP_MEAN_PNG = OUT_DIR / "ratio_heatmap_mean_detected.png"
HEATMAP_MIN_PNG = OUT_DIR / "ratio_heatmap_min_detected.png"
INVARIANCE_PNG = OUT_DIR / "scale_invariance_image_diff.png"
WIDTHS = [960, 400, 300, 200, 150, 100]


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


def draw_detection(image: np.ndarray, tag_id: int, corners: np.ndarray) -> None:
    pts = np.rint(corners.reshape(4, 2)).astype(int)
    cv2.polylines(image, [pts], True, (45, 220, 60), 2, cv2.LINE_AA)
    center = pts.mean(axis=0).astype(int)
    cv2.putText(
        image,
        str(tag_id),
        tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def detect(detector: cv2.aruco.ArucoDetector, image: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None:
        return [], []
    return [int(i) for i in ids.flatten()], corners


def safe_label(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def analyze_detections(cases: list[dict[str, str]]) -> list[dict[str, object]]:
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    detector = make_detector()
    rows: list[dict[str, object]] = []

    for case in cases:
        source = cv2.imread(str(IMAGE_DIR / case["image"]), cv2.IMREAD_COLOR)
        if source is None:
            raise SystemExit(f"Could not read {case['image']}")

        h0, w0 = source.shape[:2]
        ratio = float(case["distance_over_tag_size"])
        scale = float(case["scale_factor"])

        for width in WIDTHS:
            height = int(round(h0 * width / w0))
            resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
            detected_ids, corners = detect(detector, resized)
            unique_ids = sorted(set(detected_ids))

            annotated = resized.copy()
            for tag_id, tag_corners in zip(detected_ids, corners):
                draw_detection(annotated, tag_id, tag_corners)
            label = f"ratio {ratio:.2f}, scale {scale:g}, {width}px | {len(unique_ids)}/36"
            cv2.rectangle(annotated, (0, 0), (min(width, 390), 26), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                label,
                (5, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            image_name = f"ratio_r{safe_label(ratio)}_s{str(scale).replace('.', 'p')}_w{width}.png"
            cv2.imwrite(str(ANNOTATED_DIR / image_name), annotated)

            rows.append(
                {
                    **case,
                    "width_px": width,
                    "height_px": height,
                    "detected_count": len(unique_ids),
                    "detected_ids": " ".join(str(i) for i in unique_ids),
                    "annotated_image": str(ANNOTATED_DIR / image_name),
                }
            )

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def compare_scaled_images(cases: list[dict[str, str]]) -> dict[str, object]:
    """Compare 1x and 10x images for matching ratios."""
    by_ratio_scale = {
        (round(float(row["distance_over_tag_size"]), 6), round(float(row["scale_factor"]), 6)): row
        for row in cases
    }
    comparisons = []
    ratios = sorted({round(float(row["distance_over_tag_size"]), 6) for row in cases})
    for ratio in ratios:
        row_1 = by_ratio_scale.get((ratio, 1.0))
        row_10 = by_ratio_scale.get((ratio, 10.0))
        if not row_1 or not row_10:
            continue
        image_1 = cv2.imread(str(IMAGE_DIR / row_1["image"]), cv2.IMREAD_COLOR)
        image_10 = cv2.imread(str(IMAGE_DIR / row_10["image"]), cv2.IMREAD_COLOR)
        diff = cv2.absdiff(image_1, image_10)
        comparisons.append(
            {
                "ratio": ratio,
                "mean_abs_pixel_diff": float(diff.mean()),
                "max_abs_pixel_diff": int(diff.max()),
            }
        )

    if comparisons:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.plot(
            [c["ratio"] for c in comparisons],
            [c["mean_abs_pixel_diff"] for c in comparisons],
            marker="o",
        )
        ax.set_xlabel("Distance / tag size")
        ax.set_ylabel("Mean absolute pixel difference")
        ax.set_title("Rendered Image Difference: 1x Target vs 10x Target")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(INVARIANCE_PNG)
        plt.close(fig)

    return {
        "comparisons_1x_vs_10x": comparisons,
        "max_mean_abs_pixel_diff": max((c["mean_abs_pixel_diff"] for c in comparisons), default=0.0),
        "max_abs_pixel_diff": max((c["max_abs_pixel_diff"] for c in comparisons), default=0),
    }


def summarize(rows: list[dict[str, object]], invariance: dict[str, object]) -> dict[str, object]:
    ratios = sorted({round(float(row["distance_over_tag_size"]), 6) for row in rows})
    scales = sorted({float(row["scale_factor"]) for row in rows})

    mean_counts = np.zeros((len(ratios), len(WIDTHS)), dtype=float)
    min_counts = np.zeros_like(mean_counts)
    max_counts = np.zeros_like(mean_counts)
    std_counts = np.zeros_like(mean_counts)

    grouped: dict[tuple[float, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (round(float(row["distance_over_tag_size"]), 6), int(row["width_px"]))
        grouped[key].append(int(row["detected_count"]))

    for r_idx, ratio in enumerate(ratios):
        for w_idx, width in enumerate(WIDTHS):
            counts = grouped[(ratio, width)]
            mean_counts[r_idx, w_idx] = float(np.mean(counts))
            min_counts[r_idx, w_idx] = float(np.min(counts))
            max_counts[r_idx, w_idx] = float(np.max(counts))
            std_counts[r_idx, w_idx] = float(np.std(counts))

    summary = {
        "ratios_distance_over_tag_size": ratios,
        "widths_px": WIDTHS,
        "scale_factors": scales,
        "mean_detected_count": mean_counts.tolist(),
        "min_detected_count": min_counts.tolist(),
        "max_detected_count": max_counts.tolist(),
        "std_detected_count_across_scales": std_counts.tolist(),
        "max_std_detected_count_across_scales": float(std_counts.max()),
        "best_ratio_by_width_mean": {
            str(width): {
                "ratio": ratios[int(np.argmax(mean_counts[:, i]))],
                "mean_detected_count": float(np.max(mean_counts[:, i])),
                "min_detected_count_at_ratio": float(min_counts[int(np.argmax(mean_counts[:, i])), i]),
            }
            for i, width in enumerate(WIDTHS)
        },
        "scale_invariance_image_diff": invariance,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_heatmap(matrix: np.ndarray, ratios: list[float], title: str, output: Path, vmin: float = 0, vmax: float = 36) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(WIDTHS)))
    ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.set_yticks(np.arange(len(ratios)))
    ax.set_yticklabels([f"{r:.2f}" for r in ratios])
    ax.set_xlabel("Resized image width (px)")
    ax.set_ylabel("Camera distance / AprilTag side length")
    ax.set_title(title)
    for r in range(len(ratios)):
        for c in range(len(WIDTHS)):
            value = matrix[r, c]
            label = f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"
            ax.text(c, r, label, ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="Detected tags")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = read_cases()
    invariance = compare_scaled_images(cases)
    rows = analyze_detections(cases)
    summary = summarize(rows, invariance)

    ratios = summary["ratios_distance_over_tag_size"]
    mean_counts = np.array(summary["mean_detected_count"], dtype=float)
    min_counts = np.array(summary["min_detected_count"], dtype=float)
    plot_heatmap(mean_counts, ratios, "Mean Detected AprilTags by Distance/Tag-Size Ratio", HEATMAP_MEAN_PNG)
    plot_heatmap(min_counts, ratios, "Worst-Case Detected AprilTags Across Scale Factors", HEATMAP_MIN_PNG)

    print(json.dumps(summary, indent=2))
    print(f"Detection summary: {SUMMARY_CSV}")
    print(f"Mean heatmap: {HEATMAP_MEAN_PNG}")
    print(f"Min heatmap: {HEATMAP_MIN_PNG}")
    print(f"Image-diff plot: {INVARIANCE_PNG}")
    print(f"Annotated images: {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()

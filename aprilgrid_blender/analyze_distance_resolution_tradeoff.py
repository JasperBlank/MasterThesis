"""
Analyze whether moving closer recovers AprilTag detections at lower resolutions.

This reuses the distance renders from `render_degradation_matrix.py`, resizes
each distance view to several widths, and measures detection count.
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
MATRIX_DIR = ROOT / "degradation_matrix"
IMAGE_DIR = MATRIX_DIR / "images"
OUT_DIR = MATRIX_DIR / "distance_resolution"
ANNOTATED_DIR = OUT_DIR / "annotated"
CASES_RENDERED_CSV = MATRIX_DIR / "cases_rendered.csv"
SUMMARY_CSV = OUT_DIR / "distance_resolution_summary.csv"
SUMMARY_JSON = OUT_DIR / "distance_resolution_summary.json"
HEATMAP_PNG = OUT_DIR / "distance_resolution_heatmap.png"
RECOVERY_PNG = OUT_DIR / "distance_resolution_recovery.png"

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


def read_distance_cases() -> list[dict[str, str]]:
    with CASES_RENDERED_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cases = [
        row
        for row in rows
        if row["factor"] == "distance" or row["factor"] == "baseline"
    ]
    return sorted(cases, key=lambda row: float(row["distance_m"]))


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


def detect_ids(detector: cv2.aruco.ArucoDetector, image: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None:
        return [], []
    return [int(i) for i in ids.flatten()], corners


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    detector = make_detector()
    cases = read_distance_cases()

    rows: list[dict[str, object]] = []
    full_res_by_distance: dict[float, int] = {}

    for case in cases:
        source = cv2.imread(str(IMAGE_DIR / case["image"]), cv2.IMREAD_COLOR)
        if source is None:
            raise SystemExit(f"Could not read {case['image']}")

        distance = float(case["distance_m"])
        h0, w0 = source.shape[:2]

        distance_rows = []
        for width in WIDTHS:
            scale = width / w0
            height = int(round(h0 * scale))
            resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
            detected_ids, corners = detect_ids(detector, resized)
            unique_ids = sorted(set(detected_ids))

            if width == 960:
                full_res_by_distance[distance] = len(unique_ids)

            annotated = resized.copy()
            for tag_id, tag_corners in zip(detected_ids, corners):
                draw_detection(annotated, tag_id, tag_corners)
            label = f"{distance:.2f}m @ {width}px | {len(unique_ids)}/36"
            cv2.rectangle(annotated, (0, 0), (min(width, 260), 26), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                label,
                (5, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            image_name = f"d{distance:.2f}m_w{width}px.png".replace(".", "p")
            image_name = image_name.replace("ppng", ".png")
            cv2.imwrite(str(ANNOTATED_DIR / image_name), annotated)

            distance_rows.append(
                {
                    "distance_m": distance,
                    "width_px": width,
                    "height_px": height,
                    "detected_count": len(unique_ids),
                    "detected_ids": " ".join(str(i) for i in unique_ids),
                    "source_case_id": case["case_id"],
                    "source_image": case["image"],
                    "annotated_image": str(ANNOTATED_DIR / image_name),
                }
            )
        rows.extend(distance_rows)

    for row in rows:
        full_res_count = full_res_by_distance[float(row["distance_m"])]
        row["full_res_count_at_same_distance"] = full_res_count
        row["recovery_rate_vs_full_res"] = (
            float(row["detected_count"]) / full_res_count if full_res_count else 0.0
        )
        row["detection_rate_vs_all_36"] = float(row["detected_count"]) / 36.0

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    distances = sorted({float(row["distance_m"]) for row in rows})
    counts = np.zeros((len(distances), len(WIDTHS)), dtype=float)
    recovery = np.zeros_like(counts)
    for r, distance in enumerate(distances):
        for c, width in enumerate(WIDTHS):
            match = next(
                row
                for row in rows
                if float(row["distance_m"]) == distance and int(row["width_px"]) == width
            )
            counts[r, c] = float(match["detected_count"])
            recovery[r, c] = float(match["recovery_rate_vs_full_res"])

    summary = {
        "widths_px": WIDTHS,
        "distances_m": distances,
        "counts": counts.tolist(),
        "recovery_rate_vs_full_res": recovery.tolist(),
        "best_by_width": {
            str(width): {
                "distance_m": distances[int(np.argmax(counts[:, i]))],
                "detected_count": int(np.max(counts[:, i])),
            }
            for i, width in enumerate(WIDTHS)
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    im = ax.imshow(counts, cmap="viridis", vmin=0, vmax=36)
    ax.set_xticks(np.arange(len(WIDTHS)))
    ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.set_yticks(np.arange(len(distances)))
    ax.set_yticklabels([f"{d:.2f}" for d in distances])
    ax.set_xlabel("Resized image width (px)")
    ax.set_ylabel("Camera distance (m)")
    ax.set_title("Detected AprilTags: Distance x Resolution")
    for r in range(len(distances)):
        for c in range(len(WIDTHS)):
            ax.text(c, r, str(int(counts[r, c])), ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="Detected tags")
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    im = ax.imshow(recovery, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(WIDTHS)))
    ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.set_yticks(np.arange(len(distances)))
    ax.set_yticklabels([f"{d:.2f}" for d in distances])
    ax.set_xlabel("Resized image width (px)")
    ax.set_ylabel("Camera distance (m)")
    ax.set_title("Recovery Rate vs Full Resolution at Same Distance")
    for r in range(len(distances)):
        for c in range(len(WIDTHS)):
            ax.text(c, r, f"{recovery[r, c]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="Recovery rate")
    fig.tight_layout()
    fig.savefig(RECOVERY_PNG)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Heatmap: {HEATMAP_PNG}")
    print(f"Recovery plot: {RECOVERY_PNG}")
    print(f"Annotated images: {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()

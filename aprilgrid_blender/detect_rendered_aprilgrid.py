"""
Verify AprilTag detections on the synthetic Blender render dataset.

Run from PowerShell after rendering:
  python aprilgrid_blender\\detect_rendered_aprilgrid.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender")
RENDER_DIR = ROOT / "renders"
ANNOTATED_DIR = RENDER_DIR / "annotated"
DETECTION_CSV = RENDER_DIR / "detections.csv"
SUMMARY_JSON = RENDER_DIR / "detection_summary.json"


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


def draw_detection(image: np.ndarray, tag_id: int, corners: np.ndarray) -> None:
    pts = np.rint(corners.reshape(4, 2)).astype(int)
    color = (45, 220, 60)
    cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
    cv2.circle(image, tuple(pts[0]), 5, (0, 0, 255), -1, cv2.LINE_AA)
    center = pts.mean(axis=0).astype(int)
    cv2.putText(
        image,
        str(tag_id),
        tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    detector = make_detector()

    image_paths = sorted(RENDER_DIR.glob("aprilgrid_render_*.png"))
    if not image_paths:
        raise SystemExit(f"No rendered images found in {RENDER_DIR}")

    rows: list[dict[str, object]] = []
    per_image_counts: dict[str, int] = {}
    per_tag_counts: Counter[int] = Counter()

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        annotated = image.copy()
        detected_ids: list[int] = []
        if ids is not None:
            for tag_id, tag_corners in zip(ids.flatten(), corners):
                tag_id = int(tag_id)
                detected_ids.append(tag_id)
                per_tag_counts[tag_id] += 1
                draw_detection(annotated, tag_id, tag_corners)

                pts = np.rint(tag_corners.reshape(4, 2)).astype(int)
                for corner_index, (x, y) in enumerate(pts):
                    rows.append(
                        {
                            "image": image_path.name,
                            "id": tag_id,
                            "corner_index": corner_index,
                            "x": int(x),
                            "y": int(y),
                        }
                    )

        per_image_counts[image_path.name] = len(detected_ids)
        label = f"detected {len(detected_ids)} tags"
        cv2.rectangle(annotated, (0, 0), (230, 30), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(ANNOTATED_DIR / image_path.name), annotated)

    with DETECTION_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "id", "corner_index", "x", "y"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rendered_images": len(image_paths),
        "images_with_any_detection": sum(1 for count in per_image_counts.values() if count > 0),
        "min_tags_detected_per_image": min(per_image_counts.values()),
        "max_tags_detected_per_image": max(per_image_counts.values()),
        "mean_tags_detected_per_image": sum(per_image_counts.values()) / len(per_image_counts),
        "per_image_counts": per_image_counts,
        "per_tag_counts": dict(sorted(per_tag_counts.items())),
        "detected_unique_ids": sorted(per_tag_counts),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Annotated renders: {ANNOTATED_DIR}")
    print(f"Detection CSV: {DETECTION_CSV}")


if __name__ == "__main__":
    main()

"""
Detect AprilTags in every degradation-matrix image and summarize robustness.
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
ANNOTATED_DIR = MATRIX_DIR / "annotated"
CASES_CSV = MATRIX_DIR / "cases_all.csv"
SUMMARY_CSV = MATRIX_DIR / "detection_summary_by_case.csv"
DETECTIONS_CSV = MATRIX_DIR / "detections_by_corner.csv"
SUMMARY_JSON = MATRIX_DIR / "detection_summary_by_factor.json"
SUMMARY_PNG = MATRIX_DIR / "degradation_summary.png"
REPORT_MD = MATRIX_DIR / "degradation_report.md"


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
    cv2.circle(image, tuple(pts[0]), 4, (0, 0, 255), -1, cv2.LINE_AA)
    center = pts.mean(axis=0).astype(int)
    cv2.putText(
        image,
        str(tag_id),
        tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def parse_level_for_sort(level: str) -> float:
    cleaned = (
        level.replace("baseline", "0")
        .replace("px", "")
        .replace("mm", "")
        .replace("m", "")
        .replace("deg", "")
        .replace("pct", "")
        .replace("x", "")
    )
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def detect_all() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    detector = make_detector()
    cases = read_cases()

    summary_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []

    for case in cases:
        image_path = IMAGE_DIR / case["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        annotated = image.copy()

        detected_ids: list[int] = []
        if ids is not None:
            for tag_id, tag_corners in zip(ids.flatten(), corners):
                tag_id = int(tag_id)
                detected_ids.append(tag_id)
                draw_detection(annotated, tag_id, tag_corners)
                pts = np.rint(tag_corners.reshape(4, 2)).astype(int)
                for corner_index, (x, y) in enumerate(pts):
                    detection_rows.append(
                        {
                            "case_id": case["case_id"],
                            "image": case["image"],
                            "factor": case["factor"],
                            "level": case["level"],
                            "id": tag_id,
                            "corner_index": corner_index,
                            "x": int(x),
                            "y": int(y),
                        }
                    )

        expected = int(float(case["expected_visible_tags"]))
        detected_unique = sorted(set(detected_ids))
        detected_count = len(detected_unique)
        detection_rate = detected_count / expected if expected else 0.0

        label = f"{case['factor']} {case['level']} | {detected_count}/{expected}"
        cv2.rectangle(annotated, (0, 0), (min(520, annotated.shape[1]), 32), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(ANNOTATED_DIR / case["image"]), annotated)

        summary_rows.append(
            {
                **case,
                "detected_count": detected_count,
                "detection_rate": detection_rate,
                "detected_ids": " ".join(str(i) for i in detected_unique),
                "missed_count": max(expected - detected_count, 0),
                "has_any_detection": detected_count > 0,
                "all_expected_detected": detected_count >= expected,
            }
        )

    return summary_rows, detection_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def factor_summary(summary_rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["factor"])].append(row)

    result: dict[str, object] = {}
    for factor, rows in grouped.items():
        counts = [int(row["detected_count"]) for row in rows]
        rates = [float(row["detection_rate"]) for row in rows]
        result[factor] = {
            "cases": len(rows),
            "min_detected_count": min(counts),
            "max_detected_count": max(counts),
            "mean_detected_count": sum(counts) / len(counts),
            "min_detection_rate": min(rates),
            "mean_detection_rate": sum(rates) / len(rates),
            "levels": {
                str(row["level"]): {
                    "detected_count": int(row["detected_count"]),
                    "expected_visible_tags": int(float(str(row["expected_visible_tags"]))),
                    "detection_rate": float(row["detection_rate"]),
                }
                for row in sorted(rows, key=lambda r: parse_level_for_sort(str(r["level"])))
            },
        }
    return result


def make_plot(summary_rows: list[dict[str, object]]) -> None:
    factors = [
        "resolution",
        "distance",
        "viewing_angle",
        "motion_blur",
        "lighting",
        "partial_occlusion",
        "visible_tags",
        "tag_size",
        "spacing",
    ]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["factor"])].append(row)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=150)
    axes_flat = axes.flatten()
    for ax, factor in zip(axes_flat, factors):
        rows = sorted(grouped.get(factor, []), key=lambda r: parse_level_for_sort(str(r["level"])))
        if not rows:
            ax.axis("off")
            continue
        labels = [str(row["level"]) for row in rows]
        counts = [int(row["detected_count"]) for row in rows]
        expected = [int(float(str(row["expected_visible_tags"]))) for row in rows]
        x = np.arange(len(rows))
        ax.plot(x, expected, color="0.65", linestyle="--", marker="o", label="expected")
        ax.plot(x, counts, color="#1f77b4", marker="o", label="detected")
        ax.set_title(factor.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylim(bottom=0, top=max(max(expected), 36) + 2)
        ax.grid(True, alpha=0.25)
        if factor == "resolution":
            ax.invert_xaxis()
    axes_flat[0].legend(loc="lower left")
    fig.suptitle("AprilGrid Detection Robustness Matrix", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(SUMMARY_PNG)
    plt.close(fig)


def write_report(summary_rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    hardest = sorted(summary_rows, key=lambda row: (float(row["detection_rate"]), int(row["detected_count"])))[:8]
    lines = [
        "# AprilGrid Degradation Matrix Report",
        "",
        f"Cases tested: {len(summary_rows)}",
        "",
        "## Factor Summary",
        "",
        "| Factor | Cases | Min detected | Mean detected | Min rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for factor, data in sorted(summary.items()):
        lines.append(
            f"| {factor} | {data['cases']} | {data['min_detected_count']} | "
            f"{data['mean_detected_count']:.2f} | {data['min_detection_rate']:.2f} |"
        )
    lines.extend(["", "## Hardest Cases", ""])
    lines.append("| Case | Factor | Level | Detected / Expected |")
    lines.append("| --- | --- | --- | ---: |")
    for row in hardest:
        lines.append(
            f"| {row['case_id']} | {row['factor']} | {row['level']} | "
            f"{row['detected_count']} / {int(float(str(row['expected_visible_tags'])))} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Resolution cases preserve aspect ratio by resizing the baseline render width.",
            "- Motion blur is a horizontal linear kernel applied in image space.",
            "- Lighting is an exposure multiplier applied in image space.",
            "- Partial occlusion covers the right side of the board in Blender.",
            "- Visible-tag cases use clean textures where only the center-ranked subset of tags is drawn.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary_rows, detection_rows = detect_all()
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(DETECTIONS_CSV, detection_rows)
    summary = factor_summary(summary_rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plot(summary_rows)
    write_report(summary_rows, summary)

    print(json.dumps(summary, indent=2))
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Corner detections CSV: {DETECTIONS_CSV}")
    print(f"Plot: {SUMMARY_PNG}")
    print(f"Report: {REPORT_MD}")
    print(f"Annotated images: {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()

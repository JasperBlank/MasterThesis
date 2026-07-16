"""Calibrate both endoscopes and their stereo transform from AprilTag images.

Each detected tag is an independent metric square for the per-camera intrinsics
fit.  The stereo transform additionally uses two or more tags fixed to one flat,
rigid target; their spacing is recovered from the images and need not be entered.
Use many stationary captures with varied tilt, distance, image position, and
roll.  A separate manifest may contain different-size validation tags.
"""

from __future__ import print_function

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Motordriver"))

from apriltag_tracker import create_detector, detect_tags  # noqa: E402


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tag_edge_mm(manifest: Dict[str, object], tag_id: int) -> Optional[float]:
    overrides = manifest.get("tag_edges_mm", {})
    if str(tag_id) in overrides:
        return float(overrides[str(tag_id)])
    value = manifest.get("default_tag_edge_mm")
    return None if value is None else float(value)


def square_object_points(edge_mm: float) -> np.ndarray:
    half = 0.5 * edge_mm
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def collect_observations(manifest_path: Path) -> Tuple[List[Dict[str, object]], Tuple[int, int], Dict[str, object]]:
    manifest = load_json(manifest_path)
    root = manifest_path.parent
    detector = create_detector(str(manifest.get("family", "36h11")))
    observations = []
    image_size = None
    for capture in manifest.get("captures", []):
        left_path = root / capture["left_image"]
        right_path = root / capture["right_image"]
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None or right is None:
            print("WARNING: skipped unreadable pair %s / %s" % (left_path, right_path))
            continue
        current_size = (int(left.shape[1]), int(left.shape[0]))
        if right.shape[1] != current_size[0] or right.shape[0] != current_size[1]:
            raise ValueError("left/right size mismatch in capture %s" % capture.get("id"))
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError("all calibration images must have one resolution")

        left_by_id = dict((item.tag_id, item) for item in detect_tags(left, detector))
        right_by_id = dict((item.tag_id, item) for item in detect_tags(right, detector))
        common_ids = sorted(set(left_by_id).intersection(right_by_id))
        for tag_id in common_ids:
            edge_mm = tag_edge_mm(manifest, tag_id)
            if edge_mm is None:
                continue
            observations.append(
                {
                    "capture_id": int(capture["id"]),
                    "tag_id": int(tag_id),
                    "edge_mm": edge_mm,
                    "object_points": square_object_points(edge_mm),
                    "left_points": left_by_id[tag_id].corners.astype(np.float64),
                    "right_points": right_by_id[tag_id].corners.astype(np.float64),
                    "left_image": str(left_path),
                    "right_image": str(right_path),
                }
            )
    if image_size is None:
        raise ValueError("no readable calibration captures in %s" % manifest_path)
    return observations, image_size, manifest


def split_by_capture(
    observations: List[Dict[str, object]], validation_fraction: float
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    capture_ids = sorted(set(int(item["capture_id"]) for item in observations))
    if len(capture_ids) < 5:
        return observations, []
    shuffled = list(capture_ids)
    random.Random(20260716).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_ids = set(shuffled[:validation_count])
    training = [item for item in observations if int(item["capture_id"]) not in validation_ids]
    validation = [item for item in observations if int(item["capture_id"]) in validation_ids]
    return training, validation


def as_calibration_lists(
    observations: Sequence[Dict[str, object]], camera: str, fisheye: bool
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    objects = []
    images = []
    key = camera + "_points"
    for item in observations:
        object_points = np.asarray(item["object_points"], dtype=np.float64)
        image_points = np.asarray(item[key], dtype=np.float64)
        if fisheye:
            objects.append(object_points.reshape(-1, 1, 3))
            images.append(image_points.reshape(-1, 1, 2))
        else:
            objects.append(object_points.astype(np.float32))
            images.append(image_points.astype(np.float32))
    return objects, images


def initial_camera_matrix(image_size: Tuple[int, int], focal_px: float) -> np.ndarray:
    width, height = image_size
    return np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def calibrate_intrinsics(
    observations: Sequence[Dict[str, object]],
    camera: str,
    image_size: Tuple[int, int],
    model: str,
    initial_focal_px: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    fisheye = model == "fisheye"
    objects, images = as_calibration_lists(observations, camera, fisheye)
    camera_matrix = initial_camera_matrix(image_size, initial_focal_px)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-10)
    if fisheye:
        distortion = np.zeros((4, 1), dtype=np.float64)
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.CALIB_FIX_SKEW
        )
        rms, camera_matrix, distortion, _, _ = cv2.fisheye.calibrate(
            objects,
            images,
            image_size,
            camera_matrix,
            distortion,
            flags=flags,
            criteria=criteria,
        )
    else:
        coefficient_count = 8 if model == "rational" else 5
        distortion = np.zeros((coefficient_count, 1), dtype=np.float64)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        if model == "rational":
            flags |= cv2.CALIB_RATIONAL_MODEL
        rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
            objects,
            images,
            image_size,
            camera_matrix,
            distortion,
            flags=flags,
            criteria=criteria,
        )
    return float(rms), camera_matrix, distortion


def calibrate_stereo(
    observations: Sequence[Dict[str, object]],
    image_size: Tuple[int, int],
    model: str,
    left_matrix: np.ndarray,
    left_distortion: np.ndarray,
    right_matrix: np.ndarray,
    right_distortion: np.ndarray,
    tag_gap_mm: Optional[float] = None,
    fix_intrinsics: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, object]]:
    grouped = {}
    tag_frequency = {}
    for item in observations:
        capture_id = int(item["capture_id"])
        grouped.setdefault(capture_id, {})[int(item["tag_id"])] = item
        tag_id = int(item["tag_id"])
        tag_frequency[tag_id] = tag_frequency.get(tag_id, 0) + 1
    reference_id = max(tag_frequency, key=tag_frequency.get)

    # Recover one metric layout for the rigid tag strip.  Undistorted points
    # obey a plane homography, so the reference tag supplies a millimetre frame
    # without requiring the gaps between tags to be measured by hand.
    layout_estimates = {}
    for capture in grouped.values():
        if reference_id not in capture:
            continue
        reference = capture[reference_id]
        for camera, matrix, distortion in (
            ("left", left_matrix, left_distortion),
            ("right", right_matrix, right_distortion),
        ):
            reference_points = undistort_points_px(
                reference[camera + "_points"], matrix, distortion, model
            )
            reference_xy = np.asarray(reference["object_points"], dtype=np.float64)[:, :2]
            homography, _ = cv2.findHomography(reference_points, reference_xy)
            if homography is None:
                continue
            for tag_id, item in capture.items():
                corrected = undistort_points_px(
                    item[camera + "_points"], matrix, distortion, model
                )
                mapped = cv2.perspectiveTransform(
                    corrected.reshape(-1, 1, 2), homography
                ).reshape(-1, 2)
                layout_estimates.setdefault(tag_id, []).append(mapped)

    layout = {}
    layout_errors = []
    for tag_id, estimates in layout_estimates.items():
        stack = np.asarray(estimates, dtype=np.float64)
        median_layout = np.median(stack, axis=0)
        example = grouped[next(
            capture_id for capture_id in grouped if tag_id in grouped[capture_id]
        )][tag_id]
        canonical = np.asarray(example["object_points"], dtype=np.float64)[:, :2]
        canonical_centered = canonical - np.mean(canonical, axis=0)
        measured_center = np.mean(median_layout, axis=0)
        measured_centered = median_layout - measured_center
        left_singular, _, right_singular = np.linalg.svd(
            np.dot(canonical_centered.T, measured_centered)
        )
        planar_rotation = np.dot(left_singular, right_singular)
        if np.linalg.det(planar_rotation) < 0.0:
            left_singular[:, -1] *= -1.0
            planar_rotation = np.dot(left_singular, right_singular)
        layout[tag_id] = np.dot(canonical_centered, planar_rotation) + measured_center
        for estimate in stack:
            layout_errors.append(
                float(np.sqrt(np.mean(np.sum(np.square(estimate - layout[tag_id]), axis=1))))
            )

    layout_source = "recovered from undistorted tag images"
    if tag_gap_mm is not None:
        if tag_gap_mm <= 0.0:
            raise ValueError("--tag-gap-mm must be positive")
        if len(layout) < 2:
            raise ValueError("--tag-gap-mm needs at least two tags on the rigid target")
        tag_centers = dict(
            (tag_id, np.mean(points, axis=0)) for tag_id, points in layout.items()
        )
        center_array = np.asarray(list(tag_centers.values()), dtype=np.float64)
        centered = center_array - np.mean(center_array, axis=0)
        _, _, right_singular = np.linalg.svd(centered, full_matrices=False)
        strip_axis = right_singular[0]
        ordered_ids = sorted(
            tag_centers,
            key=lambda tag_id: float(np.dot(tag_centers[tag_id], strip_axis)),
        )
        edge_by_id = {}
        canonical_by_id = {}
        for tag_id in ordered_ids:
            example = grouped[next(
                capture_id for capture_id in grouped if tag_id in grouped[capture_id]
            )][tag_id]
            edge_by_id[tag_id] = float(example["edge_mm"])
            canonical_by_id[tag_id] = np.asarray(
                example["object_points"], dtype=np.float64
            )[:, :2]
        positions = {ordered_ids[0]: 0.0}
        for previous_id, current_id in zip(ordered_ids[:-1], ordered_ids[1:]):
            positions[current_id] = (
                positions[previous_id]
                + 0.5 * edge_by_id[previous_id]
                + tag_gap_mm
                + 0.5 * edge_by_id[current_id]
            )
        reference_center = tag_centers[reference_id]
        origin = reference_center - strip_axis * positions[reference_id]
        for tag_id in ordered_ids:
            exact_center = origin + strip_axis * positions[tag_id]
            canonical = canonical_by_id[tag_id]
            layout[tag_id] = canonical - np.mean(canonical, axis=0) + exact_center
        layout_errors = []
        for tag_id, estimates in layout_estimates.items():
            for estimate in estimates:
                layout_errors.append(
                    float(np.sqrt(np.mean(np.sum(np.square(estimate - layout[tag_id]), axis=1))))
                )
        layout_source = "measured rigid strip with %.6g mm edge-to-edge gap" % tag_gap_mm

    object_views = []
    left_views = []
    right_views = []
    used_capture_ids = []
    for capture_id, capture in sorted(grouped.items()):
        usable_ids = sorted(set(capture).intersection(layout))
        if len(usable_ids) < 2:
            continue
        object_points = []
        left_points = []
        right_points = []
        for tag_id in usable_ids:
            object_points.extend(
                [[point[0], point[1], 0.0] for point in layout[tag_id]]
            )
            left_points.extend(np.asarray(capture[tag_id]["left_points"], dtype=np.float64))
            right_points.extend(np.asarray(capture[tag_id]["right_points"], dtype=np.float64))
        object_views.append(np.asarray(object_points, dtype=np.float32))
        left_views.append(np.asarray(left_points, dtype=np.float32))
        right_views.append(np.asarray(right_points, dtype=np.float32))
        used_capture_ids.append(capture_id)

    if len(object_views) < 5:
        raise ValueError(
            "stereo calibration needs at least five captures containing two rigidly fixed tags"
        )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-10)
    result = cv2.stereoCalibrate(
        object_views,
        left_views,
        right_views,
        left_matrix.copy(),
        left_distortion.copy(),
        right_matrix.copy(),
        right_distortion.copy(),
        image_size,
        criteria=criteria,
        flags=(cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else cv2.CALIB_USE_INTRINSIC_GUESS),
    )
    if not fix_intrinsics:
        left_matrix[:, :] = np.asarray(result[1], dtype=np.float64)
        left_distortion[:] = np.asarray(result[2], dtype=np.float64).reshape(left_distortion.shape)
        right_matrix[:, :] = np.asarray(result[3], dtype=np.float64)
        right_distortion[:] = np.asarray(result[4], dtype=np.float64).reshape(right_distortion.shape)
    stereo_rms = float(result[0])
    rotation = np.asarray(result[5], dtype=np.float64)
    translation = np.asarray(result[6], dtype=np.float64).reshape(3)
    per_capture_transforms = []
    for capture_id, object_points, left_points, right_points in zip(
        used_capture_ids, object_views, left_views, right_views
    ):
        ok_left, left_rvec, left_tvec = cv2.solvePnP(
            object_points,
            left_points,
            left_matrix,
            left_distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        ok_right, right_rvec, right_tvec = cv2.solvePnP(
            object_points,
            right_points,
            right_matrix,
            right_distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok_left or not ok_right:
            continue
        left_rotation, _ = cv2.Rodrigues(left_rvec)
        right_rotation, _ = cv2.Rodrigues(right_rvec)
        relative_rotation = np.dot(right_rotation, left_rotation.T)
        relative_translation = right_tvec.reshape(3) - np.dot(
            relative_rotation, left_tvec.reshape(3)
        )
        per_capture_transforms.append(
            {
                "capture_id": int(capture_id),
                "D_mm": float(np.linalg.norm(relative_translation)),
                "rotation_error_to_fit_deg": rotation_error_deg(rotation, relative_rotation),
            }
        )
    diagnostics = {
        "method": (
            "fixed-intrinsics stereoCalibrate using rigid tag-strip layout"
            if fix_intrinsics
            else "joint intrinsics/extrinsics stereoCalibrate using rigid tag-strip layout"
        ),
        "input_tag_observations": len(observations),
        "board_capture_count": len(object_views),
        "board_capture_ids": used_capture_ids,
        "reference_tag_id": int(reference_id),
        "layout_source": layout_source,
        "tag_gap_mm": tag_gap_mm,
        "layout_estimate_count": len(layout_errors),
        "layout_rms_mm": float(np.sqrt(np.mean(np.square(layout_errors)))),
        "layout_max_error_mm": float(np.max(layout_errors)),
        "layout_tag_corners_mm": dict(
            (str(tag_id), points.tolist()) for tag_id, points in layout.items()
        ),
        "per_capture_independent_pose_diagnostics": per_capture_transforms,
        "joint_reprojection_rms_px": stereo_rms,
    }
    return stereo_rms, rotation, translation, diagnostics


def undistort_points_px(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: str,
) -> np.ndarray:
    points = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if model == "fisheye":
        corrected = cv2.fisheye.undistortPoints(
            points, camera_matrix, distortion, P=camera_matrix
        )
    else:
        corrected = cv2.undistortPoints(points, camera_matrix, distortion, P=camera_matrix)
    return corrected.reshape(-1, 2)


def project_points(
    object_points: np.ndarray,
    rotation_vector: np.ndarray,
    translation_vector: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: str,
) -> np.ndarray:
    if model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(
            np.asarray(object_points, dtype=np.float64).reshape(1, -1, 3),
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )
    else:
        projected, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )
    return projected.reshape(-1, 2)


def solve_tag_pose(
    observation: Dict[str, object],
    camera: str,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    object_points = np.asarray(observation["object_points"], dtype=np.float64)
    raw_points = np.asarray(observation[camera + "_points"], dtype=np.float64)
    corrected = undistort_points_px(raw_points, camera_matrix, distortion, model)
    ok, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        corrected,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise ValueError("solvePnP failed for validation observation")
    rotation_vector, translation_vector = cv2.solvePnPRefineLM(
        object_points,
        corrected,
        camera_matrix,
        None,
        rotation_vector,
        translation_vector,
    )
    projected = project_points(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
        model,
    )
    rms = float(np.sqrt(np.mean(np.sum(np.square(projected - raw_points), axis=1))))
    return rotation_vector.reshape(3), translation_vector.reshape(3), rms


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.dot(first.T, second)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def validate(
    observations: Sequence[Dict[str, object]],
    model: str,
    left_matrix: np.ndarray,
    left_distortion: np.ndarray,
    right_matrix: np.ndarray,
    right_distortion: np.ndarray,
    stereo_rotation: np.ndarray,
) -> Dict[str, object]:
    left_errors = []
    right_errors = []
    lateral_D_values = []
    axial_offset_values = []
    separation_3d_values = []
    rotation_errors = []
    for item in observations:
        left_rvec, left_tvec, left_error = solve_tag_pose(
            item, "left", left_matrix, left_distortion, model
        )
        right_rvec, right_tvec, right_error = solve_tag_pose(
            item, "right", right_matrix, right_distortion, model
        )
        left_rotation, _ = cv2.Rodrigues(left_rvec)
        right_rotation, _ = cv2.Rodrigues(right_rvec)
        relative_rotation = np.dot(right_rotation, left_rotation.T)
        relative_translation = right_tvec - np.dot(relative_rotation, left_tvec)
        left_errors.append(left_error)
        right_errors.append(right_error)
        lateral_D_values.append(float(np.linalg.norm(relative_translation[:2])))
        axial_offset_values.append(float(abs(relative_translation[2])))
        separation_3d_values.append(float(np.linalg.norm(relative_translation)))
        rotation_errors.append(rotation_error_deg(stereo_rotation, relative_rotation))

    def summary(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "max": float("nan")}
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "max": float(np.max(array)),
        }

    return {
        "observation_count": len(observations),
        "left_reprojection_px": summary(left_errors),
        "right_reprojection_px": summary(right_errors),
        "per_tag_lateral_D_mm": summary(lateral_D_values),
        "per_tag_axial_offset_mm": summary(axial_offset_values),
        "per_tag_camera_separation_3d_mm": summary(separation_3d_values),
        "per_tag_relative_rotation_error_deg": summary(rotation_errors),
    }


def coverage_summary(
    observations: Sequence[Dict[str, object]], image_size: Tuple[int, int], camera: str
) -> Dict[str, object]:
    width, height = image_size
    occupied = set()
    perspective_ratios = []
    for item in observations:
        points = np.asarray(item[camera + "_points"], dtype=float)
        center = np.mean(points, axis=0)
        column = min(2, max(0, int(center[0] * 3.0 / width)))
        row = min(2, max(0, int(center[1] * 3.0 / height)))
        occupied.add((row, column))
        sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        perspective_ratios.append(float(np.max(sides) / max(np.min(sides), 1e-9)))
    return {
        "occupied_grid_cells_3x3": len(occupied),
        "grid_cells": [[row, column] for row, column in sorted(occupied)],
        "max_side_ratio": max(perspective_ratios) if perspective_ratios else 1.0,
    }


def save_undistorted_preview(
    observation: Dict[str, object],
    image_size: Tuple[int, int],
    model: str,
    left_matrix: np.ndarray,
    left_distortion: np.ndarray,
    right_matrix: np.ndarray,
    right_distortion: np.ndarray,
    output_dir: Path,
) -> None:
    for camera, matrix, distortion in (
        ("left", left_matrix, left_distortion),
        ("right", right_matrix, right_distortion),
    ):
        frame = cv2.imread(str(observation[camera + "_image"]))
        if frame is None:
            continue
        if model == "fisheye":
            new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                matrix, distortion, image_size, np.eye(3), balance=0.5
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                matrix, distortion, np.eye(3), new_matrix, image_size, cv2.CV_16SC2
            )
            corrected = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        else:
            new_matrix, _ = cv2.getOptimalNewCameraMatrix(
                matrix, distortion, image_size, 0.5, image_size
            )
            corrected = cv2.undistort(frame, matrix, distortion, None, new_matrix)
        cv2.imwrite(str(output_dir / (camera + "_undistorted_preview.png")), corrected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate stereo endoscopes from AprilTag pairs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "Motordriver" / "stereo_calibration_data" / "manifest.json",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=None,
        help="Optional independent dataset, ideally using different-size tags and new positions.",
    )
    parser.add_argument(
        "--model",
        choices=["pinhole", "rational"],
        default="pinhole",
        help="Pinhole is the stable default; rational needs substantially more tilted views.",
    )
    parser.add_argument("--initial-focal-px", type=float, default=450.0)
    parser.add_argument(
        "--tag-gap-mm",
        type=float,
        default=None,
        help="Measured edge-to-edge gap between adjacent tags on the rigid strip.",
    )
    parser.add_argument(
        "--joint-refine-intrinsics",
        action="store_true",
        help="Jointly refine intrinsics and the stereo transform after separate initialization.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-captures", type=int, default=12)
    parser.add_argument(
        "--capture-id-min",
        type=int,
        default=None,
        help="Diagnostic filter: ignore captures with a smaller ID.",
    )
    parser.add_argument(
        "--capture-id-max",
        type=int,
        default=None,
        help="Diagnostic filter: ignore captures with a larger ID.",
    )
    parser.add_argument(
        "--stereo-capture-id-min",
        type=int,
        default=None,
        help="Diagnostic filter applied only to the stereo-transform fit.",
    )
    parser.add_argument(
        "--stereo-capture-id-max",
        type=int,
        default=None,
        help="Diagnostic filter applied only to the stereo-transform fit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Motordriver" / "stereo_camera_calibration.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations, image_size, manifest = collect_observations(args.manifest)
    if args.capture_id_min is not None:
        observations = [
            item for item in observations if int(item["capture_id"]) >= args.capture_id_min
        ]
    if args.capture_id_max is not None:
        observations = [
            item for item in observations if int(item["capture_id"]) <= args.capture_id_max
        ]
    capture_count = len(set(int(item["capture_id"]) for item in observations))
    if capture_count < args.minimum_captures:
        raise SystemExit(
            "need at least %d usable stereo captures; found %d" % (args.minimum_captures, capture_count)
        )
    if args.validation_manifest is None:
        training, validation_observations = split_by_capture(
            observations, args.validation_fraction
        )
        validation_source = "held-out captures from calibration dataset"
    else:
        training = observations
        validation_observations, validation_size, _ = collect_observations(
            args.validation_manifest
        )
        if validation_size != image_size:
            raise SystemExit("validation images must use the calibration resolution")
        validation_source = str(args.validation_manifest)

    print("model: %s" % args.model)
    print("training: %d tag observations from %d stereo captures" % (
        len(training), len(set(int(item["capture_id"]) for item in training))
    ))
    print("validation: %d observations (%s)" % (len(validation_observations), validation_source))
    left_coverage = coverage_summary(training, image_size, "left")
    right_coverage = coverage_summary(training, image_size, "right")
    print("coverage: left %d/9 cells, right %d/9 cells" % (
        left_coverage["occupied_grid_cells_3x3"],
        right_coverage["occupied_grid_cells_3x3"],
    ))
    if min(left_coverage["occupied_grid_cells_3x3"], right_coverage["occupied_grid_cells_3x3"]) < 7:
        print("WARNING: weak image coverage; add captures near missing corners/edges")
    if max(left_coverage["max_side_ratio"], right_coverage["max_side_ratio"]) < 1.08:
        print("WARNING: weak perspective diversity; add strongly tilted target views")

    left_rms, left_matrix, left_distortion = calibrate_intrinsics(
        training, "left", image_size, args.model, args.initial_focal_px
    )
    right_rms, right_matrix, right_distortion = calibrate_intrinsics(
        training, "right", image_size, args.model, args.initial_focal_px
    )
    stereo_training = training
    if args.stereo_capture_id_min is not None:
        stereo_training = [
            item
            for item in stereo_training
            if int(item["capture_id"]) >= args.stereo_capture_id_min
        ]
    if args.stereo_capture_id_max is not None:
        stereo_training = [
            item
            for item in stereo_training
            if int(item["capture_id"]) <= args.stereo_capture_id_max
        ]
    stereo_fit_rms, stereo_rotation, stereo_translation, stereo_diagnostics = calibrate_stereo(
        stereo_training,
        image_size,
        args.model,
        left_matrix,
        left_distortion,
        right_matrix,
        right_distortion,
        args.tag_gap_mm,
        not args.joint_refine_intrinsics,
    )
    validation = validate(
        validation_observations,
        args.model,
        left_matrix,
        left_distortion,
        right_matrix,
        right_distortion,
        stereo_rotation,
    )

    stereo_translation = np.asarray(stereo_translation, dtype=np.float64).reshape(3)
    lateral_D_mm = float(np.linalg.norm(stereo_translation[:2]))
    axial_offset_mm = float(abs(stereo_translation[2]))
    separation_3d_mm = float(np.linalg.norm(stereo_translation))
    validation_left = validation["left_reprojection_px"]["mean"]
    validation_right = validation["right_reprojection_px"]["mean"]
    quality_checks = {
        "intrinsics_validation_below_1px": bool(
            (not validation_observations)
            or (validation_left <= 1.0 and validation_right <= 1.0)
        ),
        "stereo_joint_rms_below_1px": bool(stereo_fit_rms <= 1.0),
        "lateral_D_within_1mm_of_mechanical": bool(abs(lateral_D_mm - 5.178) <= 1.0),
        "axial_offset_within_reported_2mm_allowance": bool(axial_offset_mm <= 2.25),
    }
    quality_accepted = all(quality_checks.values())

    output = {
        "schema_version": 1,
        "model": args.model,
        "image_size": list(image_size),
        "family": manifest.get("family", "36h11"),
        "source_manifest": str(args.manifest),
        "validation_source": validation_source,
        "training_capture_count": len(set(int(item["capture_id"]) for item in training)),
        "training_observation_count": len(training),
        "coverage": {"left": left_coverage, "right": right_coverage},
        "left": {
            "camera_matrix": left_matrix.tolist(),
            "distortion_coefficients": left_distortion.reshape(-1).tolist(),
            "training_rms_px": left_rms,
        },
        "right": {
            "camera_matrix": right_matrix.tolist(),
            "distortion_coefficients": right_distortion.reshape(-1).tolist(),
            "training_rms_px": right_rms,
        },
        "stereo": {
            "rotation_left_to_right": stereo_rotation.tolist(),
            "translation_left_to_right_mm": stereo_translation.tolist(),
            "D_mm": lateral_D_mm,
            "lateral_D_mm": lateral_D_mm,
            "axial_offset_mm": axial_offset_mm,
            "camera_center_separation_3d_mm": separation_3d_mm,
            "joint_reprojection_rms_px": stereo_fit_rms,
            "diagnostics": stereo_diagnostics,
        },
        "validation": validation,
        "quality": {
            "accepted": quality_accepted,
            "checks": quality_checks,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    preview_source = validation_observations[0] if validation_observations else training[0]
    save_undistorted_preview(
        preview_source,
        image_size,
        args.model,
        left_matrix,
        left_distortion,
        right_matrix,
        right_distortion,
        args.output.parent,
    )
    print("left RMS: %.3f px; right RMS: %.3f px; stereo joint RMS: %.3f px" % (
        left_rms, right_rms, stereo_fit_rms
    ))
    print("stereo lateral D: %.3f mm (mechanical D: %.3f mm)" % (
        lateral_D_mm, 5.178
    ))
    print("stereo axial offset: %.3f mm; 3D camera separation: %.3f mm" % (
        axial_offset_mm, separation_3d_mm
    ))
    if validation_observations:
        print("validation reprojection: left %.3f px, right %.3f px" % (
            validation["left_reprojection_px"]["mean"],
            validation["right_reprojection_px"]["mean"],
        ))
        print("validation per-tag lateral D: %.3f +/- %.3f mm" % (
            validation["per_tag_lateral_D_mm"]["mean"],
            validation["per_tag_lateral_D_mm"]["std"],
        ))
    if quality_accepted:
        print("quality gate: ACCEPTED")
    else:
        failed = [name for name, passed in quality_checks.items() if not passed]
        print("quality gate: REJECTED (%s)" % ", ".join(failed))
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()

"""Estimate the probe's 6-DOF pose in the AprilTag-sheet frame from the stereo endoscopes.

The scene contains exactly three 36h11 AprilTags (distinct IDs) on one flat
sheet. The pipeline is self-calibrating up to focal length:

1. Detect the tags in both endoscope images.
2. Recover the sheet layout (all tag corners in a common sheet frame, in units
   of one tag edge) from the reference tag's homography - no ruler needed.
3. solvePnP per camera -> camera pose in the sheet frame (tag-edge units).
4. The distance between the two camera centers must equal the known bore
   separation D = 5.178 mm (measured from Array_side_slip.STL), which yields
   the absolute scale (and, as a byproduct, the physical tag edge length).
5. The probe frame is built from the mean viewing axis and the baseline
   direction between the two camera centers.

Sheet frame: x/y in the sheet plane (units mm after scaling), z toward the
cameras. Camera frames follow OpenCV: z forward, x right, y down.

Offline:  python stereo_probe_pose.py --left-image L.png --right-image R.png
Live:     python stereo_probe_pose.py --left-camera 0 --right-camera 1
          (uses MSMF + tip LEDs at min, per lab policy)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apriltag_tracker import TagDetection, create_detector, detect_tags  # noqa: E402

SCOPE_SEPARATION_MM = 5.178  # bore center-to-center distance, from Array_side_slip.STL
EXPECTED_TAG_IDS = (1, 2, 3)

# Canonical corner coordinates of one tag in the sheet plane, in units of one
# tag edge, matching cv2.aruco corner order (TL, TR, BR, BL) for an upright tag
# seen from the front (sheet x right, y up, z toward the viewer).
CANONICAL_CORNERS = np.array(
    [
        [-0.5, 0.5],
        [0.5, 0.5],
        [0.5, -0.5],
        [-0.5, -0.5],
    ],
    dtype=np.float64,
)


def detections_by_id(detections: List[TagDetection]) -> Dict[int, TagDetection]:
    return dict((det.tag_id, det) for det in detections)


def recover_sheet_layout(
    detections: List[TagDetection],
    reference_id: int,
    corrected_corners: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[int, np.ndarray]:
    """Map every tag's corners into the sheet frame (tag-edge units) using the
    reference tag's homography. Assumes all tags are coplanar and equally sized."""
    by_id = detections_by_id(detections)
    if reference_id not in by_id:
        raise ValueError("reference tag %d not detected" % reference_id)
    ref = by_id[reference_id]
    ref_corners = ref.corners if corrected_corners is None else corrected_corners[reference_id]
    homography, _ = cv2.findHomography(ref_corners.astype(np.float64), CANONICAL_CORNERS)
    if homography is None:
        raise ValueError("homography for reference tag %d failed" % reference_id)
    layout = {}
    for det in detections:
        corners = det.corners if corrected_corners is None else corrected_corners[det.tag_id]
        pts = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float64), homography)
        layout[det.tag_id] = pts.reshape(4, 2)
    return layout


def camera_matrix(focal_px: float, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [focal_px, 0.0, width / 2.0],
            [0.0, focal_px, height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def undistort_points_px(
    points: np.ndarray,
    intrinsics: np.ndarray,
    distortion: Optional[np.ndarray],
    model: str,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if distortion is None or np.asarray(distortion).size == 0:
        return points.reshape(-1, 2)
    if model == "fisheye":
        corrected = cv2.fisheye.undistortPoints(
            points, intrinsics, distortion, P=intrinsics
        )
    else:
        corrected = cv2.undistortPoints(
            points, intrinsics, distortion, P=intrinsics
        )
    return corrected.reshape(-1, 2)


def project_points_px(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intrinsics: np.ndarray,
    distortion: Optional[np.ndarray],
    model: str,
) -> np.ndarray:
    if model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(
            object_points.reshape(1, -1, 3),
            rvec,
            tvec,
            intrinsics,
            distortion,
        )
    else:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            intrinsics,
            distortion,
        )
    return projected.reshape(-1, 2)


def camera_parameters(
    calibration: Dict[str, object], camera: str, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray, str]:
    calibration_width, calibration_height = calibration["image_size"]
    scale_x = float(width) / float(calibration_width)
    scale_y = float(height) / float(calibration_height)
    if abs(scale_x - scale_y) > 1e-3:
        raise ValueError("camera calibration aspect ratio does not match the captured frames")
    parameters = calibration[camera]
    intrinsics = np.asarray(parameters["camera_matrix"], dtype=np.float64).copy()
    intrinsics[0, 0] *= scale_x
    intrinsics[0, 2] *= scale_x
    intrinsics[1, 1] *= scale_y
    intrinsics[1, 2] *= scale_y
    distortion = np.asarray(
        parameters["distortion_coefficients"], dtype=np.float64
    ).reshape(-1, 1)
    return intrinsics, distortion, str(calibration["model"])


def solve_camera_pose(
    detections: List[TagDetection],
    layout: Dict[int, np.ndarray],
    intrinsics: np.ndarray,
    distortion: Optional[np.ndarray] = None,
    model: str = "pinhole",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """solvePnP against the shared sheet layout.

    Returns (R, t, reprojection_error_px) with sheet -> camera: x_cam = R.x_sheet + t.
    Translation is in tag-edge units until scaled."""
    object_points = []
    image_points = []
    for det in detections:
        if det.tag_id not in layout:
            continue
        sheet = layout[det.tag_id]
        object_points.extend([[p[0], p[1], 0.0] for p in sheet])
        image_points.extend(det.corners.tolist())
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    if len(object_points) < 8:
        raise ValueError("not enough shared tag corners for PnP (%d)" % len(object_points))

    corrected_points = undistort_points_px(
        image_points, intrinsics, distortion, model
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points, corrected_points, intrinsics, None, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        raise ValueError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points, corrected_points, intrinsics, None, rvec, tvec
    )
    projected = project_points_px(
        object_points, rvec, tvec, intrinsics, distortion, model
    )
    error = float(np.linalg.norm(projected - image_points, axis=1).mean())
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec.reshape(3), error


def camera_center_sheet(rotation: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return -rotation.T.dot(tvec)


def camera_axes_sheet(rotation: np.ndarray) -> Dict[str, np.ndarray]:
    """Camera basis vectors expressed in the sheet frame."""
    return {
        "view": rotation.T.dot(np.array([0.0, 0.0, 1.0])),
        "right": rotation.T.dot(np.array([1.0, 0.0, 0.0])),
        "down": rotation.T.dot(np.array([0.0, 1.0, 0.0])),
    }


def probe_pose_from_cameras(
    pose_left: Tuple[np.ndarray, np.ndarray],
    pose_right: Tuple[np.ndarray, np.ndarray],
    mm_per_unit: Optional[float] = None,
) -> Dict[str, object]:
    """Combine both camera poses into a probe pose.

    mm_per_unit is the physical tag edge length. If omitted, it is estimated
    from the known scope separation - beware: that becomes ill-conditioned when
    the sheet is much farther away than the 5.18 mm baseline. Prefer a stage
    -motion calibration (--calibrate) or a known tag size (--tag-edge-mm)."""
    r_left, t_left = pose_left
    r_right, t_right = pose_right
    center_left = camera_center_sheet(r_left, t_left)
    center_right = camera_center_sheet(r_right, t_right)

    separation_units = float(np.linalg.norm(center_right - center_left))
    scale_source = "tag-edge"
    if mm_per_unit is None:
        mm_per_unit = SCOPE_SEPARATION_MM / separation_units  # 1 unit = 1 tag edge
        scale_source = "scope-separation (ill-conditioned at distance!)"
    center_left_mm = center_left * mm_per_unit
    center_right_mm = center_right * mm_per_unit

    view_left = camera_axes_sheet(r_left)["view"]
    view_right = camera_axes_sheet(r_right)["view"]
    probe_axis = view_left + view_right
    probe_axis = probe_axis / np.linalg.norm(probe_axis)

    center_delta_mm = center_right_mm - center_left_mm
    axial_offset_mm = float(center_delta_mm.dot(probe_axis))
    lateral_delta_mm = center_delta_mm - axial_offset_mm * probe_axis
    lateral_d_mm = float(np.linalg.norm(lateral_delta_mm))

    baseline_dir = center_right_mm - center_left_mm
    baseline_dir = baseline_dir / np.linalg.norm(baseline_dir)
    # Orthonormal probe frame: axis (viewing direction), baseline, and normal.
    baseline_ortho = baseline_dir - probe_axis * baseline_dir.dot(probe_axis)
    baseline_ortho = baseline_ortho / np.linalg.norm(baseline_ortho)
    normal = np.cross(probe_axis, baseline_ortho)

    tip_mm = 0.5 * (center_left_mm + center_right_mm)
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -probe_axis[2]))))

    def roll_about_axis(axes: Dict[str, np.ndarray]) -> float:
        """Scope roll: angle of the camera 'right' vector against the baseline,
        measured about the probe axis."""
        right = axes["right"] - probe_axis * axes["right"].dot(probe_axis)
        right = right / np.linalg.norm(right)
        return math.degrees(math.atan2(right.dot(normal), right.dot(baseline_ortho)))

    return {
        "tip_mm": tip_mm.tolist(),
        "tip_units": (0.5 * (center_left + center_right)).tolist(),
        "probe_axis": probe_axis.tolist(),
        "baseline_dir": baseline_dir.tolist(),
        "tilt_from_sheet_normal_deg": tilt_deg,
        "camera_left_mm": center_left_mm.tolist(),
        "camera_right_mm": center_right_mm.tolist(),
        "distance_left_mm": float(abs(center_left_mm[2])),
        "distance_right_mm": float(abs(center_right_mm[2])),
        "tag_edge_mm": mm_per_unit,
        "scale_source": scale_source,
        "camera_separation_mm": separation_units * mm_per_unit,
        "camera_lateral_D_mm": lateral_d_mm,
        "camera_axial_offset_mm": axial_offset_mm,
        "roll_left_deg": roll_about_axis(camera_axes_sheet(r_left)),
        "roll_right_deg": roll_about_axis(camera_axes_sheet(r_right)),
    }


def analyse_pair(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    focal_px: float,
    family: str = "36h11",
    reference_id: int = 2,
    mm_per_unit: Optional[float] = None,
    camera_calibration: Optional[Dict[str, object]] = None,
    anchor_ids: Optional[Sequence[int]] = None,
    sheet_layout: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, object]:
    detector = create_detector(family)
    left_dets = detect_tags(left_frame, detector)
    right_dets = detect_tags(right_frame, detector)
    left_ids = sorted(d.tag_id for d in left_dets)
    right_ids = sorted(d.tag_id for d in right_dets)
    anchor_set = set(int(tag_id) for tag_id in anchor_ids) if anchor_ids else None
    left_pose_dets = [
        det for det in left_dets if anchor_set is None or det.tag_id in anchor_set
    ]
    right_pose_dets = [
        det for det in right_dets if anchor_set is None or det.tag_id in anchor_set
    ]
    left_pose_ids = sorted(det.tag_id for det in left_pose_dets)
    right_pose_ids = sorted(det.tag_id for det in right_pose_dets)
    if len(left_pose_ids) < 2 or len(right_pose_ids) < 2:
        raise ValueError(
            "need >=2 anchor tags per view, got left=%s right=%s"
            % (left_pose_ids, right_pose_ids)
        )

    left_h, left_w = left_frame.shape[:2]
    right_h, right_w = right_frame.shape[:2]
    if camera_calibration is None:
        left_intrinsics = camera_matrix(focal_px, left_w, left_h)
        right_intrinsics = camera_matrix(focal_px, right_w, right_h)
        left_distortion = None
        right_distortion = None
        camera_model = "assumed-pinhole"
    else:
        left_intrinsics, left_distortion, camera_model = camera_parameters(
            camera_calibration, "left", left_w, left_h
        )
        right_intrinsics, right_distortion, right_model = camera_parameters(
            camera_calibration, "right", right_w, right_h
        )
        if right_model != camera_model:
            raise ValueError("left/right calibration models differ")

    left_corrected = dict(
        (det.tag_id, undistort_points_px(det.corners, left_intrinsics, left_distortion, camera_model))
        for det in left_pose_dets
    )
    right_corrected = dict(
        (det.tag_id, undistort_points_px(det.corners, right_intrinsics, right_distortion, camera_model))
        for det in right_pose_dets
    )

    if sheet_layout is None:
        # Learn the rigid layout from the current stereo pair.  Live callers can
        # retain this result for the rest of the session; no tag coordinates
        # need to be stored in advance.
        layouts = []
        for dets, corrected in (
            (left_pose_dets, left_corrected),
            (right_pose_dets, right_corrected),
        ):
            by_id = detections_by_id(dets)
            if reference_id in by_id:
                layouts.append(recover_sheet_layout(dets, reference_id, corrected))
        if not layouts:
            raise ValueError("reference tag %d not seen by either camera" % reference_id)
        layout: Dict[int, np.ndarray] = {}
        for tag_id in set().union(*[set(item) for item in layouts]):
            estimates = [item[tag_id] for item in layouts if tag_id in item]
            layout[tag_id] = np.mean(estimates, axis=0)
        layout_source = "learned-from-current-stereo-pair"
    else:
        layout = dict(
            (int(tag_id), np.asarray(corners, dtype=np.float64))
            for tag_id, corners in sheet_layout.items()
        )
        layout_source = "learned-at-live-start"

    r_left, t_left, err_left = solve_camera_pose(
        left_pose_dets, layout, left_intrinsics, left_distortion, camera_model
    )
    r_right, t_right, err_right = solve_camera_pose(
        right_pose_dets, layout, right_intrinsics, right_distortion, camera_model
    )

    result = probe_pose_from_cameras((r_left, t_left), (r_right, t_right), mm_per_unit)
    result["reprojection_error_px"] = {"left": err_left, "right": err_right}
    result["tag_ids"] = {"left": left_ids, "right": right_ids}
    result["anchor_tag_ids"] = {"left": left_pose_ids, "right": right_pose_ids}
    result["layout_tag_edge_units"] = {str(k): v.tolist() for k, v in layout.items()}
    result["layout_source"] = layout_source
    result["focal_px"] = focal_px
    result["camera_model"] = camera_model
    resolved_scale = float(result["tag_edge_mm"])
    result["camera_poses"] = {
        "left": {
            "rotation_camera_to_sheet": r_left.T.tolist(),
            "center_mm": (camera_center_sheet(r_left, t_left) * resolved_scale).tolist(),
        },
        "right": {
            "rotation_camera_to_sheet": r_right.T.tolist(),
            "center_mm": (camera_center_sheet(r_right, t_right) * resolved_scale).tolist(),
        },
    }
    if camera_calibration is not None:
        result["calibrated_focal_px"] = {
            "left": [float(left_intrinsics[0, 0]), float(left_intrinsics[1, 1])],
            "right": [float(right_intrinsics[0, 0]), float(right_intrinsics[1, 1])],
        }
    return result


def print_report(result: Dict[str, object]) -> None:
    tip = result["tip_mm"]
    axis = result["probe_axis"]
    print("tags seen: left=%s right=%s" % (result["tag_ids"]["left"], result["tag_ids"]["right"]))
    print("camera model: %s" % result.get("camera_model", "unknown"))
    print(
        "reprojection error: left %.2f px, right %.2f px"
        % (result["reprojection_error_px"]["left"], result["reprojection_error_px"]["right"])
    )
    print("tag edge length: %.2f mm (scale from %s)" % (result["tag_edge_mm"], result["scale_source"]))
    print(
        "camera separation check: %.2f mm reconstructed vs %.3f mm from the STL"
        % (result["camera_separation_mm"], SCOPE_SEPARATION_MM)
    )
    print("probe tip (sheet frame): (%.2f, %.2f, %.2f) mm" % (tip[0], tip[1], tip[2]))
    print("probe axis (pointing at sheet): (%.3f, %.3f, %.3f)" % (axis[0], axis[1], axis[2]))
    print("tilt from sheet normal: %.1f deg" % result["tilt_from_sheet_normal_deg"])
    print(
        "camera distances to sheet plane: left %.1f mm, right %.1f mm"
        % (result["distance_left_mm"], result["distance_right_mm"])
    )
    print(
        "scope roll about probe axis: left %+.1f deg, right %+.1f deg (relative twist %.1f deg)"
        % (
            result["roll_left_deg"],
            result["roll_right_deg"],
            (result["roll_right_deg"] - result["roll_left_deg"] + 180.0) % 360.0 - 180.0,
        )
    )


def apply_led_min(camera_index: int) -> None:
    try:
        import comtypes

        import led_control

        comtypes.CoInitialize()
        ks = led_control.open_ks_control(camera_index)
        led_control.set_led(ks, led_control.LED_LEVELS["min"])
    except Exception as exc:
        print("camera %d: LED control failed (%s)" % (camera_index, exc))


def capture_pair(left_index: int, right_index: int) -> Tuple[np.ndarray, np.ndarray]:
    caps = []
    for index in (left_index, right_index):
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not cap.isOpened():
            raise SystemExit("could not open camera %d via MSMF" % index)
        caps.append(cap)
    for cap in caps:
        for _ in range(5):
            cap.read()
    time.sleep(0.5)
    apply_led_min(left_index)
    apply_led_min(right_index)
    time.sleep(0.8)
    # Interleave reads so the final left/right frames are as close in time as
    # possible - the pose math assumes a rigid scene between the two views.
    frames = [None, None]
    for _ in range(6):
        for idx, cap in enumerate(caps):
            ok, frame = cap.read()
            if ok:
                frames[idx] = frame
    for cap in caps:
        cap.release()
    if frames[0] is None or frames[1] is None:
        raise SystemExit("frame capture failed")
    return frames[0], frames[1]


CALIBRATION_PATH = Path(__file__).with_name("tag_sheet_calibration.json")
CAMERA_CALIBRATION_PATH = Path(__file__).with_name("stereo_camera_calibration.json")


def load_calibration() -> Optional[Dict[str, object]]:
    if CALIBRATION_PATH.exists():
        with CALIBRATION_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def load_camera_calibration(path: Optional[Path] = None) -> Optional[Dict[str, object]]:
    selected = CAMERA_CALIBRATION_PATH if path is None else path
    if selected.exists():
        with selected.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        quality = calibration.get("quality")
        if isinstance(quality, dict) and quality.get("accepted") is False:
            return None
        return calibration
    return None


def calibrate_scale(
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, object]]:
    """Move one stage axis a known distance; the tip displacement in tag-edge
    units fixes the metric scale (physical tag edge length in mm)."""
    from kcube_motion import load_axes, move_to_mm, read_position_mm

    axes = load_axes()
    if args.calibrate_axis not in axes:
        raise SystemExit("unknown axis '%s'" % args.calibrate_axis)
    axis = axes[args.calibrate_axis]
    start = read_position_mm(axis.serial)
    if start is None:
        raise SystemExit("could not read %s position" % axis.name)
    target = start + args.step_mm
    if not axis.min_mm <= target <= axis.max_mm:
        raise SystemExit("calibration move %.3f -> %.3f mm violates soft limits" % (start, target))

    print("calibration: pose A at %s = %.4f mm" % (axis.name, start))
    left, right = capture_pair(args.left_camera, args.right_camera)
    result_a = analyse_pair(
        left,
        right,
        args.focal_px,
        args.family,
        args.reference_id,
        camera_calibration=args.camera_calibration_data,
    )
    print("calibration: moving %s to %.4f mm" % (axis.name, target))
    move_to_mm(axis.serial, target, axis)
    try:
        left, right = capture_pair(args.left_camera, args.right_camera)
        result_b = analyse_pair(
            left,
            right,
            args.focal_px,
            args.family,
            args.reference_id,
            camera_calibration=args.camera_calibration_data,
        )
    finally:
        print("calibration: moving %s back to %.4f mm" % (axis.name, start))
        move_to_mm(axis.serial, start, axis)

    delta_units = np.asarray(result_b["tip_units"]) - np.asarray(result_a["tip_units"])
    magnitude = float(np.linalg.norm(delta_units))
    if magnitude < 1e-6:
        raise SystemExit("no tip displacement measured; is the scene rigid?")
    mm_per_unit = args.step_mm / magnitude
    stage_dir = (delta_units / magnitude).tolist()
    calibration = {
        "tag_edge_mm": mm_per_unit,
        "stage_axis": axis.name,
        "stage_axis_dir_sheet": stage_dir,
        "step_mm": args.step_mm,
        "focal_px": args.focal_px,
    }
    with CALIBRATION_PATH.open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)
    print(
        "calibration: tag edge = %.3f mm, stage +%s in sheet frame = (%.3f, %.3f, %.3f)"
        % (mm_per_unit, axis.name, stage_dir[0], stage_dir[1], stage_dir[2])
    )
    print("wrote %s" % CALIBRATION_PATH)
    return mm_per_unit, calibration


def resolve_scale(args: argparse.Namespace) -> Optional[float]:
    """Scale priority: explicit --tag-edge-mm, then saved calibration, then None."""
    if args.tag_edge_mm:
        return args.tag_edge_mm
    calibration = load_calibration()
    if calibration:
        print("using tag edge %.3f mm from %s" % (calibration["tag_edge_mm"], CALIBRATION_PATH.name))
        return float(calibration["tag_edge_mm"])
    print("WARNING: no scale calibration; distances will use the ill-conditioned scope-separation scale.")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe pose from the stereo endoscopes and 3 AprilTags.")
    parser.add_argument("--left-image", type=Path, help="Offline left image instead of live capture.")
    parser.add_argument("--right-image", type=Path, help="Offline right image instead of live capture.")
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--focal-px", type=float, default=800.0, help="Assumed focal length in pixels.")
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=None,
        help="Stereo camera-calibration JSON; defaults to stereo_camera_calibration.json when present.",
    )
    parser.add_argument("--family", default="36h11")
    parser.add_argument("--reference-id", type=int, default=2, help="Tag used to anchor the sheet frame.")
    parser.add_argument("--json", type=Path, help="Write the full result to this JSON file.")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Move a stage axis a known step to calibrate the metric scale, then save it.",
    )
    parser.add_argument("--calibrate-axis", default="axis1", help="Stage axis for --calibrate (default axis1).")
    parser.add_argument("--step-mm", type=float, default=2.0, help="Calibration step in mm (default 2).")
    parser.add_argument(
        "--tag-edge-mm",
        type=float,
        default=None,
        help="Known physical tag edge length; overrides the saved calibration.",
    )
    args = parser.parse_args()

    explicit_camera_calibration = args.camera_calibration is not None
    camera_calibration_path = args.camera_calibration
    if camera_calibration_path is None and CAMERA_CALIBRATION_PATH.exists():
        camera_calibration_path = CAMERA_CALIBRATION_PATH
    args.camera_calibration_data = load_camera_calibration(camera_calibration_path)
    if camera_calibration_path is not None:
        if args.camera_calibration_data is None:
            if explicit_camera_calibration:
                raise SystemExit("camera calibration not found or rejected: %s" % camera_calibration_path)
            print("WARNING: default camera calibration is rejected; using assumed focal length")
            camera_calibration_path = None
        else:
            print("using camera calibration %s" % camera_calibration_path)

    if bool(args.left_image) != bool(args.right_image):
        raise SystemExit("--left-image and --right-image must be used together")
    if args.calibrate and args.left_image:
        raise SystemExit("--calibrate needs live cameras, not images")

    if args.calibrate:
        mm_per_unit, _ = calibrate_scale(args)
    else:
        mm_per_unit = resolve_scale(args)

    if args.left_image:
        left = cv2.imread(str(args.left_image))
        right = cv2.imread(str(args.right_image))
        if left is None or right is None:
            raise SystemExit("could not read input images")
    else:
        left, right = capture_pair(args.left_camera, args.right_camera)

    result = analyse_pair(
        left,
        right,
        args.focal_px,
        args.family,
        args.reference_id,
        mm_per_unit,
        args.camera_calibration_data,
    )
    print_report(result)
    if args.json:
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()

"""Read-only diagnostic for AprilTag scale and stereo pose consistency.

Captures one near-simultaneous stereo pair, then reports how the known physical
tag edge, the known endoscope center-to-center distance D, the assumed focal
length, and the saved stage-motion calibration agree.  This script does not
connect to or move the Thorlabs stages.
"""

from __future__ import print_function

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Motordriver"))

import stereo_probe_pose  # noqa: E402


def mean_edge_px(detection):
    corners = np.asarray(detection.corners, dtype=float)
    following = np.roll(corners, -1, axis=0)
    return float(np.mean(np.linalg.norm(following - corners, axis=1)))


def camera_separation_for_focal(left_dets, right_dets, layout, width, height, focal_px, tag_edge_mm):
    intrinsics = stereo_probe_pose.camera_matrix(focal_px, width, height)
    left_pose = stereo_probe_pose.solve_camera_pose(left_dets, layout, intrinsics)[:2]
    right_pose = stereo_probe_pose.solve_camera_pose(right_dets, layout, intrinsics)[:2]
    result = stereo_probe_pose.probe_pose_from_cameras(left_pose, right_pose, tag_edge_mm)
    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose AprilTag metric-scale consistency.")
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--reference-id", type=int, default=2)
    parser.add_argument("--tag-edge-mm", type=float, default=10.62)
    parser.add_argument("--focal-px", type=float, default=800.0)
    args = parser.parse_args()

    left, right = stereo_probe_pose.capture_pair(args.left_camera, args.right_camera)
    detector = stereo_probe_pose.create_detector("36h11")
    left_dets = stereo_probe_pose.detect_tags(left, detector)
    right_dets = stereo_probe_pose.detect_tags(right, detector)
    if len(left_dets) < 2 or len(right_dets) < 2:
        raise SystemExit("need at least two tags in each view")

    layouts = []
    for detections in (left_dets, right_dets):
        by_id = stereo_probe_pose.detections_by_id(detections)
        if args.reference_id in by_id:
            layouts.append(stereo_probe_pose.recover_sheet_layout(detections, args.reference_id))
    if len(layouts) != 2:
        raise SystemExit("reference tag must be visible in both views for this diagnostic")

    common_ids = sorted(set(layouts[0]).intersection(layouts[1]))
    layout = {}
    residuals = []
    for tag_id in common_ids:
        layout[tag_id] = 0.5 * (layouts[0][tag_id] + layouts[1][tag_id])
        residuals.extend((layouts[0][tag_id] - layouts[1][tag_id]).ravel().tolist())
    layout_rms_units = float(np.sqrt(np.mean(np.square(residuals))))

    print("tag edge measurements in the captured images:")
    for name, detections in (("left", left_dets), ("right", right_dets)):
        values = ["id%d=%.2f px" % (item.tag_id, mean_edge_px(item)) for item in detections]
        print("  %s: %s" % (name, ", ".join(values)))
    print("left/right rectified-layout disagreement: %.5f tag-edge units RMS (%.3f mm RMS)" % (
        layout_rms_units,
        layout_rms_units * args.tag_edge_mm,
    ))

    height, width = left.shape[:2]
    nominal = camera_separation_for_focal(
        left_dets,
        right_dets,
        layout,
        width,
        height,
        args.focal_px,
        args.tag_edge_mm,
    )
    print("at f=%.1f px and tag edge %.3f mm:" % (args.focal_px, args.tag_edge_mm))
    print("  D reconstructed = %.3f mm (known D = %.3f mm)" % (
        nominal["camera_separation_mm"],
        stereo_probe_pose.SCOPE_SEPARATION_MM,
    ))
    print("  mean height = %.3f mm" % (0.5 * (
        nominal["distance_left_mm"] + nominal["distance_right_mm"]
    )))

    focal_values = np.linspace(250.0, 2400.0, 216)
    results = []
    for focal_px in focal_values:
        result = camera_separation_for_focal(
            left_dets,
            right_dets,
            layout,
            width,
            height,
            float(focal_px),
            args.tag_edge_mm,
        )
        results.append((
            abs(result["camera_separation_mm"] - stereo_probe_pose.SCOPE_SEPARATION_MM),
            float(focal_px),
            result,
        ))
    _, best_focal, best = min(results, key=lambda item: item[0])
    separation_values = [item[2]["camera_separation_mm"] for item in results]
    print("focal-length scan 250..2400 px:")
    print("  D range = %.3f..%.3f mm" % (
        min(separation_values),
        max(separation_values),
    ))
    print("  closest point: f=%.1f px -> D=%.3f mm, height=%.3f mm" % (
        best_focal,
        best["camera_separation_mm"],
        0.5 * (best["distance_left_mm"] + best["distance_right_mm"]),
    ))

    calibration = stereo_probe_pose.load_calibration()
    if calibration:
        saved_edge = float(calibration["tag_edge_mm"])
        step_mm = float(calibration["step_mm"])
        observed_units = step_mm / saved_edge
        expected_units = step_mm / args.tag_edge_mm
        print("saved stage-motion calibration:")
        print("  saved edge = %.6f mm; known edge = %.6f mm; ratio = %.4f" % (
            saved_edge,
            args.tag_edge_mm,
            args.tag_edge_mm / saved_edge,
        ))
        print("  calibration observed %.4f tag widths for a claimed %.3f mm move" % (
            observed_units,
            step_mm,
        ))
        print("  known edge predicts %.4f tag widths; observed relative motion implies %.3f mm" % (
            expected_units,
            observed_units * args.tag_edge_mm,
        ))


if __name__ == "__main__":
    main()

"""Measure apparent probe rotation across a verified stage translation.

This hardware diagnostic captures stereo frames at two axis positions, returns
the axis with the raw KCube home command, and compares the reconstructed probe
orientation at several assumed focal lengths.  It saves the endpoint images and
full pose results so the measurement can be audited offline.
"""

from __future__ import print_function

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Motordriver"))

import kcube_motion  # noqa: E402
import stereo_probe_pose  # noqa: E402


HOMED_BIT = 0x0400
HOMING_BIT = 0x0200


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / magnitude


def angle_between_deg(first, second):
    cosine = float(np.clip(np.dot(unit(first), unit(second)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def wrapped_difference_deg(after, before):
    return (float(after) - float(before) + 180.0) % 360.0 - 180.0


def probe_frame(pose):
    z_axis = unit(pose["probe_axis"])
    d_axis = unit(pose["baseline_dir"])
    d_axis = unit(d_axis - z_axis * float(np.dot(d_axis, z_axis)))
    y_axis = unit(np.cross(z_axis, d_axis))
    return np.column_stack((d_axis, y_axis, z_axis))


def frame_change_deg(before, after):
    relative = np.dot(probe_frame(before).T, probe_frame(after))
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def move_to_verified(axis, target_mm, tolerance_mm, timeout_s):
    kcube_motion.require_within_limits(axis, target_mm)
    kcube_motion.move_to_mm(axis.serial, target_mm, axis, wait=False)
    deadline = time.monotonic() + timeout_s
    stable = 0
    last_position = None
    while time.monotonic() < deadline:
        position = kcube_motion.read_position_mm(axis.serial)
        if position is None:
            time.sleep(0.1)
            continue
        if position < axis.min_mm - 0.05 or position > axis.max_mm + 0.05:
            kcube_motion.stop(axis.serial)
            raise RuntimeError("axis left guarded range at %.6f mm" % position)
        if abs(position - target_mm) <= tolerance_mm:
            if last_position is not None and abs(position - last_position) <= tolerance_mm * 0.1:
                stable += 1
            else:
                stable = 0
            if stable >= 3:
                return position
        else:
            stable = 0
        last_position = position
        time.sleep(0.1)
    kcube_motion.stop(axis.serial)
    raise RuntimeError("axis did not reach %.6f +/- %.6f mm" % (target_mm, tolerance_mm))


def home_verified(axis, tolerance_mm, timeout_s):
    start_position = kcube_motion.read_position_mm(axis.serial)
    if start_position is None:
        raise RuntimeError("could not read position before homing")
    kcube_motion.send_recv(
        kcube_motion.ENDPOINTS[axis.serial],
        kcube_motion.apt_short(0x0443, 1),
        timeout=0.2,
    )
    deadline = time.monotonic() + timeout_s
    started = False
    while time.monotonic() < deadline:
        status = kcube_motion.get_status(axis.serial)
        position = status.dc_status_position_mm
        bits = status.status_bits
        if bits is not None and bits & HOMING_BIT:
            started = True
        if position is not None and position < start_position - 0.05:
            started = True
        if (
            started
            and bits is not None
            and bits & HOMED_BIT
            and not bits & HOMING_BIT
            and position is not None
            and abs(position) <= tolerance_mm
        ):
            return position
        time.sleep(0.2)
    kcube_motion.stop(axis.serial)
    raise RuntimeError("raw home did not complete within %.1f s" % timeout_s)


def compare_poses(before, after):
    tip_before = np.asarray(before["tip_mm"], dtype=float)
    tip_after = np.asarray(after["tip_mm"], dtype=float)
    return {
        "probe_frame_change_deg": frame_change_deg(before, after),
        "viewing_axis_change_deg": angle_between_deg(before["probe_axis"], after["probe_axis"]),
        "D_direction_change_deg": angle_between_deg(before["baseline_dir"], after["baseline_dir"]),
        "left_roll_change_deg": wrapped_difference_deg(after["roll_left_deg"], before["roll_left_deg"]),
        "right_roll_change_deg": wrapped_difference_deg(after["roll_right_deg"], before["roll_right_deg"]),
        "tilt_change_deg": float(after["tilt_from_sheet_normal_deg"] - before["tilt_from_sheet_normal_deg"]),
        "reconstructed_tip_displacement_mm": float(np.linalg.norm(tip_after - tip_before)),
        "D_before_mm": float(before["camera_separation_mm"]),
        "D_after_mm": float(after["camera_separation_mm"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare reconstructed rotation at 0 and 20 mm.")
    parser.add_argument("--axis", default="axis1")
    parser.add_argument("--target-mm", type=float, default=20.0)
    parser.add_argument("--tag-edge-mm", type=float, default=10.62)
    parser.add_argument("--focal-px", type=float, nargs="+", default=[550.0, 660.0, 800.0])
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "digital_twin" / "motion_rotation_0_20",
    )
    args = parser.parse_args()

    axes = kcube_motion.load_axes()
    if args.axis not in axes:
        raise SystemExit("unknown axis %s" % args.axis)
    axis = axes[args.axis]
    start_status = kcube_motion.get_status(axis.serial)
    if start_status.status_bits is None or not start_status.status_bits & HOMED_BIT:
        raise SystemExit("%s is not homed" % args.axis)
    start_mm = start_status.dc_status_position_mm
    if start_mm is None or abs(start_mm) > 0.02:
        raise SystemExit("expected %s at 0 mm, read %s" % (args.axis, start_mm))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("capturing endpoint A at %.6f mm" % start_mm)
    left_a, right_a = stereo_probe_pose.capture_pair(args.left_camera, args.right_camera)

    moved = False
    end_mm = None
    try:
        print("moving %s to %.3f mm with target verification" % (args.axis, args.target_mm))
        moved = True
        end_mm = move_to_verified(axis, args.target_mm, 0.01, 30.0)
        print("verified endpoint B at %.6f mm" % end_mm)
        left_b, right_b = stereo_probe_pose.capture_pair(args.left_camera, args.right_camera)
    finally:
        if moved:
            print("returning %s with raw home command" % args.axis)
            home_mm = home_verified(axis, 0.02, 40.0)
            print("verified home at %.6f mm" % home_mm)

    cv2.imwrite(str(args.output_dir / "left_0mm.png"), left_a)
    cv2.imwrite(str(args.output_dir / "right_0mm.png"), right_a)
    cv2.imwrite(str(args.output_dir / "left_20mm.png"), left_b)
    cv2.imwrite(str(args.output_dir / "right_20mm.png"), right_b)

    output = {
        "axis": args.axis,
        "start_mm": start_mm,
        "end_mm": end_mm,
        "tag_edge_mm": args.tag_edge_mm,
        "focal_results": {},
    }
    for focal_px in args.focal_px:
        before = stereo_probe_pose.analyse_pair(
            left_a, right_a, float(focal_px), mm_per_unit=args.tag_edge_mm
        )
        after = stereo_probe_pose.analyse_pair(
            left_b, right_b, float(focal_px), mm_per_unit=args.tag_edge_mm
        )
        comparison = compare_poses(before, after)
        output["focal_results"][str(float(focal_px))] = {
            "pose_0mm": before,
            "pose_20mm": after,
            "comparison": comparison,
        }
        print("f=%.1f px:" % focal_px)
        print("  probe-frame rotation change: %.3f deg" % comparison["probe_frame_change_deg"])
        print("  viewing-axis change: %.3f deg" % comparison["viewing_axis_change_deg"])
        print("  D-direction change: %.3f deg" % comparison["D_direction_change_deg"])
        print("  scope-roll changes: left %+.3f deg, right %+.3f deg" % (
            comparison["left_roll_change_deg"],
            comparison["right_roll_change_deg"],
        ))
        print("  reconstructed displacement: %.3f mm" % comparison["reconstructed_tip_displacement_mm"])

    output_path = args.output_dir / "rotation_comparison.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print("wrote %s" % output_path)


if __name__ == "__main__":
    main()

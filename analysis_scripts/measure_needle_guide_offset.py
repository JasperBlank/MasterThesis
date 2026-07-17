"""Measure the systematic offset between the projected CAD needle axis and the
detected needle, per camera (lab-PC experiment from docs/fable_laptop_handover.md).

Runs the same LiveStereoTracker the five-panel twin uses, samples for a fixed
duration, and reports per camera the signed perpendicular offset of the detected
needle (tip and entry) from the projected guide line plus the angle difference.
Mean = systematic part (bore clearance tilt + probe-frame bias); std = jitter.

Usage (scene static, needle visible in both views):
    python analysis_scripts/measure_needle_guide_offset.py --seconds 60
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Motordriver"))
sys.path.insert(0, str(PROJECT_ROOT / "digital_twin"))


def signed_offset(px: float, py: float, line) -> float:
    x0, y0, dx, dy = line
    norm = math.hypot(dx, dy) or 1.0
    return ((px - x0) * (-dy) + (py - y0) * dx) / norm


def angle_diff_deg(det_angle: float, line) -> float:
    line_angle = math.degrees(math.atan2(line[3], line[2]))
    return (det_angle - line_angle + 90.0) % 180.0 - 90.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--tag-edge-mm", type=float, default=10.62)
    parser.add_argument("--reference-id", type=int, default=2)
    parser.add_argument("--nominal-cad-order", action="store_true",
                        help="Use the nominal (not reversed) CAD camera mapping.")
    parser.add_argument("--needle-line-samples", type=int, default=0,
                        help="Enable the per-session needle-line calibration with this "
                        "many samples before measuring (0 = measure against raw CAD axis).")
    args = parser.parse_args()

    import stereo_probe_pose
    import twin_wasd_jog as twin

    camera_calibration = stereo_probe_pose.load_camera_calibration(None)
    if camera_calibration is None:
        raise SystemExit("no accepted stereo camera calibration found")

    tracker = twin.LiveStereoTracker(
        args.left_camera,
        args.right_camera,
        args.tag_edge_mm,
        camera_calibration,
        args.reference_id,
        (1, 2, 3),
        not args.nominal_cad_order,
        needle_line_samples=args.needle_line_samples,
    )
    if args.needle_line_samples > 0:
        print("collecting needle-line calibration first (%d samples)..." % args.needle_line_samples)
        deadline = time.monotonic() + 60.0
        while tracker.needle_line_model is None and time.monotonic() < deadline:
            tracker.read(True)
        if tracker.needle_line_model is None:
            raise SystemExit("needle-line calibration did not lock within 60 s")
    samples = {"left": {"tip": [], "entry": [], "angle": []},
               "right": {"tip": [], "entry": [], "angle": []}}
    pose_errors = 0
    frames = 0
    print("sampling for %.0f s (scene must stay static)..." % args.seconds)
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            _, _, pose, error = tracker.read(True)
            frames += 1
            if pose is None:
                pose_errors += 1
                continue
            for index, side in enumerate(("left", "right")):
                line = tracker.needle_expected_lines[index]
                det = tracker.needle_detections[index]
                if line is None or det is None:
                    continue
                samples[side]["tip"].append(signed_offset(det.tip_x, det.tip_y, line))
                samples[side]["entry"].append(signed_offset(det.entry_x, det.entry_y, line))
                samples[side]["angle"].append(angle_diff_deg(det.angle_deg, line))
    finally:
        tracker.close()

    print("frames: %d, pose failures: %d" % (frames, pose_errors))
    for side in ("left", "right"):
        tips = np.asarray(samples[side]["tip"])
        entries = np.asarray(samples[side]["entry"])
        angles = np.asarray(samples[side]["angle"])
        if len(tips) == 0:
            print("%s: no needle samples" % side)
            continue
        print(
            "%s (%d samples):\n"
            "  tip offset    mean %+7.1f px  std %5.1f px\n"
            "  entry offset  mean %+7.1f px  std %5.1f px\n"
            "  angle diff    mean %+7.2f deg std %5.2f deg"
            % (
                side, len(tips),
                float(tips.mean()), float(tips.std()),
                float(entries.mean()), float(entries.std()),
                float(angles.mean()), float(angles.std()),
            )
        )
    print(
        "interpretation: |mean| <= ~15 px -> no calibration needed; "
        "larger and stable (small std) -> per-session needle-line calibration is warranted."
    )


if __name__ == "__main__":
    main()

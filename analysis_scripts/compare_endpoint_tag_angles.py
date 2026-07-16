"""Compare raw detected AprilTag image angles between saved motion endpoints."""

from __future__ import print_function

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Motordriver"))

import stereo_probe_pose  # noqa: E402


def wrapped_difference_deg(after, before):
    return (float(after) - float(before) + 180.0) % 360.0 - 180.0


def detections(path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError("could not read %s" % path)
    detector = stereo_probe_pose.create_detector("36h11")
    return dict(
        (item.tag_id, item)
        for item in stereo_probe_pose.detect_tags(frame, detector)
    )


def main():
    parser = argparse.ArgumentParser(description="Compare raw tag angles at 0 and 20 mm.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "digital_twin" / "motion_rotation_0_20",
    )
    args = parser.parse_args()

    for camera in ("left", "right"):
        before = detections(args.input_dir / (camera + "_0mm.png"))
        after = detections(args.input_dir / (camera + "_20mm.png"))
        common_ids = sorted(set(before).intersection(after))
        changes = []
        print("%s camera:" % camera)
        for tag_id in common_ids:
            change = wrapped_difference_deg(after[tag_id].angle_deg, before[tag_id].angle_deg)
            changes.append(change)
            print("  id%d: %+.3f deg" % (tag_id, change))
        print("  mean: %+.3f deg, spread: %.3f deg" % (
            float(np.mean(changes)),
            float(np.ptp(changes)),
        ))


if __name__ == "__main__":
    main()

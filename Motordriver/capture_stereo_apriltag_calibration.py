"""Capture paired AprilTag images for stereo endoscope calibration.

Use a rigid, flat AprilTag target.  Move and tilt it between captures so tags
cover the center, edges, and corners of both images.  Known tag sizes are stored
in manifest.json and may be overridden per ID for mixed-size validation tags.

Controls: SPACE/S saves a pair, Q/ESC quits.
"""

from __future__ import print_function

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from apriltag_tracker import create_detector, detect_tags, draw_overlay
from stereo_apriltag_webcams import apply_led, make_side_by_side, open_camera


def parse_tag_edges(values: List[str]) -> Dict[int, float]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("expected ID=MM, got '%s'" % value)
        raw_id, raw_mm = value.split("=", 1)
        tag_id = int(raw_id)
        edge_mm = float(raw_mm)
        if edge_mm <= 0:
            raise ValueError("tag edge must be positive: %s" % value)
        parsed[tag_id] = edge_mm
    return parsed


def known_edge(tag_id: int, default_mm: Optional[float], overrides: Dict[int, float]) -> Optional[float]:
    if tag_id in overrides:
        return overrides[tag_id]
    return default_mm


def load_or_create_manifest(
    path: Path,
    family: str,
    default_mm: Optional[float],
    overrides: Dict[int, float],
) -> Dict[str, object]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("family") != family:
            raise ValueError("existing manifest uses family %s" % manifest.get("family"))
        stored_default = manifest.get("default_tag_edge_mm")
        stored_overrides = dict(
            (int(key), float(value))
            for key, value in manifest.get("tag_edges_mm", {}).items()
        )
        if stored_default != default_mm or stored_overrides != overrides:
            raise ValueError(
                "existing manifest uses different tag sizes; choose a new --output-dir"
            )
        return manifest
    return {
        "schema_version": 1,
        "family": family,
        "default_tag_edge_mm": default_mm,
        "tag_edges_mm": dict((str(key), value) for key, value in sorted(overrides.items())),
        "captures": [],
    }


def write_manifest(path: Path, manifest: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def mean_edge_px(corners: np.ndarray) -> float:
    following = np.roll(corners, -1, axis=0)
    return float(np.mean(np.linalg.norm(following - corners, axis=1)))


def capture_metadata(
    left_detections: List[object],
    right_detections: List[object],
    default_mm: Optional[float],
    overrides: Dict[int, float],
) -> Tuple[List[int], Dict[str, object]]:
    left_by_id = dict((item.tag_id, item) for item in left_detections)
    right_by_id = dict((item.tag_id, item) for item in right_detections)
    common_ids = sorted(set(left_by_id).intersection(right_by_id))
    usable_ids = [tag_id for tag_id in common_ids if known_edge(tag_id, default_mm, overrides) is not None]
    metadata = {}
    for tag_id in usable_ids:
        left = left_by_id[tag_id]
        right = right_by_id[tag_id]
        metadata[str(tag_id)] = {
            "edge_mm": known_edge(tag_id, default_mm, overrides),
            "left_center_px": [left.center_x, left.center_y],
            "right_center_px": [right.center_x, right.center_y],
            "left_edge_px": mean_edge_px(left.corners),
            "right_edge_px": mean_edge_px(right.corners),
            "left_angle_deg": left.angle_deg,
            "right_angle_deg": right.angle_deg,
        }
    return usable_ids, metadata


def put_status(frame: np.ndarray, line1: str, line2: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(frame, line1, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)


def capture_pair(left_cap: object, right_cap: object) -> Tuple[np.ndarray, np.ndarray]:
    """Grab both streams before decoding either frame to reduce pair skew."""
    left_grabbed = left_cap.grab()
    right_grabbed = right_cap.grab()
    if not left_grabbed or not right_grabbed:
        raise RuntimeError("stereo frame grab failed")
    left_ok, left_frame = left_cap.retrieve()
    right_ok, right_frame = right_cap.retrieve()
    if not left_ok or not right_ok:
        raise RuntimeError("stereo frame retrieve failed")
    return left_frame, right_frame


def maximum_tag_motion(
    previous: List[object], current: List[object]
) -> float:
    previous_by_id = dict((item.tag_id, item) for item in previous)
    current_by_id = dict((item.tag_id, item) for item in current)
    common_ids = set(previous_by_id).intersection(current_by_id)
    if not common_ids:
        return float("inf")
    motions = []
    for tag_id in common_ids:
        delta = current_by_id[tag_id].corners - previous_by_id[tag_id].corners
        motions.append(float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1)))))
    return max(motions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture paired AprilTag camera-calibration images.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("stereo_calibration_data"))
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--family", default="36h11")
    parser.add_argument(
        "--default-tag-edge-mm",
        type=float,
        default=10.62,
        help="Edge length used for IDs without an explicit --tag-edge entry.",
    )
    parser.add_argument(
        "--tag-edge",
        action="append",
        default=[],
        metavar="ID=MM",
        help="Known outer-black-border edge for a specific tag ID; repeatable.",
    )
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.left_camera == args.right_camera:
        raise SystemExit("left and right camera indices must differ")
    if args.default_tag_edge_mm is not None and args.default_tag_edge_mm <= 0:
        raise SystemExit("--default-tag-edge-mm must be positive")
    overrides = parse_tag_edges(args.tag_edge)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = load_or_create_manifest(
        manifest_path, args.family, args.default_tag_edge_mm, overrides
    )
    captures = manifest["captures"]

    detector = create_detector(args.family)
    left_cap = open_camera(args.left_camera, args.width, args.height, "msmf")
    right_cap = open_camera(args.right_camera, args.width, args.height, "msmf")
    try:
        for _ in range(5):
            capture_pair(left_cap, right_cap)
        time.sleep(0.5)
        apply_led(args.left_camera, "min", "left")
        apply_led(args.right_camera, "min", "right")
        time.sleep(0.8)

        print("SPACE/S: save pair once STABLE; Q/ESC: quit")
        print("Move/tilt/raise the target, then hold it completely still before every save.")
        previous_left = []
        previous_right = []
        stable_frames = 0
        while True:
            left_raw, right_raw = capture_pair(left_cap, right_cap)
            left_detections = detect_tags(left_raw, detector)
            right_detections = detect_tags(right_raw, detector)
            usable_ids, metadata = capture_metadata(
                left_detections, right_detections, args.default_tag_edge_mm, overrides
            )
            left_motion = maximum_tag_motion(previous_left, left_detections)
            right_motion = maximum_tag_motion(previous_right, right_detections)
            if usable_ids and max(left_motion, right_motion) <= 0.75:
                stable_frames += 1
            else:
                stable_frames = 0
            stable = stable_frames >= 8
            previous_left = left_detections
            previous_right = right_detections

            left_view = left_raw.copy()
            right_view = right_raw.copy()
            draw_overlay(left_view, left_detections)
            draw_overlay(right_view, right_detections)
            preview = make_side_by_side(left_view, right_view)
            line1 = "pairs: %d   %s   common known-size IDs: %s" % (
                len(captures),
                "STABLE" if stable else "HOLD STILL",
                ",".join(str(item) for item in usable_ids) if usable_ids else "none",
            )
            line2 = "Move between captures; save only after STABLE appears"
            put_status(preview, line1, line2)
            cv2.imshow("Stereo AprilTag calibration capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key not in (ord("s"), ord(" ")):
                continue
            if not usable_ids:
                print("not saved: no common tag with a known physical edge")
                continue
            if not stable:
                print("not saved: target is still moving; hold it still until STABLE appears")
                continue

            capture_id = len(captures)
            stem = "pair_%03d" % capture_id
            left_name = stem + "_left.png"
            right_name = stem + "_right.png"
            if not cv2.imwrite(str(args.output_dir / left_name), left_raw):
                raise RuntimeError("failed to write %s" % left_name)
            if not cv2.imwrite(str(args.output_dir / right_name), right_raw):
                raise RuntimeError("failed to write %s" % right_name)
            captures.append(
                {
                    "id": capture_id,
                    "left_image": left_name,
                    "right_image": right_name,
                    "image_size": [int(left_raw.shape[1]), int(left_raw.shape[0])],
                    "usable_tag_ids": usable_ids,
                    "detections": metadata,
                }
            )
            write_manifest(manifest_path, manifest)
            print("saved pair %03d with tag IDs %s" % (capture_id, usable_ids))
    finally:
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()

    print("dataset: %s (%d stereo pairs)" % (args.output_dir, len(captures)))


if __name__ == "__main__":
    main()

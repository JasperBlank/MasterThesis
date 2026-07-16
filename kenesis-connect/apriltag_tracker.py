"""Live AprilTag center detection using OpenCV's aruco AprilTag dictionaries.

This script deliberately does not move motors. It only reports where the tag is
relative to the camera frame center, so the next layer can convert pixel error
into safe stage corrections.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np


APRILTAG_FAMILIES = {
    "16h5": cv2.aruco.DICT_APRILTAG_16H5,
    "25h9": cv2.aruco.DICT_APRILTAG_25H9,
    "36h10": cv2.aruco.DICT_APRILTAG_36H10,
    "36h11": cv2.aruco.DICT_APRILTAG_36H11,
}


@dataclass
class TagDetection:
    tag_id: int
    center_x: float
    center_y: float
    error_x: float
    error_y: float
    angle_deg: float
    corners: np.ndarray


def create_detector(family: str) -> object:
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILIES[family])
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def tag_angle_deg(corners: np.ndarray) -> float:
    # OpenCV returns corners in tag order. The vector from corner 0 to corner 1
    # is the tag's top edge in image coordinates.
    top_edge = corners[1] - corners[0]
    angle = math.degrees(math.atan2(float(top_edge[1]), float(top_edge[0])))
    if angle > 180.0:
        angle -= 360.0
    elif angle <= -180.0:
        angle += 360.0
    return angle


def detect_tags(frame: np.ndarray, detector: object, tag_id: int | None = None) -> list[TagDetection]:
    height, width = frame.shape[:2]
    image_center_x = width / 2.0
    image_center_y = height / 2.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    detections: list[TagDetection] = []
    if ids is None:
        return detections

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if tag_id is not None and marker_id != tag_id:
            continue
        pts = marker_corners.reshape(4, 2)
        center_x = float(pts[:, 0].mean())
        center_y = float(pts[:, 1].mean())
        detections.append(
            TagDetection(
                tag_id=marker_id,
                center_x=center_x,
                center_y=center_y,
                error_x=center_x - image_center_x,
                error_y=center_y - image_center_y,
                angle_deg=tag_angle_deg(pts),
                corners=pts,
            )
        )
    return detections


def draw_overlay(frame: np.ndarray, detections: list[TagDetection]) -> None:
    height, width = frame.shape[:2]
    image_center = (width // 2, height // 2)
    cv2.drawMarker(frame, image_center, (255, 0, 0), cv2.MARKER_CROSS, 28, 2)

    for det in detections:
        corners = det.corners.astype(np.int32)
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        center = (int(round(det.center_x)), int(round(det.center_y)))
        cv2.circle(frame, center, 5, (0, 255, 255), -1)
        cv2.line(frame, image_center, center, (0, 200, 255), 1)
        label = f"id={det.tag_id} dx={det.error_x:+.1f}px dy={det.error_y:+.1f}px angle={det.angle_deg:+.1f}"
        cv2.putText(frame, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def open_camera(index: int, width: int | None, height: int | None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--family", choices=sorted(APRILTAG_FAMILIES), default="36h11")
    parser.add_argument("--tag-id", type=int, help="Only track this tag id")
    parser.add_argument("--width", type=int, help="Requested camera width")
    parser.add_argument("--height", type=int, help="Requested camera height")
    parser.add_argument("--print-every", type=float, default=0.25, help="Seconds between console reports")
    parser.add_argument("--no-window", action="store_true", help="Run without a preview window")
    args = parser.parse_args()

    detector = create_detector(args.family)
    cap = open_camera(args.camera, args.width, args.height)
    print(f"Tracking AprilTag family {args.family} on camera {args.camera}. Press q in the preview to quit.")

    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")

            detections = detect_tags(frame, detector, args.tag_id)
            now = time.monotonic()
            if now - last_print >= args.print_every:
                if detections:
                    det = detections[0]
                    print(
                        f"id={det.tag_id} center=({det.center_x:.1f},{det.center_y:.1f}) "
                        f"error=({det.error_x:+.1f},{det.error_y:+.1f}) px angle={det.angle_deg:+.1f} deg"
                    )
                else:
                    print("no tag")
                last_print = now

            if not args.no_window:
                draw_overlay(frame, detections)
                cv2.imshow("AprilTag tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Read two webcams and detect AprilTags in both streams.

This is a live diagnostic script for checking two physical camera feeds before
using them in stereo or motor-control experiments. It uses OpenCV's aruco module
with AprilTag dictionaries, draws detections on each image, and prints the tags
seen by each camera plus the tag IDs seen by both cameras.

Install dependencies:
  pip install opencv-contrib-python numpy

Example:
  python Motordriver\\stereo_apriltag_webcams.py --left-camera 0 --right-camera 1
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    camera_name: str
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
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.005
    parameters.maxMarkerPerimeterRate = 4.0

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def tag_angle_deg(corners: np.ndarray) -> float:
    top_edge = corners[1] - corners[0]
    angle = math.degrees(math.atan2(float(top_edge[1]), float(top_edge[0])))
    if angle > 180.0:
        angle -= 360.0
    elif angle <= -180.0:
        angle += 360.0
    return angle


def detect_tags(
    frame: np.ndarray,
    detector: object,
    camera_name: str,
    tag_id: Optional[int] = None,
) -> List[TagDetection]:
    height, width = frame.shape[:2]
    image_center_x = width / 2.0
    image_center_y = height / 2.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    detections = []
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
                camera_name=camera_name,
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


def draw_overlay(frame: np.ndarray, camera_name: str, detections: Sequence[TagDetection]) -> None:
    height, width = frame.shape[:2]
    image_center = (width // 2, height // 2)
    cv2.drawMarker(frame, image_center, (255, 0, 0), cv2.MARKER_CROSS, 28, 2)

    cv2.rectangle(frame, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "%s: %d tag(s)" % (camera_name, len(detections)),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for det in detections:
        corners = np.rint(det.corners).astype(np.int32)
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        center = (int(round(det.center_x)), int(round(det.center_y)))
        cv2.circle(frame, center, 5, (0, 255, 255), -1)
        cv2.line(frame, image_center, center, (0, 200, 255), 1)
        cv2.circle(frame, tuple(corners[0]), 5, (0, 0, 255), -1)
        label = "id=%d dx=%+.1f dy=%+.1f a=%+.1f" % (
            det.tag_id,
            det.error_x,
            det.error_y,
            det.angle_deg,
        )
        cv2.putText(
            frame,
            label,
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    scale = float(target_height) / float(height)
    target_width = int(round(width * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    target_height = min(left.shape[0], right.shape[0])
    left_view = resize_to_height(left, target_height)
    right_view = resize_to_height(right, target_height)
    return np.hstack([left_view, right_view])


def backend_value(name: str) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def open_camera(index: int, width: Optional[int], height: Optional[int], backend: str) -> cv2.VideoCapture:
    if backend == "any":
        backends = ["any", "dshow", "msmf"]
    else:
        backends = [backend, "any", "msmf", "dshow"]

    tried = []
    for backend_name in backends:
        if backend_name in tried:
            continue
        tried.append(backend_name)
        cap = cv2.VideoCapture(index, backend_value(backend_name))
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            print("opened camera %d using backend %s" % (index, backend_name))
            return cap
        cap.release()

    raise RuntimeError("Could not open camera index %d with backends: %s" % (index, ", ".join(tried)))


def scan_backend_names(backend: str) -> List[str]:
    if backend == "any":
        return ["any", "dshow", "msmf"]
    return [backend]


def scan_cameras(max_index: int, backend: str) -> List[Tuple[int, str]]:
    found = []
    for backend_name in scan_backend_names(backend):
        for index in range(max_index + 1):
            cap = cv2.VideoCapture(index, backend_value(backend_name))
            ok = False
            if cap.isOpened():
                ok, _ = cap.read()
            cap.release()
            if ok:
                found.append((index, backend_name))
    return found


def detections_by_id(detections: Sequence[TagDetection]) -> Dict[int, TagDetection]:
    return dict((det.tag_id, det) for det in detections)


def format_detection(det: TagDetection) -> str:
    return "id=%d center=(%.1f,%.1f) error=(%+.1f,%+.1f)px angle=%+.1fdeg" % (
        det.tag_id,
        det.center_x,
        det.center_y,
        det.error_x,
        det.error_y,
        det.angle_deg,
    )


def print_report(left_detections: Sequence[TagDetection], right_detections: Sequence[TagDetection]) -> None:
    left_by_id = detections_by_id(left_detections)
    right_by_id = detections_by_id(right_detections)
    common_ids = sorted(set(left_by_id).intersection(set(right_by_id)))

    if left_detections:
        left_text = "; ".join(format_detection(det) for det in left_detections)
    else:
        left_text = "no tags"

    if right_detections:
        right_text = "; ".join(format_detection(det) for det in right_detections)
    else:
        right_text = "no tags"

    print("left:  %s" % left_text)
    print("right: %s" % right_text)
    if common_ids:
        print("both:  %s" % ", ".join(str(tag_id) for tag_id in common_ids))
    else:
        print("both:  no shared tag IDs")


def apply_led(camera_index: int, level: Optional[str], name: str) -> None:
    """Set the tip-LED level via the C8209 extension unit (see led_control.py).

    The firmware resets the LED to max whenever streaming starts, so this must
    run after the capture is open - which requires the MSMF backend, because
    exclusive DirectShow capture blocks the control connection.
    """
    if level is None:
        return
    try:
        import comtypes

        import led_control

        comtypes.CoInitialize()
        ks = led_control.open_ks_control(camera_index)
        led_control.set_led(ks, led_control.LED_LEVELS[level])
        print("%s camera: tip LED set to %s" % (name, level))
    except Exception as exc:
        print("%s camera: tip-LED control failed (%s); try --backend msmf" % (name, exc))


def apply_tuning(cap: cv2.VideoCapture, name: str, brightness: Optional[float], gain: Optional[float]) -> None:
    for prop_name, prop, value in (("brightness", cv2.CAP_PROP_BRIGHTNESS, brightness), ("gain", cv2.CAP_PROP_GAIN, gain)):
        if value is None:
            continue
        if cap.set(prop, value):
            print("%s camera: %s set to %g" % (name, prop_name, value))
        else:
            print("%s camera: could not set %s to %g" % (name, prop_name, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-camera", type=int, default=0, help="OpenCV index for the left webcam")
    parser.add_argument("--right-camera", type=int, default=1, help="OpenCV index for the right webcam")
    parser.add_argument("--family", choices=sorted(APRILTAG_FAMILIES), default="36h11")
    parser.add_argument("--tag-id", type=int, help="Only report this tag ID")
    parser.add_argument("--width", type=int, help="Requested camera width, e.g. 1280")
    parser.add_argument("--height", type=int, help="Requested camera height, e.g. 720")
    parser.add_argument("--backend", choices=["any", "dshow", "msmf"], default="dshow")
    parser.add_argument("--print-every", type=float, default=0.5, help="Seconds between console reports")
    parser.add_argument("--left-brightness", type=float, help="UVC brightness for the left camera (A82 default 5)")
    parser.add_argument("--right-brightness", type=float, help="UVC brightness for the right camera (A82 default 5)")
    parser.add_argument("--left-gain", type=float, help="UVC gain for the left camera (A82 default 1)")
    parser.add_argument("--right-gain", type=float, help="UVC gain for the right camera (A82 default 1)")
    parser.add_argument(
        "--left-led",
        choices=["off", "min", "medium", "max"],
        help="Tip-LED level for the left camera, applied after the stream opens. Needs --backend msmf.",
    )
    parser.add_argument(
        "--right-led",
        choices=["off", "min", "medium", "max"],
        help="Tip-LED level for the right camera, applied after the stream opens. Needs --backend msmf.",
    )
    parser.add_argument("--no-window", action="store_true", help="Run without a preview window")
    parser.add_argument("--scan-cameras", action="store_true", help="Print usable camera indices and exit")
    parser.add_argument("--scan-max-index", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scan_cameras:
        found = scan_cameras(args.scan_max_index, args.backend)
        if found:
            print("usable camera index/backend pairs:")
            for index, backend_name in found:
                print("  index=%d backend=%s" % (index, backend_name))
        else:
            print("no usable cameras found")
        return

    if args.left_camera == args.right_camera:
        raise SystemExit("left and right camera indices must be different")

    detector = create_detector(args.family)
    left_cap = open_camera(args.left_camera, args.width, args.height, args.backend)
    right_cap = open_camera(args.right_camera, args.width, args.height, args.backend)
    apply_tuning(left_cap, "left", args.left_brightness, args.left_gain)
    apply_tuning(right_cap, "right", args.right_brightness, args.right_gain)
    if args.left_led or args.right_led:
        # The firmware resets the LED to max when streaming actually starts,
        # which with MSMF is at the first frame read - so stream first.
        for _ in range(5):
            left_cap.read()
            right_cap.read()
        time.sleep(0.5)
        apply_led(args.left_camera, args.left_led, "left")
        apply_led(args.right_camera, args.right_led, "right")
    print(
        "Detecting AprilTags family %s on cameras %d and %d. Press q in the preview to quit."
        % (args.family, args.left_camera, args.right_camera)
    )

    last_print = 0.0
    try:
        while True:
            left_ok, left_frame = left_cap.read()
            right_ok, right_frame = right_cap.read()
            if not left_ok:
                raise RuntimeError("Left camera frame read failed")
            if not right_ok:
                raise RuntimeError("Right camera frame read failed")

            left_detections = detect_tags(left_frame, detector, "left", args.tag_id)
            right_detections = detect_tags(right_frame, detector, "right", args.tag_id)

            now = time.monotonic()
            if now - last_print >= args.print_every:
                print_report(left_detections, right_detections)
                last_print = now

            if not args.no_window:
                draw_overlay(left_frame, "left", left_detections)
                draw_overlay(right_frame, "right", right_detections)
                preview = make_side_by_side(left_frame, right_frame)
                cv2.imshow("Stereo AprilTag webcams", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    except KeyboardInterrupt:
        print("stopped")
    finally:
        left_cap.release()
        right_cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

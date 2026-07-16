"""Interactive held-key WASD/QE velocity control for three KCube axes.

Close Kinesis before running this script. Holding a movement key starts
continuous velocity motion; releasing the key stops that axis.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
from typing import Dict, Iterable, Optional, Tuple

import cv2

from kcube_motion import (
    AxisConfig,
    VelocityParams,
    load_axes,
    move_velocity,
    read_position_mm,
    set_velocity_scale,
    set_velocity_params,
    stop,
)


DEFAULT_X_AXIS = "axis1"
DEFAULT_Y_AXIS = "axis2"
DEFAULT_Z_AXIS = "axis3"

VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_CODES = {
    "a": 0x41,
    "d": 0x44,
    "e": 0x45,
    "h": 0x48,
    "j": 0x4A,
    "l": 0x4C,
    "q": 0x51,
    "s": 0x53,
    "w": 0x57,
    "x": 0x58,
}


def require_axis(axes: Dict[str, AxisConfig], name: str, role: str) -> None:
    if name not in axes:
        known = ", ".join(sorted(axes)) or "(none)"
        raise SystemExit("Unknown %s axis '%s'. Known axes: %s" % (role, name, known))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive KCube axes while WASD/QE keys are held, with a live webcam preview."
    )
    parser.add_argument("--x-axis", default=DEFAULT_X_AXIS, help="Axis driven by A/D.")
    parser.add_argument("--y-axis", default=DEFAULT_Y_AXIS, help="Axis driven by W/S.")
    parser.add_argument("--z-axis", default=DEFAULT_Z_AXIS, help="Axis driven by Q/E.")
    parser.add_argument("--invert-x", action="store_true", help="Invert A/D direction.")
    parser.add_argument("--invert-y", action="store_true", help="Invert W/S direction.")
    parser.add_argument("--invert-z", action="store_true", help="Invert Q/E direction.")
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=0.8,
        help="Fraction of each controller's configured max velocity to use. Default is 0.8.",
    )
    parser.add_argument(
        "--limit-guard-mm",
        type=float,
        default=0.05,
        help="Stop this far before configured soft limits during held-key driving.",
    )
    parser.add_argument(
        "--poll-s",
        type=float,
        default=0.02,
        help="Key/camera polling period in seconds.",
    )
    parser.add_argument(
        "--limit-check-s",
        type=float,
        default=0.08,
        help="How often to read positions while moving and enforce soft limits.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to show during manual control.")
    parser.add_argument("--camera-width", type=int, help="Requested camera capture width in pixels.")
    parser.add_argument("--camera-height", type=int, help="Requested camera capture height in pixels.")
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=3.0,
        help="Preview window scale factor. Default is 3x normal frame size.",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=5.0,
        help="Preview rotation step for J/L keys in degrees. Default is 5.",
    )
    parser.add_argument("--no-camera", action="store_true", help="Run without opening a webcam preview.")
    return parser.parse_args()


def key_down(vk_code: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def key_pressed_once(vk_code: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x0001)


def open_camera(index: int, width: Optional[int], height: Optional[int]):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index %s" % index)
    return cap


def rotate_frame(frame, angle_deg: float):
    if abs(angle_deg) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_width = int((height * sin_a) + (width * cos_a))
    new_height = int((height * cos_a) + (width * sin_a))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    return cv2.warpAffine(frame, matrix, (new_width, new_height))


def read_rotation(rotation_state: Dict[str, float], rotation_lock: threading.Lock) -> float:
    with rotation_lock:
        return rotation_state["angle_deg"]


def set_preview_window_size(window_name: str, frame, scale: float) -> None:
    height, width = frame.shape[:2]
    cv2.resizeWindow(window_name, int(width * scale), int(height * scale))


def camera_preview_loop(
    args: argparse.Namespace,
    stop_event: threading.Event,
    rotation_state: Dict[str, float],
    rotation_lock: threading.Lock,
) -> None:
    cap = None
    window_name = "KCube manual control webcam"
    last_angle = None
    try:
        cap = open_camera(args.camera, args.camera_width, args.camera_height)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        ok, frame = cap.read()
        if ok:
            frame = rotate_frame(frame, read_rotation(rotation_state, rotation_lock))
            set_preview_window_size(window_name, frame, args.camera_scale)
            cv2.imshow(window_name, frame)

        while not stop_event.is_set():
            ok, frame = cap.read()
            if ok:
                angle = read_rotation(rotation_state, rotation_lock)
                frame = rotate_frame(frame, angle)
                if angle != last_angle:
                    set_preview_window_size(window_name, frame, args.camera_scale)
                    last_angle = angle
                cv2.imshow(window_name, frame)
            cv2.waitKey(1)
    except Exception as exc:
        print("Webcam preview stopped: %s" % exc)
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


def start_camera_preview(
    args: argparse.Namespace,
    rotation_state: Dict[str, float],
    rotation_lock: threading.Lock,
) -> Tuple[Optional[threading.Event], Optional[threading.Thread]]:
    if args.no_camera:
        return None, None
    if args.camera_scale <= 0:
        raise SystemExit("--camera-scale must be positive.")
    stop_event = threading.Event()
    thread = threading.Thread(
        target=camera_preview_loop,
        args=(args, stop_event, rotation_state, rotation_lock),
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def stop_camera_preview(stop_event: Optional[threading.Event], thread: Optional[threading.Thread]) -> None:
    if stop_event is not None:
        stop_event.set()
    if thread is not None:
        thread.join(timeout=2.0)


def signed_direction(positive_key: str, negative_key: str, invert: bool) -> int:
    positive = key_down(VK_CODES[positive_key])
    negative = key_down(VK_CODES[negative_key])
    if positive == negative:
        return 0
    direction = 1 if positive else -1
    return -direction if invert else direction


def requested_directions(args: argparse.Namespace) -> Dict[str, int]:
    return {
        args.x_axis: signed_direction("w", "s", args.invert_x),
        args.y_axis: signed_direction("a", "d", args.invert_y),
        args.z_axis: signed_direction("e", "q", args.invert_z),
    }


def format_positions(axis_names: Iterable[str], axes: Dict[str, AxisConfig]) -> str:
    parts = []
    for name in axis_names:
        pos = read_position_mm(axes[name].serial)
        if pos is None:
            parts.append("%s=?" % name)
        else:
            parts.append("%s=%.4f mm" % (name, pos))
    return ", ".join(parts)


def print_help(args: argparse.Namespace) -> None:
    print("")
    print("KCube WASD/QE held-key drive")
    print("  Hold W/S: drive %s positive/negative" % args.x_axis)
    print("  Hold A/D: drive %s positive/negative" % args.y_axis)
    print("  Hold Q/E: drive %s negative/positive" % args.z_axis)
    print("  Speed: %g%% of each controller's configured max velocity" % (args.speed_scale * 100.0))
    print("  Space: stop all configured drive axes")
    print("  J/L: rotate webcam preview -%g/+%g degrees" % (args.rotation_step_deg, args.rotation_step_deg))
    print("  H: show this help")
    print("  X or Esc: stop and exit")
    if not args.no_camera:
        print("  Webcam preview: camera %s at %gx window scale" % (args.camera, args.camera_scale))
    print("")


def stop_axes(axis_names: Iterable[str], axes: Dict[str, AxisConfig]) -> None:
    seen = set()
    for name in axis_names:
        if name in seen:
            continue
        seen.add(name)
        stop(axes[name].serial)


def safe_to_drive(axis: AxisConfig, direction: int, guard_mm: float) -> Tuple[bool, Optional[float]]:
    position = read_position_mm(axis.serial)
    if position is None:
        return False, None
    if direction < 0 and position <= axis.min_mm + guard_mm:
        return False, position
    if direction > 0 and position >= axis.max_mm - guard_mm:
        return False, position
    return True, position


def start_velocity(axis: AxisConfig, direction: int, guard_mm: float) -> bool:
    ok, position = safe_to_drive(axis, direction, guard_mm)
    if not ok:
        if position is None:
            print("%s: position unavailable; refusing continuous drive" % axis.name)
        else:
            print("%s: at soft-limit guard %.4f mm; refusing direction %+d" % (axis.name, position, direction))
        return False
    move_velocity(axis.serial, direction)
    print("%s: driving %+d from %.4f mm" % (axis.name, direction, position))
    return True


def apply_velocity_limits(
    active: Dict[str, int],
    axes: Dict[str, AxisConfig],
    guard_mm: float,
) -> None:
    for name, direction in sorted(active.items()):
        if direction == 0:
            continue
        ok, position = safe_to_drive(axes[name], direction, guard_mm)
        if ok:
            continue
        stop(axes[name].serial)
        active[name] = 0
        if position is None:
            print("%s: stopped; position unavailable" % name)
        else:
            print("%s: stopped at soft-limit guard %.4f mm" % (name, position))


def configure_velocity(axis_names: Iterable[str], axes: Dict[str, AxisConfig], scale: float) -> Dict[str, VelocityParams]:
    originals = {}
    for name in axis_names:
        original = set_velocity_scale(axes[name].serial, scale)
        originals[name] = original
        print(
            "%s: velocity set to %.4f mm/s (%g%% of %.4f mm/s)"
            % (name, original.max_velocity_mm_s * scale, scale * 100.0, original.max_velocity_mm_s)
        )
    return originals


def restore_velocity(originals: Dict[str, VelocityParams], axes: Dict[str, AxisConfig]) -> None:
    for name, params in originals.items():
        try:
            set_velocity_params(axes[name].serial, params)
        except Exception as exc:
            print("%s: could not restore original velocity params: %s" % (name, exc))


def rotate_preview(
    rotation_state: Dict[str, float],
    rotation_lock: threading.Lock,
    delta_deg: float,
) -> None:
    with rotation_lock:
        rotation_state["angle_deg"] = (rotation_state["angle_deg"] + delta_deg) % 360.0
        angle = rotation_state["angle_deg"]
    display_angle = angle if angle <= 180.0 else angle - 360.0
    print("Webcam rotation: %.1f deg" % display_angle)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.speed_scale <= 1.0:
        raise SystemExit("--speed-scale must be greater than 0 and no more than 1.")
    if args.limit_guard_mm < 0:
        raise SystemExit("--limit-guard-mm must be non-negative.")
    if args.poll_s <= 0:
        raise SystemExit("--poll-s must be positive.")
    if args.limit_check_s <= 0:
        raise SystemExit("--limit-check-s must be positive.")
    if args.rotation_step_deg <= 0:
        raise SystemExit("--rotation-step-deg must be positive.")

    axes = load_axes()
    require_axis(axes, args.x_axis, "x")
    require_axis(axes, args.y_axis, "y")
    require_axis(axes, args.z_axis, "z")

    axis_names = [args.x_axis, args.y_axis, args.z_axis]
    if len(set(axis_names)) != len(axis_names):
        raise SystemExit("The x, y, and z controls must map to three different configured axes.")

    rotation_state = {"angle_deg": 0.0}
    rotation_lock = threading.Lock()
    camera_stop, camera_thread = start_camera_preview(args, rotation_state, rotation_lock)
    originals = {}
    active = {name: 0 for name in axis_names}
    last_limit_check = time.monotonic()

    try:
        originals = configure_velocity(axis_names, axes, args.speed_scale)
        print_help(args)
        print("Current position: %s" % format_positions(axis_names, axes))

        while True:
            if key_down(VK_ESCAPE) or key_down(VK_CODES["x"]):
                print("Stopping and exiting.")
                stop_axes(axis_names, axes)
                return
            if key_pressed_once(VK_CODES["h"]):
                print_help(args)
            if key_pressed_once(VK_CODES["j"]):
                rotate_preview(rotation_state, rotation_lock, -args.rotation_step_deg)
            if key_pressed_once(VK_CODES["l"]):
                rotate_preview(rotation_state, rotation_lock, args.rotation_step_deg)
            if key_down(VK_SPACE):
                stop_axes(axis_names, axes)
                for name in active:
                    active[name] = 0
                time.sleep(args.poll_s)
                continue

            requested = requested_directions(args)
            for name, direction in sorted(requested.items()):
                if direction == active[name]:
                    continue
                if active[name] != 0:
                    stop(axes[name].serial)
                    active[name] = 0
                if direction != 0 and start_velocity(axes[name], direction, args.limit_guard_mm):
                    active[name] = direction

            now = time.monotonic()
            if now - last_limit_check >= args.limit_check_s:
                apply_velocity_limits(active, axes, args.limit_guard_mm)
                last_limit_check = now

            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("")
        print("Interrupted; stopping axes.")
        stop_axes(axis_names, axes)
    finally:
        stop_axes(axis_names, axes)
        restore_velocity(originals, axes)
        stop_camera_preview(camera_stop, camera_thread)


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("kcube_wasd_jog.py uses Windows keyboard state polling.")
    main()

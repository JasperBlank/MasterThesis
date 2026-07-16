"""Held-key KCube control with live needle and red-dot detection.

This combines the manual WASD/QE velocity controller with the live camera
pipeline from needle_detector.py.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

from kcube_motion import AxisConfig, VelocityParams, load_axes, move_velocity, read_position_mm
from kcube_motion import set_velocity_params, set_velocity_scale, stop
from kcube_wasd_jog import (
    VK_CODES,
    VK_ESCAPE,
    VK_SPACE,
    key_down,
    key_pressed_once,
    rotate_frame,
    rotate_preview,
    signed_direction,
)
from needle_detector import (
    NeedleDetection,
    NeedleParams,
    RedDotParams,
    _debug_view_frame,
    _read_latest,
    detect_needle,
    detect_red_dot,
    draw_needle_overlay,
    draw_red_dot_overlay,
    open_camera,
)


DEFAULT_X_AXIS = "axis1"
DEFAULT_Y_AXIS = "axis2"
DEFAULT_Z_AXIS = "axis3"


@dataclass
class ParallaxSample:
    t: float
    stage_x_mm: float
    stage_y_mm: float
    dot_x_px: float
    dot_y_px: float


@dataclass
class ParallaxEstimate:
    z_mm: float
    motion_mm: float
    disparity_px: float


def require_axis(axes: Dict[str, AxisConfig], name: str, role: str) -> None:
    if name not in axes:
        known = ", ".join(sorted(axes)) or "(none)"
        raise SystemExit("Unknown %s axis '%s'. Known axes: %s" % (role, name, known))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive KCube axes while detecting the needle and red dot live."
    )

    parser.add_argument("--x-axis", default=DEFAULT_X_AXIS, help="Axis driven by W/S.")
    parser.add_argument("--y-axis", default=DEFAULT_Y_AXIS, help="Axis driven by A/D.")
    parser.add_argument("--z-axis", default=DEFAULT_Z_AXIS, help="Axis driven by Q/E.")
    parser.add_argument("--invert-x", action="store_true", help="Invert W/S direction.")
    parser.add_argument("--invert-y", action="store_true", help="Invert A/D direction.")
    parser.add_argument("--invert-z", action="store_true", help="Invert Q/E direction.")
    parser.add_argument("--speed-scale", type=float, default=0.8,
                        help="Fraction of configured max velocity to use. Default is 0.8.")
    parser.add_argument("--limit-guard-mm", type=float, default=0.05,
                        help="Stop this far before configured soft limits.")
    parser.add_argument("--poll-s", type=float, default=0.02,
                        help="Motor/key polling period in seconds.")
    parser.add_argument("--limit-check-s", type=float, default=0.08,
                        help="How often to enforce soft limits while moving.")

    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=None, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=None, help="Requested camera height.")
    parser.add_argument("--camera-scale", type=float, default=3.0,
                        help="Preview scale factor. Default is 3x.")
    parser.add_argument("--rotation-step-deg", type=float, default=5.0,
                        help="Preview/detection rotation step for J/L keys.")
    parser.add_argument("--debug", action="store_true",
                        help="Show edge map next to needle/dot overlay.")

    parser.add_argument("--entry", choices=["top", "bottom", "left", "right", "auto"],
                        default="auto", help="Border the needle enters from.")
    parser.add_argument("--canny-low", type=int, default=40)
    parser.add_argument("--canny-high", type=int, default=120)
    parser.add_argument("--min-length-frac", type=float, default=0.15)
    parser.add_argument("--hough-threshold", type=int, default=30)
    parser.add_argument("--expected-angle", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.5,
                        help="Temporal smoothing for live tip; 1.0 disables smoothing.")
    parser.add_argument("--needle-width-max", type=float, default=22.0,
                        help="Max pixel gap between the needle's two parallel sides.")
    parser.add_argument("--no-require-pair", action="store_true",
                        help="Do not require two close parallel needle edges.")
    parser.add_argument("--no-require-border", action="store_true",
                        help="Do not require the needle to touch an image edge.")

    parser.add_argument("--no-red-dot", action="store_true", help="Disable red-dot detection.")
    parser.add_argument("--red-s-min", type=int, default=50,
                        help="Min HSV saturation for red dot.")
    parser.add_argument("--red-v-min", type=int, default=60,
                        help="Min HSV value for red dot.")
    parser.add_argument("--red-min-area-frac", type=float, default=0.0006,
                        help="Smallest accepted red blob as frame-area fraction.")
    parser.add_argument("--no-parallax", action="store_true",
                        help="Disable continuous red-dot distance estimation.")
    parser.add_argument("--focal-length-px", type=float, default=800.0,
                        help="Camera focal length in pixels for parallax depth. Default is 800.")
    parser.add_argument("--parallax-min-motion-mm", type=float, default=0.5,
                        help="Minimum XY camera/stage motion before estimating depth.")
    parser.add_argument("--parallax-min-disparity-px", type=float, default=3.0,
                        help="Minimum red-dot pixel motion before estimating depth.")
    parser.add_argument("--parallax-history-s", type=float, default=5.0,
                        help="Seconds of red-dot/stage history used for parallax.")
    parser.add_argument("--parallax-ema-alpha", type=float, default=0.25,
                        help="Smoothing for displayed depth estimate.")
    parser.add_argument("--position-sample-s", type=float, default=0.12,
                        help="How often to read XY stage positions for parallax.")
    return parser.parse_args()


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


def apply_velocity_limits(active: Dict[str, int], axes: Dict[str, AxisConfig], guard_mm: float) -> None:
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


def read_rotation(rotation_state: Dict[str, float], rotation_lock: threading.Lock) -> float:
    with rotation_lock:
        return rotation_state["angle_deg"]


def set_preview_window_size(window_name: str, frame, scale: float) -> None:
    height, width = frame.shape[:2]
    cv2.resizeWindow(window_name, int(width * scale), int(height * scale))


def stage_position_loop(
    axes: Dict[str, AxisConfig],
    x_axis: str,
    y_axis: str,
    period_s: float,
    stop_event: threading.Event,
    state: Dict[str, Optional[float]],
    state_lock: threading.Lock,
) -> None:
    while not stop_event.is_set():
        try:
            x_mm = read_position_mm(axes[x_axis].serial)
            y_mm = read_position_mm(axes[y_axis].serial)
            with state_lock:
                state["x_mm"] = x_mm
                state["y_mm"] = y_mm
                state["t"] = time.monotonic()
        except Exception as exc:
            print("Parallax position read failed: %s" % exc)
        stop_event.wait(period_s)


def start_stage_position_sampler(
    axes: Dict[str, AxisConfig],
    x_axis: str,
    y_axis: str,
    period_s: float,
) -> Tuple[threading.Event, threading.Thread, Dict[str, Optional[float]], threading.Lock]:
    stop_event = threading.Event()
    state: Dict[str, Optional[float]] = {"x_mm": None, "y_mm": None, "t": None}
    state_lock = threading.Lock()
    thread = threading.Thread(
        target=stage_position_loop,
        args=(axes, x_axis, y_axis, period_s, stop_event, state, state_lock),
        daemon=True,
    )
    thread.start()
    return stop_event, thread, state, state_lock


def stop_stage_position_sampler(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    thread.join(timeout=2.0)


def read_stage_state(
    state: Dict[str, Optional[float]],
    state_lock: threading.Lock,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    with state_lock:
        return state["x_mm"], state["y_mm"], state["t"]


def estimate_parallax_depth(
    current: ParallaxSample,
    samples: List[ParallaxSample],
    focal_length_px: float,
    min_motion_mm: float,
    min_disparity_px: float,
) -> Optional[ParallaxEstimate]:
    best = None
    best_motion = 0.0
    for sample in samples:
        stage_dx = current.stage_x_mm - sample.stage_x_mm
        stage_dy = current.stage_y_mm - sample.stage_y_mm
        motion_mm = math.hypot(stage_dx, stage_dy)
        if motion_mm < min_motion_mm:
            continue

        dot_dx = current.dot_x_px - sample.dot_x_px
        dot_dy = current.dot_y_px - sample.dot_y_px
        disparity_px = math.hypot(dot_dx, dot_dy)
        if disparity_px < min_disparity_px:
            continue
        if motion_mm <= best_motion:
            continue

        best_motion = motion_mm
        z_mm = focal_length_px * motion_mm / disparity_px
        best = ParallaxEstimate(z_mm=z_mm, motion_mm=motion_mm, disparity_px=disparity_px)
    return best


def draw_parallax_overlay(
    frame,
    estimate: Optional[ParallaxEstimate],
    smoothed_z_mm: Optional[float],
    enabled: bool,
) -> None:
    if not enabled:
        text = "parallax Z: off"
    elif smoothed_z_mm is None or estimate is None:
        text = "parallax Z: collecting motion"
    else:
        text = "parallax Z=%.1f mm  move=%.2f mm  shift=%.1f px" % (
            smoothed_z_mm,
            estimate.motion_mm,
            estimate.disparity_px,
        )
    cv2.putText(frame, text, (8, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
    cv2.putText(frame, text, (8, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


def detection_preview_loop(
    args: argparse.Namespace,
    needle_params: NeedleParams,
    red_params: RedDotParams,
    stop_event: threading.Event,
    rotation_state: Dict[str, float],
    rotation_lock: threading.Lock,
    stage_state: Dict[str, Optional[float]],
    stage_lock: threading.Lock,
) -> None:
    cap = None
    window_name = "KCube + needle/dot detector"
    prev: Optional[NeedleDetection] = None
    misses = 0
    last_angle = None
    parallax_samples: List[ParallaxSample] = []
    parallax_estimate: Optional[ParallaxEstimate] = None
    smoothed_z_mm: Optional[float] = None
    try:
        cap = open_camera(args.camera, args.width, args.height)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("Detection preview live on camera %s." % args.camera)
        while not stop_event.is_set():
            frame = _read_latest(cap)
            if frame is None:
                time.sleep(0.02)
                continue

            angle = read_rotation(rotation_state, rotation_lock)
            frame = rotate_frame(frame, angle)
            det = detect_needle(frame, needle_params, prev)
            if det is not None:
                prev = det
                misses = 0
            else:
                misses += 1
                if misses > 8:
                    prev = None
            shown = prev if det is None and misses <= 8 else det
            red = detect_red_dot(frame, red_params)

            if not args.no_parallax and red is not None:
                stage_x, stage_y, stage_t = read_stage_state(stage_state, stage_lock)
                if stage_x is not None and stage_y is not None:
                    now = time.monotonic()
                    sample = ParallaxSample(
                        t=now if stage_t is None else stage_t,
                        stage_x_mm=stage_x,
                        stage_y_mm=stage_y,
                        dot_x_px=red.center_x,
                        dot_y_px=red.center_y,
                    )
                    cutoff = sample.t - args.parallax_history_s
                    parallax_samples = [item for item in parallax_samples if item.t >= cutoff]
                    estimate = estimate_parallax_depth(
                        sample,
                        parallax_samples,
                        args.focal_length_px,
                        args.parallax_min_motion_mm,
                        args.parallax_min_disparity_px,
                    )
                    parallax_samples.append(sample)
                    if estimate is not None:
                        parallax_estimate = estimate
                        if smoothed_z_mm is None:
                            smoothed_z_mm = estimate.z_mm
                        else:
                            alpha = args.parallax_ema_alpha
                            smoothed_z_mm = alpha * estimate.z_mm + (1.0 - alpha) * smoothed_z_mm

            if args.debug:
                view = _debug_view_frame(frame, needle_params, shown, red)
            else:
                view = frame.copy()
                draw_needle_overlay(view, shown)
                draw_red_dot_overlay(view, red)
            draw_parallax_overlay(view, parallax_estimate, smoothed_z_mm, not args.no_parallax)

            if angle != last_angle:
                set_preview_window_size(window_name, view, args.camera_scale)
                last_angle = angle
            cv2.imshow(window_name, view)
            cv2.waitKey(1)
    except Exception as exc:
        print("Detection preview stopped: %s" % exc)
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


def start_detection_preview(
    args: argparse.Namespace,
    needle_params: NeedleParams,
    red_params: RedDotParams,
    rotation_state: Dict[str, float],
    rotation_lock: threading.Lock,
    stage_state: Dict[str, Optional[float]],
    stage_lock: threading.Lock,
) -> Tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=detection_preview_loop,
        args=(args, needle_params, red_params, stop_event, rotation_state, rotation_lock, stage_state, stage_lock),
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def stop_detection_preview(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    thread.join(timeout=2.0)


def print_help(args: argparse.Namespace) -> None:
    print("")
    print("KCube WASD/QE + needle/dot detector")
    print("  Hold W/S: drive %s positive/negative" % args.x_axis)
    print("  Hold A/D: drive %s positive/negative" % args.y_axis)
    print("  Hold Q/E: drive %s negative/positive" % args.z_axis)
    print("  J/L: rotate detection preview -%g/+%g degrees" % (args.rotation_step_deg, args.rotation_step_deg))
    print("  Space: stop all configured drive axes")
    print("  H: show this help")
    print("  X or Esc: stop and exit")
    if args.no_parallax:
        print("  Parallax depth: off")
    else:
        print(
            "  Parallax depth: f=%g px, min move=%g mm, min shift=%g px"
            % (args.focal_length_px, args.parallax_min_motion_mm, args.parallax_min_disparity_px)
        )
    print("")


def build_needle_params(args: argparse.Namespace) -> NeedleParams:
    return NeedleParams(
        entry=args.entry,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        hough_min_length_frac=args.min_length_frac,
        hough_threshold=args.hough_threshold,
        expected_angle_deg=args.expected_angle,
        needle_width_max_px=args.needle_width_max,
        require_pair=not args.no_require_pair,
        require_border=not args.no_require_border,
        ema_alpha=args.ema_alpha,
    )


def build_red_params(args: argparse.Namespace) -> RedDotParams:
    return RedDotParams(
        s_min=args.red_s_min,
        v_min=args.red_v_min,
        min_area_frac=args.red_min_area_frac,
        enabled=not args.no_red_dot,
    )


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
    if args.camera_scale <= 0:
        raise SystemExit("--camera-scale must be positive.")
    if args.rotation_step_deg <= 0:
        raise SystemExit("--rotation-step-deg must be positive.")
    if args.focal_length_px <= 0:
        raise SystemExit("--focal-length-px must be positive.")
    if args.parallax_min_motion_mm <= 0:
        raise SystemExit("--parallax-min-motion-mm must be positive.")
    if args.parallax_min_disparity_px <= 0:
        raise SystemExit("--parallax-min-disparity-px must be positive.")
    if args.parallax_history_s <= 0:
        raise SystemExit("--parallax-history-s must be positive.")
    if not 0.0 < args.parallax_ema_alpha <= 1.0:
        raise SystemExit("--parallax-ema-alpha must be greater than 0 and no more than 1.")
    if args.position_sample_s <= 0:
        raise SystemExit("--position-sample-s must be positive.")

    axes = load_axes()
    require_axis(axes, args.x_axis, "x")
    require_axis(axes, args.y_axis, "y")
    require_axis(axes, args.z_axis, "z")
    axis_names = [args.x_axis, args.y_axis, args.z_axis]
    if len(set(axis_names)) != len(axis_names):
        raise SystemExit("The x, y, and z controls must map to three different configured axes.")

    rotation_state = {"angle_deg": 0.0}
    rotation_lock = threading.Lock()
    position_stop, position_thread, stage_state, stage_lock = start_stage_position_sampler(
        axes,
        args.x_axis,
        args.y_axis,
        args.position_sample_s,
    )
    detection_stop, detection_thread = start_detection_preview(
        args,
        build_needle_params(args),
        build_red_params(args),
        rotation_state,
        rotation_lock,
        stage_state,
        stage_lock,
    )

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
        stop_detection_preview(detection_stop, detection_thread)
        stop_stage_position_sampler(position_stop, position_thread)


if __name__ == "__main__":
    main()

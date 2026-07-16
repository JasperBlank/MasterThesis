"""Dry-run AprilTag centering loop for the KCube stage.

This is the bridge between vision and motion, but it defaults to dry-run mode:
it tracks tag ID 2, compensates image roll using the tag angle, and prints the
stage correction it would command. Motor motion should be enabled only after
pixel-to-mm calibration and sign checks are confirmed.
"""

from __future__ import annotations

import argparse
import math
import time

import cv2

from apriltag_tracker import create_detector, detect_tags, draw_overlay
from kcube_motion import load_axes, move_axes_by_mm, read_position_mm


def rotate_vector(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def draw_centering_overlay(
    frame,
    detections,
    corrected_error: tuple[float, float] | None,
    proposal: tuple[str, float, str, float] | None,
    mode: str,
    controller: str | None = None,
) -> None:
    draw_overlay(frame, detections)
    status_lines = [mode]
    if controller is not None:
        status_lines.append(controller)
    if corrected_error is not None:
        status_lines.append(f"corrected dx={corrected_error[0]:+.1f}px dy={corrected_error[1]:+.1f}px")
    if proposal is not None:
        vertical_name, vertical_mm, horizontal_name, horizontal_mm = proposal
        status_lines.append(f"{vertical_name}={vertical_mm:+.4f} mm {horizontal_name}={horizontal_mm:+.4f} mm")
    if not detections:
        status_lines.append("tag not found")

    y = 24
    for line in status_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24


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
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--family", choices=["16h5", "25h9", "36h10", "36h11"], default="36h11")
    parser.add_argument("--tag-id", type=int, default=2)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--vertical-axis", default="axis1", help="Positive move makes the tag move down in the image")
    parser.add_argument("--horizontal-axis", default="axis2", help="Positive move makes the tag move right in the image")
    parser.add_argument("--gain-vertical", type=float, default=0.1, help="P gain: mm correction per corrected vertical pixel")
    parser.add_argument("--gain-horizontal", type=float, default=0.1, help="P gain: mm correction per corrected horizontal pixel")
    parser.add_argument("--integral-scale", type=float, default=0.01, help="I gain is P gain multiplied by this scale")
    parser.add_argument("--integral-limit", type=float, default=5000.0, help="Maximum accumulated corrected pixel-seconds")
    parser.add_argument("--max-step-mm", type=float, default=5.0)
    parser.add_argument("--tolerance-px", type=float, default=5.0)
    parser.add_argument("--period", type=float, default=0.5, help="Seconds between correction proposals")
    parser.add_argument("--move", action="store_true", help="Actually move motors. Default is dry-run only.")
    parser.add_argument("--no-window", action="store_true", help="Run without a preview window")
    args = parser.parse_args()

    axes = load_axes()
    if args.vertical_axis not in axes:
        raise SystemExit(f"Unknown vertical axis '{args.vertical_axis}'. Known axes: {', '.join(sorted(axes))}")
    if args.horizontal_axis not in axes:
        raise SystemExit(f"Unknown horizontal axis '{args.horizontal_axis}'. Known axes: {', '.join(sorted(axes))}")

    vertical_axis = axes[args.vertical_axis]
    horizontal_axis = axes[args.horizontal_axis]
    detector = create_detector(args.family)
    cap = open_camera(args.camera, args.width, args.height)
    mode = "LIVE MOTOR MOVE" if args.move else "dry-run"
    gain_integral_vertical = args.gain_vertical * args.integral_scale
    gain_integral_horizontal = args.gain_horizontal * args.integral_scale
    print(
        f"Centering tag ID {args.tag_id} using vertical={args.vertical_axis} "
        f"horizontal={args.horizontal_axis} in {mode} mode. Press Ctrl+C to stop."
    )
    print(
        f"PI gains: P_vertical={args.gain_vertical:g}, I_vertical={gain_integral_vertical:g}, "
        f"P_horizontal={args.gain_horizontal:g}, I_horizontal={gain_integral_horizontal:g}, "
        f"max_step={args.max_step_mm:g} mm"
    )

    last_action = 0.0
    integral_vertical = 0.0
    integral_horizontal = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            detections = detect_tags(frame, detector, args.tag_id)
            now = time.monotonic()
            corrected_error = None
            proposal = None
            controller_text = (
                f"P={args.gain_vertical:g}/{args.gain_horizontal:g} "
                f"I={gain_integral_vertical:g}/{gain_integral_horizontal:g}"
            )
            if now - last_action < args.period:
                if not args.no_window:
                    draw_centering_overlay(frame, detections, corrected_error, proposal, mode, controller_text)
                    cv2.imshow("AprilTag centering", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            last_action = now

            if not detections:
                print("tag not found")
                integral_vertical = 0.0
                integral_horizontal = 0.0
                if not args.no_window:
                    draw_centering_overlay(frame, detections, corrected_error, proposal, mode, controller_text)
                    cv2.imshow("AprilTag centering", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            det = detections[0]
            corrected_x, corrected_y = rotate_vector(det.error_x, det.error_y, -det.angle_deg)
            corrected_error = (corrected_x, corrected_y)
            if abs(corrected_x) <= args.tolerance_px and abs(corrected_y) <= args.tolerance_px:
                integral_vertical = 0.0
                integral_horizontal = 0.0
                print(
                    f"centered: raw=({det.error_x:+.1f},{det.error_y:+.1f}) px "
                    f"corrected=({corrected_x:+.1f},{corrected_y:+.1f}) px angle={det.angle_deg:+.1f} deg"
                )
                if not args.no_window:
                    draw_centering_overlay(frame, detections, corrected_error, proposal, mode, controller_text)
                    cv2.imshow("AprilTag centering", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            integral_vertical = clamp(integral_vertical + corrected_y * args.period, args.integral_limit)
            integral_horizontal = clamp(integral_horizontal + corrected_x * args.period, args.integral_limit)
            p_vertical_mm = args.gain_vertical * corrected_y
            p_horizontal_mm = args.gain_horizontal * corrected_x
            i_vertical_mm = gain_integral_vertical * integral_vertical
            i_horizontal_mm = gain_integral_horizontal * integral_horizontal
            delta_vertical_mm = clamp(-(p_vertical_mm + i_vertical_mm), args.max_step_mm)
            delta_horizontal_mm = clamp(-(p_horizontal_mm + i_horizontal_mm), args.max_step_mm)
            proposal = (args.vertical_axis, delta_vertical_mm, args.horizontal_axis, delta_horizontal_mm)
            print(
                f"id={det.tag_id} raw=({det.error_x:+.1f},{det.error_y:+.1f}) px "
                f"angle={det.angle_deg:+.1f} deg corrected=({corrected_x:+.1f},{corrected_y:+.1f}) px "
                f"PI_v=({p_vertical_mm:+.4f},{i_vertical_mm:+.4f}) PI_h=({p_horizontal_mm:+.4f},{i_horizontal_mm:+.4f}) "
                f"proposal {args.vertical_axis}={delta_vertical_mm:+.4f} mm "
                f"{args.horizontal_axis}={delta_horizontal_mm:+.4f} mm"
            )

            if not args.no_window:
                draw_centering_overlay(frame, detections, corrected_error, proposal, mode, controller_text)
                cv2.imshow("AprilTag centering", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.move:
                # Reading positions here gives a clear error before commanding
                # either axis if the motor connection is unavailable.
                if read_position_mm(vertical_axis.serial) is None or read_position_mm(horizontal_axis.serial) is None:
                    raise RuntimeError("Could not read motor position; refusing correction move.")
                move_axes_by_mm(
                    {
                        args.vertical_axis: delta_vertical_mm,
                        args.horizontal_axis: delta_horizontal_mm,
                    },
                    axes,
                )
    except KeyboardInterrupt:
        print("stopped")
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

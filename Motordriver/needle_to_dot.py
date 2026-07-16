"""Two-phase visual servoing: center the red dot, then bring the needle to it.

Rig assumption (confirmed): the endoscope camera rides on the stage together with
the needle. So moving the stage moves the *camera*, which moves the fixed red dot
within the image; the needle is rigid relative to the camera and stays ~fixed in
the image. The thing we actually steer is therefore the DOT.

  Phase 1  CENTER_DOT : drive the dot to the image center (position the end-effector
                        over the target).
  Phase 2  APPROACH   : drive the dot onto the needle tip (align tip with target).

Rotation normalization: the needle's image direction changes only with camera roll,
so it is a roll sensor. We measure the needle angle, de-rotate everything into a
canonical "needle-comes-from-the-top-left" frame, and issue motor commands in that
frame -- so a single sign/gain calibration holds at any camera roll.

Safety (mirrors center_tag.py): DRY-RUN BY DEFAULT. --move arms the motors. Every
correction is clamped to --max-step-mm, gated by a pixel deadband, requires both
relevant detections locked for --lock-frames cycles, and is only sent when the
resulting absolute target is inside the axis soft limits (otherwise it is logged,
not sent).

Python 3.8 compatible.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from needle_detector import (
    NeedleDetection,
    NeedleParams,
    RedDotParams,
    detect_needle,
    detect_red_dot,
    open_camera,
    _read_latest,
)
from kcube_motion import load_axes, move_axes_by_mm, read_position_mm


# --------------------------------------------------------------------------- #
# Small geometry / control helpers
# --------------------------------------------------------------------------- #

def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def norm_angle(a: float) -> float:
    """Wrap to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def needle_angle_deg(needle: NeedleDetection) -> float:
    """Direction entry -> tip in raw image coords (x right, y down)."""
    ex, ey = needle.entry
    tx, ty = needle.tip
    return math.degrees(math.atan2(ty - ey, tx - ex))


def angle_ema(prev: Optional[float], new: float, alpha: float) -> float:
    """EMA on an angle using the shortest-arc difference (avoids the ±180 jump)."""
    if prev is None:
        return new
    return norm_angle(prev + alpha * norm_angle(new - prev))


def rotation_matrix(theta_deg: float, center: Tuple[float, float]) -> np.ndarray:
    return cv2.getRotationMatrix2D(center, theta_deg, 1.0)


def transform_point(M: np.ndarray, pt) -> Tuple[int, int]:
    x = M[0, 0] * pt[0] + M[0, 1] * pt[1] + M[0, 2]
    y = M[1, 0] * pt[0] + M[1, 1] * pt[1] + M[1, 2]
    return int(round(x)), int(round(y))


def rotate_vector(R: np.ndarray, vx: float, vy: float) -> Tuple[float, float]:
    return float(R[0, 0] * vx + R[0, 1] * vy), float(R[1, 0] * vx + R[1, 1] * vy)


class NeedleKalman:
    """Constant-velocity Kalman filter for needle tip and entry point in pixels."""

    def __init__(self, process_noise: float, measurement_noise: float, miss_reset: int) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.miss_reset = miss_reset
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.misses = 0
        self.last_support = 0

    def update(self, measurement: Optional[NeedleDetection]) -> Optional[NeedleDetection]:
        if measurement is None:
            return self.predict_only()

        z = np.array(
            [[measurement.tip_x],
             [measurement.tip_y],
             [measurement.entry_x],
             [measurement.entry_y]],
            dtype=np.float64,
        )

        if self.x is None:
            self.x = np.zeros((8, 1), dtype=np.float64)
            self.x[0:4, 0:1] = z
            self.P = np.eye(8, dtype=np.float64) * self.measurement_noise
        else:
            self._predict()
            H = self._measurement_matrix()
            R = np.eye(4, dtype=np.float64) * self.measurement_noise
            y = z - H.dot(self.x)
            S = H.dot(self.P).dot(H.T) + R
            K = self.P.dot(H.T).dot(np.linalg.inv(S))
            self.x = self.x + K.dot(y)
            self.P = (np.eye(8, dtype=np.float64) - K.dot(H)).dot(self.P)

        self.misses = 0
        self.last_support = measurement.n_segments
        return self.estimate()

    def predict_only(self) -> Optional[NeedleDetection]:
        if self.x is None:
            return None
        self._predict()
        self.misses += 1
        if self.misses > self.miss_reset:
            self.reset()
            return None
        return self.estimate()

    def estimate(self) -> Optional[NeedleDetection]:
        if self.x is None:
            return None
        tip_x = float(self.x[0, 0])
        tip_y = float(self.x[1, 0])
        entry_x = float(self.x[2, 0])
        entry_y = float(self.x[3, 0])
        return NeedleDetection(
            tip_x=tip_x,
            tip_y=tip_y,
            entry_x=entry_x,
            entry_y=entry_y,
            angle_deg=math.degrees(math.atan2(tip_y - entry_y, tip_x - entry_x)),
            length_px=math.hypot(tip_x - entry_x, tip_y - entry_y),
            n_segments=self.last_support,
        )

    def reset(self) -> None:
        self.x = None
        self.P = None
        self.misses = 0
        self.last_support = 0

    def _predict(self) -> None:
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = 1.0
        F[1, 5] = 1.0
        F[2, 6] = 1.0
        F[3, 7] = 1.0
        Q = np.eye(8, dtype=np.float64) * self.process_noise
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q

    @staticmethod
    def _measurement_matrix() -> np.ndarray:
        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        return H


# --------------------------------------------------------------------------- #
# Motor command planning (logs the would-be command; never raises)
# --------------------------------------------------------------------------- #

def plan_axis(name: str, delta_mm: float, axes) -> dict:
    cfg = axes.get(name)
    if cfg is None:
        return {"name": name, "delta": delta_mm, "ok": False, "note": "unknown axis (no config)"}
    try:
        current = read_position_mm(cfg.serial)
    except Exception as exc:
        return {"name": name, "delta": delta_mm, "ok": False, "note": f"read failed: {exc}"}
    if current is None:
        return {"name": name, "delta": delta_mm, "ok": False, "note": "position unreadable"}
    target = current + delta_mm
    in_range = cfg.min_mm <= target <= cfg.max_mm
    return {"name": name, "delta": delta_mm, "current": current, "target": target,
            "lo": cfg.min_mm, "hi": cfg.max_mm, "in_range": in_range, "ok": True}


def format_plan(plan: dict) -> str:
    if not plan.get("ok"):
        return f"{plan['name']}: dx={plan['delta']:+.4f} mm  [{plan['note']}]"
    flag = "IN" if plan["in_range"] else "OUT-OF-LIMITS"
    return (f"{plan['name']}: {plan['current']:+.4f} {plan['delta']:+.4f} -> "
            f"{plan['target']:+.4f} mm  limits[{plan['lo']:.3f},{plan['hi']:.3f}]  {flag}")


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #

def draw_view(frame, M, needle, red, target_pt, phase, theta, err_canon, locked):
    """Rotate the frame into the canonical view and draw upright overlays."""
    h, w = frame.shape[:2]
    view = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    center = (w // 2, h // 2)
    cv2.drawMarker(view, center, (255, 0, 0), cv2.MARKER_CROSS, 26, 2)

    if needle is not None:
        e = transform_point(M, needle.entry)
        t = transform_point(M, needle.tip)
        cv2.line(view, e, t, (0, 165, 255), 2)
        cv2.circle(view, t, 6, (0, 0, 255), -1)
    if red is not None:
        c = transform_point(M, red.center)
        cv2.circle(view, c, max(4, int(red.radius_px)), (255, 0, 255), 2)
        cv2.drawMarker(view, c, (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
        if target_pt is not None:
            cv2.line(view, c, transform_point(M, target_pt), (255, 255, 0), 1)

    lines = [f"phase={phase}  {'LOCKED' if locked else 'acquiring'}  theta={theta:+.1f} deg"]
    if err_canon is not None:
        lines.append(f"error=({err_canon[0]:+.1f},{err_canon[1]:+.1f}) px (canonical)")
    if needle is None:
        lines.append("needle not found")
    if red is None:
        lines.append("red dot not found")
    y = 22
    for ln in lines:
        cv2.putText(view, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(view, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24
    return view


def draw_panel_label(frame, label: str) -> None:
    y0 = max(0, frame.shape[0] - 32)
    cv2.rectangle(frame, (0, y0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1)


def fit_height(frame, height: int):
    if frame.shape[0] == height:
        return frame
    scale = float(height) / float(frame.shape[0])
    width = max(1, int(round(frame.shape[1] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def blank_panel_like(frame, text: str):
    panel = np.zeros_like(frame)
    cv2.putText(panel, text, (18, panel.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (220, 220, 220), 2)
    return panel


def side_by_side(live_frame, decision_frame):
    live = live_frame.copy()
    if decision_frame is None:
        decision = blank_panel_like(live, "waiting for first movement decision")
    else:
        decision = fit_height(decision_frame.copy(), live.shape[0])
    draw_panel_label(live, "LIVE CAMERA")
    draw_panel_label(decision, "FINAL FRAME USED FOR MOVEMENT")
    return np.hstack([live, decision])


def servo_worker(
    args: argparse.Namespace,
    axes,
    needle_params: NeedleParams,
    red_params: RedDotParams,
    frame_state: Dict[str, object],
    frame_lock: threading.Lock,
    decision_state: Dict[str, Optional[np.ndarray]],
    decision_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    phase = "CENTER_DOT"
    integral_v = integral_h = 0.0
    lock = 0
    theta = 0.0 if args.no_rotate else None
    kalman = NeedleKalman(args.kalman_process_noise, args.kalman_measurement_noise,
                          args.kalman_miss_reset)
    mode = "LIVE MOTOR MOVE" if args.move else "dry-run"
    gain_i_v = args.gain_vertical * args.integral_scale
    gain_i_h = args.gain_horizontal * args.integral_scale

    while not stop_event.is_set():
        loop_start = time.monotonic()

        with frame_lock:
            frames_obj = frame_state.get("frames")
            frames = [item.copy() for item in frames_obj] if frames_obj is not None else []

        if not frames:
            stop_event.wait(0.02)
            continue

        frame = frames[-1]
        h, w = frame.shape[:2]
        center_pt = (w / 2.0, h / 2.0)

        raw_needle = None
        needle = None
        detections_in_batch = 0
        for sample in frames:
            raw_needle = detect_needle(sample, needle_params)
            if raw_needle is not None:
                detections_in_batch += 1
            if args.no_kalman:
                if raw_needle is not None:
                    needle = raw_needle
            else:
                needle = kalman.update(raw_needle)
        if args.no_kalman and raw_needle is None:
            needle = None
        elif not args.no_kalman and detections_in_batch == 0:
            needle = None
        red = detect_red_dot(frame, red_params)

        if not args.no_rotate and needle is not None:
            theta_meas = norm_angle(needle_angle_deg(needle) - args.canonical_angle)
            theta = angle_ema(theta, theta_meas, args.theta_ema)
        theta_use = 0.0 if theta is None else theta
        M = rotation_matrix(theta_use, center_pt)
        R = M[:, :2]

        target_raw = None
        if phase == "CENTER_DOT":
            target_raw = center_pt
        elif phase == "APPROACH" and needle is not None:
            target_raw = (float(needle.tip[0]), float(needle.tip[1]))

        err_canon = None
        if red is not None and target_raw is not None:
            ex, ey = rotate_vector(R, red.center[0] - target_raw[0],
                                   red.center[1] - target_raw[1])
            err_canon = (ex, ey)

        decision_view = draw_view(frame, M, needle, red, target_raw, phase,
                                  theta_use, err_canon, lock >= args.lock_frames)
        source_text = ("needle target: single frame" if args.no_kalman
                       else "needle target: Kalman %d/%d detections"
                       % (detections_in_batch, len(frames)))
        cv2.putText(decision_view, source_text, (12, max(24, decision_view.shape[0] - 44)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(decision_view, source_text, (12, max(24, decision_view.shape[0] - 44)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        with decision_lock:
            decision_state["frame"] = decision_view

        if err_canon is None:
            integral_v = integral_h = 0.0
            lock = 0
            if phase == "APPROACH" and red is not None and needle is None:
                print("APPROACH waiting: needle (target) not detected")
            else:
                print(f"{phase} waiting: red dot not detected")
                if red is None:
                    phase = "CENTER_DOT"
        else:
            ex, ey = err_canon
            tol = args.center_tol_px if phase == "CENTER_DOT" else args.approach_tol_px
            within = abs(ex) <= tol and abs(ey) <= tol
            lock = lock + 1 if within else 0

            if within and phase == "CENTER_DOT" and lock >= args.lock_frames:
                print(f"dot centered (err {ex:+.1f},{ey:+.1f}) -> APPROACH")
                phase = "APPROACH"
                integral_v = integral_h = 0.0
                lock = 0
            elif within and phase == "APPROACH":
                integral_v = integral_h = 0.0
                print(f"ON TARGET: dot on needle tip (err {ex:+.1f},{ey:+.1f}) px")
            else:
                integral_v = clamp(integral_v + ey * args.period, args.integral_limit)
                integral_h = clamp(integral_h + ex * args.period, args.integral_limit)
                dv = clamp(-(args.gain_vertical * ey + gain_i_v * integral_v), args.max_step_mm)
                dh = clamp(-(args.gain_horizontal * ex + gain_i_h * integral_h), args.max_step_mm)
                plan_v = plan_axis(args.vertical_axis, dv, axes)
                plan_h = plan_axis(args.horizontal_axis, dh, axes)
                in_range = (plan_v.get("ok") and plan_h.get("ok")
                            and plan_v["in_range"] and plan_h["in_range"])
                print(f"{phase} err=({ex:+.1f},{ey:+.1f}) px theta={theta_use:+.1f}  [{mode}]")
                print("  would send -> " + format_plan(plan_v))
                print("              " + format_plan(plan_h))
                if not args.move:
                    pass
                elif not in_range:
                    print("  NOT SENDING: would exceed soft limits (or unreadable). Holding.")
                else:
                    move_axes_by_mm({args.vertical_axis: dv, args.horizontal_axis: dh}, axes)
                    print("  SENT.")

        elapsed = time.monotonic() - loop_start
        stop_event.wait(max(0.0, args.period - elapsed))


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-phase servo: center the red dot, then bring the needle to it.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)

    # Axis mapping / sign convention, defined in the CANONICAL (de-rotated) frame.
    parser.add_argument("--vertical-axis", default="axis1",
                        help="Axis whose POSITIVE move drives the RED DOT DOWN in the canonical image")
    parser.add_argument("--horizontal-axis", default="axis2",
                        help="Axis whose POSITIVE move drives the RED DOT RIGHT in the canonical image")

    # PI controller
    parser.add_argument("--gain-vertical", type=float, default=0.05)
    parser.add_argument("--gain-horizontal", type=float, default=0.05)
    parser.add_argument("--integral-scale", type=float, default=0.01)
    parser.add_argument("--integral-limit", type=float, default=5000.0)
    parser.add_argument("--max-step-mm", type=float, default=0.3)
    parser.add_argument("--period", type=float, default=0.4)
    parser.add_argument("--lock-frames", type=int, default=3)

    # Phase tolerances
    parser.add_argument("--center-tol-px", type=float, default=10.0,
                        help="Phase 1 done when the dot is within this of image center")
    parser.add_argument("--approach-tol-px", type=float, default=6.0,
                        help="Phase 2 done when the dot is within this of the needle tip")

    # Rotation normalization
    parser.add_argument("--canonical-angle", type=float, default=45.0,
                        help="Target needle direction (deg) for 'from the top-left'. 45 = down-right.")
    parser.add_argument("--theta-ema", type=float, default=0.3,
                        help="Smoothing on the estimated camera roll (1.0 = none)")
    parser.add_argument("--no-rotate", action="store_true",
                        help="Disable rotation normalization (assume needle already top-left)")

    # Detection tunables
    parser.add_argument("--needle-width-max", type=float, default=22.0)
    parser.add_argument("--kalman-frames", type=int, default=5,
                        help="Number of newest camera frames used to update the needle Kalman estimate per move decision.")
    parser.add_argument("--kalman-process-noise", type=float, default=2.0,
                        help="Needle Kalman process noise in pixel units; higher follows motion faster.")
    parser.add_argument("--kalman-measurement-noise", type=float, default=16.0,
                        help="Needle Kalman measurement noise in pixel units; higher trusts detections less.")
    parser.add_argument("--kalman-miss-reset", type=int, default=8,
                        help="Reset the needle Kalman estimate after this many consecutive missed frames.")
    parser.add_argument("--no-kalman", action="store_true",
                        help="Use the latest single-frame needle detection directly.")
    parser.add_argument("--red-s-min", type=int, default=50)
    parser.add_argument("--red-v-min", type=int, default=60)
    parser.add_argument("--red-min-area-frac", type=float, default=0.0006)

    parser.add_argument("--move", action="store_true",
                        help="ACTUALLY move motors. Default is dry-run (log only).")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    if args.kalman_frames < 1:
        raise SystemExit("--kalman-frames must be at least 1.")
    if args.kalman_process_noise <= 0:
        raise SystemExit("--kalman-process-noise must be positive.")
    if args.kalman_measurement_noise <= 0:
        raise SystemExit("--kalman-measurement-noise must be positive.")
    if args.kalman_miss_reset < 1:
        raise SystemExit("--kalman-miss-reset must be at least 1.")

    axes = load_axes()
    if args.move:
        for axis_name in (args.vertical_axis, args.horizontal_axis):
            if axis_name not in axes:
                raise SystemExit(
                    f"Unknown axis '{axis_name}'. Known axes: {', '.join(sorted(axes)) or '(none)'}")

    needle_params = NeedleParams(entry="auto", needle_width_max_px=args.needle_width_max, ema_alpha=1.0)
    red_params = RedDotParams(s_min=args.red_s_min, v_min=args.red_v_min,
                              min_area_frac=args.red_min_area_frac)

    cap = open_camera(args.camera, args.width, args.height)
    mode = "LIVE MOTOR MOVE" if args.move else "dry-run"
    print(f"Two-phase needle servo in {mode} mode. vertical={args.vertical_axis} "
          f"horizontal={args.horizontal_axis}. Press q / Ctrl+C to stop.")
    if args.move:
        print("!! MOTORS ARMED. Keep a hand on the e-stop / Ctrl+C. !!")
    if not args.no_kalman:
        print("Needle target uses Kalman estimate over the newest %d frames." % args.kalman_frames)

    frame_state: Dict[str, object] = {"frames": deque(maxlen=args.kalman_frames)}
    decision_state: Dict[str, Optional[np.ndarray]] = {"frame": None}
    frame_lock = threading.Lock()
    decision_lock = threading.Lock()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=servo_worker,
        args=(args, axes, needle_params, red_params, frame_state, frame_lock,
              decision_state, decision_lock, stop_event),
        daemon=True,
    )
    worker.start()

    try:
        while True:
            frame = _read_latest(cap)
            if frame is None:
                raise RuntimeError("Camera frame read failed")

            with frame_lock:
                frame_buffer = frame_state["frames"]
                frame_buffer.append(frame.copy())

            if not args.no_window:
                with decision_lock:
                    decision_frame = decision_state.get("frame")
                    decision_frame = None if decision_frame is None else decision_frame.copy()
                cv2.imshow("needle servo: live + movement decision",
                           side_by_side(frame, decision_frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("stopped")
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

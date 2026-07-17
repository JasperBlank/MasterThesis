"""Digital twin of the KCube-driven probe with WASD/QE held-key velocity control.

Mirrors the control scheme of Motordriver/kcube_wasd_jog.py, but drives a
simulated 3-axis stage carrying Array_side_slip.STL instead of real hardware.
Five views are rendered in one window:
  * top row    - external twin plus the virtual left/right endoscope views
  * bottom row - live left/right endoscope frames with AprilTag overlays

A needle (default 10 mm protrusion, --needle-extension-mm) runs down the
syringe channel and out of the tip face.

Controls (identical to kcube_wasd_jog.py):
  Hold W/S: axis1 positive/negative (world X)
  Hold A/D: axis2 positive/negative (world Y)
  Hold Q/E: axis3 negative/positive (world Z, insertion depth)
  Space: stop all axes    J/L: roll both endoscope views -/+ 5 deg
  C: toggle the two CAD camera/needle mappings
  H: help                 X or Esc: stop and exit

Soft limits are read from Motordriver/kcube_axes.json so the twin refuses the
same moves the real controllers would.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pyvista as pv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STL = PROJECT_ROOT / "Holderdesign" / "cad" / "cad" / "Array_side_slip.STL"
AXES_CONFIG = PROJECT_ROOT / "Motordriver" / "kcube_axes.json"

DEFAULT_X_AXIS = "axis1"
DEFAULT_Y_AXIS = "axis2"
DEFAULT_Z_AXIS = "axis3"

# Thorlabs Z925B configured max velocity; the real script scales this value.
DEFAULT_MAX_VELOCITY_MM_S = 2.6

VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_CODES = {
    "a": 0x41,
    "c": 0x43,
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

# Tip-face bore centers measured from Array_side_slip.STL (model x/z, mm).
# Model frame: tip face at y=0, probe body along +y, 8x30x8 mm bounding box.
SCOPE_BORES = {
    "left": (1.411, 4.432),
    "right": (6.589, 4.432),
}
# Syringe channel: 0.5 mm face orifice at (4, 6), 1.0 mm bore behind it.
NEEDLE_BORE_XZ = (4.0, 6.0)
NEEDLE_RADIUS_MM = 0.25
MODEL_CENTER_XZ = (4.0, 4.0)
MUC112_DIAGONAL_FOV_DEG = 120.0

# PyVista shape "3/2" creates two renderers on the bottom first, followed by
# three renderers on top.  Keep these indices explicit because the five-panel
# layout is not a rectangular row/column grid.
LIVE_LEFT_RENDERER = 0
LIVE_RIGHT_RENDERER = 1
EXTERNAL_RENDERER = 2
SCOPE_LEFT_RENDERER = 3
SCOPE_RIGHT_RENDERER = 4
TWIN_RENDERERS = (EXTERNAL_RENDERER, SCOPE_LEFT_RENDERER, SCOPE_RIGHT_RENDERER)

# World frame: floor (phantom) at z=0, probe axis vertical, tip looking down.
TIP_HEIGHT_AT_MID_TRAVEL_MM = 15.0  # muC112 working distance is 5-50 mm

# Rotation taking model coords (x right, y along probe, z up-face) to world
# coords with the probe tip pointing down: mx->wx, my->wz, mz->-wy.
MODEL_TO_WORLD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)


class SimAxis:
    """Simulated KCube axis: position, soft limits, held-key velocity state."""

    def __init__(self, name: str, min_mm: float, max_mm: float, velocity_mm_s: float):
        self.name = name
        self.min_mm = min_mm
        self.max_mm = max_mm
        self.velocity_mm_s = velocity_mm_s
        self.position_mm = 0.5 * (min_mm + max_mm)
        self.direction = 0

    def integrate(self, dt_s: float, guard_mm: float) -> None:
        if self.direction == 0:
            return
        self.position_mm += self.direction * self.velocity_mm_s * dt_s
        low = self.min_mm + guard_mm
        high = self.max_mm - guard_mm
        if self.position_mm <= low or self.position_mm >= high:
            self.position_mm = min(max(self.position_mm, low), high)
            self.direction = 0
            print("%s: stopped at soft-limit guard %.4f mm" % (self.name, self.position_mm))

    def safe_to_drive(self, direction: int, guard_mm: float) -> bool:
        if direction < 0 and self.position_mm <= self.min_mm + guard_mm:
            return False
        if direction > 0 and self.position_mm >= self.max_mm - guard_mm:
            return False
        return True

    def start(self, direction: int, guard_mm: float) -> bool:
        if not self.safe_to_drive(direction, guard_mm):
            print(
                "%s: at soft-limit guard %.4f mm; refusing direction %+d"
                % (self.name, self.position_mm, direction)
            )
            return False
        self.direction = direction
        print("%s: driving %+d from %.4f mm" % (self.name, direction, self.position_mm))
        return True

    def stop(self) -> None:
        self.direction = 0


def load_axis_limits() -> Dict[str, Tuple[float, float]]:
    if AXES_CONFIG.exists():
        with AXES_CONFIG.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {
            name: (float(item["min_mm"]), float(item["max_mm"]))
            for name, item in raw["axes"].items()
        }
    print("Warning: %s not found; using 0-24 mm travel for all axes." % AXES_CONFIG)
    return {name: (0.0, 24.0) for name in (DEFAULT_X_AXIS, DEFAULT_Y_AXIS, DEFAULT_Z_AXIS)}


def parse_tag_ids(value: str) -> Tuple[int, ...]:
    try:
        ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError:
        raise argparse.ArgumentTypeError("tag IDs must be comma-separated integers")
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("provide at least two distinct tag IDs")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Digital-twin WASD/QE drive of the probe with a simulated endoscope view."
    )
    parser.add_argument("--x-axis", default=DEFAULT_X_AXIS, help="Axis driven by W/S (world X).")
    parser.add_argument("--y-axis", default=DEFAULT_Y_AXIS, help="Axis driven by A/D (world Y).")
    parser.add_argument("--z-axis", default=DEFAULT_Z_AXIS, help="Axis driven by Q/E (world Z).")
    parser.add_argument("--invert-x", action="store_true", help="Invert W/S direction.")
    parser.add_argument("--invert-y", action="store_true", help="Invert A/D direction.")
    parser.add_argument("--invert-z", action="store_true", help="Invert Q/E direction.")
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=0.8,
        help="Fraction of the configured max velocity to use. Default is 0.8.",
    )
    parser.add_argument(
        "--max-velocity-mm-s",
        type=float,
        default=DEFAULT_MAX_VELOCITY_MM_S,
        help="Simulated controller max velocity in mm/s. Default is %g (Z925B)."
        % DEFAULT_MAX_VELOCITY_MM_S,
    )
    parser.add_argument(
        "--limit-guard-mm",
        type=float,
        default=0.05,
        help="Stop this far before configured soft limits during held-key driving.",
    )
    parser.add_argument("--poll-s", type=float, default=0.02, help="Key polling period in seconds.")
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=5.0,
        help="Endoscope view roll step for J/L keys in degrees. Default is 5.",
    )
    parser.add_argument(
        "--needle-extension-mm",
        type=float,
        default=10.0,
        help="How far the needle protrudes below the tip face in mm; 0 hides it. Default is 10.",
    )
    parser.add_argument(
        "--reverse-cad-camera-order",
        action="store_true",
        help="Start with the CAD probe/needle rotated 180 degrees around the camera-pair axis.",
    )
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL, help="Probe STL to load.")
    parser.add_argument(
        "--pose-json",
        type=Path,
        default=None,
        help="Pose file from Motordriver/stereo_probe_pose.py: places the probe at the "
        "measured pose over the real tag-sheet layout (world frame = sheet frame).",
    )
    parser.add_argument(
        "--pose-live",
        action="store_true",
        help="Continuously track the AprilTag target from both endoscopes and update the twin.",
    )
    parser.add_argument("--left-camera", type=int, default=0, help="Camera index for --pose-live.")
    parser.add_argument("--right-camera", type=int, default=1, help="Camera index for --pose-live.")
    parser.add_argument(
        "--needle-line-samples",
        type=int,
        default=15,
        help="Stereo needle observations to average before replacing the CAD "
        "needle axis with the measured one (per-session bore-tilt calibration; "
        "0 disables). Default 15.",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=None,
        help="Stereo camera-calibration JSON used by --pose-live.",
    )
    parser.add_argument(
        "--tag-edge-mm",
        type=float,
        default=None,
        help="Known AprilTag outer-black-border edge; overrides the saved motion scale.",
    )
    parser.add_argument(
        "--live-pose-hz",
        type=float,
        default=5.0,
        help="AprilTag pose updates per second in --pose-live mode. Default is 5.",
    )
    parser.add_argument(
        "--reference-id",
        type=int,
        default=2,
        help="AprilTag ID defining the session tag-frame origin. Default is 2.",
    )
    parser.add_argument(
        "--anchor-ids",
        type=parse_tag_ids,
        default=(1, 2, 3),
        help="Rigid tag IDs used to learn the session frame and estimate camera rotation. "
        "Their coordinates are learned at startup, not stored. Default is 1,2,3.",
    )
    parser.add_argument(
        "--test-render",
        type=Path,
        default=None,
        help="Render one frame to this PNG (off-screen) and exit; no keyboard loop.",
    )
    return parser.parse_args()


def key_down(vk_code: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def key_pressed_once(vk_code: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x0001)


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


def format_positions(axis_names: Iterable[str], axes: Dict[str, SimAxis]) -> str:
    return ", ".join("%s=%.4f mm" % (name, axes[name].position_mm) for name in axis_names)


def print_help(args: argparse.Namespace) -> None:
    print("")
    print("Digital twin WASD/QE held-key drive (simulation, no hardware)")
    print("  Hold W/S: drive %s positive/negative (world X)" % args.x_axis)
    print("  Hold A/D: drive %s positive/negative (world Y)" % args.y_axis)
    print("  Hold Q/E: drive %s negative/positive (world Z, insertion)" % args.z_axis)
    print("  Speed: %g%% of %g mm/s" % (args.speed_scale * 100.0, args.max_velocity_mm_s))
    print("  Space: stop all axes")
    print("  J/L: roll both endoscope views -%g/+%g degrees" % (args.rotation_step_deg, args.rotation_step_deg))
    print("  C: toggle nominal/reversed CAD camera-to-needle mapping")
    print("  H: show this help")
    print("  X or Esc: stop and exit")
    print("  Endoscopes: left+right bores, %g deg diagonal FOV each" % MUC112_DIAGONAL_FOV_DEG)
    print("  Needle: %g mm below the tip face (syringe channel)" % args.needle_extension_mm)
    print("")


def vertical_fov_deg(diagonal_fov_deg: float, aspect_w_over_h: float) -> float:
    """Vertical FOV for a viewport, given the sensor's diagonal FOV."""
    tan_half_diag = math.tan(math.radians(diagonal_fov_deg) / 2.0)
    scale = math.sqrt(1.0 + aspect_w_over_h ** 2)
    return 2.0 * math.degrees(math.atan(tan_half_diag / scale * 1.0))


def checkerboard_texture(squares: int = 12, pixels_per_square: int = 24) -> pv.Texture:
    size = squares * pixels_per_square
    rows, cols = np.indices((size, size))
    tile = (rows // pixels_per_square + cols // pixels_per_square) % 2
    board = np.where(tile[..., None] == 0, 210, 120).astype(np.uint8)
    return pv.Texture(np.repeat(board, 3, axis=2))


def needle_mesh(extension_mm: float) -> Optional[pv.PolyData]:
    """Needle in probe-model coords: down the syringe bore, tip extension_mm past the face."""
    if extension_mm <= 0:
        return None
    x0, z0 = NEEDLE_BORE_XZ
    tip_len = min(1.5, 0.5 * extension_mm)
    inside_mm = 2.0  # shaft continues this far up into the channel
    shaft = pv.Cylinder(
        center=(x0, 0.5 * (inside_mm - (extension_mm - tip_len)), z0),
        direction=(0.0, 1.0, 0.0),
        radius=NEEDLE_RADIUS_MM,
        height=inside_mm + extension_mm - tip_len,
    )
    tip = pv.Cone(
        center=(x0, -(extension_mm - 0.5 * tip_len), z0),
        direction=(0.0, -1.0, 0.0),
        radius=NEEDLE_RADIUS_MM,
        height=tip_len,
    )
    return shaft.merge(tip)


def probe_matrix(stage_off: np.ndarray) -> np.ndarray:
    """4x4 world transform of the probe model for a stage offset (mm from mid)."""
    matrix = np.eye(4)
    matrix[:3, :3] = MODEL_TO_WORLD
    model_shift = np.array([-MODEL_CENTER_XZ[0], 0.0, -MODEL_CENTER_XZ[1]])
    base = np.array([0.0, 0.0, TIP_HEIGHT_AT_MID_TRAVEL_MM])
    matrix[:3, 3] = MODEL_TO_WORLD.dot(model_shift) + base + stage_off
    return matrix


def pose_matrix_from_json(pose: Dict[str, object]) -> np.ndarray:
    """Model->world transform placing the probe at a measured pose.

    The pose comes from Motordriver/stereo_probe_pose.py: the twin's world
    frame is then the AprilTag sheet frame (sheet in the z=0 floor plane,
    z toward the probe)."""
    axis = np.asarray(pose["probe_axis"], dtype=float)
    axis = axis / np.linalg.norm(axis)
    baseline = np.asarray(pose["baseline_dir"], dtype=float)
    baseline = baseline - axis * baseline.dot(axis)
    baseline = baseline / np.linalg.norm(baseline)
    normal = np.cross(axis, baseline)
    rotation = np.column_stack([baseline, -axis, normal])
    bore_mid_model = np.array(
        [
            0.5 * (SCOPE_BORES["left"][0] + SCOPE_BORES["right"][0]),
            0.0,
            SCOPE_BORES["left"][1],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(pose["tip_mm"], dtype=float) - rotation.dot(bore_mid_model)
    return matrix


def reversed_cad_mapping_matrix() -> np.ndarray:
    """Local 180-degree rotation about the midpoint of the two camera bores."""
    pivot = np.array(
        [
            0.5 * (SCOPE_BORES["left"][0] + SCOPE_BORES["right"][0]),
            0.0,
            SCOPE_BORES["left"][1],
        ]
    )
    rotation = np.diag([-1.0, 1.0, -1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pivot - rotation.dot(pivot)
    return matrix


class LiveStereoTracker:
    """Persistent capture and a tag layout learned from the first stereo pair."""

    def __init__(
        self,
        left_index: int,
        right_index: int,
        mm_per_unit: float,
        camera_calibration: Dict[str, object],
        reference_id: int,
        anchor_ids: Tuple[int, ...],
        reverse_cad_mapping: bool,
        needle_line_samples: int = 15,
    ) -> None:
        import cv2

        sys.path.insert(0, str(PROJECT_ROOT / "Motordriver"))
        import stereo_probe_pose
        from apriltag_tracker import create_detector, detect_tags, draw_overlay
        from needle_detector import (
            NeedleParams,
            detect_needle,
            draw_needle_overlay,
        )

        self.cv2 = cv2
        self.stereo_probe_pose = stereo_probe_pose
        self.detect_tags = detect_tags
        self.draw_overlay = draw_overlay
        self.detect_needle = detect_needle
        self.draw_needle_overlay = draw_needle_overlay
        self.detector = create_detector("36h11")
        self.needle_params = [
            NeedleParams(
                entry="auto",
                hough_threshold=20,
                hough_min_length_frac=0.06,
                hough_max_gap=20,
                expected_angle_deg=None,
                angle_tol_deg=22.0,
                expected_line_px=None,
                line_distance_max_px=40.0,
                needle_width_max_px=80.0,
                # Stereo ray-gap and CAD-axis gates reject bad detections, so
                # smoothing each 2D tip here only adds lag and shrinks motion
                # during insertion/retraction. Filter once in millimetres below.
                ema_alpha=1.0,
                require_pair=False,
                require_border=False,
            )
            for _side in ("left", "right")
        ]
        self.needle_detections = [None, None]
        self.needle_misses = [0, 0]
        self.needle_expected_lines = [None, None]
        # Per-session needle-line calibration: the physical needle tilts inside
        # the bore clearance, so the nominal CAD axis projects a few degrees
        # off. Once enough consistent stereo needle observations accumulate,
        # the measured line (in probe-model coords) replaces the CAD axis.
        self.needle_line_target = int(needle_line_samples)
        self.needle_line_buffer = []  # type: List[Tuple[np.ndarray, np.ndarray]]
        self.needle_line_model = None  # type: Optional[Tuple[np.ndarray, np.ndarray]]
        self.last_raw_frames = None  # type: Optional[Tuple[np.ndarray, np.ndarray]]
        self.mm_per_unit = mm_per_unit
        self.camera_calibration = camera_calibration
        self.reference_id = reference_id
        self.anchor_ids = anchor_ids
        self.reverse_cad_mapping = reverse_cad_mapping
        self.session_layout = None  # type: Optional[Dict[int, np.ndarray]]
        self.indices = (left_index, right_index)
        self.caps = []
        try:
            for index in self.indices:
                cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if not cap.isOpened():
                    raise RuntimeError("could not open camera %d via MSMF" % index)
                self.caps.append(cap)
            for _ in range(6):
                self.capture_raw()
            time.sleep(0.5)
            stereo_probe_pose.apply_led_min(left_index)
            stereo_probe_pose.apply_led_min(right_index)
            time.sleep(0.8)
        except Exception:
            self.close()
            raise

    def capture_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        grabbed = [cap.grab() for cap in self.caps]
        if not all(grabbed):
            raise RuntimeError("stereo frame grab failed")
        frames = []
        for cap in self.caps:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError("stereo frame retrieve failed")
            frames.append(frame)
        return frames[0], frames[1]

    def cad_display_matrix(self, pose: Dict[str, object]) -> np.ndarray:
        model_to_sheet = pose_matrix_from_json(pose)
        if self.reverse_cad_mapping:
            model_to_sheet = model_to_sheet.dot(reversed_cad_mapping_matrix())
        return model_to_sheet

    def needle_line_in_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """Exit point (on the tip face, model y=0) and unit axis of the needle
        in probe-model coords: the calibrated line when locked, CAD otherwise."""
        if self.needle_line_model is not None:
            return self.needle_line_model
        return (
            np.array([NEEDLE_BORE_XZ[0], 0.0, NEEDLE_BORE_XZ[1]]),
            np.array([0.0, -1.0, 0.0]),
        )

    def collect_needle_line_sample(self, pose: Dict[str, object]) -> None:
        """Accumulate stereo needle-line observations; lock the calibration
        once enough consistent samples exist.

        Each camera's detected 2D needle (entry + tip pixels) spans a plane
        through the camera center; the physical needle axis is the
        intersection of the two planes. The triangulated tip anchors the line.
        """
        if self.needle_line_model is not None or self.needle_line_target <= 0:
            return
        estimator = pose.get("needle_estimator")
        if not isinstance(estimator, dict) or estimator.get("status") != "ok":
            return
        if float(estimator["ray_gap_mm"]) > 1.0:
            return  # calibration wants tighter stereo agreement than live use
        try:
            normals = []
            frame = self.last_raw_frames[0]
            for index, side in enumerate(("left", "right")):
                det = self.needle_detections[index]
                _, ray_entry = self.pixel_ray_in_sheet(
                    pose, side, (det.entry_x, det.entry_y), frame
                )
                _, ray_tip = self.pixel_ray_in_sheet(
                    pose, side, (det.tip_x, det.tip_y), frame
                )
                normal = np.cross(ray_entry, ray_tip)
                norm = float(np.linalg.norm(normal))
                if norm < 1e-6:
                    return
                normals.append(normal / norm)
            direction = np.cross(normals[0], normals[1])
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                return
            direction = direction / norm
            model_to_sheet = self.cad_display_matrix(pose)
            cad_axis_sheet = model_to_sheet[:3, :3].dot(np.array([0.0, -1.0, 0.0]))
            if direction.dot(cad_axis_sheet) < 0:
                direction = -direction
            tip_sheet = np.asarray(estimator["tip_sheet_mm"], dtype=float)
            # Reject wild samples: more than ~15 deg from CAD is not bore tilt.
            if direction.dot(cad_axis_sheet / np.linalg.norm(cad_axis_sheet)) < math.cos(
                math.radians(15.0)
            ):
                return
            self.needle_line_buffer.append((tip_sheet, direction))
        except Exception:
            return
        if len(self.needle_line_buffer) < self.needle_line_target:
            return

        tips = np.mean([s[0] for s in self.needle_line_buffer], axis=0)
        mean_dir = np.mean([s[1] for s in self.needle_line_buffer], axis=0)
        mean_dir = mean_dir / np.linalg.norm(mean_dir)
        model_to_sheet = self.cad_display_matrix(pose)
        rotation = model_to_sheet[:3, :3]
        dir_model = rotation.T.dot(mean_dir)
        dir_model = dir_model / np.linalg.norm(dir_model)
        if dir_model[1] > 0:
            dir_model = -dir_model  # needle protrudes toward model -y
        tip_model = rotation.T.dot(tips - model_to_sheet[:3, 3])
        if abs(dir_model[1]) < 1e-6:
            return
        t_exit = -tip_model[1] / dir_model[1]
        exit_model = tip_model + t_exit * dir_model
        tilt_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, -dir_model[1])))
        )
        self.needle_line_model = (exit_model, dir_model)
        print(
            "needle-line calibration locked (%d samples): exit (%.2f, %.2f) model-mm "
            "(CAD bore %.1f, %.1f), tilt %.2f deg from the channel axis"
            % (
                len(self.needle_line_buffer),
                exit_model[0], exit_model[2],
                NEEDLE_BORE_XZ[0], NEEDLE_BORE_XZ[1],
                tilt_deg,
            )
        )

    def projected_needle_line(
        self,
        pose: Dict[str, object],
        side: str,
        frame: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Project the needle axis (calibrated, else CAD) into one live image."""
        model_to_sheet = self.cad_display_matrix(pose)
        exit_model, axis_model = self.needle_line_in_model()
        exit_sheet = model_to_sheet.dot(np.append(exit_model, 1.0))[:3]
        axis_sheet = model_to_sheet[:3, :3].dot(axis_model)
        axis_sheet = axis_sheet / np.linalg.norm(axis_sheet)
        points_sheet = np.asarray(
            [exit_sheet + 3.0 * axis_sheet, exit_sheet + 30.0 * axis_sheet],
            dtype=np.float64,
        )

        camera_pose = pose["camera_poses"][side]
        rotation_camera_to_sheet = np.asarray(
            camera_pose["rotation_camera_to_sheet"], dtype=np.float64
        )
        rotation_sheet_to_camera = rotation_camera_to_sheet.T
        center_sheet = np.asarray(camera_pose["center_mm"], dtype=np.float64)
        translation_sheet_to_camera = -rotation_sheet_to_camera.dot(center_sheet)
        points_camera = (
            rotation_sheet_to_camera.dot(points_sheet.T).T
            + translation_sheet_to_camera
        )
        if np.any(points_camera[:, 2] <= 0.1):
            raise ValueError("projected CAD needle axis is behind the camera")
        height, width = frame.shape[:2]
        intrinsics, distortion, model = self.stereo_probe_pose.camera_parameters(
            self.camera_calibration, side, width, height
        )
        projected = self.stereo_probe_pose.project_points_px(
            points_camera,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            intrinsics,
            distortion,
            model,
        )
        delta = projected[1] - projected[0]
        if float(np.linalg.norm(delta)) < 2.0:
            raise ValueError("projected CAD needle axis is too short in the image")
        return (
            float(projected[0, 0]),
            float(projected[0, 1]),
            float(delta[0]),
            float(delta[1]),
        )

    def update_needle_detections(
        self,
        left: np.ndarray,
        right: np.ndarray,
        pose: Dict[str, object],
    ) -> None:
        for index, (side, frame) in enumerate(
            zip(("left", "right"), (left, right))
        ):
            try:
                expected_line = self.projected_needle_line(pose, side, frame)
            except Exception:
                expected_line = None
            self.needle_expected_lines[index] = expected_line
            params = self.needle_params[index]
            params.expected_line_px = expected_line
            # While the needle-line calibration has not locked, the projected
            # CAD axis can miss the real needle by more than the corridor
            # half-width (scope re-seating shifts the probe-frame estimate).
            # Search wide until locked, then track tight.
            calibrating = (
                self.needle_line_model is None and self.needle_line_target > 0
            )
            params.corridor_halfwidth_px = 100 if calibrating else 45
            params.line_distance_max_px = 100.0 if calibrating else 40.0
            params.expected_angle_deg = (
                math.degrees(math.atan2(expected_line[3], expected_line[2]))
                if expected_line is not None
                else None
            )
            detection = self.detect_needle(
                frame, params, self.needle_detections[index]
            )
            if detection is not None:
                self.needle_detections[index] = detection
                self.needle_misses[index] = 0
            else:
                self.needle_misses[index] += 1
                if self.needle_misses[index] > 3:
                    self.needle_detections[index] = None

    def draw_expected_needle_axis(
        self, frame: np.ndarray, expected_line: object
    ) -> None:
        if expected_line is None:
            return
        x0, y0, dx, dy = expected_line
        cv2 = self.cv2
        cv2.line(
            frame,
            (int(round(x0)), int(round(y0))),
            (int(round(x0 + dx)), int(round(y0 + dy))),
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    def pixel_ray_in_sheet(
        self,
        pose: Dict[str, object],
        side: str,
        pixel: Tuple[float, float],
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        intrinsics, distortion, model = self.stereo_probe_pose.camera_parameters(
            self.camera_calibration, side, width, height
        )
        corrected = self.stereo_probe_pose.undistort_points_px(
            np.asarray([pixel], dtype=np.float64),
            intrinsics,
            distortion,
            model,
        )[0]
        ray_camera = np.linalg.solve(
            intrinsics, np.array([corrected[0], corrected[1], 1.0])
        )
        ray_camera = ray_camera / np.linalg.norm(ray_camera)
        camera_pose = pose["camera_poses"][side]
        rotation_camera_to_sheet = np.asarray(
            camera_pose["rotation_camera_to_sheet"], dtype=float
        )
        ray_sheet = rotation_camera_to_sheet.dot(ray_camera)
        ray_sheet = ray_sheet / np.linalg.norm(ray_sheet)
        center_sheet = np.asarray(camera_pose["center_mm"], dtype=float)
        return center_sheet, ray_sheet

    def add_triangulated_needle_tip(
        self,
        pose: Dict[str, object],
        left: np.ndarray,
        right: np.ndarray,
    ) -> None:
        left_detection, right_detection = self.needle_detections
        if left_detection is None or right_detection is None:
            pose["needle_estimator"] = {
                "status": "need needle detection in both views",
                "left_detected": left_detection is not None,
                "right_detected": right_detection is not None,
            }
            return
        try:
            left_center, left_ray = self.pixel_ray_in_sheet(
                pose,
                "left",
                (left_detection.tip_x, left_detection.tip_y),
                left,
            )
            right_center, right_ray = self.pixel_ray_in_sheet(
                pose,
                "right",
                (right_detection.tip_x, right_detection.tip_y),
                right,
            )
            system = np.column_stack([left_ray, -right_ray])
            depths, _residuals, _rank, _singular = np.linalg.lstsq(
                system, right_center - left_center, rcond=None
            )
            left_depth = float(depths[0])
            right_depth = float(depths[1])
            left_point = left_center + left_depth * left_ray
            right_point = right_center + right_depth * right_ray
            tip_sheet = 0.5 * (left_point + right_point)
            ray_gap = float(np.linalg.norm(left_point - right_point))
            if left_depth <= 0.0 or right_depth <= 0.0:
                raise ValueError("triangulated needle tip is behind a camera")
            pose["needle_estimator"] = {
                "status": "ok",
                "tip_sheet_mm": tip_sheet.tolist(),
                "left_tip_px": [left_detection.tip_x, left_detection.tip_y],
                "right_tip_px": [right_detection.tip_x, right_detection.tip_y],
                "ray_gap_mm": ray_gap,
                "left_depth_mm": left_depth,
                "right_depth_mm": right_depth,
                "left_support_segments": left_detection.n_segments,
                "right_support_segments": right_detection.n_segments,
            }
        except Exception as exc:
            pose["needle_estimator"] = {"status": str(exc)}

    def read(
        self, solve_pose: bool
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, object]], Optional[str]]:
        left, right = self.capture_raw()
        self.last_raw_frames = (left.copy(), right.copy())
        left_detections = self.detect_tags(left, self.detector)
        right_detections = self.detect_tags(right, self.detector)
        pose = None
        error = None
        if solve_pose:
            try:
                pose = self.stereo_probe_pose.analyse_pair(
                    left,
                    right,
                    800.0,
                    reference_id=self.reference_id,
                    mm_per_unit=self.mm_per_unit,
                    camera_calibration=self.camera_calibration,
                    anchor_ids=self.anchor_ids,
                    sheet_layout=self.session_layout,
                )
                if self.session_layout is None:
                    self.session_layout = dict(
                        (int(tag_id), np.asarray(corners, dtype=np.float64))
                        for tag_id, corners in pose["layout_tag_edge_units"].items()
                    )
                    pose["layout_source"] = "learned-at-live-start"
                self.update_needle_detections(left, right, pose)
                self.add_triangulated_needle_tip(pose, left, right)
                self.collect_needle_line_sample(pose)
                exit_model, axis_model = self.needle_line_in_model()
                pose["needle_line_model"] = {
                    "exit": exit_model.tolist(),
                    "axis": axis_model.tolist(),
                    "calibrated": self.needle_line_model is not None,
                }
            except Exception as exc:
                error = str(exc)
        left_overlay = left.copy()
        right_overlay = right.copy()
        self.draw_overlay(left_overlay, left_detections)
        self.draw_overlay(right_overlay, right_detections)
        self.draw_expected_needle_axis(
            left_overlay, self.needle_expected_lines[0]
        )
        self.draw_expected_needle_axis(
            right_overlay, self.needle_expected_lines[1]
        )
        self.draw_needle_overlay(left_overlay, self.needle_detections[0])
        self.draw_needle_overlay(right_overlay, self.needle_detections[1])
        return left_overlay, right_overlay, pose, error

    def close(self) -> None:
        for cap in self.caps:
            try:
                cap.release()
            except Exception:
                pass
        self.caps = []


def camera_frame_texture(frame_bgr: np.ndarray) -> pv.Texture:
    """Convert an OpenCV BGR frame to a correctly oriented PyVista texture."""
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    return pv.Texture(rgb)


def tag_sheet_texture(pose: Dict[str, object], size_mm: float = 60.0, px_per_mm: int = 10) -> pv.Texture:
    """Floor texture showing the real 36h11 tags at their measured layout."""
    import cv2

    size_px = int(size_mm * px_per_mm)
    canvas = np.full((size_px, size_px, 3), 235, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    edge_mm = float(pose["tag_edge_mm"])
    marker_px = 160
    src = np.array(
        [[0, 0], [marker_px - 1, 0], [marker_px - 1, marker_px - 1], [0, marker_px - 1]],
        dtype=np.float32,
    )
    for tag_id, corners in pose["layout_tag_edge_units"].items():
        corners_mm = np.asarray(corners, dtype=float) * edge_mm
        dst = np.stack(
            [
                (corners_mm[:, 0] + size_mm / 2.0) * px_per_mm,
                (size_mm / 2.0 - corners_mm[:, 1]) * px_per_mm,
            ],
            axis=1,
        ).astype(np.float32)
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), marker_px)
        else:
            marker = cv2.aruco.drawMarker(dictionary, int(tag_id), marker_px)
        transform = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            marker, transform, (size_px, size_px), flags=cv2.INTER_NEAREST, borderValue=255
        )
        mask = cv2.warpPerspective(
            np.full((marker_px, marker_px), 255, dtype=np.uint8), transform, (size_px, size_px)
        )
        canvas[mask > 127] = np.repeat(warped[mask > 127][:, None], 3, axis=1)
    # Verified empirically: array row 0 maps to the plane's +y edge and column 0
    # to the -x edge, matching the canvas construction above directly.
    return pv.Texture(canvas)


def stage_offset(axes: Dict[str, SimAxis], axis_names: Iterable[str]) -> np.ndarray:
    return np.array(
        [
            axes[name].position_mm - 0.5 * (axes[name].min_mm + axes[name].max_mm)
            for name in axis_names
        ]
    )


def build_scene(
    args: argparse.Namespace,
    off_screen: bool,
    pose: Optional[Dict[str, object]] = None,
    live_frames: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[
    pv.Plotter,
    object,
    Dict[int, object],
    Dict[int, object],
    Dict[int, object],
]:
    if not args.stl.exists():
        raise SystemExit("Probe STL not found: %s" % args.stl)
    probe = pv.read(str(args.stl))
    initial_needle_extension = args.needle_extension_mm
    if args.pose_live:
        initial_needle_extension = max(0.01, initial_needle_extension)
    needle = needle_mesh(initial_needle_extension)

    plotter = pv.Plotter(
        shape="3/2",
        splitting_position=0.4,
        window_size=(1800, 1000),
        off_screen=off_screen,
    )
    plotter.set_background("white")

    floor = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=60, j_size=60)
    floor.texture_map_to_plane(inplace=True)
    if pose is not None:
        texture = tag_sheet_texture(pose)
        target = None
        markers = []
    elif args.pose_live:
        texture = checkerboard_texture()
        target = None
        markers = []
    else:
        texture = checkerboard_texture()
        target = pv.Cylinder(center=(0, 0, 0.05), direction=(0, 0, 1), radius=0.8, height=0.1)
        markers = [
            (pv.Sphere(radius=1.0, center=(8, 6, 1.0)), "seagreen"),
            (pv.Sphere(radius=1.5, center=(-9, -5, 1.5)), "steelblue"),
            (pv.Cube(center=(6, -8, 1.0), x_length=2, y_length=2, z_length=2), "goldenrod"),
        ]

    floor_actors = {}
    for renderer_index in TWIN_RENDERERS:
        plotter.subplot(renderer_index)
        floor_actors[renderer_index] = plotter.add_mesh(
            floor, texture=texture, name="floor%d" % renderer_index
        )
        if target is not None:
            plotter.add_mesh(target, color="red", name="target%d" % renderer_index)
        for idx, (mesh, color) in enumerate(markers):
            plotter.add_mesh(
                mesh, color=color, name="marker%d_%d" % (renderer_index, idx)
            )
        plotter.add_mesh(
            probe, color="lightsteelblue", name="probe%d" % renderer_index
        )
        if needle is not None:
            plotter.add_mesh(
                needle, color="dimgray", name="needle%d" % renderer_index
            )

    plotter.subplot(EXTERNAL_RENDERER)
    plotter.add_text("external view", font_size=10, name="label_ext")
    readout = plotter.add_text("", position="lower_left", font_size=9, name="readout")
    plotter.add_axes()
    plotter.camera.position = (70, -70, 60)
    plotter.camera.focal_point = (0, 0, 15)
    plotter.camera.up = (0, 0, 1)

    for renderer_index, side in (
        (SCOPE_LEFT_RENDERER, "left"),
        (SCOPE_RIGHT_RENDERER, "right"),
    ):
        plotter.subplot(renderer_index)
        plotter.add_text(
            "endoscope %s bore (muC112 %g deg diag FOV)" % (side, MUC112_DIAGONAL_FOV_DEG),
            font_size=10,
            name="label_scope_%s" % side,
        )
        plotter.camera.view_angle = vertical_fov_deg(MUC112_DIAGONAL_FOV_DEG, 1.0)

    if live_frames is None:
        blank = np.zeros((720, 720, 3), dtype=np.uint8)
        live_frames = (blank, blank)
    live_actors = {}
    rotation_readouts = {}
    for renderer_index, side, frame in (
        (LIVE_LEFT_RENDERER, "left", live_frames[0]),
        (LIVE_RIGHT_RENDERER, "right", live_frames[1]),
    ):
        plotter.subplot(renderer_index)
        plotter.set_background("black", all_renderers=False)
        image_plane = pv.Plane(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=2.0,
            j_size=2.0,
        )
        image_plane.texture_map_to_plane(inplace=True)
        live_actors[renderer_index] = plotter.add_mesh(
            image_plane,
            texture=camera_frame_texture(frame),
            lighting=False,
            name="live_%s" % side,
        )
        plotter.add_text(
            "live %s endoscope + AprilTags" % side,
            color="white",
            font_size=10,
            name="label_live_%s" % side,
        )
        rotation_readouts[renderer_index] = plotter.add_text(
            "",
            position="lower_left",
            color="white",
            font_size=9,
            shadow=True,
            name="rotation_live_%s" % side,
        )
        plotter.camera.position = (0.0, 0.0, 2.0)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 1.05

    return plotter, readout, floor_actors, live_actors, rotation_readouts


def current_probe_world_matrix(
    axes: Dict[str, SimAxis],
    axis_names: Iterable[str],
    pose_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    off = stage_offset(axes, axis_names)
    if pose_matrix is None:
        return probe_matrix(off)
    matrix = pose_matrix.copy()
    matrix[:3, 3] += off
    return matrix


def current_cad_display_matrix(
    axes: Dict[str, SimAxis],
    axis_names: Iterable[str],
    pose_matrix: Optional[np.ndarray],
    reverse_cad_mapping: bool,
) -> np.ndarray:
    matrix = current_probe_world_matrix(axes, axis_names, pose_matrix)
    if reverse_cad_mapping:
        return matrix.dot(reversed_cad_mapping_matrix())
    return matrix


def update_views(
    plotter: pv.Plotter,
    axes: Dict[str, SimAxis],
    axis_names: Iterable[str],
    roll_deg: float,
    pose_matrix: Optional[np.ndarray] = None,
    scope_rolls: Tuple[float, float] = (0.0, 0.0),
    reverse_cad_mapping: bool = False,
) -> None:
    matrix = current_probe_world_matrix(axes, axis_names, pose_matrix)
    cad_display_matrix = current_cad_display_matrix(
        axes, axis_names, pose_matrix, reverse_cad_mapping
    )
    for col, renderer in enumerate(plotter.renderers):
        for base in ("probe", "needle"):
            actor = renderer.actors.get("%s%d" % (base, col))
            if actor is not None:
                actor.user_matrix = cad_display_matrix

    view_dir = -matrix[:3, 1]  # model +y runs from the tip face into the probe body
    baseline = matrix[:3, 0]
    normal = matrix[:3, 2]
    for (renderer_index, side), extra_roll in zip(
        ((SCOPE_LEFT_RENDERER, "left"), (SCOPE_RIGHT_RENDERER, "right")),
        scope_rolls,
    ):
        bore = SCOPE_BORES[side]
        cam_pos = matrix.dot(np.array([bore[0], 0.0, bore[1], 1.0]))[:3]
        roll = math.radians(roll_deg + extra_roll)
        scope_cam = plotter.renderers[renderer_index].camera
        scope_cam.position = tuple(cam_pos)
        scope_cam.focal_point = tuple(cam_pos + 10.0 * view_dir)
        scope_cam.up = tuple(math.sin(roll) * baseline - math.cos(roll) * normal)
        scope_cam.clipping_range = (0.2, 400.0)


def update_live_panels(
    live_actors: Dict[int, object], left_frame: np.ndarray, right_frame: np.ndarray
) -> None:
    live_actors[LIVE_LEFT_RENDERER].SetTexture(camera_frame_texture(left_frame))
    live_actors[LIVE_RIGHT_RENDERER].SetTexture(camera_frame_texture(right_frame))


def update_needle_geometry(plotter: pv.Plotter, extension_mm: float) -> None:
    mesh = needle_mesh(max(0.01, extension_mm))
    if mesh is None:
        return
    for renderer_index in TWIN_RENDERERS:
        actor = plotter.renderers[renderer_index].actors.get(
            "needle%d" % renderer_index
        )
        if actor is not None:
            mapper = actor.GetMapper()
            mapper.SetInputData(mesh)
            mapper.Update()


def needle_extension_measurement(
    pose: Dict[str, object],
    sheet_to_world: np.ndarray,
    cad_display_matrix: np.ndarray,
) -> Dict[str, object]:
    estimator = pose.get("needle_estimator")
    if not isinstance(estimator, dict) or estimator.get("status") != "ok":
        return {
            "status": estimator.get("status", "needle estimator waiting")
            if isinstance(estimator, dict)
            else "needle estimator waiting"
        }
    tip_sheet = np.asarray(estimator["tip_sheet_mm"], dtype=float)
    tip_world = sheet_to_world.dot(np.append(tip_sheet, 1.0))[:3]
    line = pose.get("needle_line_model")
    if isinstance(line, dict):
        exit_model = np.append(np.asarray(line["exit"], dtype=float), 1.0)
        axis_model = np.asarray(line["axis"], dtype=float)
    else:
        exit_model = np.array(
            [NEEDLE_BORE_XZ[0], 0.0, NEEDLE_BORE_XZ[1], 1.0], dtype=float
        )
        axis_model = np.array([0.0, -1.0, 0.0])
    exit_world = cad_display_matrix.dot(exit_model)[:3]
    needle_axis_world = cad_display_matrix[:3, :3].dot(axis_model)
    needle_axis_world = needle_axis_world / np.linalg.norm(needle_axis_world)
    tip_delta = tip_world - exit_world
    extension_mm = float(tip_delta.dot(needle_axis_world))
    lateral_error_mm = float(
        np.linalg.norm(tip_delta - extension_mm * needle_axis_world)
    )
    return {
        "status": "ok",
        "extension_mm": extension_mm,
        "lateral_error_mm": lateral_error_mm,
        "ray_gap_mm": float(estimator["ray_gap_mm"]),
        "tip_world_mm": tip_world.tolist(),
    }


def apply_live_needle_measurement(
    plotter: pv.Plotter,
    pose: Dict[str, object],
    sheet_to_world: np.ndarray,
    cad_display_matrix: np.ndarray,
    current_extension_mm: float,
    initialized: bool,
) -> Tuple[float, bool, str]:
    measurement = needle_extension_measurement(
        pose, sheet_to_world, cad_display_matrix
    )
    if measurement.get("status") != "ok":
        return (
            current_extension_mm,
            initialized,
            "needle estimator: %s" % measurement.get("status", "waiting"),
        )
    raw_extension = float(measurement["extension_mm"])
    lateral_error = float(measurement["lateral_error_mm"])
    ray_gap = float(measurement["ray_gap_mm"])
    rejection = None
    if not 0.0 <= raw_extension <= 40.0:
        rejection = "extension %.1f mm outside 0-40 mm" % raw_extension
    elif ray_gap > 2.5:
        rejection = "stereo ray gap %.2f mm exceeds 2.50 mm" % ray_gap
    elif lateral_error > 6.0:
        rejection = "tip is %.2f mm from CAD needle axis" % lateral_error
    if rejection is not None:
        return current_extension_mm, initialized, "needle estimate rejected: %s" % rejection

    if initialized:
        # Keep a small amount of damping for Hough endpoint jitter while
        # following real needle travel closely at the 5 Hz pose-update rate.
        extension_mm = 0.75 * raw_extension + 0.25 * current_extension_mm
    else:
        extension_mm = raw_extension
    update_needle_geometry(plotter, extension_mm)
    estimator = pose.get("needle_estimator")
    if isinstance(estimator, dict):
        estimator["raw_extension_mm"] = raw_extension
        estimator["filtered_extension_mm"] = extension_mm
        estimator["lateral_error_mm"] = lateral_error
    status = (
        "needle extension: %.2f mm live (raw %.2f, ray gap %.2f, axis offset %.2f mm)"
        % (extension_mm, raw_extension, ray_gap, lateral_error)
    )
    return extension_mm, True, status


def camera_rotation_matrix(
    pose: Dict[str, object], side: str
) -> Optional[np.ndarray]:
    camera_poses = pose.get("camera_poses")
    if not isinstance(camera_poses, dict) or side not in camera_poses:
        return None
    rotation = np.asarray(
        camera_poses[side]["rotation_camera_to_sheet"], dtype=float
    )
    if rotation.shape != (3, 3):
        return None
    return rotation


def rotation_change_deg(current: np.ndarray, reference: np.ndarray) -> float:
    delta = reference.T.dot(current)
    cosine = 0.5 * (float(np.trace(delta)) - 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def live_rotation_metrics(
    pose: Dict[str, object], initial_rotations: Dict[str, np.ndarray]
) -> Optional[Dict[str, float]]:
    left = camera_rotation_matrix(pose, "left")
    right = camera_rotation_matrix(pose, "right")
    if left is None or right is None or not initial_rotations:
        return None
    initial_left = initial_rotations["left"]
    initial_right = initial_rotations["right"]
    relative_initial = initial_left.T.dot(initial_right)
    relative_current = left.T.dot(right)
    return {
        "left_delta_deg": rotation_change_deg(left, initial_left),
        "right_delta_deg": rotation_change_deg(right, initial_right),
        "relative_delta_deg": rotation_change_deg(relative_current, relative_initial),
    }


def update_rotation_readouts(
    readouts: Dict[int, object],
    pose: Dict[str, object],
    initial_rotations: Dict[str, np.ndarray],
    anchor_ids: Tuple[int, ...],
) -> None:
    metrics = live_rotation_metrics(pose, initial_rotations)
    if metrics is None:
        return
    anchor_text = ",".join(str(tag_id) for tag_id in anchor_ids)
    errors = pose["reprojection_error_px"]
    for renderer_index, side in (
        (LIVE_LEFT_RENDERER, "left"),
        (LIVE_RIGHT_RENDERER, "right"),
    ):
        delta = metrics["%s_delta_deg" % side]
        roll = float(pose["roll_%s_deg" % side])
        readouts[renderer_index].SetText(
            0,
            "IDs %s: layout learned at start\n"
            "orientation change: %.2f deg\n"
            "roll: %+.2f deg | reproj: %.2f px"
            % (anchor_text, delta, roll, float(errors[side])),
        )


def tracking_readout_text(
    axis_names: Iterable[str],
    axes: Dict[str, SimAxis],
    tracking_status: str,
    pose: Optional[Dict[str, object]],
    initial_rotations: Dict[str, np.ndarray],
    reverse_cad_mapping: bool,
    needle_status: str,
) -> str:
    lines = [
        "stage: %s" % format_positions(axis_names, axes),
        tracking_status[:120],
        "CAD camera-to-needle mapping: %s (press C to toggle)"
        % ("REVERSED 180 deg" if reverse_cad_mapping else "nominal"),
        needle_status[:140],
    ]
    if pose is not None:
        metrics = live_rotation_metrics(pose, initial_rotations)
        if metrics is not None:
            lines.append(
                "rotation change: left %.2f deg, right %.2f deg, relative %.2f deg"
                % (
                    metrics["left_delta_deg"],
                    metrics["right_delta_deg"],
                    metrics["relative_delta_deg"],
                )
            )
        if "camera_lateral_D_mm" in pose:
            lines.append(
                "PnP check: D %.3f mm, axial offset %+.3f mm"
                % (
                    float(pose["camera_lateral_D_mm"]),
                    float(pose["camera_axial_offset_mm"]),
                )
            )
    return "\n".join(lines)


def update_scope_views_from_live_pose(
    plotter: pv.Plotter,
    pose: Dict[str, object],
    sheet_to_world: np.ndarray,
) -> None:
    """Use each tag-derived camera rotation for its virtual endoscope view."""
    camera_poses = pose.get("camera_poses")
    if not isinstance(camera_poses, dict):
        return
    for renderer_index, side in (
        (SCOPE_LEFT_RENDERER, "left"),
        (SCOPE_RIGHT_RENDERER, "right"),
    ):
        rotation_camera_to_sheet = camera_rotation_matrix(pose, side)
        if rotation_camera_to_sheet is None:
            continue
        center_sheet = np.asarray(camera_poses[side]["center_mm"], dtype=float)
        center_world = sheet_to_world.dot(np.append(center_sheet, 1.0))[:3]
        rotation_camera_to_world = sheet_to_world[:3, :3].dot(
            rotation_camera_to_sheet
        )
        view_world = rotation_camera_to_world[:, 2]
        up_world = -rotation_camera_to_world[:, 1]
        scope_camera = plotter.renderers[renderer_index].camera
        scope_camera.position = tuple(center_world)
        scope_camera.focal_point = tuple(center_world + 10.0 * view_world)
        scope_camera.up = tuple(up_world)
        scope_camera.clipping_range = (0.2, 400.0)


def update_tag_sheet(
    floor_actors: Dict[int, object],
    initial_pose_matrix: np.ndarray,
    current_pose: Dict[str, object],
) -> np.ndarray:
    """Move the current tag sheet in the initial probe/sheet world frame.

    Live mode deliberately treats the probe as the fixed reference.  The pose
    solver reports probe->current-sheet; composing its inverse with the initial
    probe->sheet transform therefore makes real target motion visible in the
    twin.  Updating the texture also handles tags moved relative to each other.
    """
    current_pose_matrix = pose_matrix_from_json(current_pose)
    sheet_to_world = np.dot(initial_pose_matrix, np.linalg.inv(current_pose_matrix))
    texture = tag_sheet_texture(current_pose)
    for actor in floor_actors.values():
        actor.user_matrix = sheet_to_world
        actor.SetTexture(texture)
    return sheet_to_world


def main() -> None:
    args = parse_args()
    if args.reference_id not in args.anchor_ids:
        raise SystemExit("--reference-id must be included in --anchor-ids")
    if not 0.0 < args.speed_scale <= 1.0:
        raise SystemExit("--speed-scale must be greater than 0 and no more than 1.")
    if args.limit_guard_mm < 0:
        raise SystemExit("--limit-guard-mm must be non-negative.")
    if args.poll_s <= 0:
        raise SystemExit("--poll-s must be positive.")
    if args.rotation_step_deg <= 0:
        raise SystemExit("--rotation-step-deg must be positive.")
    if args.max_velocity_mm_s <= 0:
        raise SystemExit("--max-velocity-mm-s must be positive.")
    if args.needle_extension_mm < 0:
        raise SystemExit("--needle-extension-mm must be non-negative.")
    if args.live_pose_hz <= 0:
        raise SystemExit("--live-pose-hz must be positive.")

    limits = load_axis_limits()
    for role, name in (("x", args.x_axis), ("y", args.y_axis), ("z", args.z_axis)):
        if name not in limits:
            known = ", ".join(sorted(limits)) or "(none)"
            raise SystemExit("Unknown %s axis '%s'. Known axes: %s" % (role, name, known))

    axis_names = [args.x_axis, args.y_axis, args.z_axis]
    if len(set(axis_names)) != len(axis_names):
        raise SystemExit("The x, y, and z controls must map to three different configured axes.")

    velocity = args.max_velocity_mm_s * args.speed_scale
    axes = {
        name: SimAxis(name, limits[name][0], limits[name][1], velocity)
        for name in axis_names
    }
    for name in axis_names:
        print(
            "%s: velocity set to %.4f mm/s (%g%% of %.4f mm/s)"
            % (name, velocity, args.speed_scale * 100.0, args.max_velocity_mm_s)
        )

    pose = None
    pose_matrix = None
    scope_rolls = (0.0, 0.0)
    live_tracker = None
    live_frames = None
    tracking_status = "live tracking inactive"
    if args.pose_live and args.pose_json is not None:
        raise SystemExit("use either --pose-live or --pose-json, not both")
    if args.pose_live:
        sys.path.insert(0, str(PROJECT_ROOT / "Motordriver"))
        import stereo_probe_pose

        scale_calibration = stereo_probe_pose.load_calibration()
        if args.tag_edge_mm is not None:
            mm_per_unit = args.tag_edge_mm
        else:
            mm_per_unit = float(scale_calibration["tag_edge_mm"]) if scale_calibration else None
        camera_calibration = stereo_probe_pose.load_camera_calibration(args.camera_calibration)
        if args.camera_calibration is not None and camera_calibration is None:
            raise SystemExit("camera calibration not found: %s" % args.camera_calibration)
        if camera_calibration is None:
            raise SystemExit("--pose-live needs an accepted stereo camera calibration")
        if mm_per_unit is None:
            raise SystemExit("--pose-live needs --tag-edge-mm or a valid saved tag scale")
        print("pose-live: opening persistent stereo streams...")
        live_tracker = LiveStereoTracker(
            args.left_camera,
            args.right_camera,
            float(mm_per_unit),
            camera_calibration,
            args.reference_id,
            args.anchor_ids,
            bool(args.reverse_cad_camera_order),
            needle_line_samples=args.needle_line_samples,
        )
        left_live, right_live, pose, pose_error = live_tracker.read(True)
        live_frames = (left_live, right_live)
        if pose is None:
            left_debug = Path(__file__).with_name("last_live_left.png")
            right_debug = Path(__file__).with_name("last_live_right.png")
            live_tracker.cv2.imwrite(str(left_debug), left_live)
            live_tracker.cv2.imwrite(str(right_debug), right_live)
            tracking_status = (
                "waiting for IDs %s: %s"
                % (
                    ",".join(str(tag_id) for tag_id in args.anchor_ids),
                    pose_error,
                )
            )
            print(
                "initial live AprilTag pose unavailable: %s (saved %s and %s)"
                % (pose_error, left_debug.name, right_debug.name)
            )
        else:
            tracking_status = "live camera rotation from IDs %s at %.1f Hz" % (
                ",".join(str(tag_id) for tag_id in args.anchor_ids),
                args.live_pose_hz,
            )
            stereo_probe_pose.print_report(pose)
    elif args.pose_json is not None:
        with args.pose_json.open("r", encoding="utf-8") as handle:
            pose = json.load(handle)
    if pose is not None:
        pose_matrix = pose_matrix_from_json(pose)
        scope_rolls = (float(pose["roll_left_deg"]), float(pose["roll_right_deg"]))
        print(
            "pose mode: tip at (%.1f, %.1f, %.1f) mm over the tag sheet, scope rolls %+.1f/%+.1f deg"
            % (pose["tip_mm"][0], pose["tip_mm"][1], pose["tip_mm"][2], scope_rolls[0], scope_rolls[1])
        )

    initial_rotations = {}  # type: Dict[str, np.ndarray]
    if pose is not None:
        for side in ("left", "right"):
            rotation = camera_rotation_matrix(pose, side)
            if rotation is not None:
                initial_rotations[side] = rotation.copy()

    plotter, readout, floor_actors, live_actors, rotation_readouts = build_scene(
        args,
        off_screen=args.test_render is not None,
        pose=pose,
        live_frames=live_frames,
    )
    if pose is not None:
        tip = np.asarray(pose["tip_mm"], dtype=float)
        plotter.subplot(EXTERNAL_RENDERER)
        plotter.camera.focal_point = tuple(0.5 * tip)
        plotter.camera.position = tuple(0.5 * tip + np.array([70.0, -70.0, 45.0]))
    roll_deg = 0.0
    reverse_cad_mapping = bool(args.reverse_cad_camera_order)
    sheet_to_world = np.eye(4)
    live_needle_extension = float(args.needle_extension_mm)
    needle_estimator_initialized = False
    needle_status = "needle estimator waiting for both camera tips"
    update_views(
        plotter,
        axes,
        axis_names,
        roll_deg,
        pose_matrix,
        scope_rolls,
        reverse_cad_mapping,
    )
    if pose is not None:
        update_scope_views_from_live_pose(plotter, pose, sheet_to_world)
        update_rotation_readouts(
            rotation_readouts, pose, initial_rotations, args.anchor_ids
        )
        cad_display_matrix = current_cad_display_matrix(
            axes, axis_names, pose_matrix, reverse_cad_mapping
        )
        (
            live_needle_extension,
            needle_estimator_initialized,
            needle_status,
        ) = apply_live_needle_measurement(
            plotter,
            pose,
            sheet_to_world,
            cad_display_matrix,
            live_needle_extension,
            needle_estimator_initialized,
        )
        print(needle_status)
    readout.SetText(
        0,
        tracking_readout_text(
            axis_names,
            axes,
            tracking_status,
            pose,
            initial_rotations,
            reverse_cad_mapping,
            needle_status,
        ),
    )

    if args.test_render is not None:
        plotter.screenshot(str(args.test_render))
        plotter.close()
        if live_tracker is not None:
            if live_tracker.last_raw_frames is not None:
                live_tracker.cv2.imwrite(
                    str(Path(__file__).with_name("last_live_raw_left.png")),
                    live_tracker.last_raw_frames[0],
                )
                live_tracker.cv2.imwrite(
                    str(Path(__file__).with_name("last_live_raw_right.png")),
                    live_tracker.last_raw_frames[1],
                )
            live_tracker.cv2.imwrite(
                str(Path(__file__).with_name("last_live_overlay_left.png")),
                left_live,
            )
            live_tracker.cv2.imwrite(
                str(Path(__file__).with_name("last_live_overlay_right.png")),
                right_live,
            )
            live_tracker.close()
        print("Wrote test render to %s" % args.test_render)
        return

    print_help(args)
    print("Current position: %s" % format_positions(axis_names, axes))
    plotter.show(title="Probe digital twin", interactive_update=True, auto_close=False)

    last_time = time.monotonic()
    next_pose_update = last_time + 1.0 / args.live_pose_hz
    last_tracking_error = None
    try:
        while True:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            if live_tracker is not None:
                solve_live_pose = now >= next_pose_update
                if solve_live_pose:
                    next_pose_update = now + 1.0 / args.live_pose_hz
                try:
                    left_live, right_live, live_pose, live_error = live_tracker.read(
                        solve_live_pose
                    )
                    update_live_panels(live_actors, left_live, right_live)
                    if solve_live_pose:
                        if live_pose is not None:
                            first_live_lock = pose_matrix is None
                            if first_live_lock:
                                pose_matrix = pose_matrix_from_json(live_pose)
                                scope_rolls = (
                                    float(live_pose["roll_left_deg"]),
                                    float(live_pose["roll_right_deg"]),
                                )
                                initial_rotations.clear()
                                for side in ("left", "right"):
                                    rotation = camera_rotation_matrix(live_pose, side)
                                    if rotation is not None:
                                        initial_rotations[side] = rotation.copy()
                            pose = live_pose
                            sheet_to_world = update_tag_sheet(
                                floor_actors, pose_matrix, live_pose
                            )
                            update_rotation_readouts(
                                rotation_readouts,
                                live_pose,
                                initial_rotations,
                                args.anchor_ids,
                            )
                            cad_display_matrix = current_cad_display_matrix(
                                axes,
                                axis_names,
                                pose_matrix,
                                reverse_cad_mapping,
                            )
                            (
                                live_needle_extension,
                                needle_estimator_initialized,
                                needle_status,
                            ) = apply_live_needle_measurement(
                                plotter,
                                live_pose,
                                sheet_to_world,
                                cad_display_matrix,
                                live_needle_extension,
                                needle_estimator_initialized,
                            )
                            tracking_status = (
                                "live camera rotation from IDs %s at %.1f Hz"
                                % (
                                    ",".join(
                                        str(tag_id) for tag_id in args.anchor_ids
                                    ),
                                    args.live_pose_hz,
                                )
                            )
                            if last_tracking_error is not None:
                                print("Live AprilTag tracking restored.")
                            last_tracking_error = None
                        else:
                            tracking_status = "tracking lost: %s" % live_error
                            if live_error != last_tracking_error:
                                print("Live AprilTag pose unavailable: %s" % live_error)
                            last_tracking_error = live_error
                except Exception as exc:
                    live_error = str(exc)
                    tracking_status = "camera error: %s" % live_error
                    if live_error != last_tracking_error:
                        print("Live stereo capture error: %s" % live_error)
                    last_tracking_error = live_error

            if key_down(VK_ESCAPE) or key_down(VK_CODES["x"]):
                print("Stopping and exiting.")
                return
            if key_pressed_once(VK_CODES["h"]):
                print_help(args)
            if key_pressed_once(VK_CODES["c"]):
                reverse_cad_mapping = not reverse_cad_mapping
                print(
                    "CAD camera-to-needle mapping: %s"
                    % ("REVERSED 180 deg" if reverse_cad_mapping else "nominal")
                )
            if key_pressed_once(VK_CODES["j"]):
                roll_deg = (roll_deg - args.rotation_step_deg) % 360.0
                print("Endoscope roll: %.1f deg" % roll_deg)
            if key_pressed_once(VK_CODES["l"]):
                roll_deg = (roll_deg + args.rotation_step_deg) % 360.0
                print("Endoscope roll: %.1f deg" % roll_deg)

            if key_down(VK_SPACE):
                for axis in axes.values():
                    axis.stop()
            else:
                requested = requested_directions(args)
                for name in sorted(requested):
                    direction = requested[name]
                    axis = axes[name]
                    if direction == axis.direction:
                        continue
                    axis.stop()
                    if direction != 0:
                        axis.start(direction, args.limit_guard_mm)

            for axis in axes.values():
                axis.integrate(dt, args.limit_guard_mm)

            update_views(
                plotter,
                axes,
                axis_names,
                roll_deg,
                pose_matrix,
                scope_rolls,
                reverse_cad_mapping,
            )
            if pose is not None:
                update_scope_views_from_live_pose(plotter, pose, sheet_to_world)
            readout.SetText(
                0,
                tracking_readout_text(
                    axis_names,
                    axes,
                    tracking_status,
                    pose,
                    initial_rotations,
                    reverse_cad_mapping,
                    needle_status,
                ),
            )

            if getattr(plotter, "_closed", False) or plotter.render_window is None:
                print("Window closed; exiting.")
                return
            plotter.update()
            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("")
        print("Interrupted; exiting.")
    finally:
        if live_tracker is not None:
            live_tracker.close()
            if pose is not None:
                try:
                    with (Path(__file__).with_name("last_pose.json")).open(
                        "w", encoding="utf-8"
                    ) as handle:
                        json.dump(pose, handle, indent=2)
                except Exception as exc:
                    print("Could not save last live pose: %s" % exc)
        try:
            plotter.close()
        except Exception:
            pass


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("twin_wasd_jog.py uses Windows keyboard state polling.")
    main()

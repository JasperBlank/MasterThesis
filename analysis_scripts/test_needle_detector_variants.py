"""Exercise the shared needle detector on the last saved stereo frames.

This is an offline tuning helper: it never opens a camera or touches hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Motordriver"))

from needle_detector import NeedleParams, detect_needle, draw_needle_overlay


def variants(side: str):
    expected_angle = -124.0 if side == "left" else -73.0
    expected_line = (
        (430.0, 392.0, 220.0, 328.0)
        if side == "left"
        else (310.0, 460.0, -80.0, 260.0)
    )
    common = dict(
        entry="bottom",
        hough_min_length_frac=0.06,
        hough_threshold=20,
        hough_max_gap=20,
        needle_width_max_px=80.0,
        ema_alpha=1.0,
    )
    return [
        ("pair-border", NeedleParams(require_pair=True, require_border=True, **common)),
        ("pair-free", NeedleParams(require_pair=True, require_border=False, **common)),
        (
            "single-border",
            NeedleParams(require_pair=False, require_border=True, **common),
        ),
        (
            "single-free",
            NeedleParams(require_pair=False, require_border=False, **common),
        ),
        (
            "pair-angle",
            NeedleParams(
                require_pair=True,
                require_border=False,
                expected_angle_deg=expected_angle,
                angle_tol_deg=15.0,
                **common
            ),
        ),
        (
            "single-angle",
            NeedleParams(
                require_pair=False,
                require_border=False,
                expected_angle_deg=expected_angle,
                angle_tol_deg=15.0,
                **common
            ),
        ),
        (
            "pair-line",
            NeedleParams(
                require_pair=True,
                require_border=False,
                expected_angle_deg=expected_angle,
                angle_tol_deg=18.0,
                expected_line_px=expected_line,
                line_distance_max_px=20.0,
                **common
            ),
        ),
        (
            "single-line",
            NeedleParams(
                require_pair=False,
                require_border=False,
                expected_angle_deg=expected_angle,
                angle_tol_deg=18.0,
                expected_line_px=expected_line,
                line_distance_max_px=20.0,
                **common
            ),
        ),
        (
            "single-line40",
            NeedleParams(
                require_pair=False,
                require_border=False,
                expected_angle_deg=expected_angle,
                angle_tol_deg=22.0,
                expected_line_px=expected_line,
                line_distance_max_px=40.0,
                **common
            ),
        ),
    ]


def main() -> None:
    output_dir = PROJECT_ROOT / "digital_twin" / "needle_detector_variants"
    output_dir.mkdir(exist_ok=True)
    for side in ("left", "right"):
        path = PROJECT_ROOT / "digital_twin" / ("last_live_raw_%s.png" % side)
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit("could not read %s" % path)
        for name, params in variants(side):
            detection = detect_needle(frame, params)
            shown = frame.copy()
            draw_needle_overlay(shown, detection)
            cv2.putText(
                shown,
                name,
                (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            out = output_dir / ("%s_%s.png" % (side, name))
            cv2.imwrite(str(out), shown)
            if detection is None:
                print("%s %-13s none" % (side, name))
            else:
                print(
                    "%s %-13s tip=(%.1f, %.1f) entry=(%.1f, %.1f) "
                    "angle=%+.1f len=%.1f segments=%d"
                    % (
                        side,
                        name,
                        detection.tip_x,
                        detection.tip_y,
                        detection.entry_x,
                        detection.entry_y,
                        detection.angle_deg,
                        detection.length_px,
                        detection.n_segments,
                    )
                )


if __name__ == "__main__":
    main()

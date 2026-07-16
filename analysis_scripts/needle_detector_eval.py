"""Evaluate the needle detector against hand-labeled ground truth.

Dataset: needle_calib/{set}_{led}_x{dx}_y{dy}_{side}.png  (2 sets x 3 LEDs x 4 positions x 2 sides)
Hand labels (tip pixel, from gridded zooms of the max_x0_y0 frames) are propagated
to every frame of a set via template matching; the needle is rigid w.r.t. the
cameras, so the match offsets double as a sanity check.

The detector runs with the guide line perturbed (offset/rotation) to simulate the
live twin's pose jitter. Metrics per configuration: median / p95 / max tip error,
fraction of runs within 15 px, and miss rate.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\Labuser\OneDrive - University of Twente\Masterproject\Motordriver")
from needle_detector import NeedleParams, detect_needle

BASE = Path(__file__).resolve().parents[1] / "Motordriver" / "needle_calib_data"

# side/set -> (hand tip label, unit direction pointing tip-ward along the needle)
LABELS = {
    ("extended", "left"): ((417.0, 391.0), (-0.695, -0.719)),
    ("extended", "right"): ((309.0, 433.0), (0.262, -0.965)),
    ("retracted", "left"): ((448.0, 412.0), (-0.727, -0.687)),
    ("retracted", "right"): ((273.0, 480.0), (0.519, -0.855)),
}
LEDS = ["min", "medium", "max"]
POSITIONS = ["x0_y0", "x5_y0", "x0_y5", "x5_y5"]
PATCH = 45  # half-size of the template patch

# guide perturbations (perp offset px, angle deg): (0,0) plus jitter cases
PERTURBATIONS = [(0.0, 0.0), (8.0, 2.0), (-8.0, -2.0), (15.0, 4.0), (-15.0, -4.0)]


def ground_truth():
    """Per-(set, position, side) verified labels, shared across LED levels."""
    import json
    with (BASE / "labels.json").open() as handle:
        labels = json.load(handle)
    gt = {}
    for (set_name, side), (_, direction) in LABELS.items():
        for pos in POSITIONS:
            tx, ty, score = labels["%s_%s_%s" % (set_name, pos, side)]
            for led in LEDS:
                name = "%s_%s_%s_%s.png" % (set_name, led, pos, side)
                gt[name] = ((float(tx), float(ty)), direction, score)
    return gt


def make_params(**overrides):
    base = dict(
        entry="bottom",
        hough_min_length_frac=0.06,
        hough_threshold=20,
        hough_max_gap=20,
        needle_width_max_px=80.0,
        ema_alpha=1.0,
        require_pair=False,
        require_border=False,
        line_distance_max_px=40.0,
        angle_tol_deg=22.0,
    )
    base.update(overrides)
    return NeedleParams(**base)


def evaluate(gt, label="baseline", **overrides):
    errors = []
    misses = 0
    worst = []
    for name, (tip, direction, _) in sorted(gt.items()):
        img = cv2.imread(str(BASE / name))
        for off, dang in PERTURBATIONS:
            a = math.atan2(direction[1], direction[0]) + math.radians(dang)
            dx, dy = math.cos(a), math.sin(a)
            px, py = -dy, dx  # unit perpendicular
            gx = tip[0] - 200.0 * dx + off * px
            gy = tip[1] - 200.0 * dy + off * py
            params = make_params(
                expected_angle_deg=math.degrees(a),
                expected_line_px=(gx, gy, dx, dy),
                **overrides
            )
            det = detect_needle(img, params)
            if det is None:
                misses += 1
                worst.append((float("inf"), name, off, dang))
                continue
            err = math.hypot(det.tip_x - tip[0], det.tip_y - tip[1])
            errors.append(err)
            worst.append((err, name, off, dang))
    errors_arr = np.array(errors) if errors else np.array([float("inf")])
    n_total = len(gt) * len(PERTURBATIONS)
    ok15 = float((errors_arr <= 15.0).sum()) / n_total * 100.0
    print("%-24s median %6.1f px | p95 %6.1f | max %6.1f | <=15px %5.1f%% | miss %d/%d"
          % (label, float(np.median(errors_arr)), float(np.percentile(errors_arr, 95)),
             float(errors_arr.max()), ok15, misses, n_total))
    worst.sort(key=lambda t: -t[0] if t[0] != float("inf") else -1e9)
    return worst


def main():
    gt = ground_truth()
    scores = [v[2] for v in gt.values()]
    print("ground truth: %d frames, template match score min/mean %.3f/%.3f"
          % (len(gt), min(scores), sum(scores) / len(scores)))
    # constancy check: spread of matched tips per set/side
    for (set_name, side), _ in LABELS.items():
        pts = np.array([gt["%s_%s_%s_%s.png" % (set_name, led, pos, side)][0]
                        for led in LEDS for pos in POSITIONS])
        spread = pts.max(axis=0) - pts.min(axis=0)
        print("  %s/%s: tip spread across frames dx=%.1f dy=%.1f px"
              % (set_name, side, spread[0], spread[1]))

    worst = evaluate(gt, "current (cone+clahe)")
    evaluate(gt, "cone off", cone_fit=False)
    evaluate(gt, "clahe off", clahe_clip=0.0)
    print("worst 8 cases (current):")
    for err, name, off, dang in worst[:8]:
        print("  %-34s off=%+5.1f ang=%+4.1f -> %s"
              % (name, off, dang, ("MISS" if err == float("inf") else "%.1f px" % err)))


if __name__ == "__main__":
    main()

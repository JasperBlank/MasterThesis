"""Classical-CV needle detection for the endoscope feed.

The needle is a rigid, roughly straight tool that enters the frame from a known
edge (e.g. the top). That prior is what lets us separate it from every other
straight line in view (the card edges, the perforated backing plate).

Pipeline:  gray -> blur -> Canny -> HoughLinesP -> filter by entry/length/angle
           -> robust line fit -> tip = deepest endpoint.

This module is deliberately standalone so it can be tuned on a single saved
frame (``python needle_detector.py some_frame.png --debug``) and then dropped
into apriltag_tracker.py for live use.

Python 3.8 compatible (annotations are lazy via __future__ import).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Maps an entry edge to a function giving "depth into the frame" for a point.
# Larger depth == farther from the entry edge == nearer the needle tip.
def _depth_from_entry(entry: str, width: int, height: int):
    if entry == "top":
        return lambda x, y: y
    if entry == "bottom":
        return lambda x, y: height - y
    if entry == "left":
        return lambda x, y: x
    if entry == "right":
        return lambda x, y: width - x
    raise ValueError("entry must be one of: top, bottom, left, right, auto")


def _dist_to_border(x: float, y: float, width: int, height: int) -> float:
    """Distance from a point to the nearest image edge (0 == on the border)."""
    return min(x, y, width - 1 - x, height - 1 - y)


@dataclass
class NeedleDetection:
    tip_x: float
    tip_y: float
    entry_x: float
    entry_y: float
    angle_deg: float          # direction from entry point toward tip, image coords
    length_px: float
    n_segments: int           # supporting Hough segments (detection confidence)

    @property
    def tip(self) -> Tuple[int, int]:
        return int(round(self.tip_x)), int(round(self.tip_y))

    @property
    def entry(self) -> Tuple[int, int]:
        return int(round(self.entry_x)), int(round(self.entry_y))


@dataclass
class NeedleParams:
    """All the tunables in one place so the live loop and the debug tool share them."""
    entry: str = "auto"                   # top/bottom/left/right, or "auto" = whichever border the needle crosses
    blur_ksize: int = 5
    canny_low: int = 40
    canny_high: int = 120
    hough_threshold: int = 30
    hough_min_length_frac: float = 0.15   # min segment length as fraction of max(H, W)
    hough_max_gap: int = 12
    expected_angle_deg: Optional[float] = None  # if set, keep segments within angle_tol of it
    angle_tol_deg: float = 35.0
    expected_line_px: Optional[Tuple[float, float, float, float]] = None
    line_distance_max_px: float = 40.0
    ema_alpha: float = 0.5                # temporal smoothing; 1.0 = no smoothing
    roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h to restrict the search

    # --- Needle-shape priors (the two cues that separate the needle from the card) ---
    require_pair: bool = True             # the needle shows up as two close, parallel edges (its sides)
    needle_width_min_px: float = 1.5      # min perpendicular gap between the two sides
    needle_width_max_px: float = 22.0     # max gap; the card's edges are far wider apart than this
    pair_angle_tol_deg: float = 12.0      # how parallel the two sides must be
    require_border: bool = True           # the needle must enter from off-screen (touch an image edge)
    border_margin_frac: float = 0.06      # "touching" tolerance, as a fraction of max(H, W)
    pair_fallback_single: bool = False    # if no pair is found, fall back to the single best line


def _segment_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _angle_within(a: float, target: float, tol: float) -> bool:
    # Compare undirected line orientations, so 170 and -10 are "close".
    d = abs((a - target + 90.0) % 180.0 - 90.0)
    return d <= tol


def _distance_to_expected_line(
    x: float,
    y: float,
    expected_line: Tuple[float, float, float, float],
) -> float:
    """Perpendicular pixel distance to an expected image-space centerline."""
    x0, y0, dx, dy = expected_line
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return float("inf")
    return abs((x - x0) * (-dy) + (y - y0) * dx) / norm


def detect_needle(
    frame_bgr: np.ndarray,
    params: NeedleParams,
    prev: Optional[NeedleDetection] = None,
) -> Optional[NeedleDetection]:
    """Detect the needle in one BGR frame. Returns None if nothing convincing is found.

    ``prev`` (the last good detection) is used only for EMA smoothing of the tip.
    """
    height, width = frame_bgr.shape[:2]

    # Optional ROI restriction (offset added back so coordinates stay frame-global).
    ox, oy = 0, 0
    work = frame_bgr
    if params.roi is not None:
        rx, ry, rw, rh = params.roi
        rx = max(0, rx); ry = max(0, ry)
        rw = min(rw, width - rx); rh = min(rh, height - ry)
        work = frame_bgr[ry:ry + rh, rx:rx + rw]
        ox, oy = rx, ry

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    if params.blur_ksize > 1:
        k = params.blur_ksize | 1  # force odd
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(gray, params.canny_low, params.canny_high)

    min_len = params.hough_min_length_frac * max(height, width)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=params.hough_threshold,
        minLineLength=int(min_len),
        maxLineGap=params.hough_max_gap,
    )
    if raw is None:
        return None

    # Collect Hough segments in frame-global coords, filtered only by the optional
    # expected-angle band. Pairing/border tests happen below.
    segments: List[np.ndarray] = []
    # OpenCV 4 commonly returns (N, 1, 4), while OpenCV 5 may return (N, 4).
    # Reshaping keeps the detector compatible with both layouts.
    for seg in np.asarray(raw).reshape(-1, 4):
        x1, y1, x2, y2 = (float(v) for v in seg)
        x1 += ox; x2 += ox; y1 += oy; y2 += oy
        if params.expected_angle_deg is not None and not _angle_within(
            _segment_angle_deg(x1, y1, x2, y2), params.expected_angle_deg, params.angle_tol_deg
        ):
            continue
        if params.expected_line_px is not None:
            midpoint_x = 0.5 * (x1 + x2)
            midpoint_y = 0.5 * (y1 + y2)
            if _distance_to_expected_line(
                midpoint_x, midpoint_y, params.expected_line_px
            ) > params.line_distance_max_px:
                continue
        segments.append(np.array([x1, y1, x2, y2]))

    if not segments:
        return None

    border_margin = params.border_margin_frac * max(width, height)

    # Primary cue: the needle's two parallel sides. Find the best such pair and
    # use all four endpoints as support for the centerline fit.
    pts = None
    n_support = 0
    if params.require_pair:
        pair = _best_needle_pair(segments, params, width, height, border_margin)
        if pair is not None:
            pts, n_support = pair

    # Fallback: single dominant line touching a border (old behaviour).
    if pts is None:
        if params.require_pair and not params.pair_fallback_single:
            return None
        single = _best_single_line(segments, params, width, height, border_margin)
        if single is None:
            return None
        pts, n_support = single

    return _detection_from_points(pts, n_support, params, width, height, prev)


def _segment_endpoints(seg: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (float(seg[0]), float(seg[1])), (float(seg[2]), float(seg[3]))


def _seg_unit_dir(seg: np.ndarray) -> Tuple[float, float]:
    dx, dy = seg[2] - seg[0], seg[3] - seg[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _touches_border(pts: List[Tuple[float, float]], width: int, height: int,
                    margin: float) -> bool:
    return any(_dist_to_border(x, y, width, height) <= margin for x, y in pts)


def _best_needle_pair(segments: List[np.ndarray], params: NeedleParams,
                      width: int, height: int, border_margin: float):
    """Find the best pair of near-parallel segments spaced like a needle's width.

    Returns (four_endpoints, n_support) for the winning pair, or None.
    """
    best_score = -1.0
    best_pts: Optional[List[Tuple[float, float]]] = None
    n = len(segments)
    for i in range(n):
        ai = _segment_angle_deg(*segments[i])
        ux, uy = _seg_unit_dir(segments[i])
        (xi1, yi1), (xi2, yi2) = _segment_endpoints(segments[i])
        len_i = math.hypot(xi2 - xi1, yi2 - yi1)
        for j in range(i + 1, n):
            aj = _segment_angle_deg(*segments[j])
            if not _angle_within(ai, aj, params.pair_angle_tol_deg):
                continue
            (xj1, yj1), (xj2, yj2) = _segment_endpoints(segments[j])

            # Perpendicular gap = distance from j's midpoint to i's line.
            mjx, mjy = (xj1 + xj2) / 2.0, (yj1 + yj2) / 2.0
            gap = abs((mjx - xi1) * (-uy) + (mjy - yi1) * ux)
            if not (params.needle_width_min_px <= gap <= params.needle_width_max_px):
                continue

            # Require longitudinal overlap (the sides run alongside each other,
            # not collinear end-to-end). Project all endpoints onto i's direction.
            ti = sorted([(xi1 - xi1) * ux + (yi1 - yi1) * uy,
                         (xi2 - xi1) * ux + (yi2 - yi1) * uy])
            tj = sorted([(xj1 - xi1) * ux + (yj1 - yi1) * uy,
                         (xj2 - xi1) * ux + (yj2 - yi1) * uy])
            overlap = min(ti[1], tj[1]) - max(ti[0], tj[0])
            if overlap <= 0:
                continue

            pts = [(xi1, yi1), (xi2, yi2), (xj1, yj1), (xj2, yj2)]
            if params.require_border and not _touches_border(pts, width, height, border_margin):
                continue

            len_j = math.hypot(xj2 - xj1, yj2 - yj1)
            score = overlap + 0.25 * (len_i + len_j)
            if score > best_score:
                best_score = score
                best_pts = pts
    if best_pts is None:
        return None
    return best_pts, 2


def _best_single_line(segments: List[np.ndarray], params: NeedleParams,
                      width: int, height: int, border_margin: float):
    """Fallback: longest border-touching line plus its orientation-mates."""
    scored: List[Tuple[float, np.ndarray]] = []
    for seg in segments:
        (x1, y1), (x2, y2) = _segment_endpoints(seg)
        if params.require_border and not _touches_border(
            [(x1, y1), (x2, y2)], width, height, border_margin
        ):
            continue
        scored.append((math.hypot(x2 - x1, y2 - y1), seg))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    best_angle = _segment_angle_deg(*scored[0][1])
    pts: List[Tuple[float, float]] = []
    n_support = 0
    for _, seg in scored:
        if _angle_within(_segment_angle_deg(*seg), best_angle, params.angle_tol_deg):
            (x1, y1), (x2, y2) = _segment_endpoints(seg)
            pts.extend([(x1, y1), (x2, y2)])
            n_support += 1
    return pts, n_support


def _detection_from_points(pts: List[Tuple[float, float]], n_support: int,
                           params: NeedleParams, width: int, height: int,
                           prev: Optional[NeedleDetection]) -> NeedleDetection:
    """Fit the centerline through support points; orient tip via entry/border prior."""
    pts_arr = np.array(pts, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    ts = (pts_arr[:, 0] - x0) * vx + (pts_arr[:, 1] - y0) * vy
    p_lo = (x0 + ts.min() * vx, y0 + ts.min() * vy)
    p_hi = (x0 + ts.max() * vx, y0 + ts.max() * vy)

    # A projected CAD axis gives an explicit entry-to-tip direction and remains
    # valid when a scope rolls. Otherwise use the configured image-border prior.
    if params.expected_line_px is not None:
        expected_x, expected_y, expected_dx, expected_dy = params.expected_line_px
        score_lo = (
            (p_lo[0] - expected_x) * expected_dx
            + (p_lo[1] - expected_y) * expected_dy
        )
        score_hi = (
            (p_hi[0] - expected_x) * expected_dx
            + (p_hi[1] - expected_y) * expected_dy
        )
        if score_hi >= score_lo:
            tip, entry_pt = p_hi, p_lo
        else:
            tip, entry_pt = p_lo, p_hi
    elif params.entry == "auto":
        if _dist_to_border(*p_hi, width, height) >= _dist_to_border(*p_lo, width, height):
            tip, entry_pt = p_hi, p_lo
        else:
            tip, entry_pt = p_lo, p_hi
    else:
        depth = _depth_from_entry(params.entry, width, height)
        if depth(*p_hi) >= depth(*p_lo):
            tip, entry_pt = p_hi, p_lo
        else:
            tip, entry_pt = p_lo, p_hi

    tip_x, tip_y = float(tip[0]), float(tip[1])
    if prev is not None and 0.0 < params.ema_alpha < 1.0:
        tip_x = params.ema_alpha * tip_x + (1.0 - params.ema_alpha) * prev.tip_x
        tip_y = params.ema_alpha * tip_y + (1.0 - params.ema_alpha) * prev.tip_y

    return NeedleDetection(
        tip_x=tip_x,
        tip_y=tip_y,
        entry_x=float(entry_pt[0]),
        entry_y=float(entry_pt[1]),
        angle_deg=math.degrees(math.atan2(tip_y - entry_pt[1], tip_x - entry_pt[0])),
        length_px=math.hypot(tip_x - entry_pt[0], tip_y - entry_pt[1]),
        n_segments=n_support,
    )


def draw_needle_overlay(frame: np.ndarray, det: Optional[NeedleDetection]) -> None:
    """Draw the needle centerline, tip, and a label. Mirrors the tag overlay style."""
    if det is None:
        cv2.putText(frame, "needle: none", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        return
    cv2.line(frame, det.entry, det.tip, (0, 165, 255), 2)          # orange centerline
    cv2.circle(frame, det.tip, 6, (0, 0, 255), -1)                 # red tip
    cv2.drawMarker(frame, det.tip, (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
    label = (f"needle tip=({det.tip[0]},{det.tip[1]}) "
             f"angle={det.angle_deg:+.1f} n={det.n_segments}")
    cv2.putText(frame, label, (det.tip[0] + 10, det.tip[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)


# ----------------------------------------------------------------------------
# Red-dot detection (HSV colour blob). Independent of the needle pipeline.
# ----------------------------------------------------------------------------

@dataclass
class RedDotDetection:
    center_x: float
    center_y: float
    radius_px: float
    area_px: float

    @property
    def center(self) -> Tuple[int, int]:
        return int(round(self.center_x)), int(round(self.center_y))


@dataclass
class RedDotParams:
    """Tunables for the red-dot colour blob."""
    h_lo: int = 10            # upper hue of the low-red band (0..h_lo)
    h_hi: int = 170           # lower hue of the high-red band (h_hi..180)
    s_min: int = 50           # min saturation: rejects pale/orange tissue.
                              # 50 (not 90): the endoscope's warm cast desaturates the red dot.
    v_min: int = 60           # min value: rejects dark reds/shadows
    min_area_frac: float = 0.0006   # smallest accepted blob, as a fraction of frame area
    morph_ksize: int = 5      # close/open kernel to clean the mask
    enabled: bool = True


def _red_mask(frame_bgr: np.ndarray, p: RedDotParams) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, p.s_min, p.v_min), (p.h_lo, 255, 255))
    hi = cv2.inRange(hsv, (p.h_hi, p.s_min, p.v_min), (180, 255, 255))
    mask = cv2.bitwise_or(lo, hi)
    if p.morph_ksize > 1:
        k = np.ones((p.morph_ksize, p.morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_red_dot(frame_bgr: np.ndarray, p: RedDotParams) -> Optional[RedDotDetection]:
    """Find the largest red blob and return its centroid + radius. None if too small."""
    if not p.enabled:
        return None
    height, width = frame_bgr.shape[:2]
    mask = _red_mask(frame_bgr, p)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(biggest))
    if area < p.min_area_frac * width * height:
        return None

    m = cv2.moments(biggest)
    if m["m00"] == 0:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    radius = math.sqrt(area / math.pi)   # equivalent-circle radius
    return RedDotDetection(center_x=cx, center_y=cy, radius_px=radius, area_px=area)


def draw_red_dot_overlay(frame: np.ndarray, det: Optional[RedDotDetection]) -> None:
    if det is None:
        return
    cv2.circle(frame, det.center, int(round(det.radius_px)), (255, 0, 255), 2)  # magenta ring
    cv2.drawMarker(frame, det.center, (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(frame, f"red dot=({det.center[0]},{det.center[1]})",
                (det.center[0] + 10, det.center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


def _edge_map(frame: np.ndarray, params: NeedleParams) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = params.blur_ksize | 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.Canny(gray, params.canny_low, params.canny_high)


def _debug_view_frame(frame: np.ndarray, params: NeedleParams,
                      det: Optional[NeedleDetection],
                      red: Optional[RedDotDetection] = None) -> np.ndarray:
    """Side-by-side of the edge map and an already-computed detection."""
    edges_bgr = cv2.cvtColor(_edge_map(frame, params), cv2.COLOR_GRAY2BGR)
    annotated = frame.copy()
    draw_needle_overlay(annotated, det)
    draw_red_dot_overlay(annotated, red)
    return np.hstack([edges_bgr, annotated])


def _debug_view(frame: np.ndarray, params: NeedleParams,
                red_params: RedDotParams) -> np.ndarray:
    """Offline tuning view: detect on the frame, then show edges + detection."""
    det = detect_needle(frame, params)
    if det is not None:
        print(f"tip=({det.tip_x:.1f},{det.tip_y:.1f}) entry=({det.entry_x:.1f},"
              f"{det.entry_y:.1f}) angle={det.angle_deg:+.1f} len={det.length_px:.0f} "
              f"n_segments={det.n_segments}")
    else:
        print("no needle detected")
    red = detect_red_dot(frame, red_params)
    print(f"red dot=({red.center[0]},{red.center[1]}) r={red.radius_px:.0f}"
          if red is not None else "no red dot detected")
    return _debug_view_frame(frame, params, det, red)


def open_camera(index: int, width: Optional[int], height: Optional[int]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Keep the driver buffer tiny so read() returns the newest frame, not a
    # backlog of stale ones (otherwise slow per-frame processing makes the
    # preview lag seconds behind reality).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap


def _read_latest(cap: cv2.VideoCapture, drain: int = 2) -> Optional[np.ndarray]:
    """Read the freshest frame, discarding any buffered backlog.

    Capture and processing run on the same thread (DSHOW returns black frames if
    read from a worker thread), so to avoid lag we grab a few frames and keep
    only the last one. With CAP_PROP_BUFFERSIZE honored, drain=1 is enough; if
    not, the extra grabs pop the stale backlog.
    """
    frame = None
    for _ in range(max(1, drain)):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    return frame


def run_live(camera: int, params: NeedleParams, width: Optional[int],
             height: Optional[int], debug: bool, red_params: RedDotParams,
             miss_hold: int = 8) -> None:
    """Detect the needle live on the raw webcam feed (no overlay baked into the input)."""
    cap = open_camera(camera, width, height)
    print(f"Needle detector live on camera {camera}. Press q to quit.")
    prev: Optional[NeedleDetection] = None
    misses = 0
    try:
        while True:
            frame = _read_latest(cap)
            if frame is None:
                raise RuntimeError("Camera frame read failed")

            # Always show the untouched live feed in its own window.
            cv2.imshow("camera (raw)", frame)

            det = detect_needle(frame, params, prev)
            if det is not None:
                prev = det
                misses = 0
            else:
                misses += 1
                if misses > miss_hold:
                    prev = None            # give up holding the stale tip
            shown = prev if det is None and misses <= miss_hold else det
            red = detect_red_dot(frame, red_params)

            if debug:
                view = _debug_view_frame(frame, params, shown, red)
            else:
                view = frame.copy()
                draw_needle_overlay(view, shown)
                draw_red_dot_overlay(view, red)
            cv2.imshow("needle detector (live)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect the needle on a saved frame or live on the raw webcam.")
    parser.add_argument("image", nargs="?", help="Path to a saved endoscope frame "
                        "(omit and use --camera for live mode)")
    parser.add_argument("--camera", type=int, default=None,
                        help="OpenCV camera index for live raw-feed detection")
    parser.add_argument("--width", type=int, default=None, help="Requested camera width")
    parser.add_argument("--height", type=int, default=None, help="Requested camera height")
    parser.add_argument("--entry", choices=["top", "bottom", "left", "right", "auto"],
                        default="auto", help="Border the needle enters from "
                        "('auto' = whichever edge it crosses)")
    parser.add_argument("--canny-low", type=int, default=40)
    parser.add_argument("--canny-high", type=int, default=120)
    parser.add_argument("--min-length-frac", type=float, default=0.15)
    parser.add_argument("--hough-threshold", type=int, default=30)
    parser.add_argument("--expected-angle", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.5,
                        help="Temporal smoothing for live tip (1.0 = off)")
    parser.add_argument("--needle-width-max", type=float, default=22.0,
                        help="Max pixel gap between the needle's two parallel sides")
    parser.add_argument("--no-require-pair", action="store_true",
                        help="Do not require two close parallel edges (use single-line mode)")
    parser.add_argument("--no-require-border", action="store_true",
                        help="Do not require the needle to touch an image edge")
    parser.add_argument("--no-red-dot", action="store_true",
                        help="Disable red-dot detection")
    parser.add_argument("--red-s-min", type=int, default=50,
                        help="Min HSV saturation for red (raise to reject pale/orange)")
    parser.add_argument("--red-v-min", type=int, default=60,
                        help="Min HSV value for red (raise to reject dark reds)")
    parser.add_argument("--red-min-area-frac", type=float, default=0.0006,
                        help="Smallest accepted red blob, as a fraction of frame area")
    parser.add_argument("--debug", action="store_true", help="Show edges + detection")
    args = parser.parse_args()

    if args.camera is None and args.image is None:
        parser.error("provide an image path, or --camera N for live mode")

    params = NeedleParams(
        entry=args.entry,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        hough_min_length_frac=args.min_length_frac,
        hough_threshold=args.hough_threshold,
        expected_angle_deg=args.expected_angle,
        needle_width_max_px=args.needle_width_max,
        require_pair=not args.no_require_pair,
        require_border=not args.no_require_border,
        ema_alpha=args.ema_alpha if args.camera is not None else 1.0,
    )
    red_params = RedDotParams(
        s_min=args.red_s_min,
        v_min=args.red_v_min,
        min_area_frac=args.red_min_area_frac,
        enabled=not args.no_red_dot,
    )

    if args.camera is not None:
        run_live(args.camera, params, args.width, args.height, args.debug, red_params)
        return

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    view = _debug_view(frame, params, red_params) if args.debug else None
    if view is None:
        det = detect_needle(frame, params)
        red = detect_red_dot(frame, red_params)
        annotated = frame.copy()
        draw_needle_overlay(annotated, det)
        draw_red_dot_overlay(annotated, red)
        view = annotated
    cv2.imshow("needle detector", view)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Handover: continuing on the laptop (from the lab-PC Fable session, 2026-07-16)

You are picking up work on Jasper's master-thesis repo
(https://github.com/JasperBlank/MasterThesis). Read `AGENTS.md` first — it is the
project-wide source of truth (hardware, constraints, conventions). This file only
adds what the laptop context needs: what just happened, what is in flight, and
what is possible *without* the lab hardware.

## Ground rules that still apply on the laptop

- **All code must stay Python 3.8 compatible** (the lab PC runs 3.8.10). Even if
  the laptop has a newer Python, do not use `match`, `X | Y` annotations,
  `dict | dict`, or `list[int]` runtime generics.
- **D = endoscope center-to-center distance**, never "baseline".
- Deps for the vision work: `pip install numpy scipy opencv-python`. The digital
  twin additionally needs `"pyvista<0.44" "vtk<9.4"` (only on Python 3.8; on a
  newer laptop Python, current pyvista is fine for *viewing*, but don't commit
  code that requires it).
- No cameras and no motors on the laptop: anything "live" is out of reach. That
  is fine — the current work item is deliberately offline.

## What just happened (last lab session)

The live needle-tip detection flickered by up to half a screen on a static
scene. We captured a **labeled calibration dataset** and rewrote the detector:

- Dataset: `Motordriver/needle_calib_data/` — 24 stereo pairs = 48 images
  (2 needle extensions × 3 LED levels {min,medium,max} × 4 stage positions),
  with hand-verified tip labels in `labels.json`, keyed
  `{set}_{position}_{side}` (positions: x0_y0, x5_y0, x0_y5, x5_y5). Labels are
  per position and shared across LED levels (the tip does not move with LED).
  Beware: in the extended set the tip moves ~100 px at position x0_y5 — the
  needle physically dragged onto the red target sticker there (real, not an
  error).
- Detector: `Motordriver/needle_detector.py`. Primary method (when a guide line
  `expected_line_px` is set, which is how the twin calls it) is
  `_corridor_trace`: rectify a 90×420 px corridor around the projected CAD
  needle axis, find per-column dark runs, trace them with **lateral continuity**
  (a run must overlap the previous column's row interval; after a >2-column gap
  the resuming run must be narrow — this is what stops the trace from crossing
  into AprilTag patterns), mask saturated pixels **dilated by 9 px** (removes
  the red sticker incl. its dark rim), and place the tip at the last column
  whose contrast is comparable to the needle's shaft
  (`>= max(corridor_min_contrast, 0.45 * median(first-half contrast))`).
  Fallbacks in order: cone-fit (intersect the needle's two converging side
  edges), single-Hough-line. CLAHE preprocessing is on by default.
- Evaluation harness: `analysis_scripts/needle_detector_eval.py`. Runs the
  detector on all 48 images × 5 guide-line perturbations (simulating the live
  pose jitter) and scores against the labels. Just run it — it finds the
  dataset by relative path.

Current numbers (laptop, 2026-07-17, ratio sweep + CLAHE off): median 11.4 px,
p95 24.2 px, max 89.1 px, 73.3 % within 15 px, 0 misses.
Previous (commit `7b7fc64`, ratio hard-coded at 0.45, CLAHE on): median 12.6,
p95 53, max 89.7, 65.4 % within 15 px. Baseline (old Hough selection):
median 42.6, p95 196, max 818.

## In-flight experiment — DONE (laptop, 2026-07-16)

The min-LED light-falloff sweep described here was completed. The `0.45`
strong-contrast ratio is now the `NeedleParams.corridor_strong_ratio` field,
and the sweep over {0.45, 0.3, 0.2, 0.1} picked **0.30** by (max, then p95):

| ratio | median | p95  | max  | <=15px |
|-------|--------|------|------|--------|
| 0.45  | 12.6   | 53.0 | 89.7 | 65.4 % |
| 0.30  | 12.2   | 28.0 | 89.1 | 72.5 % |
| 0.20  | 12.0   | 28.0 | 90.1 | 72.5 % |
| 0.10  | 11.5   | 28.0 | 90.1 | 72.5 % |

The hypothesis held: 0.30 fixes the min-LED undershoot family
(`extended_min_x0_y0_right`, `extended_min_x5_y0_right` leave the worst list)
because the absolute floor (`corridor_min_contrast` = 28) keeps rejecting the
soft cast shadow. Below 0.30 the shadow creeps back in (at 0.20/0.10 the
`retracted_min_x0_y0_right` +8 px perturbation degrades 40.8 → ~71 px), so
0.30 is a real optimum, not just "lower is better". The sticker-rim watch case
(`extended_min_x0_y5_left`, 67.6 px) is unchanged at every ratio — it is an
independent residual, not a regression.

Follow-up (2026-07-17): the CLAHE lead was tested and resolved — **CLAHE is
now off by default** (`clahe_clip = 0.0`). A per-run diff (all 240 runs,
CLAHE 2.0 vs 0.0 at ratio 0.30) showed CLAHE never touches the corridor
trace (it reads the raw frame); it only affects the Hough/cone fallback, and
the *only* image where that fires is the sticker-rim case
(`extended_min_x0_y5_left`) — where CLAHE makes it worse (67.6 px at two
perturbations vs 7.1 px at all five without CLAHE). So disabling CLAHE fixed
the sticker-rim residual outright. CLAHE was only introduced in `682c329`
(the same lab session), so the old 2D servoing layer simply gets its original
behavior back. The ratio re-sweep with CLAHE off confirms 0.30 stays the
right default (max 89.1 vs 89.7/90.1). Remaining worst family: min-LED
right-camera cases in the 33–41 px range plus one 89 px outlier
(`retracted_min_x0_y0_right` at the +15 px guide perturbation).

## Other laptop-friendly work (in priority order)

1. **Supervisor's optimization questions** (see AGENTS.md → Open work): make
   needle-axis offset Δy a first-class variable in the layout optimizer and add
   insertion length L to the multi-objective sweep (`analysis_scripts/`).
   Purely offline, thesis-critical.
2. Pose-pipeline sensitivity study: simulate focal-length / distortion /
   corner-noise errors through `Motordriver/stereo_probe_pose.py` math to
   quantify what the accepted camera calibration is worth.
3. Detector robustness: the 15 px success gate is partly limited by label
   precision (~±3 px); a sub-pixel tip refinement (local gradient maximum along
   the traced centerline) could tighten the median further.

## Needs the lab PC (do NOT attempt on the laptop)

- The decisive **needle travel test** (retract → extend a known distance,
  compare raw vs rendered extension in the five-panel twin) — waiting on
  bench time, detector is now good enough for it to be meaningful.
- Any live validation of the detector in the twin
  (`python digital_twin\twin_wasd_jog.py --pose-live --tag-edge-mm 10.62
  --anchor-ids 1,2,3 --reference-id 2 --reverse-cad-camera-order`).
- LED control, camera capture, motor homing (see AGENTS.md for all quirks —
  notably: LEDs reset to max at every stream start; USB descriptor failures
  need a physical replug; motors lose homing on power-off).

## Repo/git notes for the laptop

- Commit style: descriptive body explaining the why, and
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Correction (2026-07-16):** OneDrive DOES sync the repo to the laptop,
  `.git` included — the lab PC and the laptop share one working copy through
  OneDrive. No clone/pull is needed to see each other's commits, but never run
  git operations on both machines at the same time, and let OneDrive finish
  syncing before starting a session, or `.git` can end up with sync conflicts.

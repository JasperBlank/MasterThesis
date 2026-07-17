# Handover / agent instructions

Master-thesis project: **dual-endoscope + needle-injection end-effector for minimally
invasive surgery** (University of Twente). A Ø8 mm probe carries two Comedia muC112
micro-endoscopes (stereo pair), a Ø4 mm working channel, and a Ø1 mm syringe/needle
channel; it is moved by three Thorlabs stages. This file is the handover for AI coding
agents. Last updated: 2026-07-16, after the probe-pose / digital-twin milestone.

## Hard constraints — read first

1. **Python 3.8 only** (lab PC, Windows 10). No `match`, no `X | Y` annotations, no
   `dict|dict`, no `list[int]` runtime generics. Pinned deps that break otherwise:
   `pyvista<0.44`, `vtk<9.4`, `pygrabber==0.1` (≥0.2 uses 3.9+ syntax), plus
   `numpy scipy matplotlib opencv-python comtypes pyusb libusb-package`.
2. **Naming:** D = endoscope center-to-center distance. Do **not** call it "baseline"
   in new code, labels, or docs — deliberately renamed by the project owner.
3. **Tip-LED policy: both endoscope LEDs run at "min"** on the bench (heat/aging).
   Details under Hardware; this is a standing user preference, not a suggestion.
4. **Close the Thorlabs Kinesis GUI before touching motors** — it holds the device
   connections (check `Get-Process Thorlabs.MotionControl.Kinesis`).
5. Motors move real hardware. Respect the soft limits in `Motordriver/kcube_axes.json`
   (0–24 mm per axis) and the 0.05 mm limit guard used by the jog scripts.

## End-effector geometry (current design)

- Outer shell **Ø8 mm**; cross-section hole radii `RADII = [2.0, 0.6, 0.6, 0.5]` mm:
  Ø4 working channel locked to the vertical centerline (x = 0), 2× Ø1.2 endoscope
  channels symmetric at (±D/2, y_e), Ø1 syringe channel locked to x = 0.
- Layout optimizer: two-phase SLSQP / 1-D scan maximizing minimum wall thickness
  (`analysis_scripts/end_effector_layout.py`; reusable `optimize_layout`,
  `reposition_syringe`, `min_clearance`).
- As-printed part (`Holderdesign/cad/cad/Array_side_slip.STL`, 8×30×8 mm, tip face =
  model y=0): endoscope bores Ø2.05 at (1.411, 4.432) and (6.589, 4.432) model-mm →
  **D = 5.178 mm**; Ø4 working channel at (4.0, 2.5); syringe channel at (4.0, 6.0)
  (Ø0.5 face orifice, Ø1.0 bore behind).

## Hardware and its quirks

### Endoscopes — Comedia muC112 (×2)
- Ø2.0/2.2 mm, 120° diagonal FOV, working distance 5–50 mm, 720×720@30 via UVC.
- USB backend board: **COMedia C8209HL** (`VID_0573&PID_0A82`). No exposure control
  (`CAP_PROP_EXPOSURE` rejected). UVC knobs: brightness (def 5), contrast (4),
  saturation (6), sharpness (2), gain (0–2, but "gain" is really the board's
  snapshot-trigger channel, not an amplifier).
- **LED control** is via a vendor UVC extension unit — GUID
  `{DD880F8A-1CBA-4954-8A25-F7875967F0F7}`, unit ID 8, KSP_NODE node 3, selector 1,
  32-byte commands: `0x04 <level>` LED (0 off / 1 min / 2 medium / 3 max),
  `0x02 <mode>` WB (0 auto / 1 fixed), `0x01` firmware version (SET then GET),
  `0x05/0x06` sensor register write/read. Protocol source: "C8209 Controller User
  Manual" (shop.comedia.com.hk). Implemented in `Motordriver/led_control.py`.
- **Firmware resets the LED to max at every stream start.** The level must be applied
  *while frames are flowing*. Exclusive DirectShow capture blocks the control
  channel, so capture with **MSMF** (`cv2.CAP_MSMF`); the Windows frame server then
  allows a concurrent DirectShow property connection for the XU command.
  `stereo_apriltag_webcams.py --backend msmf --left-led min --right-led min` does the
  whole dance (opens streams, warms up, then sets LEDs).
- LEDs are lit only while some program streams; stopping all capture turns them off.
- Board keys: Key1 = white-balance calibration (hold 2–3 s at a white target while
  streaming). Key2/Key3 only set the UVC "gain" value to 1/2 as snapshot/record
  triggers for user software — they appear dead unless something polls that value.
- UVC brightness/contrast settings **persist across replugs** (Windows caches them
  per device). If one camera looks dimmer than the other, dump both cameras' values
  (IAMVideoProcAmp) before suspecting hardware.
- **Chronic USB flakiness:** the scopes repeatedly fail enumeration with "Device
  Descriptor Request Failed" (code 43), across multiple ports, typically after
  sitting idle/overnight. Only a physical replug fixes it. Suspects: cables or USB
  selective suspend (not yet ruled out). If a DirectShow bind *hangs*, the UVC stack
  is wedged — replug. Note: replugging can change camera index order (0↔1).

### Motors — Thorlabs KDC101 + Z925B servos on XR25X/M stages (×3)
- Behind a KEH ethernet hub at `192.168.0.200`; raw APT-protocol control (no Kinesis
  needed) lives in `Motordriver/kcube_motion.py`. Serial→port map in `ENDPOINTS`:
  27271413:40307 (axis1), 27271464:40308 (axis2), 27271523:40309 (axis3).
- **Homing is lost on power-off** (position reads 0.0000 and the homed status bit is
  clear). Home via raw APT: send `apt_short(0x0443, 1)`, poll `get_status()` until
  status bit 0x0400 (homed) is set and 0x0200 (homing) clear — ~12 s per axis. The
  Kinesis .NET path (`kcube_control.py home`) has been unreliable (DeviceNotReady).
- Velocity moves: 2.6 mm/s max (Z925B); jog scripts default to 80 %.

### Syringe pump
NE-300 "Just Infusion" (targets vasculature Ø 100–500 µm; needle channel ≈ Ø 0.8–1 mm).

## Key scripts

| Path | Purpose |
|---|---|
| `Motordriver/kcube_motion.py` | Raw-ethernet APT motor control; import this, not Kinesis. |
| `Motordriver/kcube_wasd_jog.py` | Held-key WASD/QE velocity jog of the real stages. |
| `Motordriver/led_control.py` | Tip-LED / WB / firmware via the C8209 extension unit. |
| `Motordriver/stereo_apriltag_webcams.py` | Live stereo AprilTag view; `--backend msmf`, `--left/right-led`, `--left/right-brightness/gain`, `--scan-cameras`. |
| `Motordriver/stereo_probe_pose.py` | **6-DOF probe pose** in the tag-sheet frame from the two endoscopes (see below). |
| `Motordriver/tag_sheet_calibration.json` | Saved metric scale (tag edge mm) + stage-axis direction from `--calibrate`. |
| `Motordriver/apriltag_tracker.py` | Single-camera 2D tag tracking (detection utilities imported by the pose script). |
| `Motordriver/needle_to_dot.py`, `kcube_wasd_needle_control.py` | Existing 2D needle→target servoing layer. |
| `digital_twin/twin_wasd_jog.py` | PyVista digital twin: 3 views (external + stereo pair), needle, WASD/QE with real soft limits; `--pose-live` / `--pose-json`. |
| `analysis_scripts/end_effector_layout.py` | Cross-section layout optimizer (matplotlib GUI). |
| `analysis_scripts/multiobjective_sweep.py` | Depth/FOV/reconstructability objectives vs D. |
| `docs/supervisor_notes.md`, `docs/stereo_endoscope_baseline_report.md` | Specs + stereo theory. |

## Probe-pose pipeline (state as of 2026-07-16, post camera calibration)

Scene: exactly **three 36h11 AprilTags, ids 1–3, coplanar on one sheet**.
**Measured tag edge = 10.62 mm** (ruler, across the outside of the black border);
white gap between adjacent tags = 2.14 mm. Print-ready tags: `docs/apriltags_10_62mm/`.

- **Accepted intrinsics: `Motordriver/stereo_camera_calibration.json`** (five-term
  pinhole, 26 captures / 78 tag observations; left RMS 0.185 px, right 0.244 px,
  joint 0.611 px; fitted lateral D = 5.282 mm vs mechanical 5.178 mm). Workflow:
  `docs/apriltag_camera_calibration.md`. **Do not silently overwrite it.**
- `Motordriver/tag_sheet_calibration.json` now stores the *measured* 10.62 mm edge
  (an earlier motion-derived 6.83 mm value was invalid and has been replaced). The
  stage-motion `--calibrate` path is unreliable; prefer the measured tag size.
- `stereo_probe_pose.py` detects tags (OpenCV aruco), recovers the sheet layout via
  the reference tag's homography (id 2), solvePnP per camera with the calibrated
  intrinsics, and returns per-camera transforms + the probe frame (mean view axis +
  camera-center line). The scale from the 5.178 mm scope separation is
  ill-conditioned at working distance — cross-check only.
- Regression test: `analysis_scripts/test_stereo_apriltag_calibration.py`
  (synthetic; last run recovered D = 5.138 mm vs expected 5.178 mm).

Physical caveats: one scope protrudes ~0.5–1.5 mm from the tip face, the other
~1–2 mm, and both can rotate/shift slightly when touched — per-session orientation
estimation is mandatory (the twin does this at startup and updates at 5 Hz);
never trust yesterday's rolls.

## Digital twin (five-panel)

`digital_twin/twin_wasd_jog.py` mirrors the real controls (same keys, soft limits
from `kcube_axes.json`, velocity scaling) and shows **five panels**: external twin,
virtual left/right endoscope views (120° diag FOV), and the two *live* endoscope
frames with detection overlays. Simulated axes only — it never commands motors.

- Startup with `--pose-live` learns the session tag layout from the first stereo
  pair (reference id 2 = origin) and keeps updating sheet/camera orientations at
  5 Hz. Standard invocation (note the required tag size):
  `python digital_twin\twin_wasd_jog.py --pose-live --tag-edge-mm 10.62
  --anchor-ids 1,2,3 --reference-id 2 --reverse-cad-camera-order`
- **Per-session needle-line calibration** (2026-07-17): the needle tilts inside
  the bore clearance (measured 2.8°, exit ~0.7 mm off the nominal bore), so the
  raw CAD axis projects up to ~12° off in an image. The tracker triangulates the
  needle line (intersection of the two cameras' needle planes, anchored by the
  triangulated tip) over the first `--needle-line-samples` (default 15) good
  stereo observations and then uses it for the guide-line projection and the
  extension measurement. Requires the needle visible in both views at startup;
  re-seat the needle → restart the twin. Validation tool:
  `analysis_scripts/measure_needle_guide_offset.py --seconds 60
  [--needle-line-samples 15]` (left angle error 12.0°→3.4° with calibration;
  residual ±15 px opposite-signed offsets are the stereo-intrinsics floor).
- **`--reverse-cad-camera-order` is the currently correct camera↔bore mapping**;
  the `C` key toggles it at runtime.
- **Live needle-extension estimation**: projects the CAD needle axis into each
  calibrated image to guide the Canny/Hough edge detector
  (`Motordriver/needle_detector.py`), triangulates the detected tip, projects it
  onto the CAD axis, and rebuilds the needle mesh. Overlays: cyan = projected CAD
  axis, orange = detected edge, red = detected endpoint. Updates are rejected if
  extension ∉ 0–40 mm, stereo-ray miss > 2.5 mm, or tip > 6 mm off-axis. Filtering:
  single mm-domain blend (75 % new / 25 % previous); image-tip smoothing disabled.
  **Pending user test:** deliberate retract→extend travel comparing raw vs rendered
  values; do not add a scale factor unless the *raw* range is wrong.
- `--test-render out.png`: headless single frame — use it to verify changes.
- Floor-texture orientation is verified (numpy row 0 → +y edge, col 0 → −x edge);
  don't "fix" it without re-running the render-and-detect check.
- Detailed milestone narrative: `docs/fable_handover.md`.

## Standard commands

```bat
:: live stereo view, LEDs at min (the standard bench command)
python Motordriver\stereo_apriltag_webcams.py --backend msmf --left-led min --right-led min

:: LED level while an MSMF stream runs (or before, but it resets at stream start)
python Motordriver\led_control.py --camera 0 --led min

:: pose snapshot (scale = measured 10.62 mm tag edge; avoid --calibrate, see above)
python Motordriver\stereo_probe_pose.py --tag-edge-mm 10.62 --json pose.json

:: five-panel digital twin initialized from reality (recommended invocation)
python digital_twin\twin_wasd_jog.py --pose-live --tag-edge-mm 10.62 --anchor-ids 1,2,3 --reference-id 2 --reverse-cad-camera-order

:: synthetic calibration regression test
python analysis_scripts\test_stereo_apriltag_calibration.py
```

## Stereo-vision conventions

- f ≈ 800 px, sub-pixel matching error Δd ≈ 0.25 px, ΔZ = Z²·Δd/(f·D). Keep these
  consistent with `analysis_scripts/`.
- Sheet/world frame: reference tag center = origin, z toward the probe; the twin's
  floor plane is the sheet plane.

## Open work (rough priority order)

1. **Needle-extension travel test** (needs the user at the bench): record raw vs
   rendered extension at a known retracted and extended position; only then decide
   about scale/offset corrections (raw wrong → Hough/stereo; rendered wrong →
   filtering).
2. Independent validation dataset (different tag sizes/positions) for the accepted
   camera calibration.
3. Supervisor's design questions: needle-axis offset Δy as a first-class
   optimization variable; syringe insertion length L in the multi-objective
   optimization (maximize usable reach with focus + both frustums + ΔZ tolerance).
4. If the scopes keep shifting in their bores: fixed mechanical reference tags +
   continuous camera-to-probe transform estimation instead of session-startup-only.
5. Needle↔floor collision guard in the twin (task-space limit mirroring the
   soft-limit guard); feed the live pose into the needle→red-dot servoing pipeline.
6. USB flakiness root cause — selective suspend is now disabled (AC+DC, High
   Performance plan), so cables/connectors are the prime suspects.

## Gotchas that cost hours — don't relearn them

- Setting the LED while idle does nothing lasting: **stream first** (MSMF), then set.
- cv2 DSHOW capture hides the device from DirectShow enumeration entirely (binds
  fail with NULL pointer); use MSMF for capture whenever control must coexist.
- The two views of a stereo capture must be near-simultaneous — interleave reads
  (already done in `capture_pair`); a moving scene between frames corrupts the scale.
- PowerShell 5.1 mangles embedded double quotes in `python -c` one-liners; write a
  temp .py file instead.
- `Get-PnpDevice -Class Camera -PresentOnly` errors (exit 1) when zero cameras are
  present — wrap in try/catch.
- Kinesis leaves a background process after its window closes; it may need
  `Stop-Process` before the .NET path connects (or just use raw APT).

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

## Probe-pose pipeline (state as of 2026-07-16)

Scene: exactly **three 36h11 AprilTags, ids 1–3, coplanar on one sheet** (measured
edge ≈ 2.61 mm — pending a ruler check). `stereo_probe_pose.py`:

1. Detects tags in both views (OpenCV aruco, sub-pixel corners).
2. Recovers the sheet layout from the reference tag's homography (tag-edge units, no
   manual measurements). Reference tag: id 2 (`--reference-id`).
3. solvePnP (IPPE + LM refine) per camera with assumed f = 800 px, 720×720, no
   distortion model.
4. Probe frame = mean viewing axis + line between the two camera centers; tip =
   their midpoint. Per-scope roll about the probe axis is reported.
5. **Metric scale** comes from a one-time `--calibrate` run (moves one stage axis a
   known 2 mm, measures the tip displacement in tag units; saved to
   `tag_sheet_calibration.json`, auto-loaded afterwards). The alternative scale from
   the known 5.178 mm camera separation is **ill-conditioned at working distance —
   do not rely on it**; it is printed only as a cross-check.

Current measured state (changes whenever someone touches the setup — remeasure, do
not trust these numbers): probe ⊥ sheet within ~3°, tip ~34.5 mm above the sheet,
scope rolls ≈ −103° / +77° (relative twist ≈ 180°: one scope is mounted half a turn
from the other; fine for stereo after rotating one image).

Validated: reprojection ~0.3 px; the twin's rendered scope views re-detect the
correct tag ids at the correct angles. **Known limitation:** the 120° lens's fisheye
distortion is not calibrated; it biases the reconstructed camera separation (~2 mm
vs true 5.18 mm). Rotations and motion-calibrated distances are good; for sub-mm
absolute accuracy do a proper intrinsics calibration (small checkerboard at
5–50 mm) — top of the wishlist.

## Digital twin

`digital_twin/twin_wasd_jog.py` mirrors the real controls (same keys, same soft
limits from `kcube_axes.json`, same velocity scaling) and renders external +
left/right endoscope views (correct 120° diagonal FOV) plus the needle out of the
syringe channel (`--needle-extension-mm`, default 10; 0 hides it).

- `--pose-live`: capture a stereo pair at startup, run the pose pipeline, initialize
  the scene from it (probe pose, real tag layout as floor texture, per-scope camera
  rolls). Writes `digital_twin/last_pose.json`. Nothing is hard-coded.
- `--pose-json file`: same from a saved pose file.
- `--test-render out.png`: headless single frame — use it to verify changes.
- Texture orientation on the floor plane is verified: numpy row 0 → +y edge,
  col 0 → −x edge. Don't "fix" it without re-running the render-and-detect check.

## Standard commands

```bat
:: live stereo view, LEDs at min (the standard bench command)
python Motordriver\stereo_apriltag_webcams.py --backend msmf --left-led min --right-led min

:: LED level while an MSMF stream runs (or before, but it resets at stream start)
python Motordriver\led_control.py --camera 0 --led min

:: one-time metric-scale calibration (moves axis1 +2 mm and back)
python Motordriver\stereo_probe_pose.py --calibrate --json pose.json

:: pose snapshot using the saved calibration
python Motordriver\stereo_probe_pose.py --json pose.json

:: digital twin initialized from reality
python digital_twin\twin_wasd_jog.py --pose-live
```

## Stereo-vision conventions

- f ≈ 800 px, sub-pixel matching error Δd ≈ 0.25 px, ΔZ = Z²·Δd/(f·D). Keep these
  consistent with `analysis_scripts/`.
- Sheet/world frame: reference tag center = origin, z toward the probe; the twin's
  floor plane is the sheet plane.

## Open work (rough priority order)

1. **Lens calibration** (fisheye/pinhole intrinsics per scope) — unlocks accurate
   absolute pose and lets the 5.18 mm separation act as a true cross-check.
2. Ruler check of the tag edge (expected ≈ 2.61 mm) → validates/corrects f.
3. Supervisor's design questions: needle-axis offset Δy as a first-class
   optimization variable; syringe insertion length L in the multi-objective
   optimization (maximize usable reach with focus + both frustums + ΔZ tolerance).
4. Needle↔floor collision guard in the twin (task-space limit mirroring the
   soft-limit guard).
5. Live pose streaming into the twin (currently snapshot-at-startup) and feeding the
   pose into the needle→red-dot servoing pipeline.
6. USB flakiness root cause (cables vs selective suspend).

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

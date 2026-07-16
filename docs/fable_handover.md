# Handover to Fable: stereo endoscope digital twin

Date: 2026-07-16

## Purpose of the project

This University of Twente master project develops an 8 mm minimally invasive
probe containing two Comedia muC112 endoscopes, a 4 mm working channel, and a
needle/syringe channel. Three Thorlabs stages position the probe. The present
work focuses on camera calibration, AprilTag-based live pose estimation, and a
five-panel digital twin that also estimates the visible needle length.

## Non-negotiable constraints

- The lab PC runs Windows 10 and Python 3.8. New code must remain Python 3.8
  compatible.
- `D` means the endoscope center-to-center distance. Use that name in new code,
  plots, and documentation.
- Both endoscope LEDs must run at `min`. The camera firmware resets them to
  maximum whenever a stream starts, so start the MSMF streams first and then
  apply the minimum setting.
- Close the Thorlabs Kinesis GUI before controlling real motors.
- Real motor travel is limited to 0-24 mm per axis, with a 0.05 mm guard. The
  five-panel twin itself simulates the axes and does not command the motors.

## Physical geometry and measurements

- CAD file: `Holderdesign/cad/cad/Array_side_slip.STL`.
- Printed probe cross-section: 8 mm by 8 mm; model tip face is at model `y=0`.
- Endoscope bore centers: `(1.411, 4.432)` and `(6.589, 4.432)` model-mm.
- Mechanical `D`: **5.178 mm**.
- Syringe/needle face position: `(4.0, 6.0)` model-mm.
- AprilTags: 36h11, IDs 1, 2, and 3, black on white.
- Measured tag edge: **10.62 mm**, measured across the outside of the black
  border.
- Measured white gap between adjacent tag edges: **2.14 mm**. The tags remain a
  rigid group, but their absolute position is not stored.
- The user reported that one endoscope protrudes about 0.5-1.5 mm from the probe
  and the other about 1-2 mm. The cameras can rotate or shift axially slightly
  when touched.

## Accepted stereo camera calibration

The accepted calibration is:

`Motordriver/stereo_camera_calibration.json`

It was fit from the stationary dataset in
`Motordriver/stereo_calibration_data_static/manifest.json` using a five-term
pinhole model, 26 training captures, and 78 tag observations.

Important result values:

| Quantity | Value |
|---|---:|
| Left training RMS | 0.185 px |
| Right training RMS | 0.244 px |
| Joint stereo RMS | 0.611 px |
| Fitted lateral `D` | 5.282 mm |
| Mechanical `D` | 5.178 mm |
| Fitted axial offset | 0.036 mm |
| Held-out left reprojection mean | 0.165 px |
| Held-out right reprojection mean | 0.326 px |

The calibration quality gate is accepted. The fitted `D` differs from the CAD
value by about 0.10 mm.

Do not use `Motordriver/tag_sheet_calibration.json` for tag size. It currently
contains an invalid motion-derived value of 6.8346 mm. Continue passing
`--tag-edge-mm 10.62` explicitly.

The capture and calibration workflow is documented in:

`docs/apriltag_camera_calibration.md`

## Current five-panel digital twin

Main program:

`digital_twin/twin_wasd_jog.py`

The window contains:

1. External digital-twin view.
2. Virtual left-endoscope view.
3. Virtual right-endoscope view.
4. Live left-endoscope frame with detections.
5. Live right-endoscope frame with detections.

At startup, IDs 1-3 define a session tag-sheet layout. The layout is learned
from the first stereo pair rather than loaded from fixed world coordinates.
Reference ID 2 defines the origin. Subsequent observations update the sheet and
camera orientations at 5 Hz.

The physical camera-to-needle orientation currently matches the twin when the
program starts with `--reverse-cad-camera-order`. Pressing `C` toggles this
mapping. The reversed state is the recommended current state.

The left virtual-bore panel showed a strange artifact once. A later diagnostic
run did not reproduce a persistent failure, and the user confirmed it was a
one-time event. Do not change the left-view mapping unless the fault recurs.

## Live needle endpoint and extension estimation

The implementation reuses the classical computer-vision detector in:

`Motordriver/needle_detector.py`

The live pipeline is:

1. Project the known CAD needle axis into each calibrated camera image.
2. Use that projected line to restrict the existing Canny/Hough needle-edge
   detector and reject AprilTag/card edges.
3. Select the longest continuous in-band metal edge in each image.
4. Mark the two detected image endpoints.
5. Undistort both endpoints and triangulate their two camera rays in the tag
   sheet frame.
6. Project the reconstructed 3D tip onto the CAD needle axis.
7. Rebuild the virtual needle mesh with the estimated protrusion.

Live overlay meanings:

- Cyan line: projected CAD needle axis used as the search guide.
- Orange line: detected physical needle edge/centerline.
- Red marker: detected needle endpoint.

The digital twin rejects an update when:

- the extension is outside 0-40 mm;
- the closest distance between the stereo rays exceeds 2.5 mm; or
- the reconstructed tip is more than 6 mm from the CAD needle axis.

The external readout displays both the newly measured `raw` extension and the
filtered/rendered extension. A verified static example produced 13.15 mm
extension, 0.13 mm stereo-ray disagreement, and 0.68 mm CAD-axis offset.

The user then reported that the virtual needle did not retract and extend far
enough during motion. The cause was likely two consecutive smoothing stages.
The current code therefore:

- disables image-tip smoothing (`ema_alpha=1.0`); and
- applies only one millimetre-domain filter using 75% new measurement and 25%
  previous rendered value.

This change passed a live smoke test. It still needs a deliberate physical
two-endpoint travel test by the user. Do not add a scale factor until raw
measurements at known retracted and extended positions demonstrate a real
scale error. If the raw number covers the correct range but the mesh does not,
the problem is rendering/filtering. If the raw range is also too small, inspect
the Hough endpoint and stereo geometry.

## Commands to resume work

Open PowerShell in the project:

```powershell
cd "C:\Users\Labuser\OneDrive - University of Twente\Masterproject"
```

Start the current live five-panel twin:

```powershell
python digital_twin\twin_wasd_jog.py `
  --pose-live `
  --tag-edge-mm 10.62 `
  --anchor-ids 1,2,3 `
  --reference-id 2 `
  --reverse-cad-camera-order
```

Create a one-frame diagnostic render without opening the interactive loop:

```powershell
python digital_twin\twin_wasd_jog.py `
  --pose-live `
  --tag-edge-mm 10.62 `
  --anchor-ids 1,2,3 `
  --reference-id 2 `
  --reverse-cad-camera-order `
  --test-render digital_twin\diagnostic.png
```

Run the ordinary stereo AprilTag preview with both LEDs at minimum:

```powershell
python Motordriver\stereo_apriltag_webcams.py `
  --backend msmf `
  --left-led min `
  --right-led min
```

Re-run the synthetic stereo calibration regression test:

```powershell
python analysis_scripts\test_stereo_apriltag_calibration.py
```

The last successful regression result was:

- synthetic RMS: left 0.0203 px, right 0.0209 px, joint 0.1130 px;
- expected `D`: 5.1783 mm;
- recovered `D`: 5.1377 mm.

## Camera and USB troubleshooting

- Use MSMF capture when LED control must coexist with streaming.
- If a camera hangs during binding or Windows reports code 43, physically
  replug it. A software restart is usually insufficient.
- Replugging can swap camera indices 0 and 1. Verify which physical camera is
  shown in each live panel before trusting direction labels.
- USB selective suspend is currently disabled for AC and DC on the active High
  Performance power plan.
- Camera brightness and contrast persist across replugs because Windows caches
  them per device.

## Files changed or added during this milestone

- `Motordriver/stereo_camera_calibration.json`: accepted pinhole calibration.
- `Motordriver/stereo_probe_pose.py`: calibrated camera poses and per-camera
  transforms returned to the twin.
- `Motordriver/apriltag_tracker.py`: more robust AprilTag detection.
- `Motordriver/needle_detector.py`: OpenCV 4/5 Hough output compatibility,
  projected-line restriction, and continuous-edge selection.
- `digital_twin/twin_wasd_jog.py`: five-panel live view, session tag layout,
  camera rotations, reversed CAD mapping, stereo needle triangulation, quality
  gates, and dynamic needle geometry.
- `analysis_scripts/test_needle_detector_variants.py`: offline detector tuning
  against the last saved stereo frames.
- `docs/apriltag_camera_calibration.md`: current calibration workflow.
- `docs/apriltags_10_62mm/`: print-ready AprilTags IDs 1-5.

## Recommended next actions

1. With the five-panel twin running, hold the needle fully retracted and record
   both `raw` and rendered extension after they settle.
2. Extend the needle by a measured physical distance, hold still, and record the
   same two values.
3. Compare the measured physical change with the raw reconstructed change. Only
   then decide whether a linear scale/offset calibration is necessary.
4. Capture an independent validation dataset using other measured tag sizes and
   locations.
5. If the cameras keep moving in the probe, add mechanically fixed reference
   tags and estimate camera-to-probe transforms continuously rather than
   treating each session startup as permanent.
6. Continue investigating the recurring USB enumeration failures, with cables
   and connectors now more likely suspects because selective suspend is off.

## Most important warning for the next session

Do not silently overwrite the accepted camera calibration or use the invalid
6.8346 mm motion-derived tag scale. Keep the measured 10.62 mm tag edge, the
2.14 mm gap, mechanical `D=5.178 mm`, MSMF streaming, and both LEDs at `min`
unless a new measured experiment justifies a change.

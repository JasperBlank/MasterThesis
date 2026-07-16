# Stereo endoscope calibration with AprilTags

This workflow calibrates both endoscopes independently and estimates their
rigid stereo transform. The measured tag edge is the outside edge of the black
border. The current target has a measured edge of 10.62 mm.

The intrinsics fit treats every detected AprilTag as an independent metric
square. The stereo fit uses at least two tags fixed to one stiff, flat target;
their spacing is recovered from the images, so it does not need to be measured.

## 1. Capture a stationary calibration dataset

Use a new output directory so rejected or moving captures are not mixed into
the new dataset:

```bat
python Motordriver\capture_stereo_apriltag_calibration.py ^
  --output-dir Motordriver\stereo_calibration_data_static ^
  --default-tag-edge-mm 10.62
```

The program uses MSMF and sets both tip LEDs to `min`. Move the target to a new
pose, then stop completely. Wait until the preview says `STABLE`, press Space or
S while continuing to hold the target still, and release it only after the pair
has been saved. Q or Escape exits.

Capture 25–40 pairs. Keep at least two rigidly fixed tags visible in both
cameras for every saved pair. Between pairs, vary all of the following:

- position: center, four edges, and four corners of both images;
- out-of-plane tilt: roughly 20–40° about both target axes;
- roll: several rotations about the viewing direction;
- distance/apparent size, concentrated around the intended working range.

Raising the target while tilting it is useful and its exact height is not
needed. Rotation between captures is intended; motion during a saved left/right
pair is not. Avoid glare over tag corners, bent paper, and partly cropped tags.

For mixed tag sizes, pass measured per-ID overrides, for example:

```bat
python Motordriver\capture_stereo_apriltag_calibration.py ^
  --output-dir Motordriver\stereo_validation_data ^
  --default-tag-edge-mm 10.62 ^
  --tag-edge 4=6.00 ^
  --tag-edge 5=15.00
```

Replace the example 6.00 and 15.00 mm values with measured outer-black-border
edges.

## 2. Solve the calibration

The five-coefficient pinhole model is the stable default for the present data.
The initial 450 px focal estimate is optimized during calibration.

```bat
python analysis_scripts\calibrate_stereo_apriltags.py ^
  --manifest Motordriver\stereo_calibration_data_static\manifest.json ^
  --model pinhole ^
  --initial-focal-px 450 ^
  --tag-gap-mm 2.14 ^
  --joint-refine-intrinsics
```

The measured 2.14 mm value is the white edge-to-edge gap between adjacent
10.62 mm tags, giving 12.76 mm center spacing. Joint refinement removes the
focal-length/axial-translation ambiguity after the separate per-camera fit.

The result is written to `Motordriver/stereo_camera_calibration.json`, together
with undistorted previews. The quality gate reports `ACCEPTED` only when held-out
intrinsics error is below 1 px, the joint stereo error is below 1 px, lateral D
is within 1 mm of 5.178 mm, and the estimated axial camera offset is compatible
with the reported approximately 2 mm allowance.

The output reports three separate translation quantities:

- lateral D, expected near 5.178 mm;
- axial offset, allowed to differ by approximately 2 mm;
- full 3D camera-center separation, up to about 5.55 mm for a 2 mm offset.

A rejected JSON is retained for diagnosis but is not auto-loaded by the probe
pose script.

## 3. Independent validation with other tag sizes

Capture a second stationary dataset after moving the tags to new positions and,
if available, use different measured tag sizes. Then run:

```bat
python analysis_scripts\calibrate_stereo_apriltags.py ^
  --manifest Motordriver\stereo_calibration_data_static\manifest.json ^
  --validation-manifest Motordriver\stereo_validation_data\manifest.json ^
  --model pinhole ^
  --initial-focal-px 450 ^
  --tag-gap-mm 2.14 ^
  --joint-refine-intrinsics
```

Different tag sizes should give similarly low held-out reprojection errors
without retuning the model.

## 4. Use an accepted calibration

Probe pose with the measured tag edge:

```bat
python Motordriver\stereo_probe_pose.py ^
  --tag-edge-mm 10.62 ^
  --camera-calibration Motordriver\stereo_camera_calibration.json ^
  --json pose_calibrated.json
```

Digital twin initialized from the calibrated live pose:

```bat
python digital_twin\twin_wasd_jog.py ^
  --pose-live ^
  --tag-edge-mm 10.62 ^
  --camera-calibration Motordriver\stereo_camera_calibration.json
```

Always pass `--tag-edge-mm 10.62` until the invalid motion-derived scale file is
replaced. It must not be used as the physical tag size.

# AprilGrid Blender Workflow

This folder contains a repeatable synthetic-render workflow for the AprilGrid image at:

`C:\Users\jjbla\Downloads\april_grid.png`

The render script treats the PNG as a textured 24 in x 24 in planar board and renders deterministic camera views in Blender. The detector script then verifies AprilTag `36h11` detections on every rendered image.

## Run

From PowerShell:

```powershell
.\aprilgrid_blender\run_pipeline.ps1
```

Or run the steps separately:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python .\aprilgrid_blender\render_aprilgrid_dataset.py
python .\aprilgrid_blender\detect_rendered_aprilgrid.py
```

## Outputs

Rendered images and metadata are written to:

`C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender\renders`

Key files:

- `aprilgrid_render_###.png`: synthetic Blender renders.
- `camera_poses.csv`: ground-truth Blender camera pose for each render.
- `render_config.json`: board size, resolution, synthetic intrinsics, and coordinate notes.
- `annotated\aprilgrid_render_###.png`: rendered images with detected tag outlines and IDs.
- `detections.csv`: detected tag corner pixels per rendered image.
- `detection_summary.json`: per-image detection counts and per-tag detection counts.

## Current Configuration

- AprilTag family: `36h11`
- Grid: `6x6`, IDs `0-35`
- Board size: `0.6096 m` square, matching the PNG text `24in x 24in`
- Render resolution: `960x720`
- Synthetic focal length: `35 mm`
- Synthetic sensor width: `32 mm`
- Intrinsics in pixels: `fx=1050`, `fy=1050`, `cx=480`, `cy=360`

The source PNG appears to be a screenshot-like image with gray margins and UI text around the board. For a cleaner synthetic target, replace it with a pure AprilGrid texture or crop the target region before rendering.

## Degradation Matrix

The systematic robustness workflow is:

```powershell
.\aprilgrid_blender\run_degradation_matrix.ps1
```

It runs:

1. `generate_degradation_textures.py`
2. `render_degradation_matrix.py`
3. `postprocess_degradation_matrix.py`
4. `detect_degradation_matrix.py`

Outputs are written to:

`C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender\degradation_matrix`

Key files:

- `textures\`: clean AprilGrid textures with controlled tag size, spacing, and visible-tag count.
- `images\`: rendered and post-processed degradation cases.
- `annotated\`: every case annotated with detected tag outlines and IDs.
- `cases_all.csv`: complete matrix metadata, including factor, level, camera pose, intrinsics, and texture metadata.
- `detection_summary_by_case.csv`: one row per case with detected count, expected count, and detection rate.
- `detections_by_corner.csv`: detected corner pixels for every tag in every case.
- `detection_summary_by_factor.json`: aggregate factor-level results.
- `degradation_summary.png`: visual summary plot.
- `degradation_report.md`: compact Markdown report.

Current one-factor-at-a-time factors:

- Resolution: `960px`, `400px`, `300px`, `200px`, `150px`, `100px`
- Distance: `0.45m`, `0.60m`, `0.80m`, `1.00m`, `1.25m`, `1.60m`, `2.00m`
- Viewing angle: `0deg`, `15deg`, `30deg`, `45deg`, `60deg`, `70deg`
- Motion blur: `0px`, `3px`, `7px`, `11px`, `15px`, `21px`, `31px`
- Lighting/exposure: `0.20x`, `0.35x`, `0.50x`, `0.75x`, `1.00x`, `1.50x`, `2.00x`
- Partial occlusion: `0pct`, `10pct`, `20pct`, `35pct`, `50pct`, `65pct`
- Visible tags: `36`, `24`, `18`, `12`, `6`, `3`, `1`
- Tag size: `30mm`, `45mm`, `60mm`, `75mm`
- Spacing: `0mm`, `10mm`, `25mm`, `40mm`

This is intentionally not a full factorial experiment. It isolates each factor around a baseline so failures are easier to attribute. Once thresholds are clear, add a second-stage combined stress matrix for the most relevant conditions.

## Distance vs Resolution Tradeoff

To test whether moving the camera closer recovers detections at lower image resolutions:

```powershell
python .\aprilgrid_blender\analyze_distance_resolution_tradeoff.py
```

Outputs are written to:

`C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender\degradation_matrix\distance_resolution`

Key files:

- `distance_resolution_summary.csv`: detected count for every distance-width combination.
- `distance_resolution_summary.json`: same result in structured form.
- `distance_resolution_heatmap.png`: detected tag count heatmap.
- `distance_resolution_recovery.png`: recovery rate compared with full resolution at the same distance.
- `annotated\`: annotated resized images for visual inspection.

This experiment shows the tradeoff clearly: moving closer makes each tag larger and helps low-resolution detection, but too-close views crop the board and reduce the maximum number of recoverable tags.

## Distance / Tag-Size Ratio

To test whether the relevant variable is camera distance divided by AprilTag side length:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python .\aprilgrid_blender\render_ratio_scale_test.py
python .\aprilgrid_blender\analyze_tag_distance_ratio.py
```

Outputs are written to:

`C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender\degradation_matrix\ratio_scale`

Key files:

- `ratio_scale_cases.csv`: rendered scale/ratio cases.
- `ratio_scale_detection_summary.csv`: detection counts for every scale, ratio, and resolution.
- `ratio_scale_summary.json`: aggregate result and scale-invariance image-diff measurements.
- `ratio_heatmap_mean_detected.png`: detected count by image width and distance/tag-size ratio.
- `ratio_heatmap_min_detected.png`: worst-case detected count across tested scale factors.
- `scale_invariance_image_diff.png`: rendered image difference between the `1x` and `10x` physical target scale.

The current controlled test uses target scale factors `0.5x`, `1x`, `2x`, and `10x`. The board size, tag size, spacing, and camera distance are all scaled together, while camera intrinsics stay fixed. Detection counts are identical across all scale factors for every tested ratio and resolution, so the experiment supports using `distance / tag side length` as the normalized distance variable when the entire target geometry scales uniformly.

# AprilGrid Degradation Matrix Report

Cases tested: 55

## Factor Summary

| Factor | Cases | Min detected | Mean detected | Min rate |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1 | 36 | 36.00 | 1.00 |
| distance | 7 | 8 | 28.57 | 0.22 |
| lighting | 7 | 30 | 35.14 | 0.83 |
| motion_blur | 7 | 0 | 20.00 | 0.00 |
| partial_occlusion | 6 | 12 | 26.00 | 0.33 |
| resolution | 6 | 0 | 13.67 | 0.00 |
| spacing | 4 | 0 | 27.00 | 0.00 |
| tag_size | 4 | 36 | 36.00 | 1.00 |
| viewing_angle | 6 | 6 | 28.83 | 0.17 |
| visible_tags | 7 | 1 | 14.29 | 1.00 |

## Hardest Cases

| Case | Factor | Level | Detected / Expected |
| --- | --- | --- | ---: |
| render_031_spacing_0mm | spacing | 0mm | 0 / 36 |
| post_003_resolution_200px | resolution | 200px | 0 / 36 |
| post_004_resolution_150px | resolution | 150px | 0 / 36 |
| post_005_resolution_100px | resolution | 100px | 0 / 36 |
| post_010_motion_blur_15px | motion_blur | 15px | 0 / 36 |
| post_011_motion_blur_21px | motion_blur | 21px | 0 / 36 |
| post_012_motion_blur_31px | motion_blur | 31px | 0 / 36 |
| render_013_viewing_angle_70deg | viewing_angle | 70deg | 6 / 36 |

## Notes

- Resolution cases preserve aspect ratio by resizing the baseline render width.
- Motion blur is a horizontal linear kernel applied in image space.
- Lighting is an exposure multiplier applied in image space.
- Partial occlusion covers the right side of the board in Blender.
- Visible-tag cases use clean textures where only the center-ranked subset of tags is drawn.
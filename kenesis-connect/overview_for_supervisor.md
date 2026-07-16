# KCube Ethernet Motion and AprilTag Centering Overview

This project contains two main parts:

1. AprilTag image detection, which looks at the camera image and measures where the tag is.
2. KCube motion control, which moves the Thorlabs motorized axes over Ethernet.

The long-term goal is to combine these into a closed-loop centering system: the camera sees AprilTag ID 2, the software measures how far it is from the image center, and the motorized axes move to bring the tag back to the center.

## AprilTag Detection

The AprilTag detection code is in `apriltag_tracker.py`.

It uses OpenCV's built-in AprilTag support through the `cv2.aruco` module. No extra AprilTag package was needed on this machine because the installed OpenCV version already includes AprilTag dictionaries such as `36h11`.

The tracker opens a camera stream, detects AprilTags in each frame, and reports:

- the AprilTag ID,
- the tag center in image pixel coordinates,
- the error between the tag center and the image center,
- the apparent 2D rotation angle of the tag in the image.

The image error is reported as:

```text
error_x = tag_center_x - image_center_x
error_y = tag_center_y - image_center_y
```

So a positive `error_x` means the tag is to the right of image center, and a positive `error_y` means the tag is below image center.

The code also estimates the tag angle from the ordered AprilTag corners. This is useful because the endoscope may be rotated during production. If the camera image is rotated, the raw image x/y error is no longer aligned with the physical stage axes. The tag angle gives a way to compensate for that roll before converting image error into motor motion.

The tracker displays a live preview window showing:

- the camera image,
- the image center cross,
- the detected tag outline,
- the detected tag center,
- the tag ID,
- the pixel error,
- the tag angle.

By default, the later centering code tracks AprilTag ID 2.

## Motion Control

The KCube motion code is split into two files:

- `kcube_motion.py`: reusable motion-control library,
- `kcube_raw_ethernet.py`: command-line interface for manual testing and debugging.

The motors are Thorlabs KDC101 DC servo controllers connected through a KEH/KCube Ethernet hub. The hub is at:

```text
192.168.0.200
```

The controllers are reached through separate TCP ports:

```text
axis1 -> serial 27271413 -> 192.168.0.200:40307
axis2 -> serial 27271464 -> 192.168.0.200:40308
axis3 -> serial 27271523 -> 192.168.0.200:40309
hub   -> serial 120000166 -> 192.168.0.200:40303
```

The axis names and software limits are stored in `kcube_axes.json`.

Current software limits are:

```text
axis1: 0 to 20 mm
axis2: 0 to 20 mm
axis3: 0 to 20 mm
```

The software talks directly to the Ethernet TCP endpoints using the Thorlabs APT-style binary command messages. This was necessary because the normal Kinesis device-manager API did not reliably enumerate the Ethernet-connected KCubes outside the Kinesis GUI on this machine.

The motion library supports:

- reading hardware information,
- reading current position,
- moving by a relative distance in millimeters,
- moving to an absolute position in millimeters,
- stopping an axis,
- zeroing one or all axes,
- moving multiple axes with near-simultaneous starts.

Important operational note: the Kinesis GUI must be closed while Python controls the motors. Kinesis opens the same TCP connections and prevents the Python code from connecting.

Manual examples:

```powershell
python kcube_raw_ethernet.py axes
python kcube_raw_ethernet.py status-all
python kcube_raw_ethernet.py move-to axis1=10 axis2=10 axis3=10
python kcube_raw_ethernet.py move-by axis1=0.5 axis2=0.5
python kcube_raw_ethernet.py zero-all
```

For multi-axis moves, the code first checks all requested final positions against the soft limits. If any move would be unsafe, none of the axes are moved. If all moves are safe, commands are sent to all requested axes first, and then the code waits for their final positions.

## How Detection and Motion Come Together

The bridge between image detection and motion is `center_tag.py`.

This script tracks AprilTag ID 2, measures the pixel error from the image center, compensates the error using the detected tag angle, and converts the corrected pixel error into motor corrections.

The current physical convention is:

```text
positive axis1 movement makes the tag move down in the image
positive axis2 movement makes the tag move right in the image
```

Therefore:

- vertical image error is corrected with `axis1`,
- horizontal image error is corrected with `axis2`,
- the motor command uses negative feedback, so the tag moves toward the image center.

The centering loop uses a PI controller:

```text
motor_step = -(P * corrected_error + I * accumulated_error)
```

Current default controller settings are:

```text
P vertical   = 0.1 mm/pixel
P horizontal = 0.1 mm/pixel
I vertical   = P vertical / 100 = 0.001
I horizontal = P horizontal / 100 = 0.001
max step     = 5.0 mm per correction cycle
tolerance    = 5 pixels
```

The integral term is reset when:

- the tag is lost,
- the tag is already within the pixel tolerance.

This avoids accumulating old error when the visual target is unavailable or already centered.

By default, `center_tag.py` runs in dry-run mode. In dry-run mode it opens the camera preview and prints the motor corrections it would make, but it does not move the motors.

Dry-run example:

```powershell
python center_tag.py --camera 0
```

Live motor-control example:

```powershell
python center_tag.py --camera 0 --move
```

The live mode should only be used after confirming in dry-run mode that:

- AprilTag ID 2 is detected reliably,
- the reported angle is sensible,
- the proposed axis signs are correct,
- the proposed step sizes are reasonable,
- Kinesis is closed.

The intended workflow is:

1. Use `apriltag_tracker.py` to verify camera detection and tag angle.
2. Use `kcube_raw_ethernet.py` to verify safe manual motor movement.
3. Use `center_tag.py` in dry-run mode to verify proposed corrections.
4. Use `center_tag.py --move` for closed-loop centering.


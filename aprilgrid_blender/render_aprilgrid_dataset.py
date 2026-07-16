"""
Render a repeatable synthetic AprilGrid image dataset in Blender.

Run from PowerShell:
  & "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" --background --python aprilgrid_blender\\render_aprilgrid_dataset.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(r"C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender")
APRILGRID_IMAGE = Path(r"C:\Users\jjbla\Downloads\april_grid.png")
OUT_DIR = ROOT / "renders"

BOARD_SIZE_M = 0.6096  # 24 in x 24 in, matching the printed metadata in april_grid.png.
RESOLUTION_X = 960
RESOLUTION_Y = 720
CAMERA_LENS_MM = 35.0
CAMERA_SENSOR_WIDTH_MM = 32.0
RANDOM_SEED = 42


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_aprilgrid_board() -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    board = bpy.context.object
    board.name = "aprilgrid_board_24in"
    board.dimensions = (BOARD_SIZE_M, BOARD_SIZE_M, 0.0)
    bpy.context.view_layer.update()

    image = bpy.data.images.load(str(APRILGRID_IMAGE))
    image.colorspace_settings.name = "Non-Color"

    mat = bpy.data.materials.new("aprilgrid_emission_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = image
    tex.extension = "CLIP"

    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0

    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    board.data.materials.append(mat)

    return board


def create_camera() -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0.0, -0.2, 1.0))
    camera = bpy.context.object
    camera.name = "synthetic_camera"
    camera.data.lens = CAMERA_LENS_MM
    camera.data.sensor_width = CAMERA_SENSOR_WIDTH_MM
    camera.data.sensor_fit = "HORIZONTAL"
    bpy.context.scene.camera = camera
    return camera


def configure_scene() -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.72, 0.72, 0.72)


def get_intrinsics() -> dict[str, float]:
    fx = RESOLUTION_X * CAMERA_LENS_MM / CAMERA_SENSOR_WIDTH_MM
    fy = fx
    cx = RESOLUTION_X / 2.0
    cy = RESOLUTION_Y / 2.0
    return {"fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy}


def make_camera_poses() -> list[dict[str, float]]:
    """Return deterministic poses that vary distance, lateral offset, and obliqueness."""
    random.seed(RANDOM_SEED)

    poses: list[dict[str, float]] = []

    # A structured sweep keeps the dataset easy to reason about.
    sweep = [
        (-0.22, -0.18, 0.85),
        (-0.10, -0.10, 0.75),
        (0.00, -0.05, 0.70),
        (0.12, -0.04, 0.78),
        (0.24, -0.12, 0.90),
        (0.18, 0.10, 0.82),
        (0.04, 0.18, 0.76),
        (-0.16, 0.16, 0.88),
        (-0.28, 0.02, 1.00),
        (0.00, 0.00, 1.15),
    ]
    for i, (x, y, z) in enumerate(sweep):
        poses.append(
            {
                "frame": i,
                "camera_x_m": x,
                "camera_y_m": y,
                "camera_z_m": z,
                "target_x_m": 0.0,
                "target_y_m": 0.0,
                "target_z_m": 0.0,
                "label": "sweep",
            }
        )

    # Randomized-but-seeded views make robustness tests less overfit to the sweep.
    for i in range(10, 30):
        z = random.uniform(0.62, 1.25)
        x = random.uniform(-0.30, 0.30)
        y = random.uniform(-0.26, 0.26)
        target_x = random.uniform(-0.08, 0.08)
        target_y = random.uniform(-0.08, 0.08)
        poses.append(
            {
                "frame": i,
                "camera_x_m": x,
                "camera_y_m": y,
                "camera_z_m": z,
                "target_x_m": target_x,
                "target_y_m": target_y,
                "target_z_m": 0.0,
                "label": "seeded_random",
            }
        )

    return poses


def render_dataset() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reset_scene()
    configure_scene()
    create_aprilgrid_board()
    camera = create_camera()

    rows = []
    poses = make_camera_poses()
    intrinsics = get_intrinsics()

    for pose in poses:
        camera.location = (
            pose["camera_x_m"],
            pose["camera_y_m"],
            pose["camera_z_m"],
        )
        look_at(
            camera,
            Vector((pose["target_x_m"], pose["target_y_m"], pose["target_z_m"])),
        )
        bpy.context.view_layer.update()

        frame = int(pose["frame"])
        image_name = f"aprilgrid_render_{frame:03d}.png"
        bpy.context.scene.render.filepath = str(OUT_DIR / image_name)
        bpy.ops.render.render(write_still=True)

        rows.append(
            {
                **pose,
                "image": image_name,
                "camera_rot_x_rad": camera.rotation_euler.x,
                "camera_rot_y_rad": camera.rotation_euler.y,
                "camera_rot_z_rad": camera.rotation_euler.z,
                **intrinsics,
            }
        )

    with (OUT_DIR / "camera_poses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "aprilgrid_image": str(APRILGRID_IMAGE),
        "board_size_m": BOARD_SIZE_M,
        "resolution_x": RESOLUTION_X,
        "resolution_y": RESOLUTION_Y,
        "camera_lens_mm": CAMERA_LENS_MM,
        "camera_sensor_width_mm": CAMERA_SENSOR_WIDTH_MM,
        "random_seed": RANDOM_SEED,
        "intrinsics": intrinsics,
        "pose_count": len(poses),
        "coordinate_note": "Board lies on Blender XY plane at Z=0; camera looks along its local -Z axis.",
    }
    with (OUT_DIR / "render_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Rendered {len(poses)} images to {OUT_DIR}")


if __name__ == "__main__":
    render_dataset()

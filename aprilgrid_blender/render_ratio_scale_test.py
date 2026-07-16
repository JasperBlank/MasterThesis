"""
Render a pure scale-invariance experiment for AprilTag detection.

For a fixed camera model, a planar target should render identically when the
target's physical size and camera distance are scaled by the same factor.
This script renders that test for several tag-size scales and distance/tag-size
ratios.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(r"C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender")
TEXTURE_PATH = ROOT / "degradation_matrix" / "textures" / "baseline_t60_s25_v36.png"
OUT_DIR = ROOT / "degradation_matrix" / "ratio_scale"
IMAGE_DIR = OUT_DIR / "images"
CASES_CSV = OUT_DIR / "ratio_scale_cases.csv"

BASE_BOARD_SIZE_M = 0.6096
BASE_TAG_SIZE_M = 0.060
RESOLUTION_X = 960
RESOLUTION_Y = 720
CAMERA_LENS_MM = 35.0
CAMERA_SENSOR_WIDTH_MM = 32.0

SCALE_FACTORS = [0.5, 1.0, 2.0, 10.0]
RATIOS_DISTANCE_OVER_TAG = [7.5, 10.0, 13.333333, 15.0, 16.666667, 20.833333, 26.666667, 33.333333]


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


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
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.45, 0.45, 0.45)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("ratio_baseline_aprilgrid")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(TEXTURE_PATH))
    tex.image.colorspace_settings.name = "Non-Color"
    tex.extension = "CLIP"

    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0

    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def create_board(material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    board = bpy.context.object
    board.name = "ratio_scaled_board"
    board.data.materials.append(material)
    return board


def create_camera() -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0.0, 0.0, 1.0))
    camera = bpy.context.object
    camera.name = "ratio_camera"
    camera.data.lens = CAMERA_LENS_MM
    camera.data.sensor_width = CAMERA_SENSOR_WIDTH_MM
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.clip_start = 0.001
    camera.data.clip_end = 1000.0
    bpy.context.scene.camera = camera
    return camera


def intrinsics() -> dict[str, float]:
    fx = RESOLUTION_X * CAMERA_LENS_MM / CAMERA_SENSOR_WIDTH_MM
    return {
        "fx_px": fx,
        "fy_px": fx,
        "cx_px": RESOLUTION_X / 2.0,
        "cy_px": RESOLUTION_Y / 2.0,
    }


def render_cases() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    reset_scene()
    configure_scene()
    material = create_material()
    board = create_board(material)
    camera = create_camera()

    rows: list[dict[str, object]] = []

    for scale in SCALE_FACTORS:
        board_size_m = BASE_BOARD_SIZE_M * scale
        tag_size_m = BASE_TAG_SIZE_M * scale
        board.dimensions = (board_size_m, board_size_m, 0.0)
        bpy.context.view_layer.update()

        for ratio in RATIOS_DISTANCE_OVER_TAG:
            distance_m = ratio * tag_size_m
            camera.location = (0.0, 0.0, distance_m)
            look_at(camera, Vector((0.0, 0.0, 0.0)))
            bpy.context.view_layer.update()

            scale_label = f"{scale:g}".replace(".", "p")
            ratio_label = f"{ratio:.3f}".replace(".", "p")
            image_name = f"ratio_s{scale_label}_r{ratio_label}.png"
            bpy.context.scene.render.filepath = str(IMAGE_DIR / image_name)
            bpy.ops.render.render(write_still=True)

            rows.append(
                {
                    "image": image_name,
                    "scale_factor": scale,
                    "board_size_m": board_size_m,
                    "tag_size_m": tag_size_m,
                    "distance_m": distance_m,
                    "distance_over_tag_size": ratio,
                    "render_width_px": RESOLUTION_X,
                    "render_height_px": RESOLUTION_Y,
                    "camera_x_m": camera.location.x,
                    "camera_y_m": camera.location.y,
                    "camera_z_m": camera.location.z,
                    "camera_rot_x_rad": camera.rotation_euler.x,
                    "camera_rot_y_rad": camera.rotation_euler.y,
                    "camera_rot_z_rad": camera.rotation_euler.z,
                    **intrinsics(),
                }
            )

    with CASES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rendered {len(rows)} ratio-scale cases to {IMAGE_DIR}")
    print(f"Wrote {CASES_CSV}")


if __name__ == "__main__":
    render_cases()

"""
Render a one-factor-at-a-time degradation matrix for AprilGrid detection.

Run after generating textures:
  & "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" --background --python aprilgrid_blender\\render_degradation_matrix.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(r"C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender")
MATRIX_DIR = ROOT / "degradation_matrix"
IMAGE_DIR = MATRIX_DIR / "images"
TEXTURE_CONFIG = MATRIX_DIR / "textures" / "texture_config.json"
CASES_CSV = MATRIX_DIR / "cases_rendered.csv"

BOARD_SIZE_M = 0.6096
RESOLUTION_X = 960
RESOLUTION_Y = 720
CAMERA_LENS_MM = 35.0
CAMERA_SENSOR_WIDTH_MM = 32.0
BASE_TEXTURE = "baseline_t60_s25_v36"


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


def create_emission_material(name: str, color: tuple[float, float, float] | None = None) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    if color is not None:
        emission.inputs["Color"].default_value = (*color, 1.0)

    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def create_texture_material(texture_name: str, texture_path: str) -> bpy.types.Material:
    mat = create_emission_material(f"mat_{texture_name}")
    nodes = mat.node_tree.nodes
    emission = next(node for node in nodes if node.bl_idname == "ShaderNodeEmission")

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = bpy.data.images.load(texture_path)
    tex.image.colorspace_settings.name = "Non-Color"
    tex.extension = "CLIP"
    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    return mat


def create_board(materials: dict[str, bpy.types.Material]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    board = bpy.context.object
    board.name = "aprilgrid_board"
    board.dimensions = (BOARD_SIZE_M, BOARD_SIZE_M, 0.0)
    board.data.materials.append(materials[BASE_TEXTURE])
    bpy.context.view_layer.update()
    return board


def create_occluder(material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.002))
    occluder = bpy.context.object
    occluder.name = "occluder"
    occluder.data.materials.append(material)
    occluder.hide_render = True
    return occluder


def create_camera() -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.9))
    camera = bpy.context.object
    camera.name = "matrix_camera"
    camera.data.lens = CAMERA_LENS_MM
    camera.data.sensor_width = CAMERA_SENSOR_WIDTH_MM
    camera.data.sensor_fit = "HORIZONTAL"
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


def base_case() -> dict[str, object]:
    return {
        "factor": "baseline",
        "level": "baseline",
        "texture_name": BASE_TEXTURE,
        "camera_x_m": 0.0,
        "camera_y_m": 0.0,
        "camera_z_m": 0.9,
        "target_x_m": 0.0,
        "target_y_m": 0.0,
        "target_z_m": 0.0,
        "view_angle_deg": 0.0,
        "distance_m": 0.9,
        "occlusion_fraction": 0.0,
    }


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = [base_case()]

    for z in [0.45, 0.60, 0.80, 1.00, 1.25, 1.60, 2.00]:
        case = base_case()
        case.update({"factor": "distance", "level": f"{z:.2f}m", "camera_z_m": z, "distance_m": z})
        cases.append(case)

    radius = 1.10
    for angle in [0, 15, 30, 45, 60, 70]:
        theta = math.radians(angle)
        case = base_case()
        case.update(
            {
                "factor": "viewing_angle",
                "level": f"{angle}deg",
                "camera_x_m": math.sin(theta) * radius,
                "camera_z_m": math.cos(theta) * radius,
                "view_angle_deg": float(angle),
                "distance_m": radius,
            }
        )
        cases.append(case)

    for fraction in [0.0, 0.10, 0.20, 0.35, 0.50, 0.65]:
        case = base_case()
        case.update(
            {
                "factor": "partial_occlusion",
                "level": f"{int(round(fraction * 100))}pct",
                "occlusion_fraction": fraction,
            }
        )
        cases.append(case)

    for count in [36, 24, 18, 12, 6, 3, 1]:
        case = base_case()
        case.update({"factor": "visible_tags", "level": str(count), "texture_name": f"visible_{count:02d}"})
        cases.append(case)

    for tag_size in [30, 45, 60, 75]:
        case = base_case()
        case.update({"factor": "tag_size", "level": f"{tag_size}mm", "texture_name": f"tag_size_{tag_size:02d}mm"})
        cases.append(case)

    for spacing in [0, 10, 25, 40]:
        case = base_case()
        case.update({"factor": "spacing", "level": f"{spacing}mm", "texture_name": f"spacing_{spacing:02d}mm"})
        cases.append(case)

    return cases


def apply_occluder(occluder: bpy.types.Object, fraction: float) -> None:
    if fraction <= 0:
        occluder.hide_render = True
        return

    width = BOARD_SIZE_M * fraction
    occluder.hide_render = False
    occluder.location = (BOARD_SIZE_M / 2.0 - width / 2.0, 0.0, 0.002)
    occluder.dimensions = (width, BOARD_SIZE_M, 0.0)
    bpy.context.view_layer.update()


def render_cases() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with TEXTURE_CONFIG.open("r", encoding="utf-8") as f:
        texture_config = json.load(f)["textures"]

    reset_scene()
    configure_scene()

    materials = {
        name: create_texture_material(name, cfg["path"])
        for name, cfg in texture_config.items()
    }
    black_material = create_emission_material("black_occluder", (0.0, 0.0, 0.0))
    board = create_board(materials)
    occluder = create_occluder(black_material)
    camera = create_camera()

    rows: list[dict[str, object]] = []
    for index, case in enumerate(build_cases()):
        texture_name = str(case["texture_name"])
        board.data.materials.clear()
        board.data.materials.append(materials[texture_name])

        camera.location = (
            float(case["camera_x_m"]),
            float(case["camera_y_m"]),
            float(case["camera_z_m"]),
        )
        look_at(
            camera,
            Vector(
                (
                    float(case["target_x_m"]),
                    float(case["target_y_m"]),
                    float(case["target_z_m"]),
                )
            ),
        )
        apply_occluder(occluder, float(case["occlusion_fraction"]))
        bpy.context.view_layer.update()

        case_id = f"render_{index:03d}_{case['factor']}_{case['level']}".replace(".", "p")
        image_name = f"{case_id}.png"
        bpy.context.scene.render.filepath = str(IMAGE_DIR / image_name)
        bpy.ops.render.render(write_still=True)

        texture_meta = texture_config[texture_name]
        rows.append(
            {
                "case_id": case_id,
                "source_case_id": "",
                "image": image_name,
                "case_origin": "blender_render",
                "factor": case["factor"],
                "level": case["level"],
                "texture_name": texture_name,
                "expected_visible_tags": texture_meta["visible_count"],
                "tag_size_mm": texture_meta["tag_size_mm"],
                "spacing_mm": texture_meta["spacing_mm"],
                "render_width_px": RESOLUTION_X,
                "render_height_px": RESOLUTION_Y,
                "camera_x_m": case["camera_x_m"],
                "camera_y_m": case["camera_y_m"],
                "camera_z_m": case["camera_z_m"],
                "target_x_m": case["target_x_m"],
                "target_y_m": case["target_y_m"],
                "target_z_m": case["target_z_m"],
                "view_angle_deg": case["view_angle_deg"],
                "distance_m": case["distance_m"],
                "occlusion_fraction": case["occlusion_fraction"],
                "postprocess": "",
                **intrinsics(),
                "camera_rot_x_rad": camera.rotation_euler.x,
                "camera_rot_y_rad": camera.rotation_euler.y,
                "camera_rot_z_rad": camera.rotation_euler.z,
            }
        )

    with CASES_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rendered {len(rows)} matrix cases to {IMAGE_DIR}")
    print(f"Wrote {CASES_CSV}")


if __name__ == "__main__":
    render_cases()

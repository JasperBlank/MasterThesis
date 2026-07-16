"""Synthetic smoke test for calibrate_stereo_apriltags.py."""

from __future__ import print_function

import math

import cv2
import numpy as np

import calibrate_stereo_apriltags as calibration


def rotation_vector(rx_deg, ry_deg, rz_deg):
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    rotation_x = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    rotation_y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    rotation_z = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    vector, _ = cv2.Rodrigues(rotation)
    return vector.reshape(3), rotation


def main():
    rng = np.random.RandomState(7)
    image_size = (720, 720)
    left_matrix = np.array([[458.0, 0.0, 361.0], [0.0, 452.0, 357.0], [0.0, 0.0, 1.0]])
    right_matrix = np.array([[465.0, 0.0, 356.0], [0.0, 455.0, 363.0], [0.0, 0.0, 1.0]])
    left_distortion = np.array([-0.075, 0.012, -0.001, 0.0002, -0.001], dtype=float)
    right_distortion = np.array([-0.068, 0.010, -0.0008, 0.0001, -0.001], dtype=float)
    stereo_rvec, stereo_rotation = rotation_vector(0.4, -0.7, 0.3)
    del stereo_rvec
    stereo_translation = np.array([-5.178, 0.05, -0.03], dtype=float)

    observations = []
    tag_centers = {1: np.array([-14.0, 0.0, 0.0]), 2: np.zeros(3), 3: np.array([14.0, 0.0, 0.0])}
    for capture_id in range(45):
        object_rvec, object_rotation = rotation_vector(
            rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(-60, 60)
        )
        object_translation = np.array(
            [rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(65, 125)], dtype=float
        )
        for tag_id, tag_center in tag_centers.items():
            edge_mm = 8.0
            local_points = calibration.square_object_points(edge_mm)
            board_points = local_points + tag_center
            points_left = np.dot(object_rotation, board_points.T).T + object_translation
            points_right = np.dot(stereo_rotation, points_left.T).T + stereo_translation
            left_points, _ = cv2.projectPoints(
                points_left, np.zeros(3), np.zeros(3), left_matrix, left_distortion
            )
            right_points, _ = cv2.projectPoints(
                points_right, np.zeros(3), np.zeros(3), right_matrix, right_distortion
            )
            left_points = left_points.reshape(-1, 2) + rng.normal(0.0, 0.03, (4, 2))
            right_points = right_points.reshape(-1, 2) + rng.normal(0.0, 0.03, (4, 2))
            observations.append(
                {
                    "capture_id": capture_id,
                    "tag_id": tag_id,
                    "edge_mm": edge_mm,
                    "object_points": local_points,
                    "left_points": left_points,
                    "right_points": right_points,
                    "left_image": "",
                    "right_image": "",
                }
            )

    training = [item for item in observations if item["capture_id"] < 36]
    validation = [item for item in observations if item["capture_id"] >= 36]
    model = "pinhole"
    left_rms, estimated_left_matrix, estimated_left_distortion = calibration.calibrate_intrinsics(
        training, "left", image_size, model, 450.0
    )
    right_rms, estimated_right_matrix, estimated_right_distortion = calibration.calibrate_intrinsics(
        training, "right", image_size, model, 450.0
    )
    stereo_rms, estimated_rotation, estimated_translation, _ = calibration.calibrate_stereo(
        training,
        image_size,
        model,
        estimated_left_matrix,
        estimated_left_distortion,
        estimated_right_matrix,
        estimated_right_distortion,
    )
    validation_result = calibration.validate(
        validation,
        model,
        estimated_left_matrix,
        estimated_left_distortion,
        estimated_right_matrix,
        estimated_right_distortion,
        estimated_rotation,
    )
    estimated_D = float(np.linalg.norm(estimated_translation))
    print("synthetic RMS: left %.4f px, right %.4f px, stereo joint %.4f px" % (
        left_rms, right_rms, stereo_rms
    ))
    print("synthetic D: expected %.4f mm, recovered %.4f mm" % (
        float(np.linalg.norm(stereo_translation)), estimated_D
    ))
    print("validation reprojection: left %.4f, right %.4f px" % (
        validation_result["left_reprojection_px"]["mean"],
        validation_result["right_reprojection_px"]["mean"],
    ))
    if abs(estimated_D - float(np.linalg.norm(stereo_translation))) > 0.15:
        raise SystemExit("synthetic D recovery failed")


if __name__ == "__main__":
    main()

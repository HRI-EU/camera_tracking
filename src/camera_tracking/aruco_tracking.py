#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Class that offers tracking of Aruco markers.
#
#  Copyright (C)
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#

from __future__ import annotations

from collections import defaultdict
import json
import time
import cv2
import numpy
from scipy.spatial.transform import Rotation

from camera_tracking.camera_helper import load_camera_parameters, save_camera_parameters
from camera_tracking.base_tracking import BaseTracking


class ArucoTracking(BaseTracking):
    def __init__(
        self,
        camera_matrix: numpy.ndarray,
        distortion_coefficients: numpy.ndarray,
        visualize: bool = True,
        save_images: bool = False,
    ):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_parameters.cornerRefinementWinSize = 4
        self.aruco_parameters.minDistanceToBorder = 2
        self.aruco_parameters.minMarkerPerimeterRate = 0.03
        self.aruco_parameters.maxMarkerPerimeterRate = 0.3

        # We use a default length of 1m, and scale the markers on the receiving side.
        self.default_marker_length = 1.0
        self.object_points = numpy.array(
            [
                [-self.default_marker_length / 2, self.default_marker_length / 2, 0],
                [self.default_marker_length / 2, self.default_marker_length / 2, 0],
                [self.default_marker_length / 2, -self.default_marker_length / 2, 0],
                [-self.default_marker_length / 2, -self.default_marker_length / 2, 0],
            ],
        )

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.save_images = save_images

        if self.save_images:
            save_camera_parameters(self.camera_matrix, self.distortion_coefficients, "camera.yaml")

        super().__init__("aruco", visualize=visualize)

    def write_detector_parameters(self, filename: str) -> None:
        self.aruco_parameters.writeDetectorParameters(
            cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE),
            "aruco_parameters",
        )

    def process(self, data: numpy.ndarray) -> dict:
        """
        Process an image.
        @param data: The image to be processed. If the image is colored we assume BGR.
        @return: The found aruco landmarks.
        """

        if self.save_images:
            cv2.imwrite(f"{time.time()}.png", data)

        # Find all markers in the image.
        corners, ids, _ = cv2.aruco.detectMarkers(data, self.aruco_dict, parameters=self.aruco_parameters)

        if self.visualize:
            # Make a copy of the image, as the outside might assume that we do not alter the image.
            # In case of a grayscale image, convert it to BGR as this looks nicer.
            if len(data.shape) == 2:
                self.visualization = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
            else:
                self.visualization = data.copy()

            # Visualize the found markers.
            cv2.aruco.drawDetectedMarkers(self.visualization, corners, ids)

        landmarks = defaultdict(list)
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                # Estimate pose of the marker.

                # Solutions are sorted by reprojection error of SOLVEPNP_IPPE_SQUARE in undistorted image coordinates.
                # But the returned reprojection error is from solvePnPGeneric which is in distorted image coordinates.
                # Therefore, it can happen that the given reprojection error is not sorted (see data/log/aruco.txt)
                number, rotations, translations, reprojection_errors = cv2.solvePnPGeneric(
                    self.object_points,
                    marker_corners[0],
                    self.camera_matrix,
                    self.distortion_coefficients,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,  # default is iterative
                )

                if number != 2:
                    raise AssertionError(f"Expected two solutions but got {number}.")

                solutions = [
                    {
                        "position": dict(zip("xyz", translation.flatten())),
                        "orientation": dict(zip("xyzw", Rotation.from_rotvec(rotation.flatten()).as_quat())),
                        "reprojection_error": reprojection_error[0],
                    }
                    for translation, rotation, reprojection_error in zip(translations, rotations, reprojection_errors)
                ]

                # if abs(reprojection_errors[0] - reprojection_errors[1]) < 0.2:
                #    print(f"{marker_id}.\n   {solutions[0]}\n   {solutions[1]}")

                # Store information as list of 3 + 4 values.
                landmarks[f"aruco_{marker_id}"].append({**solutions[0], "solutions": solutions})

                # Draw the axis of the marker.
                if self.visualize:
                    cv2.drawFrameAxes(
                        self.visualization,
                        self.camera_matrix,
                        self.distortion_coefficients,
                        rotations[0],
                        translations[0],
                        0.40,
                        2,
                    )

        return landmarks


def main():
    camera_parameters = load_camera_parameters("data/calibration/Logitech-C920.yaml")
    image = cv2.imread("data/test/aruco_test.jpg")
    aruco_tracking = ArucoTracking(camera_parameters["camera_matrix"], camera_parameters["distortion_coefficients"])
    landmarks = aruco_tracking.process(image)
    print(f"Landmarks are:\n{landmarks}")

    # Store the found landmarks.
    with open("data/test/aruco_test_landmarks.json", "w", encoding="utf-8") as file:
        json.dump(landmarks, file)

    # Compare to reference landmarks.
    with open("data/test/aruco_test_reference_landmarks.json", encoding="utf-8") as file:
        reference_landmarks = json.load(file)
    if landmarks != reference_landmarks:
        print(f"Mismatch detected. Reference landmarks are:\n{reference_landmarks}")


if __name__ == "__main__":
    main()

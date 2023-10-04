#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (c) Honda Research Institute Europe GmbH.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
#  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
#  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
#  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from typing import Dict
from collections import defaultdict
import json
import time
import cv2
import numpy
from scipy.spatial.transform import Rotation

from camera_tracking.camera_helper import load_camera_parameters
from camera_tracking.base_tracking import BaseTracking


class ArucoTracking(BaseTracking):
    def __init__(self, camera_matrix: numpy.ndarray, distortion_coefficients: numpy.ndarray, visualize: bool = True):

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_parameters.cornerRefinementWinSize = 4
        self.aruco_parameters.minDistanceToBorder = 2
        self.aruco_parameters.writeDetectorParameters(
            cv2.FileStorage("aruco_parameters.txt", cv2.FILE_STORAGE_WRITE),
            "aruco_parameters",
        )

        # We use a default length of 1m, and scale the markers on the receiving side.
        self.default_marker_length = 1.0

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

        super().__init__("aruco", visualize=visualize)

    def process(self, image: numpy.ndarray) -> Dict:
        """
        Process an image.
        @param image: The image to be processed. If the image is colored we assume BGR.
        @return: The found aruco landmarks.
        """

        start_time = time.time()

        # Find all markers in the image.
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_parameters)

        if self.visualize:
            # Make a copy of the image, as the outside might assume that we do not alter the image.
            self.visualization = image.copy()
            # Visualize the found markers.
            cv2.aruco.drawDetectedMarkers(self.visualization, corners, ids)

        landmarks = defaultdict(list)
        if ids is not None:
            ids = ids.flatten()
            for marker_corners, marker_id in zip(corners, ids):
                # Estimate pose of the marker.
                rotation_vector, translation_vector, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corners, self.default_marker_length, self.camera_matrix, self.distortion_coefficients
                )

                # Store information as list of 3 + 4 values.
                landmarks[f"aruco_{marker_id}"].append(
                    {
                        "pose": [
                            *translation_vector.flatten(),
                            *Rotation.from_rotvec(rotation_vector.flatten()).as_quat(),
                        ]
                    }
                )

                # Draw the axis of the marker.
                if self.visualize:
                    cv2.drawFrameAxes(
                        self.visualization,
                        self.camera_matrix,
                        self.distortion_coefficients,
                        rotation_vector,
                        translation_vector,
                        0.30,
                    )

        self.sum_processing_time += time.time() - start_time

        return landmarks


def main():
    camera_parameters = load_camera_parameters("data/calibration/Logitech-C920.yaml")
    image = cv2.imread("data/test/aruco_test.jpg")
    aruco_tracking = ArucoTracking(camera_parameters["camera_matrix"], camera_parameters["distortion_coefficients"])
    landmarks = aruco_tracking.process(image)
    print(f"Landmarks are:\n{landmarks}")

    # Store the found landmarks.
    with open("data/test/aruco_test_landmarks.json", "w") as file:
        json.dump(landmarks, file)

    # Compare to reference landmarks.
    with open("data/test/aruco_test_reference_landmarks.json") as file:
        reference_landmarks = json.load(file)
    if landmarks != reference_landmarks:
        print(f"Mismatch detected. Reference landmarks are:\n{reference_landmarks}")


if __name__ == "__main__":
    main()

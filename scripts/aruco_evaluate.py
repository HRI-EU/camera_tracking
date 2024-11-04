#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Store aruco tracking result for a set of images.
#
#  Copyright (c) Honda Research Institute Europe GmbH
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

from __future__ import annotations
import glob
import os
import json
import cv2

from camera_tracking.aruco_tracking import ArucoTracking
from camera_tracking.camera_helper import load_camera_parameters


def main():
    load_path = "/hri/localdisk/stephanh/aruco_evaluation/data"
    save_path = "/hri/localdisk/stephanh/aruco_evaluation/result2"
    camera_parameters = load_camera_parameters(os.path.join(load_path, "camera.yaml"))
    aruco_tracking = ArucoTracking(
        camera_parameters["camera_matrix"],
        camera_parameters["distortion_coefficients"],
        visualize=True,
        with_tracking=True,
    )
    aruco_tracking.write_detector_parameters(os.path.join(save_path, "aruco_parameters.yaml"))

    all_landmarks = []
    filenames = glob.glob(os.path.join(load_path, "*.png"))
    filenames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    for filename in filenames:
        image = cv2.imread(filename)
        index = int(os.path.basename(filename).split("_")[0])
        print(f"{index}. {filename}")
        landmarks = aruco_tracking.process(image)
        print(landmarks)
        all_landmarks.append({"image_index": index, "data": landmarks})
        cv2.imwrite(os.path.join(save_path, f"{index}_visualization.png"), aruco_tracking.visualization)

    with open(os.path.join(save_path, "landmarks.json"), "w", encoding="utf-8") as file:
        json.dump(all_landmarks, file, indent=2)


if __name__ == "__main__":
    main()

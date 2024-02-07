#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Store aruco tracking result for a set of images.
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
import glob
import os
import json
import cv2

from camera_tracking.aruco_tracking import ArucoTracking
from camera_tracking.camera_helper import load_camera_parameters


def main():
    load_path = "/hri/localdisk/stephanh/aruco_evaluation/data"
    save_path = "/hri/localdisk/stephanh/aruco_evaluation/result1"
    camera_parameters = load_camera_parameters(os.path.join(load_path, "camera.yaml"))
    aruco_tracking = ArucoTracking(
        camera_parameters["camera_matrix"], camera_parameters["distortion_coefficients"], visualize=True
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
        all_landmarks.append({"image_index": index, "data": landmarks})
        cv2.imwrite(os.path.join(save_path, f"{index}_visualization.png"), aruco_tracking.visualization)

    with open(os.path.join(save_path, "landmarks.json"), "w", encoding="utf-8") as file:
        json.dump(all_landmarks, file, indent=2)


if __name__ == "__main__":
    main()

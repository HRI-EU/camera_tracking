#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Plot stored aruco result.
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
import math
import os
import json
import matplotlib.pyplot as plt

from camera_tracking.aruco_tracking import angle_between_quaternions


def main():
    result_path = "/hri/localdisk/stephanh/aruco_evaluation/result2"

    with open(os.path.join(result_path, "landmarks.json"), "r") as file:
        samples = json.load(file)

    marker_id = "aruco_40"
    result = defaultdict(list)

    last_landmarks = {}
    for sample in samples:
        print(f"{sample['image_index']}")
        result["index"].append(sample["image_index"])
        landmarks = sample["data"]
        print(landmarks)
        for landmark_id, data in landmarks.items():
            if len(data) != 1:
                print(f"{landmark_id} has multiple detections.")

        if marker_id in landmarks:
            if marker_id in last_landmarks:
                result["angle_0_to_last_0"].append(
                    angle_between_quaternions(
                        landmarks[marker_id][0]["orientation"],
                        last_landmarks[marker_id][0]["orientation"],
                    )
                )
            else:
                result["angle_0_to_last_0"].append(None)

            result["track_id"].append(landmarks[marker_id][0]["track_id"])

        else:
            result["angle_0_to_last_0"].append(None)
            result["track_id"].append(None)

        last_landmarks = landmarks

    print(result["angle_0_to_last_0"])
    print(result["track_id"])

    plt.figure(1, figsize=(25, 10))
    plt.plot(result["index"], result["angle_0_to_last_0"], "py")

    plt.show()


if __name__ == "__main__":
    main()

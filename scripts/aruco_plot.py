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


def get_angle(quaternion_1: dict, quaternion_2: dict) -> float:
    delta_w = (
        quaternion_1["x"] * quaternion_2["x"]
        + quaternion_1["y"] * quaternion_2["y"]
        + quaternion_1["z"] * quaternion_2["z"]
        + quaternion_1["w"] * quaternion_2["w"]
    )
    return 2 * math.acos(abs(min(delta_w, 1.0)))


def main():
    result_path = "/hri/localdisk/stephanh/aruco_evaluation/result1"

    with open(os.path.join(result_path, "landmarks.json"), "r") as file:
        samples = json.load(file)

    marker_id = "aruco_40"
    result = defaultdict(list)

    reference_landmarks = samples[0]["data"]
    last_landmarks = samples[0]["data"]
    for sample in samples:
        print(f"{sample['image_index']}")
        result["index"].append(sample["image_index"])
        landmarks = sample["data"]

        # Compare against reference.
        for landmark_id, data in landmarks.items():
            if landmark_id not in reference_landmarks:
                print(f"{landmark_id} is not in reference landmarks.")
            if len(data) != 1:
                print(f"{landmark_id} has multiple detections.")

            reference_data = reference_landmarks[landmark_id]
            angle = get_angle(data[0]["orientation"], reference_data[0]["orientation"])
            angle_degrees = math.degrees(angle)
            if angle_degrees > 10:
                print(f"{landmark_id}. angle[deg] {angle_degrees}")

        if marker_id in landmarks:
            result["reprojection_error_0"].append(landmarks[marker_id][0]["solutions"][0]["reprojection_error"])
            result["reprojection_error_1"].append(landmarks[marker_id][0]["solutions"][1]["reprojection_error"])

            result["angle_0_to_reference_0"].append(
                get_angle(
                    landmarks[marker_id][0]["solutions"][0]["orientation"],
                    reference_landmarks[marker_id][0]["orientation"],
                )
            )
            result["angle_1_to_reference_0"].append(
                get_angle(
                    landmarks[marker_id][0]["solutions"][1]["orientation"],
                    reference_landmarks[marker_id][0]["orientation"],
                )
            )
            result["angle_0_to_last_0"].append(
                get_angle(
                    landmarks[marker_id][0]["solutions"][0]["orientation"],
                    last_landmarks[marker_id][0]["solutions"][0]["orientation"],
                )
            )
            result["angle_1_to_last_1"].append(
                get_angle(
                    landmarks[marker_id][0]["solutions"][1]["orientation"],
                    last_landmarks[marker_id][0]["solutions"][1]["orientation"],
                )
            )
            result["angle_0_to_last_1"].append(
                get_angle(
                    landmarks[marker_id][0]["solutions"][0]["orientation"],
                    last_landmarks[marker_id][0]["solutions"][1]["orientation"],
                )
            )
            result["angle_1_to_last_0"].append(
                get_angle(
                    landmarks[marker_id][0]["solutions"][1]["orientation"],
                    last_landmarks[marker_id][0]["solutions"][0]["orientation"],
                )
            )

        else:
            result["reprojection_error_0"].append(None)
            result["reprojection_error_1"].append(None)
            result["angle_0_to_reference_0"].append(None)
            result["angle_1_to_reference_0"].append(None)
            result["angle_0_to_last_0"].append(None)
            result["angle_1_to_last_1"].append(None)
            result["angle_0_to_last_1"].append(None)
            result["angle_1_to_last_0"].append(None)

        last_landmarks = landmarks

    print(result["angle_0_to_last_0"])
    print(result["angle_1_to_last_1"])

    plt.figure(1, figsize=(25, 10))
    plt.plot(result["index"], result["reprojection_error_0"], "pr")
    plt.plot(result["index"], result["reprojection_error_1"], "pb")
    plt.plot(result["index"], result["angle_0_to_reference_0"], "pr")
    # plt.plot(result["index"], result["angle_1_to_reference_0"], "pb")
    # plt.plot(result["index"], result["angle_0_to_last_0"], "py")
    # plt.plot(result["index"], result["angle_1_to_last_1"], "pg")
    # plt.plot(result["index"], result["angle_0_to_last_1"], "pw")
    # plt.plot(result["index"], result["angle_1_to_last_0"], "pc")

    plt.show()


if __name__ == "__main__":
    main()

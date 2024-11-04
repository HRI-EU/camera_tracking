#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Plot stored aruco result.
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
from collections import defaultdict
import math
import os
import json
import matplotlib.pyplot as plt

from camera_tracking.aruco_tracking import angle_between_quaternions


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
            angle = angle_between_quaternions(data[0]["orientation"], reference_data[0]["orientation"])
            angle_degrees = math.degrees(angle)
            if angle_degrees > 10:
                print(f"{landmark_id}. angle[deg] {angle_degrees}")

        if marker_id in landmarks:
            result["reprojection_error_0"].append(landmarks[marker_id][0]["solutions"][0]["reprojection_error"])
            result["reprojection_error_1"].append(landmarks[marker_id][0]["solutions"][1]["reprojection_error"])

            result["angle_0_to_reference_0"].append(
                angle_between_quaternions(
                    landmarks[marker_id][0]["solutions"][0]["orientation"],
                    reference_landmarks[marker_id][0]["orientation"],
                )
            )
            result["angle_1_to_reference_0"].append(
                angle_between_quaternions(
                    landmarks[marker_id][0]["solutions"][1]["orientation"],
                    reference_landmarks[marker_id][0]["orientation"],
                )
            )
            result["angle_0_to_last_0"].append(
                angle_between_quaternions(
                    landmarks[marker_id][0]["solutions"][0]["orientation"],
                    last_landmarks[marker_id][0]["solutions"][0]["orientation"],
                )
            )
            result["angle_1_to_last_1"].append(
                angle_between_quaternions(
                    landmarks[marker_id][0]["solutions"][1]["orientation"],
                    last_landmarks[marker_id][0]["solutions"][1]["orientation"],
                )
            )
            result["angle_0_to_last_1"].append(
                angle_between_quaternions(
                    landmarks[marker_id][0]["solutions"][0]["orientation"],
                    last_landmarks[marker_id][0]["solutions"][1]["orientation"],
                )
            )
            result["angle_1_to_last_0"].append(
                angle_between_quaternions(
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

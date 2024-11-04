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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Helper functions for using a camera.
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
import yaml
import numpy


def load_camera_parameters(config_file: str) -> dict:
    """Load camera parameters from a YAML file.
    @param config_file: The path of the YAML file.
    @return: The camera parameters.
    """
    try:
        with open(config_file, encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    except IOError as e:
        raise AssertionError(f"Could not open '{config_file}'.") from e
    except Exception as e:
        raise AssertionError(f"Could not parse '{config_file}':\n{e}'") from e

    return camera_parameters_from_config(config)


def save_camera_parameters(
    camera_matrix: numpy.ndarray, distortion_coefficients: numpy.ndarray, config_file: str
) -> None:
    """Save camera parameters to a YAML file.
    @param camera_matrix: The 3x3 camera matrix.
    @param distortion_coefficients: The distortion coefficients.
    @param config_file: The path of the YAML file.
    """
    distortion_coefficient_names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    config = {
        "fx": float(camera_matrix[0][0]),
        "fy": float(camera_matrix[1][1]),
        "cx": float(camera_matrix[0][2]),
        "cy": float(camera_matrix[1][2]),
        **{key: float(value) for key, value in zip(distortion_coefficient_names, distortion_coefficients)},
    }

    with open(config_file, "w", encoding="utf-8") as file:
        yaml.dump(config, file, sort_keys=False)


def camera_parameters_from_config(config: dict) -> dict:
    # fx 0  cx
    # 0  fy cy
    # 0  0  1
    camera_matrix = numpy.array([[config["fx"], 0, config["cx"]], [0, config["fy"], config["cy"]], [0, 0, 1]])
    distortion_coefficients = [config["k1"], config["k2"], config["p1"], config["p2"], config["k3"]]
    if "k6" in config:
        distortion_coefficients.extend([config["k4"], config["k5"], config["k6"]])
    distortion_coefficients = numpy.array(distortion_coefficients)

    return {
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion_coefficients,
        "width": config.get("width"),
        "height": config.get("height"),
        "frames_per_second": config.get("frames_per_second"),
    }

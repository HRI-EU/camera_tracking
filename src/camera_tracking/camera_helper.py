#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Helper functions for using a camera.
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

from typing import Dict
import yaml
import numpy


def load_camera_parameters(config_file: str) -> Dict:
    """Load camera parameters from a YAML file.
    @param config_file: The path of the YAML file.
    @return: The camera parameters.
    """
    try:
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    except IOError as e:
        raise AssertionError(f"Could not open '{config_file}'.") from e
    except Exception as e:
        raise AssertionError(f"Could not parse '{config_file}':\n{e}'") from e

    return camera_parameters_from_config(config)


def camera_parameters_from_config(config: Dict) -> Dict:
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

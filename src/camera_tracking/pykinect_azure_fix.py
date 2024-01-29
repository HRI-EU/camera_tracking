#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Fixes and improvements for pykinect_azure.
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

import os
from enum import Enum
import ctypes
from pykinect_azure.k4a import _k4atypes
from pykinect_azure.k4abt import _k4abt, _k4abtTypes, Tracker, Frame, Body2d
import cv2


class K4ABT_JOINTS(Enum):
    K4ABT_JOINT_PELVIS = 0
    K4ABT_JOINT_SPINE_NAVEL = 1
    K4ABT_JOINT_SPINE_CHEST = 2
    K4ABT_JOINT_NECK = 3
    K4ABT_JOINT_CLAVICLE_LEFT = 4
    K4ABT_JOINT_SHOULDER_LEFT = 5
    K4ABT_JOINT_ELBOW_LEFT = 6
    K4ABT_JOINT_WRIST_LEFT = 7
    K4ABT_JOINT_HAND_LEFT = 8
    K4ABT_JOINT_HANDTIP_LEFT = 9
    K4ABT_JOINT_THUMB_LEFT = 10
    K4ABT_JOINT_CLAVICLE_RIGHT = 11
    K4ABT_JOINT_SHOULDER_RIGHT = 12
    K4ABT_JOINT_ELBOW_RIGHT = 13
    K4ABT_JOINT_WRIST_RIGHT = 14
    K4ABT_JOINT_HAND_RIGHT = 15
    K4ABT_JOINT_HANDTIP_RIGHT = 16
    K4ABT_JOINT_THUMB_RIGHT = 17
    K4ABT_JOINT_HIP_LEFT = 18
    K4ABT_JOINT_KNEE_LEFT = 19
    K4ABT_JOINT_ANKLE_LEFT = 20
    K4ABT_JOINT_FOOT_LEFT = 21
    K4ABT_JOINT_HIP_RIGHT = 22
    K4ABT_JOINT_KNEE_RIGHT = 23
    K4ABT_JOINT_ANKLE_RIGHT = 24
    K4ABT_JOINT_FOOT_RIGHT = 25
    K4ABT_JOINT_HEAD = 26
    K4ABT_JOINT_NOSE = 27
    K4ABT_JOINT_EYE_LEFT = 28
    K4ABT_JOINT_EAR_LEFT = 29
    K4ABT_JOINT_EYE_RIGHT = 30
    K4ABT_JOINT_EAR_RIGHT = 31


def get_custom_sdk_path():
    """Returns a custom location of the azure libraries."""
    return os.getenv("AZURE_CUSTOM_PATH", "/usr")


def get_custom_k4a_path():
    """Returns a custom location of the k4a library."""
    path = os.getenv("AZURE_CUSTOM_PATH")
    if path is None:
        return None

    return os.path.join(path, "lib", "libk4a.so")


def get_custom_k4abt_path():
    """Returns a custom location of the k4abt library."""
    path = os.getenv("AZURE_CUSTOM_PATH")
    if path is None:
        return None

    return os.path.join(path, "lib", "libk4abt.so")


def setup_onnx_provider_linux():
    # The original code checks whether it can load libonnxruntime_providers_cuda.so (usually under /usr/lib).
    # This fails with: /usr/lib/libonnxruntime_providers_cuda.so: undefined symbol: Provider_GetHost
    # Instead we directly load libonnxruntime.so.1.10.0 which later seems to load libonnxruntime_providers_cuda.so.
    _k4abtTypes.k4abt_tracker_default_configuration.processing_mode = _k4abtTypes.K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA
    ctypes.cdll.LoadLibrary(os.path.join(get_custom_sdk_path(), "lib", "libonnxruntime.so.1.10.0"))


_k4abt.setup_onnx_provider_linux = setup_onnx_provider_linux


def get_tracker_configuration(self, model_type):
    if model_type == _k4abt.K4ABT_LITE_MODEL:
        model_name = "dnn_model_2_0_lite_op11.onnx"
    elif model_type == _k4abt.K4ABT_DEFAULT_MODEL:
        model_name = "dnn_model_2_0_op11.onnx"
    else:
        raise AssertionError(f"Cannot handle model type {model_type}.")

    model_path = os.path.join(get_custom_sdk_path(), "bin", model_name)
    print(f"Using dnn model path '{model_path}'.")

    tracker_config = _k4abtTypes.k4abt_tracker_default_configuration
    tracker_config.model_path = model_path.encode("utf-8")

    return tracker_config


Tracker.get_tracker_configuration = get_tracker_configuration


# Manually fix bug:
#  color = (int (body_colors[self.id][0]), int (body_colors[self.id][1]), int (body_colors[self.id][2]))
#  IndexError: index 256 is out of bounds for axis 0 with size 256
def draw(self, image, only_segments=False):
    color_number = _k4abtTypes.body_colors.shape[0]
    color = (
        int(_k4abtTypes.body_colors[self.id % color_number][0]),
        int(_k4abtTypes.body_colors[self.id % color_number][1]),
        int(_k4abtTypes.body_colors[self.id % color_number][2]),
    )

    for segmentId in range(len(_k4abtTypes.K4ABT_SEGMENT_PAIRS)):
        segment_pair = _k4abtTypes.K4ABT_SEGMENT_PAIRS[segmentId]
        point1 = self.joints[segment_pair[0]].get_coordinates()
        point2 = self.joints[segment_pair[1]].get_coordinates()

        if (point1[0] == 0 and point1[1] == 0) or (point2[0] == 0 and point2[1] == 0):
            continue
        image = cv2.line(image, point1, point2, color, 2)

    if only_segments:
        return image

    for joint in self.joints:
        image = cv2.circle(image, joint.get_coordinates(), 3, color, 3)

    return image


Body2d.draw = draw


# Manually fix bodyIdx/body.id bug, see https://github.com/ibaiGorordo/pyKinectAzure/pull/110
def get_body2d(self, bodyIdx=0, dest_camera=_k4atypes.K4A_CALIBRATION_TYPE_DEPTH):
    body_handle = self.get_body(bodyIdx).handle()
    return Body2d.create(body_handle, self.calibration, body_handle.id, dest_camera)


Frame.get_body2d = get_body2d


import pykinect_azure

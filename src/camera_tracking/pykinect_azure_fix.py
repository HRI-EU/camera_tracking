#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Fixes and improvements for pykinect_azure.
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

import os
import ctypes
from pykinect_azure.k4a import _k4atypes, _k4a
from pykinect_azure.k4abt import _k4abt, _k4abtTypes, Tracker, Frame, Body2d
import cv2
from .pykinect_azure_add_on import K4ABT_JOINTS


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

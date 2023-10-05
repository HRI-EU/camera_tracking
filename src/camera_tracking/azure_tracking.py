#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (c) Honda Research Institute Europe GmbH.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
#  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
#  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
#  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from typing import Dict, Optional
from collections import OrderedDict
import time
import cv2

from .base_tracking import BaseTracking, ThreadedTracker
from .camera_helper import camera_parameters_from_config
from .pykinect_azure_fix import pykinect_azure as pykinect
from .pykinect_azure_fix import K4ABT_JOINTS


def get_color_camera_calibration(device, depth_mode, color_resolution) -> Dict:
    calibration = device.get_calibration(depth_mode, color_resolution)
    config = {name: getattr(calibration.color_params, name) for name, _ in calibration.color_params._fields_}
    return camera_parameters_from_config(config)


class BodyTracking(BaseTracking):
    millimeter_to_meter_ratio = 1000.0

    def __init__(self, visualize: bool = True):
        # Initialize the body tracker.
        self.body_tracker = pykinect.start_body_tracker(pykinect.K4ABT_DEFAULT_MODEL)
        super().__init__("body", visualize=visualize)

    @staticmethod
    def body_frame_to_landmarks(body_frame: pykinect.Frame) -> Dict[str, Dict]:
        # body_<joint_id>_<body_id>, e.g. body_right_hand_0
        landmarks = {}
        for body in body_frame.get_bodies():
            joints = {}
            for joint in body.joints:
                joint_name = K4ABT_JOINTS(joint.id).name[12:].lower()
                joints[f"{joint_name}"] = {
                    "position": {
                        "x": joint.position.x / BodyTracking.millimeter_to_meter_ratio,
                        "y": joint.position.y / BodyTracking.millimeter_to_meter_ratio,
                        "z": joint.position.z / BodyTracking.millimeter_to_meter_ratio,
                    },
                    "orientation": {
                        "x": joint.orientation.x,
                        "y": joint.orientation.y,
                        "z": joint.orientation.z,
                        "w": joint.orientation.w,
                    },
                    "confidence": joint.confidence_level,
                }

            landmarks[body.handle().id] = joints

        return landmarks

    def process(self, capture: pykinect.Capture) -> Dict:
        start_time = time.time()

        # Get the updated body tracker frame.
        body_frame = self.body_tracker.update()
        body_landmarks = self.body_frame_to_landmarks(body_frame)

        if self.visualize:
            # Get the depth image from the capture.
            success, depth_image = capture.get_depth_image()
            if not success:
                print("Could not get depth image from capture.")
            else:
                # Convert the depth image to color and draw the skeletons.
                self.visualization = cv2.convertScaleAbs(depth_image, alpha=0.05)
                self.visualization = cv2.cvtColor(self.visualization, cv2.COLOR_GRAY2RGB)
                self.visualization = body_frame.draw_bodies(self.visualization)

        self.sum_processing_time += time.time() - start_time

        return body_landmarks


class AzureTracking:
    color_resolution_mapping = {
        "720P": pykinect.K4A_COLOR_RESOLUTION_720P,
        "1080P": pykinect.K4A_COLOR_RESOLUTION_1080P,
        "1440P": pykinect.K4A_COLOR_RESOLUTION_1440P,
        "1536P": pykinect.K4A_COLOR_RESOLUTION_1536P,
        "2160P": pykinect.K4A_COLOR_RESOLUTION_2160P,
        "3072P": pykinect.K4A_COLOR_RESOLUTION_3072P,
    }

    def __init__(
        self,
        with_aruco: bool = True,
        with_body: bool = True,
        with_mediapipe: bool = True,
        visualize: bool = True,
        color_resolution: str = "1536P",
        module_k4a_path: Optional[str] = None,
        module_k4abt_path: Optional[str] = None,
    ):
        if not any((with_aruco, with_mediapipe, with_body)):
            raise AssertionError("No tracker is enabled.")

        if color_resolution not in AzureTracking.color_resolution_mapping:
            raise AssertionError(
                f"Unknown color resolution '{color_resolution}'. "
                f"Known ones are {AzureTracking.color_resolution_mapping}."
            )

        # Initialize the library, if the library is not found, add the library path as argument.
        pykinect.initialize_libraries(
            module_k4a_path=module_k4a_path, module_k4abt_path=module_k4abt_path, track_body=with_body
        )

        # Define the device configuration.
        device_config = pykinect.default_configuration
        device_config.color_resolution = (
            AzureTracking.color_resolution_mapping[color_resolution]
            if (with_aruco or with_mediapipe)
            else pykinect.K4A_COLOR_RESOLUTION_OFF
        )
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED if with_body else pykinect.K4A_DEPTH_MODE_OFF
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

        # Start device.
        self.device = pykinect.start_device(config=device_config)

        self.trackers = OrderedDict()

        # We add trackers in order of expected processing time (decreasingly).
        if with_body:
            body_tracking = BodyTracking(visualize=visualize)
            self.trackers["body"] = ThreadedTracker(body_tracking, input_function=lambda capture: (True, capture))

        if with_mediapipe:
            from .mediapipe_tracking import MediapipeTracking

            mediapipe_tracking = MediapipeTracking(visualize=visualize)
            self.trackers["mediapipe"] = ThreadedTracker(
                mediapipe_tracking, input_function=lambda capture: capture.get_color_image()
            )

        if with_aruco:
            from .aruco_tracking import ArucoTracking

            # Get the resolution specific calibration parameters of the color camera.
            color_camera_parameters = get_color_camera_calibration(
                self.device, device_config.depth_mode, device_config.color_resolution
            )
            aruco_tracking = ArucoTracking(
                color_camera_parameters["camera_matrix"],
                color_camera_parameters["distortion_coefficients"],
                visualize=visualize,
            )
            self.trackers["aruco"] = ThreadedTracker(
                aruco_tracking, input_function=lambda capture: capture.get_color_image()
            )

        # Initialize statistics.
        self.step_count = 0
        self.sum_overall_time = 0.0
        self.sum_capture_time = 0.0
        self.report_interval = 30

    def step(self) -> Dict:
        start_time = time.time()

        # Get capture.
        capture = self.device.update()
        capture_time = time.time()
        self.sum_capture_time += capture_time - start_time

        # Trigger trackers in given order (decreasing processing time).
        for tracker in self.trackers.values():
            tracker.trigger(capture)

        # Wait for tracker results in reversed order (increasing processing time)
        landmarks = {"header": {"timestamp": capture_time, "frame_id": "camera", "seq": self.step_count}, "data": {}}
        for tracker in reversed(self.trackers.values()):
            landmarks["data"][tracker.tracker.name] = tracker.output.get()
            tracker.tracker.show_visualization()

        self.sum_overall_time += time.time() - start_time
        if self.step_count % self.report_interval == 0:
            status = (
                f"Step {self.step_count} mean times: "
                f"overall {self.sum_overall_time / self.report_interval:.3f}s"
                f" | capture {self.sum_capture_time / self.report_interval:.3f}s"
                f" | processing: {(self.sum_overall_time - self.sum_capture_time) / self.report_interval:.3f}s"
            )
            for tracker in self.trackers.values():
                status += f" | {tracker.tracker.name} {tracker.tracker.sum_processing_time / self.report_interval:.3f}s"
                tracker.tracker.sum_processing_time = 0.0

            print(status)
            self.sum_overall_time = 0.0
            self.sum_capture_time = 0.0

        self.step_count += 1

        return landmarks

    def stop(self):
        self.device.close()

        for tracker in self.trackers.values():
            tracker.input.put((True, None))
            tracker.thread.join()

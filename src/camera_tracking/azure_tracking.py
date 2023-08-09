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

from typing import Dict
import queue
import threading
import time
import cv2

from .base_tracking import BaseTracking
from .camera_helper import camera_parameters_from_config
from .pykinect_azure_fix import pykinect_azure as pykinect


def get_color_camera_calibration(device, depth_mode, color_resolution) -> Dict:
    calibration = device.get_calibration(depth_mode, color_resolution)
    config = {name: getattr(calibration.color_params, name) for name, _ in calibration.color_params._fields_}
    return camera_parameters_from_config(config)


class BodyTracking(BaseTracking):
    def __init__(self, visualize: bool = True):
        # Initialize the body tracker.
        self.body_tracker = pykinect.start_body_tracker(pykinect.K4ABT_DEFAULT_MODEL)

        super().__init__("body", visualize=visualize)

    @staticmethod
    def body_frame_to_landmarks(body_frame: pykinect.Frame) -> Dict:
        return {
            f"azure_{i:02}_{body.handle().id:02}": [
                body.joints[i].position.x,
                body.joints[i].position.y,
                body.joints[i].position.z,
                body.joints[i].orientation.x,
                body.joints[i].orientation.y,
                body.joints[i].orientation.z,
                body.joints[i].orientation.w,
                body.joints[i].confidence_level,
            ]
            for body in body_frame.get_bodies()
            for i in range(pykinect.K4ABT_JOINT_COUNT)
        }

    def process(self, capture) -> Dict:
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


# class ThreadedTracker:
#     def __init__(self, tracker):
#         self.input = queue.Queue()
#         self.output = queue.Queue()
#         self.tracker = tracker
#         self.thread =


class AzureTracking:
    def __init__(
        self, with_aruco: bool = True, with_body: bool = True, with_mediapipe: bool = True, visualize: bool = True
    ):
        self.with_aruco = with_aruco
        self.with_body = with_body
        self.with_mediapipe = with_mediapipe

        # Initialize the library, if the library is not found, add the library path as argument.
        pykinect.initialize_libraries(track_body=self.with_body)

        # Define the device configuration.
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1536P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

        # Start device.
        self.device = pykinect.start_device(config=device_config)

        self.trackers = []

        if self.with_aruco:
            from .aruco_tracking import ArucoTracking

            # Get the resolution specific calibration parameters of the color camera.
            color_camera_parameters = get_color_camera_calibration(
                self.device, device_config.depth_mode, device_config.color_resolution
            )
            self.aruco_tracking = ArucoTracking(
                color_camera_parameters["camera_matrix"],
                color_camera_parameters["distortion_coefficients"],
                visualize,
            )
            self.trackers.append(self.aruco_tracking)
            self.aruco_input = queue.Queue()
            self.aruco_output = queue.Queue()
            self.aruco_thread = threading.Thread(target=self.aruco_worker)
            self.aruco_thread.start()

        if self.with_body:
            self.body_tracking = BodyTracking(visualize)
            self.trackers.append(self.body_tracking)
            self.body_input = queue.Queue()
            self.body_output = queue.Queue()
            self.body_thread = threading.Thread(target=self.body_worker)
            self.body_thread.start()

        if self.with_mediapipe:
            from .mediapipe_tracking import MediapipeTracking

            self.mediapipe_tracking = MediapipeTracking(visualize)
            self.trackers.append(self.mediapipe_tracking)
            self.mediapipe_input = queue.Queue()
            self.mediapipe_output = queue.Queue()
            self.mediapipe_thread = threading.Thread(target=self.mediapipe_worker)
            self.mediapipe_thread.start()

        # Initialize statistics.
        self.step_count = 0
        self.sum_overall_time = 0.0
        self.sum_capture_time = 0.0
        self.report_interval = 20

    def step(self) -> Dict:
        start_time = time.time()

        # Get capture.
        capture = self.device.update()
        self.sum_capture_time += time.time() - start_time

        if self.with_body:
            self.body_input.put((True, capture))

        if self.with_mediapipe:
            self.mediapipe_input.put(capture.get_color_image())

        if self.with_aruco:
            self.aruco_input.put(capture.get_color_image())

        landmarks = {}

        if self.with_aruco:
            aruco_landmarks = self.aruco_output.get()
            landmarks.update(aruco_landmarks)
            self.aruco_tracking.show_visualization()

        if self.with_mediapipe:
            mediapipe_landmarks = self.mediapipe_output.get()
            landmarks.update(mediapipe_landmarks)
            self.mediapipe_tracking.show_visualization()

        if self.with_body:
            body_landmarks = self.body_output.get()
            landmarks.update(body_landmarks)
            self.body_tracking.show_visualization()

        self.sum_overall_time += time.time() - start_time
        self.step_count += 1
        if self.step_count % self.report_interval == 0:
            status = (
                f"Step {self.step_count} mean times: "
                f"overall {(self.sum_overall_time) / self.report_interval:.3f}s"
                f" | capture {self.sum_capture_time / self.report_interval:.3f}s"
                f" | processing: {(self.sum_overall_time - self.sum_capture_time) / self.report_interval:.3f}s"
            )
            for tracker in self.trackers:
                status += f" | {tracker.name} {tracker.sum_processing_time / self.report_interval:.3f}s"
                tracker.sum_processing_time = 0.0

            print(status)
            self.sum_overall_time = 0.0
            self.sum_capture_time = 0.0

        return landmarks

    def body_worker(self):
        while True:
            success, capture = self.body_input.get()
            if not success:
                print("Could not get capture.")
                self.body_output.put({})

            if capture is None:
                break

            # Compute body landmarks and put them into the output buffer.
            body_landmarks = self.body_tracking.process(capture)
            self.body_output.put(body_landmarks)

    def aruco_worker(self):
        while True:
            success, color_image = self.aruco_input.get()
            if not success:
                print("Could not get color image from capture.")
                self.aruco_output.put({})
                continue

            if color_image is None:
                break

            # Compute aruco landmarks and put them into the output buffer.
            aruco_landmarks = self.aruco_tracking.process(color_image)
            self.aruco_output.put(aruco_landmarks)

    def mediapipe_worker(self):
        while True:
            success, color_image = self.mediapipe_input.get()
            if not success:
                print("Could not get color image from capture.")
                self.mediapipe_output.put({})
                continue

            if color_image is None:
                break

            # Compute mediapipe landmarks and put them into the output buffer.
            mediapipe_landmarks = self.mediapipe_tracking.process(color_image, {"face": True, "hands": True})
            self.mediapipe_output.put(mediapipe_landmarks)

    def stop(self):
        self.device.close()

        if self.with_aruco:
            self.aruco_input.put((True, None))
            self.aruco_thread.join()

        if self.with_mediapipe:
            self.mediapipe_input.put((True, None))
            self.mediapipe_thread.join()

        if self.with_body:
            self.body_input.put((True, None))
            self.body_thread.join()

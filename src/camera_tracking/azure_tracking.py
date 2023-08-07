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
import sys
import os
import time
import logging
import cv2

from .track_aruco import ArucoTracking
from .camera_helper import camera_parameters_from_config

sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "pyKinectAzure3rdParty", "pyKinectAzure"))
from pyKinectAzure import pyKinectAzure
import _k4a
from kinectBodyTracker import _k4abt


# Azure library paths.
module_path = "/usr/lib/x86_64-linux-gnu/libk4a.so"
body_tracking_module_path = "/usr/lib/libk4abt.so"
body_tracking_model_directory = "/usr/bin"


def bodies_to_landmarks(bodies):
    return {
        f"azure_{i:02}_{body.id:02}": [
            *body.skeleton.joints[i].position.v,
            *body.skeleton.joints[i].orientation.v,
            body.skeleton.joints[i].confidence_level,
        ]
        for body in bodies
        for i in range(_k4abt.K4ABT_JOINT_COUNT)
    }


def get_color_camera_calibration(pyk4a, depth_mode, color_resolution) -> Dict:
    calibration = _k4a.k4a_calibration_t()
    pyk4a.device_get_calibration(depth_mode, color_resolution, calibration)
    config = {
        "width": calibration.color_camera_calibration.resolution_width,
        "height": calibration.color_camera_calibration.resolution_height,
    }
    for name, value in zip(
        calibration.color_camera_calibration.intrinsics.parameters.param._fields_,
        calibration.color_camera_calibration.intrinsics.parameters.v,
    ):
        config[name[0]] = value

    return camera_parameters_from_config(config)


class AzureTracking:
    def __init__(self, with_aruco=True, with_body=True, visualize=True):
        self.with_aruco = with_aruco
        self.with_body = with_body
        self.visualize = visualize

        # Initialize the library with the path containing the module.
        self.pyk4a = pyKinectAzure(module_path)

        # Open the device.
        self.pyk4a.device_open()

        # Define the device configuration.
        device_config = self.pyk4a.config
        device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1536P
        device_config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_2X2BINNED
        device_config.camera_fps = _k4a.K4A_FRAMES_PER_SECOND_30

        # Start the camera.
        self.pyk4a.device_start_cameras(device_config)

        if self.with_aruco:
            # Get the resolution specific calibration parameters of the color camera.
            color_camera_parameters = get_color_camera_calibration(
                self.pyk4a, device_config.depth_mode, device_config.color_resolution
            )
            self.aruco_tracking = ArucoTracking(
                color_camera_parameters["camera_matrix"],
                color_camera_parameters["distortion_coefficients"],
                self.visualize,
            )
            self.aruco_input = queue.Queue()
            self.aruco_output = queue.Queue()
            self.aruco_thread = threading.Thread(target=self.aruco_worker)
            self.aruco_thread.start()

        if self.with_body:
            # Initialize the body tracker.
            self.pyk4a.bodyTracker_start(
                body_tracking_module_path, _k4abt.K4ABT_DEFAULT_MODEL, body_tracking_model_directory
            )
            self.body_input = queue.Queue()
            self.body_output = queue.Queue()
            self.body_thread = threading.Thread(target=self.body_worker)
            self.body_thread.start()

        # Initialize statistics.
        self.step_count = 0
        self.sum_overall_time = 0.0
        self.sum_capture_time = 0.0
        self.report_interval = 20

    def step(self) -> Dict:
        start_time = time.time()

        # Capture Azure.
        self.pyk4a.device_get_capture()
        self.sum_capture_time += time.time() - start_time

        if self.with_body:
            self.body_input.put(self.step_count)

        aruco_running = False
        if self.with_aruco:
            # Get the color image from the capture.
            color_image_handle = self.pyk4a.capture_get_color_image()

            # Check the image has been read correctly.
            if color_image_handle:
                self.aruco_input.put(color_image_handle)
                aruco_running = True
            else:
                logging.warning("Could not get color image from capture.")

        landmarks = {}

        if aruco_running:
            aruco_landmarks = self.aruco_output.get()
            landmarks.update(aruco_landmarks)
            self.aruco_tracking.show_visualization()

        if self.with_body:
            body_landmarks = self.body_output.get()
            landmarks.update(body_landmarks)
            if self.visualize:
                cv2.namedWindow("Body tracking", cv2.WINDOW_NORMAL)
                cv2.imshow("Body tracking", self.combined_image)
                cv2.waitKey(1)

        self.pyk4a.capture_release()

        self.sum_overall_time += time.time() - start_time
        self.step_count += 1
        if self.step_count % self.report_interval == 0:
            status = (
                f"Step {self.step_count} / "
                f"Mean capture time: {self.sum_capture_time / self.report_interval:.4f}s / "
                f"Mean processing time: {(self.sum_overall_time - self.sum_capture_time) / self.report_interval:.4f}s"
            )
            if self.with_aruco:
                status += f" / Mean aruco time: {self.aruco_tracking.sum_processing_time / self.report_interval:.4f}s"
                self.aruco_tracking.sum_processing_time = 0.0

            print(status)
            self.sum_overall_time = 0.0
            self.sum_capture_time = 0.0

        return landmarks

    def body_worker(self):
        while True:
            item = self.body_input.get()
            if item is None:
                break

            # Update the body tracker, this will use the adapted depth and infrared image.
            self.pyk4a.bodyTracker_update()
            body_landmarks = bodies_to_landmarks(self.pyk4a.body_tracker.bodiesNow)

            if self.visualize:
                # Overlay body segmentation on depth image.

                # Get the depth image from the capture.
                depth_image_handle = self.pyk4a.capture_get_depth_image()
                if not depth_image_handle:
                    logging.warning("Could not get depth image from capture.")
                else:
                    depth_image = self.pyk4a.image_convert_to_numpy(depth_image_handle)

                    # Convert the depth image to numpy array.
                    depth_color_image = cv2.convertScaleAbs(depth_image, alpha=0.05)
                    depth_color_image = cv2.cvtColor(depth_color_image, cv2.COLOR_GRAY2RGB)

                    # Get body segmentation image.
                    body_image_color = self.pyk4a.bodyTracker_get_body_segmentation()

                    # Combine depth and body segmentation.
                    self.combined_image = cv2.addWeighted(depth_color_image, 0.8, body_image_color, 0.2, 0)

                    # Add the skeleton of each body to the combined image.
                    for body in self.pyk4a.body_tracker.bodiesNow:
                        skeleton_2d = self.pyk4a.bodyTracker_project_skeleton(body.skeleton)
                        self.combined_image = self.pyk4a.body_tracker.draw2DSkeleton(
                            skeleton_2d, body.id, self.combined_image
                        )

                    self.pyk4a.image_release(self.pyk4a.body_tracker.segmented_body_img)

                    # Release the image handles.
                    self.pyk4a.image_release(depth_image_handle)

            self.body_output.put(body_landmarks)
            self.pyk4a.body_tracker.release_frame()

    def aruco_worker(self):
        while True:
            color_image_handle = self.aruco_input.get()
            if color_image_handle is None:
                break

            # Read and convert the image data to numpy array.
            color_image = self.pyk4a.image_convert_to_numpy(color_image_handle)

            # Compute aruco landmarks and put them into the output buffer.
            aruco_landmarks = self.aruco_tracking.process(color_image)
            self.aruco_output.put(aruco_landmarks)

            # Release the image.
            self.pyk4a.image_release(color_image_handle)

    def stop(self):
        self.pyk4a.device_stop_cameras()
        self.pyk4a.device_close()

        if self.with_aruco:
            self.aruco_input.put(None)
            self.aruco_thread.join()

        if self.with_body:
            self.body_input.put(None)
            self.body_thread.join()

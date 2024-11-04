#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  #  Class that offers tracking of markers with a webcam.
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

import cv2

from .base_tracking import ThreadedTracker, BaseCamera
from .camera_helper import load_camera_parameters


class WebcamTracking(BaseCamera):
    def __init__(
        self,
        camera_config_file: str,
        with_aruco: bool = True,
        with_mediapipe: bool = True,
        visualize: bool = True,
        frame_id: str = "",
    ):
        super().__init__(frame_id=frame_id)

        camera_parameters = load_camera_parameters(camera_config_file)

        self.capture = cv2.VideoCapture(0)
        # Depends on fourcc available camera.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.capture.set(cv2.CAP_PROP_FPS, camera_parameters["frames_per_second"])
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_parameters["width"])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_parameters["height"])

        # We add trackers in order of expected processing time (decreasingly).
        if with_mediapipe:
            from .mediapipe_tracking import MediapipeTracking

            mediapipe_tracking = MediapipeTracking(visualize=visualize)
            self.trackers["mediapipe"] = ThreadedTracker(mediapipe_tracking, input_function=lambda capture: capture)

        if with_aruco:
            from .aruco_tracking import ArucoTracking

            aruco_tracking = ArucoTracking(
                camera_parameters["camera_matrix"], camera_parameters["distortion_coefficients"], visualize=visualize
            )
            self.trackers["aruco"] = ThreadedTracker(aruco_tracking, input_function=lambda capture: capture)

    def get_capture(self):
        return self.capture.read()

    def stop(self):
        self.capture.release()
        super().stop()

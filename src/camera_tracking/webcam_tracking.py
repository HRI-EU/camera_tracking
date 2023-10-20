#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  #  Class that offers tracking of markers with a webcam.
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

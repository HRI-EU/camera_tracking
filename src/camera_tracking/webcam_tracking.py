#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  BLAAA
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

import time
from collections import OrderedDict
import cv2

from .base_tracking import ThreadedTracker
from .camera_helper import load_camera_parameters


class WebcamTracking:
    def __init__(
        self, camera_config_file: str, with_aruco: bool = True, with_mediapipe: bool = True, visualize: bool = True
    ):
        camera_parameters = load_camera_parameters(camera_config_file)

        self.capture = cv2.VideoCapture(0)
        # Depends on fourcc available camera.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.capture.set(cv2.CAP_PROP_FPS, camera_parameters["frames_per_second"])
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_parameters["width"])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_parameters["height"])

        self.trackers = OrderedDict()
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

        # Initialize statistics.
        self.step_count = 0
        self.sum_overall_time = 0.0
        self.sum_capture_time = 0.0
        self.report_interval = 30

    def step(self):
        start_time = time.time()

        # Get capture.
        capture = self.capture.read()
        self.sum_capture_time += time.time() - start_time

        # Trigger trackers in given order (decreasing processing time).
        for tracker in self.trackers.values():
            tracker.trigger(capture)

        landmarks = {}

        # Wait for tracker results in reversed order (increasing processing time)
        for tracker in reversed(self.trackers.values()):
            tracker_landmarks = tracker.output.get()
            landmarks.update(tracker_landmarks)
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
        self.capture.release()

        for tracker in self.trackers.values():
            tracker.input.put((True, None))
            tracker.thread.join()

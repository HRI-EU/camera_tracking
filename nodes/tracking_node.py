#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Node that provides marker tracking using ROS image topics.
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
import queue
from typing import Optional

import logging
import json
import numpy

import rospy
import std_msgs.msg
import sensor_msgs.msg
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError

from camera_tracking.base_tracking import ThreadedTracker, BaseCamera


class Tracking(BaseCamera):
    def __init__(
        self,
        with_aruco: bool = True,
        with_mediapipe: bool = True,
        visualize: bool = True,
        frame_id: str = "",
        aruco_with_tracking: bool = False,
        camera_matrix: Optional[numpy.ndarray] = None,
        distortion_coefficients: Optional[numpy.ndarray] = None,
    ):
        super().__init__(frame_id=frame_id)

        self.input_buffer = queue.Queue()

        # We add trackers in order of expected processing time (decreasingly).
        if with_mediapipe:
            from camera_tracking.mediapipe_tracking import MediapipeTracking

            mediapipe_tracking = MediapipeTracking(visualize=visualize)
            self.trackers["mediapipe"] = ThreadedTracker(mediapipe_tracking, input_function=lambda capture: capture)

        if with_aruco:
            from camera_tracking.aruco_tracking import ArucoTracking

            aruco_tracking = ArucoTracking(
                camera_matrix, distortion_coefficients, visualize=visualize, with_tracking=aruco_with_tracking
            )
            self.trackers["aruco"] = ThreadedTracker(aruco_tracking, input_function=lambda capture: capture)

    def get_capture(self):
        return self.input_buffer.get()

    def stop(self):
        super().stop()


class TrackingNode:
    def __init__(self):
        self.visualize = rospy.get_param("~visualize")
        self.image_topic = rospy.get_param("~image_topic")
        self.camera_info_topic = rospy.get_param("~camera_info_topic")
        with_aruco = rospy.get_param("~with_aruco")

        self.cv_bridge = CvBridge()

        camera_matrix = None
        distortion_coefficients = None

        if with_aruco:
            rospy.loginfo(f"Waiting for topic '{self.camera_info_topic}' ...")
            camera_info_msg = rospy.wait_for_message(self.camera_info_topic, sensor_msgs.msg.CameraInfo)
            rospy.loginfo(f"Topic '{self.camera_info_topic}' received.")
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(camera_info_msg)
            camera_matrix = camera_model.intrinsicMatrix()
            distortion_coefficients = camera_model.distortionCoeffs()

        self.tracking = Tracking(
            with_aruco=rospy.get_param("~with_aruco"),
            with_mediapipe=rospy.get_param("~with_mediapipe"),
            visualize=self.visualize,
            frame_id=rospy.get_param("~frame_id"),
            aruco_with_tracking=rospy.get_param("~aruco_with_tracking"),
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
        )
        self.landmarks_publisher = rospy.Publisher(
            f"/landmarks/{self.tracking.frame_id}", std_msgs.msg.String, queue_size=1
        )

        # Initialize subscribers.
        rospy.Subscriber(self.image_topic, sensor_msgs.msg.Image, self.image_callback, queue_size=2)

        rospy.loginfo("Initialization done.")

    @staticmethod
    def run():
        rospy.spin()

    def stop(self):
        self.tracking.stop()

    def image_callback(self, image_msg):
        # Convert msgs to cv2.
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self.tracking.input_buffer.put((True, cv_image))
        landmarks = self.tracking.step()
        self.landmarks_publisher.publish(json.dumps(landmarks))


def main():
    rospy.init_node("tracking")
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    tracking = TrackingNode()
    tracking.run()
    tracking.stop()


if __name__ == "__main__":
    main()

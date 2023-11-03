#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Node that provides marker tracking using a webcam.
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

import logging
import json

import rospy
import std_msgs.msg
from camera_tracking.webcam_tracking import WebcamTracking


class WebcamTrackingNode:
    def __init__(self):
        self.webcam_tracking = WebcamTracking(
            rospy.get_param("~camera_config_file"),
            with_aruco=rospy.get_param("~with_aruco"),
            with_mediapipe=rospy.get_param("~with_mediapipe"),
            visualize=rospy.get_param("~visualize"),
            frame_id=rospy.get_param("~frame_id"),
        )
        self.landmarks_publisher = rospy.Publisher(
            f"/landmarks/{self.webcam_tracking.frame_id}", std_msgs.msg.String, queue_size=1
        )
        rospy.loginfo("Initialization done.")

    def run(self):
        while not rospy.is_shutdown():
            landmarks = self.webcam_tracking.step()
            self.landmarks_publisher.publish(json.dumps(landmarks))

    def stop(self):
        self.webcam_tracking.stop()


def main():
    rospy.init_node("webcam_tracking")
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    webcam_tracking = WebcamTrackingNode()
    webcam_tracking.run()
    webcam_tracking.stop()


if __name__ == "__main__":
    main()

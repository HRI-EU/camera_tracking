#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Node that provides marker tracking using Azure Kinect.
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
from camera_tracking.azure_tracking import AzureTracking


class AzureTrackingNode:
    def __init__(self):
        self.azure_tracking = AzureTracking(
            with_aruco=rospy.get_param("~with_aruco"),
            with_body=rospy.get_param("~with_body"),
            with_mediapipe=rospy.get_param("~with_mediapipe"),
            visualize=rospy.get_param("~visualize"),
            color_resolution=rospy.get_param("~color_resolution"),
            depth_mode=rospy.get_param("~depth_mode"),
            fps=rospy.get_param("~fps"),
            frame_id=rospy.get_param("~frame_id"),
        )
        self.landmarks_publisher = rospy.Publisher(
            f"/landmarks/{self.azure_tracking.frame_id}", std_msgs.msg.String, queue_size=1
        )
        rospy.loginfo("Initialization done.")

    def run(self):
        while not rospy.is_shutdown():
            landmarks = self.azure_tracking.step()
            self.landmarks_publisher.publish(json.dumps(landmarks))

    def stop(self):
        self.azure_tracking.stop()


def main():
    rospy.init_node("azure_tracking")
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    azure_tracking = AzureTrackingNode()
    azure_tracking.run()
    azure_tracking.stop()


if __name__ == "__main__":
    main()

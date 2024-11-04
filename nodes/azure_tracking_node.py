#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Node that provides marker tracking using Azure Kinect.
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

import logging
import json

import rospy
import std_msgs.msg
from camera_tracking.azure_tracking import AzureTracking


class AzureTrackingNode:
    def __init__(self):
        self.visualize = rospy.get_param("~visualize")
        self.azure_tracking = AzureTracking(
            with_aruco=rospy.get_param("~with_aruco"),
            with_body=rospy.get_param("~with_body"),
            with_mediapipe=rospy.get_param("~with_mediapipe"),
            visualize=self.visualize,
            color_resolution=rospy.get_param("~color_resolution"),
            depth_mode=rospy.get_param("~depth_mode"),
            fps=rospy.get_param("~fps"),
            frame_id=rospy.get_param("~frame_id"),
            body_max_distance=rospy.get_param("~body_max_distance"),
            aruco_with_tracking=rospy.get_param("~aruco_with_tracking"),
            color_control_filename=rospy.get_param("~color_control_filename"),
        )
        self.landmarks_publisher = rospy.Publisher(
            f"/landmarks/{self.azure_tracking.frame_id}", std_msgs.msg.String, queue_size=1
        )
        rospy.loginfo("Initialization done.")

    def run(self):
        while not rospy.is_shutdown():
            landmarks = self.azure_tracking.step(self.landmarks_publisher.get_num_connections() > 0 or self.visualize)
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

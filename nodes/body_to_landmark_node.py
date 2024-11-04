#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Node that provides marker tracking using ROS image topics.
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

from typing import Optional
import logging
import json
from collections import defaultdict

import numpy
from scipy.spatial.transform import Rotation

import rospy
import actionlib
import tf2_msgs.msg
import std_msgs.msg
import visualization_msgs.msg

from camera_tracking.pykinect_azure_add_on import K4ABT_JOINTS


class BodyToLandmarkNode:
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id")
        self.body_topic = rospy.get_param("~body_topic")

        self.transformation_matrix = None
        self.step_count = 0

        self.landmarks_publisher = rospy.Publisher(f"/landmarks/{self.frame_id}", std_msgs.msg.String, queue_size=1)

        self.lookup_transform_client = actionlib.SimpleActionClient(
            "/tf2_buffer_server", tf2_msgs.msg.LookupTransformAction
        )

        # Initialize subscribers.
        rospy.Subscriber(self.body_topic, visualization_msgs.msg.MarkerArray, self.body_callback, queue_size=2)

        rospy.loginfo("Initialization done.")

    @staticmethod
    def run():
        rospy.spin()

    def stop(self):
        pass

    def _get_transformation_matrix(self) -> Optional[numpy.ndarray]:
        if self.transformation_matrix is None:
            if not self.lookup_transform_client.wait_for_server(rospy.Duration(1)):
                message = f"Timeout waiting for ActionServer '{self.lookup_transform_client.action_client.ns}'."
                rospy.logwarn(message)
                return None

            goal_msg = tf2_msgs.msg.LookupTransformGoal(
                source_frame=f"{self.frame_id}/depth_camera_link", target_frame=f"{self.frame_id}/rgb_camera_link"
            )
            action_status = self.lookup_transform_client.send_goal_and_wait(goal_msg)
            if action_status != actionlib.GoalStatus.SUCCEEDED:
                return None

            action_result = self.lookup_transform_client.get_result()

            rotation = Rotation.from_quat(
                [
                    action_result.transform.transform.rotation.x,
                    action_result.transform.transform.rotation.y,
                    action_result.transform.transform.rotation.z,
                    action_result.transform.transform.rotation.w,
                ]
            )
            translation = numpy.array(
                [
                    action_result.transform.transform.translation.x,
                    action_result.transform.transform.translation.y,
                    action_result.transform.transform.translation.z,
                ]
            )

            transformation_matrix = numpy.identity(4)
            transformation_matrix[:3, :3] = rotation.as_matrix()
            transformation_matrix[:3, 3] = translation

            self.transformation_matrix = transformation_matrix

        return self.transformation_matrix

    def body_callback(self, body_msg: visualization_msgs.msg.MarkerArray) -> None:
        transformation_matrix = self._get_transformation_matrix()
        if transformation_matrix is None:
            rospy.logwarn("No transformation matrix found.")
            return

        if len(body_msg.markers) == 0:
            return

        landmarks = defaultdict(dict)

        for marker in body_msg.markers:
            joint_id = marker.id % 100
            body_id = marker.id // 100

            # Convert to rgb frame
            rotation = Rotation.from_quat(
                [
                    marker.pose.orientation.x,
                    marker.pose.orientation.y,
                    marker.pose.orientation.z,
                    marker.pose.orientation.w,
                ]
            )
            translation = numpy.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])

            pose_matrix_in_have = numpy.identity(4)
            pose_matrix_in_have[:3, :3] = rotation.as_matrix()
            pose_matrix_in_have[:3, 3] = translation

            pose_matrix_in_want = numpy.dot(self.transformation_matrix, pose_matrix_in_have)
            translation = pose_matrix_in_want[:3, 3]
            rotation = Rotation.from_matrix(pose_matrix_in_want[:3, :3])

            joint_name = K4ABT_JOINTS(joint_id).name[12:].lower()
            landmarks[body_id][joint_name] = {
                "position": dict(zip("xyz", translation)),
                "orientation": dict(zip("xyzw", rotation.as_quat())),
                "confidence": 1.0,
            }

        landmarks = {
            "header": {
                "timestamp": body_msg.markers[0].header.stamp.to_sec(),
                "frame_id": self.frame_id,
                "seq": self.step_count,
            },
            "data": {"body": landmarks},
        }

        self.step_count += 1

        self.landmarks_publisher.publish(json.dumps(landmarks))


def main():
    rospy.init_node("body_to_landmark")
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    body_to_landmark = BodyToLandmarkNode()
    body_to_landmark.run()
    body_to_landmark.stop()


if __name__ == "__main__":
    main()

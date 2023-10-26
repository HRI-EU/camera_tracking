#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Class for doing mediapipe-based tracking.
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

from typing import Dict, NamedTuple, Optional
import json
import time
import numpy
import cv2
import mediapipe as mp

from camera_tracking.base_tracking import BaseTracking


mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def face_to_dict(results_face: NamedTuple) -> Dict:
    # face_<face_index>_<landmark_index>, e.g. face_0_12
    landmarks = {}
    if results_face.multi_face_landmarks:
        for face_index, face in enumerate(results_face.multi_face_landmarks):
            for landmark_index, landmark in enumerate(face.landmark):
                landmarks[f"face_{face_index}_{landmark_index}"] = [landmark.x, landmark.y, landmark.z]

    return landmarks


def hands_to_dict(results_hands: NamedTuple) -> Dict:
    # hand_<hand_index>_<handedness>_<landmark_id>, e.g. hand_0_left_index_finger_pip
    landmarks = {}
    if results_hands.multi_hand_landmarks:
        for hand_index, hand in enumerate(results_hands.multi_hand_landmarks):
            landmark = hand.landmark
            handedness = results_hands.multi_handedness[hand_index].classification[0].label.lower()
            for landmark_id in mp_hands.HandLandmark:
                landmarks[f"hand_{hand_index}_{handedness}_{str(landmark_id).lower()[13:]}"] = [
                    landmark[landmark_id].x,
                    landmark[landmark_id].y,
                    landmark[landmark_id].z,
                    landmark[landmark_id].visibility,
                ]

    return landmarks


def pose_to_dict(results_pose: NamedTuple) -> Dict:
    # pose_<landmark_id>, e.g. pose_right_shoulder
    landmarks = {}
    if results_pose.pose_world_landmarks:
        landmark = results_pose.pose_world_landmarks.landmark
        for landmark_id in mp_pose.PoseLandmark:
            landmarks[f"pose_{str(landmark_id).lower()[13:]}"] = [
                landmark[landmark_id].x,
                landmark[landmark_id].y,
                landmark[landmark_id].z,
                landmark[landmark_id].visibility,
            ]

    return landmarks


# Create a string - string dictionary like this:
# 0   HandLandmark.THUMB_IP            HandLandmark.THUMB_TIP
# 1   HandLandmark.WRIST               HandLandmark.INDEX_FINGER_MCP
# 2   HandLandmark.PINKY_MCP           HandLandmark.PINKY_PIP
def get_connections(connections) -> Dict:
    result_connections = {}
    for connection_index, connection_id in enumerate(connections):
        result_connections[f"{connection_index}"] = [str(connection_id[0]), str(connection_id[1])]

    return result_connections


class MediapipeTracking(BaseTracking):
    default_options = {"face": True, "hands": True, "pose": True}

    def __init__(self, options: Optional[Dict] = None, visualize: bool = True):
        self.options = self.default_options if options is None else options
        super().__init__("mediapipe", visualize=visualize)

    def process(self, data: numpy.ndarray, options: Optional[Dict] = None):
        """
        Process an image.
        @param data: The image to be processed. If the image is colored we assume BGR.
        @param options: A dictionary defining which landmarks to compute.
        @return: The found landmarks.
        """

        if options is None:
            options = self.default_options

        landmarks = {}

        if not options:
            return landmarks

        # Check whether connection data should be returned.
        if options.get("face_connections", False):
            landmarks.update(get_connections(mp_face.FACE_CONNECTIONS))
        if options.get("hand_connections", False):
            landmarks.update(get_connections(mp_hands.HAND_CONNECTIONS))
        if options.get("pose_connections", False):
            landmarks.update(get_connections(mp_pose.POSE_CONNECTIONS))

        with_face = options.get("face", False)
        with_hands = options.get("hands", False)
        with_pose = options.get("pose", False)

        if not with_face and not with_hands and not with_pose:
            return landmarks

        # Convert the BGR image to RGB.
        rgb_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        rgb_image.flags.writeable = False

        if with_hands:
            results_hands = hands.process(rgb_image)
            landmarks.update(hands_to_dict(results_hands))

        if with_face:
            results_face = face_mesh.process(rgb_image)
            landmarks.update(face_to_dict(results_face))

        if with_pose:
            results_pose = pose.process(rgb_image)
            landmarks.update(pose_to_dict(results_pose))

        rgb_image.flags.writeable = True

        if self.visualize:
            # Draw the hand annotations on the image.
            self.visualization = data.copy()

            if with_pose and results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(self.visualization, results_pose.pose_landmarks)

            if with_face and results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    mp_drawing.draw_landmarks(self.visualization, face_landmarks)

            if with_hands and results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(self.visualization, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return landmarks


def main():
    image = cv2.imread("data/test/mediapipe_test.jpg")
    mediapipe_tracking = MediapipeTracking()
    landmarks = mediapipe_tracking.process(image, options={"face": True, "hands": True, "pose": True})
    print(f"Landmarks are:\n{landmarks}")

    # Store the found landmarks.
    with open("data/test/mediapipe_test_landmarks.json", "w", encoding="utf-8") as file:
        json.dump(landmarks, file)

    # Compare to reference landmarks.
    with open("data/test/mediapipe_test_reference_landmarks.json", encoding="utf-8") as file:
        reference_landmarks = json.load(file)
    if landmarks != reference_landmarks:
        print(f"Mismatch detected. Reference landmarks are:\n{reference_landmarks}")


if __name__ == "__main__":
    main()

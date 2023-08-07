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

import json
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def face_to_dict(landmarks, results_face):
    # face_<face_index>_<landmark_index>, e.g. face_0_12
    if results_face.multi_face_landmarks:
        for face_index, face in enumerate(results_face.multi_face_landmarks):
            for landmark_index, landmark in enumerate(face.landmark):
                landmarks[f"face_{face_index}_{landmark_index}"] = [landmark.x, landmark.y, landmark.z]

    return landmarks


def hands_to_dict(landmarks, results_hands):
    # hand_<hand_index>_<handedness>_<landmark_id>, e.g. hand_0_left_index_finger_pip
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


def pose_to_dict(landmarks, results_pose):
    # pose_<landmark_id>, e.g. pose_right_shoulder
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
def get_connections(connections):
    result_connections = {}
    for connection_index, connection_id in enumerate(connections):
        result_connections[f"{connection_index}"] = {str(connection_id[0]), str(connection_id[1])}

    return result_connections


def process(image, process_command, show_image_overlay=True):
    received_json = json.loads(process_command)
    if not received_json:
        return {}

    # Check whether just connection data should be returned.
    if "face_connections" in received_json:
        return get_connections(mp_face.FACE_CONNECTIONS)
    if "hand_connections" in received_json:
        return get_connections(mp_hands.HAND_CONNECTIONS)
    if "pose_connections" in received_json:
        return get_connections(mp_pose.POSE_CONNECTIONS)

    with_face = "face" in received_json
    with_hands = "hands" in received_json
    with_pose = "pose" in received_json

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False

    landmarks = {}

    if with_hands:
        results_hands = hands.process(image)
        hands_to_dict(landmarks, results_hands)

    if with_face:
        results_face = face_mesh.process(image)
        face_to_dict(landmarks, results_face)

    if with_pose:
        results_pose = pose.process(image)
        pose_to_dict(landmarks, results_pose)

    image.flags.writeable = True

    # Draw the hand annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if show_image_overlay:
        # image = cv2.resize(image, (640, 480))

        if with_pose and results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(image, results_pose.pose_landmarks)

        if with_face and results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks)

        if with_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                print(
                    "Index finger tip coordinates: (",
                    f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x}, "
                    f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y})",
                )

        cv2.namedWindow("MediaPipe", cv2.WINDOW_NORMAL)
        cv2.imshow("MediaPipe", image)

    return landmarks


def main():
    image = cv2.imread("mediapipe_test.jpg")
    landmarks = process(image, process_command=b'{"face":true,"hands":true,"pose":true}')
    print(f"Landmarks are:\n{landmarks}")

    # Store the found landmarks.
    with open("mediapipe_test_landmarks.json", "w") as file:
        json.dump(landmarks, file)

    # Compare to reference landmarks.
    with open("mediapipe_test_reference_landmarks.json") as file:
        reference_landmarks = json.load(file)
    if landmarks != reference_landmarks:
        print(f"Mismatch detected. Reference landmarks are:\n{reference_landmarks}")


if __name__ == "__main__":
    main()

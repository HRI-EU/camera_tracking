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

import time
import logging
import json
import argparse
import zmq
import cv2
import track_mediapipe
import track_aruco
from camera_helper import load_camera_parameters


# See: https://stackoverflow.com/questions/8230315/how-to-json-serialize-sets
def serialize_sets(object_):
    if isinstance(object_, set):
        return list(object_)
    return object_


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Python visual perception framework."
    )
    parser.add_argument(
        "camera_config_file",
        type=str,
        default="Logitech-C920.yaml",
        help="the config file of the camera to use",
    )
    parser.add_argument("--standalone", default=False, action="store_true", help="run without networking")
    parser.add_argument("--mediapipe", default=False, action="store_true", help="enable tracking of mediapipe markers")
    parser.add_argument("--aruco", default=False, action="store_true", help="enable tracking of aruco markers")
    parser.add_argument("--visualize", default=False, action="store_true", help="visualize markers")
    parser.add_argument(
        "--log-level",
        default="info",
        type=str.lower,
        choices=["error", "warning", "info", "debug"],
        help="the logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=args.log_level.upper())

    if args.standalone:
        print("Running stand-alone.")
    else:
        print("Running with networking.")

        # Start socket server
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")

    camera_parameters = load_camera_parameters(args.camera_config_file)

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))  # Depends on fourcc available camera.
    capture.set(cv2.CAP_PROP_FPS, camera_parameters["frames_per_second"])
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_parameters["width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_parameters["height"])

    # Initialize statistics.
    iteration = 0
    report_interval = 20
    start_time = time.time()

    # Define default request.
    mediapipe_command = b'{"face":true,"hands":true}'

    while capture.isOpened():
        success, image = capture.read()

        # Override request by client.
        if not args.standalone:
            mediapipe_command = socket.recv()
            logging.debug(f"Received request '{mediapipe_command}'.")

        landmarks = {}
        if args.mediapipe:
            mediapipe_landmarks = track_mediapipe.process(image, mediapipe_command, show_image_overlay=args.visualize)
            landmarks.update(mediapipe_landmarks)
        if args.aruco:
            aruco_landmarks = track_aruco.process(
                image,
                camera_parameters["camera_matrix"],
                camera_parameters["distortion_coefficients"],
                show_image_overlay=args.visualize,
            )
            landmarks.update(aruco_landmarks)

        if not args.standalone:
            json_object = json.dumps(landmarks, default=serialize_sets)
            socket.send_string(json_object)

        iteration += 1
        if iteration % report_interval == 0:
            end_time = time.time()
            print(
                f"Iteration: {iteration} / "
                f"Average time of last {report_interval} iterations: {(end_time - start_time) / report_interval:.4f}s."
            )
            start_time = end_time

        # Break on press of ESC.
        if cv2.waitKey(5) & 0xFF == 27:
            break

    capture.release()


if __name__ == "__main__":
    main()

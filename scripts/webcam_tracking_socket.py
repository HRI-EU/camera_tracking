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

import logging
import json
import argparse
import zmq

from camera_tracking.webcam_tracking import WebcamTracking


class WebcamTrackingSocket:
    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Perception using webcam."
        )
        parser.add_argument("camera_config_file", type=str, help="the config file of the camera to use")
        parser.add_argument("--standalone", default=False, action="store_true", help="run without networking")
        parser.add_argument("--aruco", default=False, action="store_true", help="enable tracking of aruco markers")
        parser.add_argument("--mediapipe", default=False, action="store_true", help="enable tracking with mediapipe")
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

        self.webcam_tracking = WebcamTracking(
            args.camera_config_file, with_aruco=args.aruco, with_mediapipe=args.mediapipe, visualize=args.visualize
        )
        self.standalone = args.standalone

        if self.standalone:
            print("Running stand-alone.")
        else:
            print("Running with networking.")

            # Start socket server.
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind("tcp://*:5555")

    def run(self):
        while self.webcam_tracking.capture.isOpened():
            # Override request by client.
            if not self.standalone:
                mediapipe_command = self.socket.recv()
                logging.debug(f"Received request '{mediapipe_command}'.")

            landmarks = self.webcam_tracking.step()

            if not self.standalone:
                json_object = json.dumps(landmarks)
                self.socket.send_string(json_object)

    def stop(self):
        self.webcam_tracking.stop()


def main():
    webcam_tracking = WebcamTrackingSocket()
    try:
        webcam_tracking.run()
    except KeyboardInterrupt:
        pass
    finally:
        webcam_tracking.stop()


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Azure tracking using socket interface.
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
import argparse
import zmq

from camera_tracking.azure_tracking import AzureTracking


class AzureTrackingSocket:
    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Perception using Kinect Azure."
        )
        parser.add_argument("--standalone", default=False, action="store_true", help="run without networking")
        parser.add_argument("--aruco", default=False, action="store_true", help="enable tracking of aruco markers")
        parser.add_argument("--body", default=False, action="store_true", help="enable tracking of body skeleton")
        parser.add_argument("--mediapipe", default=False, action="store_true", help="enable tracking with mediapipe")
        parser.add_argument("--visualize", default=False, action="store_true", help="visualize markers")
        parser.add_argument(
            "--frame-id", default="", type=str, help="the ID of the frame that is attached to the landmarks"
        )
        parser.add_argument(
            "--color-resolution",
            default="1536P",
            type=str,
            choices=AzureTracking.color_resolution_mapping,
            help="the color resolution to use",
        )
        parser.add_argument(
            "--depth-mode",
            default="NFOV_2X2BINNED",
            type=str,
            choices=AzureTracking.depth_mode_mapping,
            help="the depth mode to use",
        )

        parser.add_argument(
            "--log-level",
            default="info",
            type=str.lower,
            choices=["error", "warning", "info", "debug"],
            help="the logging level",
        )
        args = parser.parse_args()

        logging.basicConfig(format="%(levelname)s: %(message)s", level=args.log_level.upper())

        self.azure_tracking = AzureTracking(
            with_aruco=args.aruco,
            with_body=args.body,
            with_mediapipe=args.mediapipe,
            visualize=args.visualize,
            color_resolution=args.color_resolution,
            depth_mode=args.depth_mode,
            frame_id=args.frame_id,
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
        while True:
            # Override request by client.
            if not self.standalone:
                mediapipe_command = self.socket.recv()
                logging.debug(f"Received request '{mediapipe_command}'.")

            landmarks = self.azure_tracking.step()

            if not self.standalone:
                json_object = json.dumps(landmarks)
                self.socket.send_string(json_object)

    def stop(self):
        self.azure_tracking.stop()


def main():
    azure_tracking = AzureTrackingSocket()
    try:
        azure_tracking.run()
    except KeyboardInterrupt:
        pass
    finally:
        azure_tracking.stop()


if __name__ == "__main__":
    main()

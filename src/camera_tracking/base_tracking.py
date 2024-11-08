#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Base classes used for different marker trackers.
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

from __future__ import annotations

import time
import queue
import threading
from typing import Callable
from collections import OrderedDict
import cv2


class BaseTracking:
    def __init__(self, name: str, visualize: bool = True):
        self.name = name
        self.visualize = visualize
        self.visualization = None
        self.sum_processing_time = 0.0
        self.window_name = f"Tracking {self.name}"

    def show_visualization(self):
        if self.visualization is not None:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, self.visualization)
            cv2.waitKey(1)

    def process(self, data):
        raise NotImplementedError("Function 'process' is not implemented in the base class.")


class ThreadedTracker:
    def __init__(self, tracker: BaseTracking, input_function: Callable):
        self.input_function = input_function
        self.tracker = tracker
        self.input = queue.Queue()
        self.output = queue.Queue()
        self.thread = threading.Thread(target=self.worker)
        self.thread.start()

    def worker(self):
        while True:
            data = self.input.get()
            if data is None:
                break

            # Compute landmarks and put them into the output buffer.
            try:
                start_time = time.time()
                success, input_data = self.input_function(data)
                if not success:
                    print(f"Data is not available for '{self.tracker.name}'.")
                    self.output.put({})
                    continue

                landmarks = self.tracker.process(input_data)
                self.tracker.sum_processing_time += time.time() - start_time
                self.output.put(landmarks)

            except Exception:
                # A 'None' signals the outside that the thread crashed.
                self.output.put(None)
                raise

    def trigger(self, data):
        self.input.put(data)


class BaseCamera:
    def __init__(self, frame_id=""):
        self.trackers = OrderedDict()

        self.frame_id = frame_id

        # Initialize statistics.
        self.step_count = 0
        self.sum_overall_time = 0.0
        self.sum_capture_time = 0.0
        self.report_interval = 30

    def get_capture(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    def step(self, process: bool = True) -> dict:
        start_time = time.time()

        # Get capture.
        capture = self.get_capture()
        capture_time = time.time()
        landmarks = {
            "header": {"timestamp": capture_time, "frame_id": self.frame_id, "seq": self.step_count},
            "data": {},
        }

        if process:
            capture_time = time.time()
            self.sum_capture_time += capture_time - start_time

            # Trigger trackers in given order (decreasing processing time).
            for tracker in self.trackers.values():
                tracker.trigger(capture)

            # Wait for tracker results in reversed order (increasing processing time)
            for tracker in reversed(self.trackers.values()):
                landmarks["data"][tracker.tracker.name] = tracker.output.get()
                tracker.tracker.show_visualization()

            self.sum_overall_time += time.time() - start_time
            if self.step_count % self.report_interval == 0:
                status = (
                    f"Step {self.step_count} mean times: "
                    f"overall {self.sum_overall_time / self.report_interval:.3f}s"
                    f" | capture {self.sum_capture_time / self.report_interval:.3f}s"
                    f" | processing: {(self.sum_overall_time - self.sum_capture_time) / self.report_interval:.3f}s"
                )
                for tracker in self.trackers.values():
                    status += (
                        f" | {tracker.tracker.name} {tracker.tracker.sum_processing_time / self.report_interval:.3f}s"
                    )
                    tracker.tracker.sum_processing_time = 0.0

                print(status)
                self.sum_overall_time = 0.0
                self.sum_capture_time = 0.0
        else:
            if self.step_count % self.report_interval == 0:
                print(f"Step {self.step_count} idle.")

        self.step_count += 1

        return landmarks

    def stop(self):
        for tracker in self.trackers.values():
            tracker.input.put(None)
            tracker.thread.join()

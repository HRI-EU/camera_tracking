#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Base classes used for different marker trackers.
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


import queue
import threading
from typing import Callable
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
            success, data = self.input.get()
            if not success:
                print(f"Data is not available for {self.tracker.name}.")
                self.output.put({})
                continue

            if data is None:
                break

            # Compute landmarks and put them into the output buffer.
            try:
                landmarks = self.tracker.process(data)
                self.output.put(landmarks)
            except Exception:
                # A 'None' signals the outside that the thread crashed.
                self.output.put(None)
                raise

    def trigger(self, data):
        self.input.put(self.input_function(data))

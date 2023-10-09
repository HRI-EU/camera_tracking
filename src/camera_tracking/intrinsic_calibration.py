#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Class for doing intrinsic camera calibration.
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

import cv2
import numpy


class IntrinsicCalibration:
    def __init__(self):
        self.chessboard_shape = (7, 6)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.image_points = []
        self.image_size = None

    def add_image(self, image: numpy.ndarray) -> bool:
        """
        Add an image to the calibration process.
        @param image: The image to be added. If the image is colored we assume BGR.
        @return: Whether the calibration board was detected in the image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.image_size is None:
            self.image_size = gray.shape
        else:
            if self.image_size != gray.shape:
                raise AssertionError(f"Expected image size {self.image_size} but got {gray.shape}.")

        # Find the chess board corners.
        success, corners = cv2.findChessboardCorners(gray, self.chessboard_shape, None)
        # If found, add image points (after refining them).
        if success:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            self.image_points.append(corners2)

        return success

    def calibrate(self):
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0).
        object_point_grid = numpy.zeros((self.chessboard_shape[0] * self.chessboard_shape[1], 3), numpy.float32)
        object_point_grid[:, :2] = numpy.mgrid[0 : self.chessboard_shape[0], 0 : self.chessboard_shape[1]].T.reshape(
            -1, 2
        )
        object_points = [object_point_grid] * len(self.image_points)

        success, camera_matrix, distortion_coefficients, rotations, translations = cv2.calibrateCamera(
            object_points, self.image_points, self.image_size[::-1], None, None
        )

        if not success:
            raise AssertionError("Something went wrong.")

        mean_error = 0
        for i in range(len(self.image_points)):
            projected_image_points, _ = cv2.projectPoints(
                object_points[i], rotations[i], translations[i], camera_matrix, distortion_coefficients
            )
            error = cv2.norm(self.image_points[i], projected_image_points, cv2.NORM_L2) / len(projected_image_points)
            mean_error += error

        print(f"Total error: {mean_error / len(object_points)}.")


def main():
    intrinsic_calibration = IntrinsicCalibration()
    intrinsic_calibration.calibrate()


if __name__ == "__main__":
    main()

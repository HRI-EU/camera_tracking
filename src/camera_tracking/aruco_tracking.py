#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Class that offers tracking of Aruco markers.
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

from __future__ import annotations

import math
from typing import Optional, Callable
from collections import defaultdict, deque
import json
import time
from difflib import SequenceMatcher
import cv2
import numpy
from scipy.spatial.transform import Rotation

from camera_tracking.camera_helper import load_camera_parameters, save_camera_parameters
from camera_tracking.base_tracking import BaseTracking


def get_highlighted_string_difference(before: Optional[str], after: str) -> tuple[bool, str]:
    if before is None:
        return True, after

    if before == after:
        return False, after

    matches = SequenceMatcher(None, before, after).get_matching_blocks()
    highlighted_after = ""
    index_after = 0
    for match in matches:
        if match.b > index_after:
            highlighted_after += f"\033[94m{after[index_after : match.b]}\033[0m"
        index_after = match.b + match.size
        highlighted_after += after[match.b : index_after]

    return True, highlighted_after


class TrackID:
    track_id = 0

    @classmethod
    def get(cls) -> int:
        cls.track_id += 1
        return cls.track_id


class ArucoTrackOrientation:
    def __init__(self, solution: dict, get_track_id: Callable = TrackID.get) -> None:
        self.track_id = get_track_id()
        self.queue = deque(maxlen=5)
        self.last_position = None
        self.last_orientation = None
        self.update(solution)

    def update(self, solution: Optional[dict]) -> bool:
        self.queue.append(solution)
        if solution is not None:
            self.last_position = solution["position"]
            self.last_orientation = solution["orientation"]
            return True
        else:
            return self.get_sample_number() > 0

    def get_sample_number(self) -> int:
        return sum(1 for item in self.queue if item is not None)

    def get_reprojection_error(self) -> float:
        return sum(item["reprojection_error"] for item in self.queue if item is not None)

    def get_mean_position(self) -> dict:
        count = 0
        mean_position = numpy.zeros(shape=(3,))
        for item in reversed(self.queue):
            if item is not None:
                position = numpy.array([item["position"]["x"], item["position"]["y"], item["position"]["z"]])
                mean_position += position
                count += 1
                if count == 3:
                    break

        mean_position /= count
        return dict(zip("xyz", mean_position))

    def get_mean_orientation(self) -> dict:
        count = 0
        mean_orientation = numpy.zeros(shape=(4,))
        last_orientation = None
        for item in reversed(self.queue):
            if item is not None:
                orientation = numpy.array(
                    [
                        item["orientation"]["x"],
                        item["orientation"]["y"],
                        item["orientation"]["z"],
                        item["orientation"]["w"],
                    ]
                )

                if last_orientation is not None:
                    angle = math.degrees(angle_between_quaternions(last_orientation, item["orientation"]))
                    if angle > 20:
                        print(
                            f"Strong angle [deg] {angle}: last {last_orientation} "
                            f"current {item['orientation']} track_id {self.track_id} count {count}."
                        )

                if count and numpy.dot(mean_orientation, orientation) < 0.0:
                    orientation *= -1

                mean_orientation += orientation
                last_orientation = item["orientation"]

                count += 1
                if count == 3:
                    break

        mean_orientation /= numpy.linalg.norm(mean_orientation)
        return dict(zip("xyzw", mean_orientation))


class ArucoTrackPosition:
    def __init__(self, solutions: list[dict], marker_id: int) -> None:
        self.orientation_tracks = []
        self.marker_id = marker_id
        self.best_orientation = None
        for solution_index, solution in enumerate(solutions):
            self.orientation_tracks.append(ArucoTrackOrientation(solution))

    def update(self, solutions: Optional[list[dict]] = None) -> bool:
        if solutions is None:
            non_empty_track = self.orientation_tracks[0].update(None)
            if not non_empty_track:
                return False

            self.orientation_tracks[1].update(None)
            return True

        # Compute angles between all pairs of tracks and solutions.
        angles = numpy.empty(shape=(2, 2))
        for orientation_track_index, orientation_track in enumerate(self.orientation_tracks):
            for solution_index, solution in enumerate(solutions):
                angles[orientation_track_index, solution_index] = angle_between_quaternions(
                    orientation_track.last_orientation, solution["orientation"]
                )

        a = numpy.array([angles[0, 0], angles[1, 1]])
        b = numpy.array([angles[0, 1], angles[1, 0]])

        if sum(a) < sum(b):
            min_pair = [0, 0]
        else:
            min_pair = [0, 1]

        angle = math.degrees(
            angle_between_quaternions(
                self.orientation_tracks[0].last_orientation, self.orientation_tracks[1].last_orientation
            )
        )
        print(
            f"s0e={solutions[0]['reprojection_error']:5.3f} s1e={solutions[1]['reprojection_error']:5.3f} "
            f"angle={angle:5.3f} max_a={math.degrees(max(a)):5.2f} max_b={math.degrees(max(b)):5.2f} "
            f"sum_a={math.degrees(sum(a)):5.2f} sum_b={math.degrees(sum(b)):5.2f} "
            f"angles=[{math.degrees(angles[0,0]):5.2f}, {math.degrees(angles[0,1]):5.2f}, {math.degrees(angles[1,0]):5.2f}, {math.degrees(angles[1,1]):5.2f}]"
            f"min_pair = {min_pair}"
        )

        # Get the best matching pair.
        # min_pair = numpy.unravel_index(numpy.argmin(angles), angles.shape)
        # Update the best matching pair.
        self.orientation_tracks[min_pair[0]].update(solutions[min_pair[1]])
        # Update the remaining pair.
        self.orientation_tracks[(min_pair[0] + 1) % 2].update(solutions[(min_pair[1] + 1) % 2])

        return True

    def get_sample_number(self):
        return self.orientation_tracks[0].get_sample_number()

    def get_best_orientation_track(self) -> Optional[dict]:
        reprojection_errors = numpy.array([track.get_reprojection_error() for track in self.orientation_tracks])
        self.best_orientation = reprojection_errors.argmin()

        sample_number = self.get_sample_number()
        if sample_number < 3:
            return None

        return {
            "position": self.orientation_tracks[self.best_orientation].get_mean_position(),
            "orientation": self.orientation_tracks[self.best_orientation].get_mean_orientation(),
            "sample_number": sample_number,
            "track_id": self.orientation_tracks[self.best_orientation].track_id,
        }

    def status(self) -> str:
        return f"{self.orientation_tracks[self.best_orientation].track_id}:{self.get_sample_number()}"


def distance_between_points(point_1: dict, point_2: dict) -> float:
    return math.sqrt(
        (point_1["x"] - point_2["x"]) ** 2 + (point_1["y"] - point_2["y"]) ** 2 + (point_1["z"] - point_2["z"]) ** 2
    )


def angle_between_quaternions(quaternion_1: dict, quaternion_2: dict) -> float:
    delta_w = (
        quaternion_1["x"] * quaternion_2["x"]
        + quaternion_1["y"] * quaternion_2["y"]
        + quaternion_1["z"] * quaternion_2["z"]
        + quaternion_1["w"] * quaternion_2["w"]
    )
    return 2 * math.acos(abs(min(delta_w, 1.0)))


class ArucoTrackMarker:
    distance_threshold = 0.3 * 25

    def __init__(self, marker_id: int) -> None:
        self.marker_id = marker_id
        self.position_tracks = []

    def update(self, marker_detections: list[dict]) -> bool:
        track_to_detection_assignment = {}
        min_number = min(len(marker_detections), len(self.position_tracks))

        if min_number:
            # Compute distances between all pairs of tracks and detections.
            distances = numpy.empty(shape=(len(self.position_tracks), len(marker_detections)))
            for position_track_index, position_track in enumerate(self.position_tracks):
                for marker_detection_index, marker_detection in enumerate(marker_detections):
                    distances[position_track_index, marker_detection_index] = distance_between_points(
                        position_track.orientation_tracks[0].last_position, marker_detection["position"]
                    )

            # Greedily assign tracks to detections.
            for _ in range(min_number):
                min_pair = numpy.unravel_index(numpy.argmin(distances), distances.shape)
                distance = distances[min_pair]
                if distance > ArucoTrackMarker.distance_threshold:
                    print(
                        f"Distance exceeded {self.marker_id} for pair {min_pair}: {distance} > {ArucoTrackMarker.distance_threshold}"
                    )
                    break

                track_to_detection_assignment[min_pair[0]] = min_pair[1]
                distances[min_pair[0], :] = numpy.inf
                distances[:, min_pair[1]] = numpy.inf

        # Update assigned pairs.
        tracks_to_delete = []
        for position_track_index, position_track in enumerate(self.position_tracks):
            if position_track_index in track_to_detection_assignment:
                position_track.update(
                    marker_detections[track_to_detection_assignment[position_track_index]]["solutions"]
                )
            else:
                if not position_track.update():
                    tracks_to_delete.append(position_track_index)

        self.position_tracks = [
            track for index, track in enumerate(self.position_tracks) if index not in tracks_to_delete
        ]

        # New track for unassigned detections.
        for marker_detection_index, marker_detection in enumerate(marker_detections):
            if marker_detection_index not in track_to_detection_assignment.values():
                self.position_tracks.append(ArucoTrackPosition(marker_detection["solutions"], self.marker_id))

        return len(self.position_tracks) > 0

    def get_best_tracks(self) -> Optional[list[dict]]:
        # Choose position track with the largest number of detections.
        # Ignore tracks with less than 3 detections.
        # From the chosen position track take the orientation track with the lowest reprojection_error.

        best_tracks = [
            best_orientation_track
            for position_track in self.position_tracks
            if (best_orientation_track := position_track.get_best_orientation_track()) is not None
        ]
        if not best_tracks:
            return None

        # Sort tracks by descending sample number.
        best_tracks.sort(key=lambda x: x["sample_number"], reverse=True)
        return best_tracks

    def status(self) -> str:
        position_tracks_status = [position_track.status() for position_track in self.position_tracks]
        return f"{self.marker_id}=[" + "|".join(position_tracks_status) + "]"


class ArucoTrackAll:
    def __init__(self) -> None:
        self.marker_tracks = {}
        self.last_status = None

    def update(self, landmarks: dict) -> dict:
        # Add an empty marker track for previously untracked markers.
        for landmark in landmarks.values():
            marker_id = landmark[0]["id"]
            if marker_id not in self.marker_tracks:
                self.marker_tracks[marker_id] = ArucoTrackMarker(marker_id)

        tracks_to_delete = []
        for marker_id, marker_track in self.marker_tracks.items():
            non_empty_track = marker_track.update(landmarks.get(f"aruco_{marker_id}", []))
            if not non_empty_track:
                tracks_to_delete.append(marker_id)

        for marker_id in tracks_to_delete:
            del self.marker_tracks[marker_id]

        tracked_landmarks = {
            f"aruco_{marker_id}": tracked_landmark
            for marker_id, marker_track in self.marker_tracks.items()
            if (tracked_landmark := marker_track.get_best_tracks()) is not None
        }

        self.print_status()

        return tracked_landmarks

    def print_status(self) -> None:
        marker_tracks_status = [self.marker_tracks[marker_id].status() for marker_id in sorted(self.marker_tracks)]
        status = "Aruco tracks: " + " ".join(marker_tracks_status)
        status_differs, highlighted_status = get_highlighted_string_difference(self.last_status, status)
        if status_differs:
            print(highlighted_status)
        self.last_status = status


class ArucoTracking(BaseTracking):
    def __init__(
        self,
        camera_matrix: numpy.ndarray,
        distortion_coefficients: numpy.ndarray,
        visualize: bool = True,
        with_tracking: bool = False,
        save_images: bool = False,
    ):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_parameters.cornerRefinementWinSize = 11
        self.aruco_parameters.relativeCornerRefinmentWinSize = 0.4  # 0.3
        # self.aruco_parameters.cornerRefinementMaxIterations = 100  # 30
        # self.aruco_parameters.cornerRefinementMinAccuracy = 0.01  # 0.1
        self.aruco_parameters.minDistanceToBorder = 2
        self.aruco_parameters.minMarkerPerimeterRate = 0.03
        self.aruco_parameters.maxMarkerPerimeterRate = 0.3

        # We use a default length of 1m, and scale the markers on the receiving side.
        self.default_marker_length = 1.0
        self.object_points = numpy.array(
            [
                [-self.default_marker_length / 2, self.default_marker_length / 2, 0],
                [self.default_marker_length / 2, self.default_marker_length / 2, 0],
                [self.default_marker_length / 2, -self.default_marker_length / 2, 0],
                [-self.default_marker_length / 2, -self.default_marker_length / 2, 0],
            ],
        )

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.save_images = save_images

        self.tracking = ArucoTrackAll() if with_tracking else None

        if self.save_images:
            save_camera_parameters(self.camera_matrix, self.distortion_coefficients, "camera.yaml")

        super().__init__("aruco", visualize=visualize)

    def write_detector_parameters(self, filename: str) -> None:
        self.aruco_parameters.writeDetectorParameters(
            cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE),
            "aruco_parameters",
        )

    def process(self, data: numpy.ndarray) -> dict:
        """
        Process an image.
        @param data: The image to be processed. If the image is colored we assume BGR.
        @return: The found aruco landmarks.
        """

        if self.save_images:
            cv2.imwrite(f"{time.time()}.png", data)

        # Find all markers in the image.
        corners, ids, _ = cv2.aruco.detectMarkers(data, self.aruco_dict, parameters=self.aruco_parameters)

        if self.visualize:
            # Make a copy of the image, as the outside might assume that we do not alter the image.
            # In case of a grayscale image, convert it to BGR as this looks nicer.
            if len(data.shape) == 2:
                self.visualization = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
            else:
                self.visualization = data.copy()

            # Visualize the found markers.
            cv2.aruco.drawDetectedMarkers(self.visualization, corners, ids)

        landmarks = defaultdict(list)
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                # Solutions are sorted by reprojection error of SOLVEPNP_IPPE_SQUARE in undistorted image coordinates.
                # But the returned reprojection error is from solvePnPGeneric which is in distorted image coordinates.
                # Therefore, it can happen that the given reprojection error is not sorted (see data/log/aruco.txt)
                number, rotations, translations, reprojection_errors = cv2.solvePnPGeneric(
                    self.object_points,
                    marker_corners[0],
                    self.camera_matrix,
                    self.distortion_coefficients,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,  # default is iterative
                )

                if number != 2:
                    raise AssertionError(f"Expected two solutions but got {number}.")

                solutions = [
                    {
                        "id": marker_id,
                        "position": dict(zip("xyz", translation.flatten())),
                        "orientation": dict(zip("xyzw", Rotation.from_rotvec(rotation.flatten()).as_quat())),
                        "reprojection_error": reprojection_error[0],
                    }
                    for translation, rotation, reprojection_error in zip(translations, rotations, reprojection_errors)
                ]

                # Store information as list of 3 + 4 values.
                landmarks[f"aruco_{solutions[0]['id']}"].append({**solutions[0], "solutions": solutions})

                # Draw the axis of the marker.
                if self.visualize:
                    cv2.drawFrameAxes(
                        self.visualization,
                        self.camera_matrix,
                        self.distortion_coefficients,
                        rotations[0],
                        translations[0],
                        0.40,
                        2,
                    )

        if self.tracking is None:
            return landmarks

        return self.tracking.update(landmarks)


def main():
    camera_parameters = load_camera_parameters("data/calibration/Logitech-C920.yaml")
    image = cv2.imread("data/test/aruco_test.jpg")
    aruco_tracking = ArucoTracking(camera_parameters["camera_matrix"], camera_parameters["distortion_coefficients"])
    landmarks = aruco_tracking.process(image)
    print(f"Landmarks are:\n{landmarks}")

    # Store the found landmarks.
    with open("data/test/aruco_test_landmarks.json", "w", encoding="utf-8") as file:
        json.dump(landmarks, file)

    # Compare to reference landmarks.
    with open("data/test/aruco_test_reference_landmarks.json", encoding="utf-8") as file:
        reference_landmarks = json.load(file)
    if landmarks != reference_landmarks:
        print(f"Mismatch detected. Reference landmarks are:\n{reference_landmarks}")


if __name__ == "__main__":
    main()

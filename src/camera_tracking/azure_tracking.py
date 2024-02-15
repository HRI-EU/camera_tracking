#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Class that offers tracking of markers with the azure camera.
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

from typing import Optional
import ctypes
import numpy
import yaml
import cv2
from scipy.spatial.transform import Rotation

from .base_tracking import BaseTracking, ThreadedTracker, BaseCamera
from .camera_helper import camera_parameters_from_config
from .pykinect_azure_fix import pykinect_azure as pykinect, _k4a
from .pykinect_azure_fix import K4ABT_JOINTS, get_custom_k4a_path, get_custom_k4abt_path


def get_color_camera_calibration(device: pykinect.Device, depth_mode: int, color_resolution: int) -> dict:
    calibration = device.get_calibration(depth_mode, color_resolution)

    config = {name: getattr(calibration.color_params, name) for name, _ in calibration.color_params._fields_}
    camera_parameters = camera_parameters_from_config(config)

    # We assume depth camera is the reference and thus should have an identity transform.
    depth_camera_rotation = numpy.array(calibration._handle.depth_camera_calibration.extrinsics.rotation).reshape(3, 3)
    depth_camera_translation = numpy.array(calibration._handle.depth_camera_calibration.extrinsics.translation)
    assert numpy.allclose(depth_camera_rotation, numpy.eye(3, dtype=float)) or not numpy.any(depth_camera_rotation)
    assert not numpy.any(depth_camera_translation)

    color_camera_rotation = numpy.array(calibration._handle.color_camera_calibration.extrinsics.rotation).reshape(3, 3)
    color_camera_translation = numpy.array(calibration._handle.color_camera_calibration.extrinsics.translation) / 1000.0

    # Convert to homogeneous transformation matrix.
    transformation_matrix_depth_to_color = numpy.identity(4)
    transformation_matrix_depth_to_color[:3, :3] = color_camera_rotation
    transformation_matrix_depth_to_color[:3, 3] = color_camera_translation
    camera_parameters["transformation_matrix_depth_to_color"] = transformation_matrix_depth_to_color

    return camera_parameters


# fmt: off
capabilities = [
    {"id": 0, "name": "EXPOSURE_TIME_ABSOLUTE", "supports_auto": True,
     "min_value": 500,  "max_value": 133330, "step_value": 100, "default_value": 16670, "default_mode": 0},
    { "id": 1, "name": "AUTO_EXPOSURE_PRIORITY", "supports_auto": False,
      "min_value": 0, "max_value": 0, "step_value": 0, "default_value": 0, "default_mode": 1},
    {"id": 2,  "name": "BRIGHTNESS", "supports_auto": False,
     "min_value": 0, "max_value": 255, "step_value": 1, "default_value": 128,"default_mode": 1},
    {"id": 3, "name": "CONTRAST", "supports_auto": False, "min_value": 0,
     "max_value": 10, "step_value": 1, "default_value": 5, "default_mode": 1},
    { "id": 4, "name": "SATURATION", "supports_auto": False,
      "min_value": 0, "max_value": 63, "step_value": 1, "default_value": 32, "default_mode": 1},
    {"id": 5, "name": "SHARPNESS", "supports_auto": False,
     "min_value": 0, "max_value": 4, "step_value": 1, "default_value": 2, "default_mode": 1},
    {"id": 6, "name": "WHITEBALANCE", "supports_auto": True,
     "min_value": 2500, "max_value": 12500, "step_value": 10, "default_value": 4500, "default_mode": 0},
    {"id": 7, "name": "BACKLIGHT_COMPENSATION", "supports_auto": False,
     "min_value": 0, "max_value": 1, "step_value": 1, "default_value": 0, "default_mode": 1},
    {"id": 8, "name": "GAIN", "supports_auto": False,
     "min_value": 0, "max_value": 255, "step_value": 1, "default_value": 128, "default_mode": 1},
    {"id": 9, "name": "POWERLINE_FREQUENCY", "supports_auto": False,
     "min_value": 1, "max_value": 2, "step_value": 1, "default_value": 2, "default_mode": 1},
]
# fmt: on


def set_color_control_commands(device: pykinect.Device, commands: list[dict], capabilities: dict) -> None:
    for command in commands:
        # Fill missing values with defaults.
        mode = command.get("mode", capabilities[command["name"]]["default_mode"])
        value = command.get("value", capabilities[command["name"]]["default_value"])
        id_ = capabilities[command["name"]]["id"]
        result = _k4a.k4a_device_set_color_control(device.handle(), id_, mode, value)
        if result != 0:
            raise AssertionError(
                "Could not set color control with name: {command['name']} id: {id_} mode: {mode} value: {value]}."
            )


def get_color_control_capabilities(device: pykinect.Device, commands_mapping: dict) -> dict:
    commands = {}
    for command_name, command_id in commands_mapping.items():
        supports_auto = ctypes.c_bool()
        min_value = ctypes.c_int()
        max_value = ctypes.c_int()
        step_value = ctypes.c_int()
        default_value = ctypes.c_int()
        default_mode = ctypes.c_int()
        _k4a.k4a_device_get_color_control_capabilities(
            device.handle(),
            command_id,
            supports_auto,
            min_value,
            max_value,
            step_value,
            default_value,
            default_mode,
        )
        commands[command_name] = {
            "id": command_id,
            "name": command_name,
            "supports_auto": supports_auto.value,
            "min_value": min_value.value,
            "max_value": max_value.value,
            "step_value": step_value.value,
            "default_value": default_value.value,
            "default_mode": default_mode.value,
        }

    return commands


class BodyTracking(BaseTracking):
    millimeter_to_meter_ratio = 1000.0

    def __init__(
        self,
        visualize: bool = True,
        body_max_distance: float = 0,
        transformation_matrix: Optional[numpy.ndarray] = None,
    ):
        self.transformation_matrix = transformation_matrix
        self.body_max_distance = body_max_distance
        # Initialize the body tracker.
        self.body_tracker = pykinect.start_body_tracker(pykinect.K4ABT_DEFAULT_MODEL)
        super().__init__("body", visualize=visualize)

    def body_frame_to_landmarks(self, body_frame: pykinect.Frame) -> dict[str, dict]:
        # body_<joint_id>_<body_id>, e.g. body_right_hand_0
        landmarks = {}
        for body in body_frame.get_bodies():
            joints = {}
            for joint in body.joints:
                rotation = Rotation.from_quat(
                    [joint.orientation.x, joint.orientation.y, joint.orientation.z, joint.orientation.w]
                )
                translation = (
                    numpy.array([joint.position.x, joint.position.y, joint.position.z])
                    / BodyTracking.millimeter_to_meter_ratio
                )

                if self.transformation_matrix is not None:
                    pose_matrix_in_have = numpy.identity(4)
                    pose_matrix_in_have[:3, :3] = rotation.as_matrix()
                    pose_matrix_in_have[:3, 3] = translation

                    pose_matrix_in_want = numpy.dot(self.transformation_matrix, pose_matrix_in_have)
                    translation = pose_matrix_in_want[:3, 3]
                    rotation = Rotation.from_matrix(pose_matrix_in_want[:3, :3])

                joint_name = K4ABT_JOINTS(joint.id).name[12:].lower()
                joints[joint_name] = {
                    "position": dict(zip("xyz", translation)),
                    "orientation": dict(zip("xyzw", rotation.as_quat())),
                    "confidence": joint.confidence_level,
                }

            landmarks[body.handle().id] = joints

        return landmarks

    @staticmethod
    def image_as_numpy_array(image: pykinect.Image) -> numpy.ndarray:
        image_size = image.get_size()
        buffer_array = numpy.ctypeslib.as_array(image.buffer_pointer, shape=(image_size,))

        if image.format in [pykinect.K4A_IMAGE_FORMAT_DEPTH16, pykinect.K4A_IMAGE_FORMAT_IR16]:
            return numpy.frombuffer(buffer_array, dtype="<u2")

        raise AssertionError(f"Cannot handle image format '{image.format}'.")

    def process(self, data: pykinect.Capture) -> dict:
        # We check is depth and infrared are valid to prevent of body_tracker.update() from raising exception.
        depth_image_object = data.get_depth_image_object()
        ir_image_object = data.get_ir_image_object()
        if not depth_image_object.is_valid() or not ir_image_object.is_valid():
            print(f"Data is not available for '{self.name}'.")
            return {}

        # TODO: We change the original data of the capture. We assume no other tracker is using depth and IR.
        # Instead we should make a copy of the capture and update depth an IR.
        if self.body_max_distance > 0:
            # Crop the captured depth and infrared images.
            depth_image = self.image_as_numpy_array(depth_image_object)
            ir_image = self.image_as_numpy_array(ir_image_object)

            # Filter images w.r.t the depth.
            mask = depth_image > int(round(self.body_max_distance * self.millimeter_to_meter_ratio))
            depth_image[mask] = 0
            mask = depth_image == 0
            ir_image[mask] = 0

        # Get the updated body tracker frame.
        try:
            body_frame = self.body_tracker.update(data)
            body_landmarks = self.body_frame_to_landmarks(body_frame)
        except Exception as e:
            print(f"Body tracker update raised exception: {e}")
            return {}

        if self.visualize:
            # Get the depth image from the capture.
            success, depth_image = data.get_depth_image()
            if not success:
                print("Could not get depth image from capture.")
            else:
                # Convert the depth image to color and draw the skeletons.
                self.visualization = cv2.convertScaleAbs(depth_image, alpha=0.05)
                self.visualization = cv2.cvtColor(self.visualization, cv2.COLOR_GRAY2RGB)
                mask = self.visualization[:, :, 0] == 0
                self.visualization[:, :, 0][mask] = 200
                self.visualization = body_frame.draw_bodies(self.visualization)

        return body_landmarks


def get_gray_from_capture(capture):
    success, color_image = capture.get_color_image()
    if not success:
        return False, None
    return True, cv2.cvtColor(color_image, cv2.COLOR_BGRA2GRAY)


class AzureTracking(BaseCamera):
    color_resolution_mapping = {
        "720P": pykinect.K4A_COLOR_RESOLUTION_720P,
        "1080P": pykinect.K4A_COLOR_RESOLUTION_1080P,
        "1440P": pykinect.K4A_COLOR_RESOLUTION_1440P,
        "1536P": pykinect.K4A_COLOR_RESOLUTION_1536P,
        "2160P": pykinect.K4A_COLOR_RESOLUTION_2160P,
        "3072P": pykinect.K4A_COLOR_RESOLUTION_3072P,
    }

    depth_mode_mapping = {
        "NFOV_2X2BINNED": pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED,
        "NFOV_UNBINNED": pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED,
        "WFOV_2X2BINNED": pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED,
        "WFOV_UNBINNED": pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED,
    }

    fps_mapping = {
        "5": pykinect.K4A_FRAMES_PER_SECOND_5,
        "15": pykinect.K4A_FRAMES_PER_SECOND_15,
        "30": pykinect.K4A_FRAMES_PER_SECOND_30,
    }

    color_commands_mapping = {
        "EXPOSURE_TIME_ABSOLUTE": pykinect.K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
        "AUTO_EXPOSURE_PRIORITY": pykinect.K4A_COLOR_CONTROL_AUTO_EXPOSURE_PRIORITY,
        "BRIGHTNESS": pykinect.K4A_COLOR_CONTROL_BRIGHTNESS,
        "CONTRAST": pykinect.K4A_COLOR_CONTROL_CONTRAST,
        "SATURATION": pykinect.K4A_COLOR_CONTROL_SATURATION,
        "SHARPNESS": pykinect.K4A_COLOR_CONTROL_SHARPNESS,
        "WHITEBALANCE": pykinect.K4A_COLOR_CONTROL_WHITEBALANCE,
        "BACKLIGHT_COMPENSATION": pykinect.K4A_COLOR_CONTROL_BACKLIGHT_COMPENSATION,
        "GAIN": pykinect.K4A_COLOR_CONTROL_GAIN,
        "POWERLINE_FREQUENCY": pykinect.K4A_COLOR_CONTROL_POWERLINE_FREQUENCY,
    }

    def __init__(
        self,
        with_aruco: bool = True,
        with_body: bool = True,
        with_mediapipe: bool = True,
        visualize: bool = True,
        color_resolution: str = "1536P",
        depth_mode: str = "NFOV_2X2BINNED",
        fps: str = "30",
        frame_id: str = "",
        body_max_distance: float = 0.0,
        aruco_with_tracking: bool = False,
        color_control_filename: Optional[str] = None,
    ):
        super().__init__(frame_id=frame_id)

        if not any((with_aruco, with_mediapipe, with_body)):
            raise AssertionError("No tracker is enabled.")

        if color_resolution not in AzureTracking.color_resolution_mapping:
            raise AssertionError(
                f"Unknown color resolution '{color_resolution}'. "
                f"Known ones are {AzureTracking.color_resolution_mapping}."
            )

        if depth_mode not in AzureTracking.depth_mode_mapping:
            raise AssertionError(
                f"Unknown depth mode '{depth_mode}'. Known ones are {AzureTracking.depth_mode_mapping}."
            )

        if fps not in AzureTracking.fps_mapping:
            raise AssertionError(f"Unknown fps '{fps}'. Known ones are {AzureTracking.fps_mapping}.")

        # Initialize the library.
        pykinect.initialize_libraries(
            module_k4a_path=get_custom_k4a_path(), module_k4abt_path=get_custom_k4abt_path(), track_body=with_body
        )

        # Define the device configuration.
        device_config = pykinect.default_configuration
        device_config.color_resolution = (
            AzureTracking.color_resolution_mapping[color_resolution]
            if (with_aruco or with_mediapipe)
            else pykinect.K4A_COLOR_RESOLUTION_OFF
        )
        device_config.depth_mode = (
            AzureTracking.depth_mode_mapping[depth_mode] if with_body else pykinect.K4A_DEPTH_MODE_OFF
        )
        device_config.camera_fps = AzureTracking.fps_mapping[fps]
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32

        # Start device.
        self.device = pykinect.start_device(config=device_config)

        # Get the resolution specific calibration parameters of the color camera.
        color_camera_calibration = get_color_camera_calibration(
            self.device, device_config.depth_mode, device_config.color_resolution
        )

        # Get control capabilities of the color camera and set those back to defaults.
        color_camera_control_capabilities = get_color_control_capabilities(self.device, self.color_commands_mapping)
        color_camera_control_defaults = [{"name": name} for name in color_camera_control_capabilities]
        set_color_control_commands(self.device, color_camera_control_defaults, color_camera_control_capabilities)
        if color_control_filename:
            with open(color_control_filename, encoding="utf-8") as file:
                color_control_commands = yaml.load(file, Loader=yaml.SafeLoader)

            print(f"Setting color control {color_control_commands}.")
            set_color_control_commands(self.device, color_control_commands, color_camera_control_capabilities)

        # We add trackers in order of expected processing time (decreasingly).
        if with_aruco:
            from .aruco_tracking import ArucoTracking

            aruco_tracking = ArucoTracking(
                color_camera_calibration["camera_matrix"],
                color_camera_calibration["distortion_coefficients"],
                visualize=visualize,
                with_tracking=aruco_with_tracking,
            )
            self.trackers["aruco"] = ThreadedTracker(
                aruco_tracking, input_function=lambda capture: get_gray_from_capture(capture)
            )

        if with_mediapipe:
            from .mediapipe_tracking import MediapipeTracking

            mediapipe_tracking = MediapipeTracking(visualize=visualize)
            self.trackers["mediapipe"] = ThreadedTracker(
                mediapipe_tracking, input_function=lambda capture: capture.get_color_image()
            )

        if with_body:
            body_tracking = BodyTracking(
                visualize=visualize,
                body_max_distance=body_max_distance,
                transformation_matrix=color_camera_calibration["transformation_matrix_depth_to_color"],
            )
            self.trackers["body"] = ThreadedTracker(body_tracking, input_function=lambda capture: (True, capture))

    def get_capture(self):
        return self.device.update()

    def stop(self):
        self.device.close()
        super().stop()

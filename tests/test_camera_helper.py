#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Unit tests for camera_helper.py.
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

import pytest

from camera_tracking.camera_helper import camera_parameters_from_config


def test_camera_parameters_from_config_missing_key():
    """Tests that missing key in config raises an exception."""
    config = {"fx": 100}

    with pytest.raises(KeyError) as e_info:
        parameters = camera_parameters_from_config(config)

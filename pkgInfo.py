#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Custom package settings
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

category = "Modules/ROS"
sqLevel = "basic"
sqOptOutDirs = ["venv"]
sqOptOutFiles = ["ci-bst-checks.sh", "src/camera_tracking/azure_tracking.py"]
scripts = {"unittest": "ci-unit-tests.sh"}
sqOptOutRules = [
    "C09",
    "C10",
]
sqComments = {
    "C09": "BST.py is not compatible with ROS.",
    "C10": "BST.py is not compatible with ROS.",
}

# EOF

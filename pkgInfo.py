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
sqOptOutRules = [
    "C09",
    "C10",
    "BASH07",
    "PY02",
    "PY05",
]
sqComments = {
    "C09": "BST.py is not compatible with ROS.",
    "C10": "BST.py is not compatible with ROS.",
    "BASH07": "Failing due to external BashSrc.",
    "PY02": "We need to access some internals of the underlying 3rd party planner.",
    "PY05": "We execute pylint ourselves for sonarqube.",
}

# EOF

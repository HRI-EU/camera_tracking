#!/bin/bash
#
#  Execute unit tests.
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

set -euo pipefail

python -m pytest \
    -v \
    --cov-branch \
    --cov-report=term \
    --cov-report=html \
    --cov-report=xml \
    --cov=src/camera_tracking \
    --junitxml junit-report.xml \
    tests "$@"

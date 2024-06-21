#!/bin/bash
#
#  Perform static type checking continuous integration pipeline job using mypy.
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

source local.env
export MYPYPATH=${PYTHONPATH}

mypy ./nodes ./src ./scripts

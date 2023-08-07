#!/bin/bash
#
#  Perform "lint" continuous integration pipeline job using pylint.
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

pylint --version
pylint --recursive y \
       -f parseable \
       -v \
       --exit-zero .

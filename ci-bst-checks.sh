#!/bin/bash
#
#  Perform "bst check" continuous integration pipeline job using BST.py.
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

set -eo pipefail

# shellcheck source=/dev/null
source /hri/sit/latest/DevelopmentTools/ToolBOSCore/4.3.2/BashSrc
source /hri/sit/latest/External/anaconda/envs/common/3.11/BashSrc
BST.py -q

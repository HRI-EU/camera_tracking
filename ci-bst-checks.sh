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

# shellcheck source=/dev/null
source /hri/sit/latest/DevelopmentTools/ToolBOSCore/4.2.5/BashSrc
source /hri/sit/latest/External/anaconda3/envs/common/3.10/BashSrc
BST.py -q

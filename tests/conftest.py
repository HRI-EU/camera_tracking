#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Configuration for data_server unit tests.
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

import os
import sys

src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
assert os.path.exists(src_dir)
sys.path.insert(0, src_dir)

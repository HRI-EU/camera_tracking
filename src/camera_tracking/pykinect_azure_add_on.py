#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Add on for pykinect_azure.
#
#  Copyright (c) Honda Research Institute Europe GmbH
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

from enum import Enum


class K4ABT_JOINTS(Enum):
    K4ABT_JOINT_PELVIS = 0
    K4ABT_JOINT_SPINE_NAVEL = 1
    K4ABT_JOINT_SPINE_CHEST = 2
    K4ABT_JOINT_NECK = 3
    K4ABT_JOINT_CLAVICLE_LEFT = 4
    K4ABT_JOINT_SHOULDER_LEFT = 5
    K4ABT_JOINT_ELBOW_LEFT = 6
    K4ABT_JOINT_WRIST_LEFT = 7
    K4ABT_JOINT_HAND_LEFT = 8
    K4ABT_JOINT_HANDTIP_LEFT = 9
    K4ABT_JOINT_THUMB_LEFT = 10
    K4ABT_JOINT_CLAVICLE_RIGHT = 11
    K4ABT_JOINT_SHOULDER_RIGHT = 12
    K4ABT_JOINT_ELBOW_RIGHT = 13
    K4ABT_JOINT_WRIST_RIGHT = 14
    K4ABT_JOINT_HAND_RIGHT = 15
    K4ABT_JOINT_HANDTIP_RIGHT = 16
    K4ABT_JOINT_THUMB_RIGHT = 17
    K4ABT_JOINT_HIP_LEFT = 18
    K4ABT_JOINT_KNEE_LEFT = 19
    K4ABT_JOINT_ANKLE_LEFT = 20
    K4ABT_JOINT_FOOT_LEFT = 21
    K4ABT_JOINT_HIP_RIGHT = 22
    K4ABT_JOINT_KNEE_RIGHT = 23
    K4ABT_JOINT_ANKLE_RIGHT = 24
    K4ABT_JOINT_FOOT_RIGHT = 25
    K4ABT_JOINT_HEAD = 26
    K4ABT_JOINT_NOSE = 27
    K4ABT_JOINT_EYE_LEFT = 28
    K4ABT_JOINT_EAR_LEFT = 29
    K4ABT_JOINT_EYE_RIGHT = 30
    K4ABT_JOINT_EAR_RIGHT = 31

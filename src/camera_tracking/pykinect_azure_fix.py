#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Fix for loading the cuda version of the body tracking.
#
#  Copyright (c) Honda Research Institute Europe GmbH.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
#  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
#  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
#  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import ctypes
from pykinect_azure.k4abt import _k4abt, _k4abtTypes


def setup_onnx_provider_linux():
    # The original code checks whether it can load libonnxruntime_providers_cuda.so (usually under /usr/lib).
    # This fails with: /usr/lib/libonnxruntime_providers_cuda.so: undefined symbol: Provider_GetHost
    # Instead we directly load libonnxruntime.so.1.10.0 which later seems to load libonnxruntime_providers_cuda.so.
    _k4abtTypes.k4abt_tracker_default_configuration.processing_mode = _k4abtTypes.K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA
    ctypes.cdll.LoadLibrary("libonnxruntime.so.1.10.0")


_k4abt.setup_onnx_provider_linux = setup_onnx_provider_linux

import pykinect_azure

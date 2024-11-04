#!/bin/bash
#
#  Script for replacing aruco markers inside aruco_11_to_22_4cm.svg.
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

set -euo pipefail

want_marker_ids=(10 11 12 14 15 16 18 19 20 21 22 23)

if [[ ${#want_marker_ids[@]} != 12 ]]; then
    echo "You must provide 12 marker IDs."
    exit 1
fi

have_marker_ids=(11 12 13 14 15 16 17 18 19 20 21 22)

# The dummy is used to prevent multiple sequential replacements of a pattern. It is later removed again.
dummy="alreadydone"

replacement=""
filename=""

for (( i=0; i<12; i++ )); do
    have_id=${have_marker_ids[i]}
    want_id=${want_marker_ids[i]}
    replacement+="s:marker\_${have_id}:marker_${dummy}${want_id}:g;"
    replacement+="s:>${have_id}<:>${want_id}${dummy}<:g;"
    filename+="_${want_id}"
done

fullname=aruco_4cm"${filename}".svg

sed "${replacement}" aruco_4cm_11_to_22.svg > "${fullname}"
sed -i "s:${dummy}::g" "${fullname}"

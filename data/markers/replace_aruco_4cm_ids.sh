#!/bin/bash
#
#  Script for replacing aruco markers inside aruco_11_to_22_4cm.svg.
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
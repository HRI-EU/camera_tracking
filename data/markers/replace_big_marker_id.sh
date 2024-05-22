#!/bin/bash
#
#  Script for replacing aruco marker inside aruco_0_14cm.svg.
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

want_marker_ids=(1)

if [[ ${#want_marker_ids[@]} != 1 ]]; then
    echo "You must provide 1 marker ID."
    exit 1
fi

have_marker_ids=(0)

# The dummy is used to prevent multiple sequential replacements of a pattern. It is later removed again.
dummy="alreadydone"

replacement=""
filename=""

for (( i=0; i<1; i++ )); do
    have_id=${have_marker_ids[i]}
    want_id=${want_marker_ids[i]}
    replacement+="s:marker\_${have_id}:marker_${dummy}${want_id}:g;"
    replacement+="s:>${have_id}<:>${want_id}${dummy}<:g;"
    filename+="_${want_id}"
done

fullname=aruco"${filename}"_14cm.svg

sed "${replacement}" aruco_0_14cm.svg > "${fullname}"
sed -i "s:${dummy}::g" "${fullname}"

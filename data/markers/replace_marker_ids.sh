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

want_marker_ids=(15 16 18 19 20 25 26 28 29 39 45 46)

if [[ ${#want_marker_ids[@]} != 12 ]]; then
    echo "You must provide 12 marker IDs."
    exit 1
fi

have_marker_ids=(11 12 13 14 15 16 17 18 19 20 21 22)

# The dummy is used to prevent multiple sequential replacements of a pattern.
dummy="blablu"

replacement=""
filename=""

for (( i=0; i<12; i++ )); do
    have_id=${have_marker_ids[i]}
    want_id=${want_marker_ids[i]}
    replacement+="s:marker\_${have_id}:marker_${dummy}${want_id}:g;"
    replacement+="s:>${have_id}<:>${want_id}${dummy}<:g; "
    filename+="_${want_id}"
done


echo "$replacement"

fullname=aruco"${filename}"_4cm.svg

sed "${replacement}" aruco_11_to_22_4cm.svg > "${fullname}"
sed -i "s:${stopper}::g" "${fullname}"

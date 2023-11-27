#!/bin/bash
#
#  Script for shifting aruco markers inside aruco_11_to_22_4cm.svg.
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

start=11
end=22
offset=12
old_file=aruco_${start}_to_${end}_4cm.svg
new_file=aruco_$((start+offset))_to_$((end+offset))_4cm.svg
echo "${old_file} -> ${new_file}"
cp ${old_file} ${new_file}
for (( a=start; a<=end; a++ )); do b=$((a+offset)); echo "marker_$a -> marker_$b"; sed -i "s:marker_$a:marker_$b:g" ${new_file}; done
for (( a=start; a<=end; a++ )); do b=$((a+offset)); echo ">$a< -> >$b<"; sed -i "s:>$a<:>$b<:g" ${new_file}; done

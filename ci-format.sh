#!/bin/bash
#
#  Perform format check using black.
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

BLACK_ADDITIONAL_ARG="--check"
if [ "$1" == "--apply" ]; then
    printf "*** Applying format inplace ***\n\n"
    BLACK_ADDITIONAL_ARG=""
else
    printf "*** For applying format inplace start with './ci-format.sh --apply' ***\n\n"
fi

# shellcheck disable=SC2086,SC2248
black . --required-version "22.3.0" ${BLACK_ADDITIONAL_ARG}

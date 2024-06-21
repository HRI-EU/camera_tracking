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

set -euo pipefail

BLACK_ADDITIONAL_ARG="--check"

if [ "${1:-}" == "--apply" ]; then
    printf "*** Applying format inplace ***\n\n"
    BLACK_ADDITIONAL_ARG=
else
    printf "*** For applying format inplace start with './ci-format.sh --apply' ***\n\n"
fi

black . --required-version "24.4.2" ${BLACK_ADDITIONAL_ARG:+"${BLACK_ADDITIONAL_ARG}"}

#!/bin/bash
#
#  Creates the project's virtual python environment and installs dependencies defined in requirements.txt.
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

if [[ ! -f requirements.txt ]]; then
    echo "requirements.txt not found. Aborting."
    exit 1
fi

# Create virtualenv when the directory does not exist.
if [[ ! -d venv ]]; then
    python -m venv venv
fi

# Activate venv.
source ./venv/bin/activate

# Upgrade pip.
pip install --upgrade pip

# Install python requirements using pip.
pip install --no-cache-dir -r requirements.txt

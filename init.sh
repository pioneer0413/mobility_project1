#!/bin/bash

# 1. symbolic links for shared data and experiments
ln -s /DATA/mobility/project1/data data 2>/dev/null
ln -s /DATA/mobility/project1/exp exp 2>/dev/null
ln -s /DATA/mobility/project1/model model 2>/dev/null

# 2. update/init git submodules
git submodule update --init --recursive

# 3. environment setup
if [ -f env/environment.yml ]; then
    echo "Setting up Conda environment..."
    conda env create -f env/environment.yml 2>/dev/null || \
    conda env update -f env/environment.yml
elif [ -f env/requirements.txt ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r env/requirements.txt
fi

# 4. optional directory creation
mkdir -p log

# 5. show results
echo "Initialization complete."
echo "Symbolic links:"
ls -l | grep ^l

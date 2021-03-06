#!/bin/bash -e

# This script re-runs most of the experiments, except the long-running
# plane-bvp experiment.

# Install/activate environment
source install.sh

# Tell PyOpenCL to use POCL
export PYOPENCL_TEST=portable

# Run a simple test
py.test --disable-warnings utils.py inteq_tests.py

# Generate data
nice ./generate-data.py --all --except '*-calibration-params' --except plane-bvp

# Postprocess data
./generate-figures-and-tables.py --all

# Build summary PDF
make -f makefile.summary

# Numerical experiments: Optimization of fast algorithms for Quadrature by Expansion using target-specific expansions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3542253.svg)](https://doi.org/10.5281/zenodo.3542253)

This repository contains the code for numerical experiments in the
paper 'Optimization of fast algorithms for Quadrature by Expansion
using target-specific expansions,' available at
[doi:10.1016/j.jcp.2019.108976](https://doi.org/10.1016/j.jcp.2019.108976)
or on [arXiv](https://arxiv.org/abs/1811.01110).

The code that is included reproduces the experiments in Section 4 of
the paper, including:

* Tables 3 and 4;
* Data presented in Sections 4.2.2, 4.2.3, and 4.3; and
* Figures 6, 7, 8, 9, 10, and 11.

There's also a Jupyter notebook for numerically verifying the
identities in Appendix A.

## Running Everything

**Important:** A machine with a large amount of memory is required for
some experiments. This issue is currently being
[investigated](https://gitlab.tiker.net/inducer/pytential/issues/137). We
have tested these successfully on a 20-core 2.30 GHz Intel Xeon
E5-2650 v3 machine with 256 GB of RAM.

To run almost everything, install the [Docker image](https://doi.org/10.5281/zenodo.3523410),
and from a shell running in a container, ensure you are in the code directory and
type:
```
./run.sh
```
This script re-runs experiments, and generates an output file
`summary/summary.pdf` containing generated figures, tables, and
data.

For ease of reference, the original paper data, saved in JSON format, is
included in this repository and is overwitten when running `./run.sh` (but
trivially recoverable through git). If you just want to postprocess the saved
data to generate tables and figures, which takes much less time than re-running
all the experiments, see [below](#running-experiments).

`./run.sh` doesn't run the plane BVP experiment from Section 4.3,
which takes quite a bit of time to run. See
[below](#running-experiments) for how to run this experiment and
others individually. If you are using the Docker image, data from
that experiment is already saved in the `raw-data-bvp` directory. The
script also doesn't run the experiment to regenerate the calibration
parameters for the various geometries, as doing so would likely cause
the results to differ.

## Installation Hints

Two options are available for installation.

### Docker Image

The simplest way to install is to use the [Docker
image](https://doi.org/10.5281/zenodo.3523410). The code,
software, and saved results are installed in the image directory
`/home/inteq/gigaqbx-ts-results`.

Note that the version of GCC and the compiler flags used for the
software in the Docker image differ from the version used in the paper
as described in Section 4.1. This does not affect any of the results,
but will affect the calibration parameters in the `params` directory
if those are regenerated.

### Manual Installation

If you don't want to use the Docker image, you can install necessary software
manually using the command:
```
./install.sh
```
This script downloads and installs software in an isolated environment in this
directory using Conda and pip.

For producing figures and outputs, you also need LaTeX and the
[latexrun](https://github.com/aclements/latexrun) script.

### Post-Installation

Before directly running the Python scripts in this directory, activate the
Conda environment and instruct PyOpenCL to use POCL, as follows:
```
source .miniconda3/bin/activate inteq
export PYOPENCL_TEST=portable
```

To ensure that everything works, you can run a few short tests:
```
py.test --disable-warnings utils.py inteq_tests.py
```

A more extensive set of tests can be found in the Pytential test suite (included
in `src/pytential/test`), which should also pass.

## Running Experiments

The scripts `generate-data.py` and `generate-figures-and-tables.py` can be used
to run individual experiments or groups of experiments, and postprocess
experimental outputs, respectively. Pass the `--help` option for more
documentation and the list of available experiments.

The `params`, `raw-data`, and `raw-data-bvp` directories are (over)written by
`generate-data.py` and hold experimental outputs. Some experiments depend on
others: the data in the `params` directory is written by a number of experiments
and serves as the input parameters for other experiments. The `out` directory
contains generated figures and tables and is written to by
`generate-figures-and-tables.py`.

To regenerate all outputs from the data that is already in the `raw-data` and `params`
directories, run
```
./generate-figures-and-tables.py --all
make -f makefile.summary
```

Figure 10, unlike the rest of the figures, is generated using Paraview
(which is not installed by the installation script). The provided file
`plane-workflow.pvsm` was tested on Paraview 5.5.2.

To run an individual experiment or to regenerate the data for a single
experiments, supply the command line option `-x experiment-name`. For instance, to
regenerate the results for the `plane-bvp` experiment, run
```
./generate-data.py -x plane-bvp
```

## Contents

The following files and directories in this repository are included and/or
generated:

| Name | Description |
|----------------------------------|------------------------------------------------------------------------------------------------------------|
| `.miniconda3` | Conda install directory |
| `Dockerfile` | Used for generating the Docker image |
| `Expansion Identities.ipynb` | Jupyter notebook for identities in Appendix A |
| `LICENSE` | License file for the code in this directory |
| `README.md` | This file |
| `env` | Files used by the installer to set up the Conda and pip environments |
| `generate-data.py` | Script for running experiments |
| `generate-figures-and-tables.py` | Script for postprocessing experiments and producing figures and tables. Puts output in the `out` directory |
| `install.sh` | Installation script |
| `inteq_tests.py` | Integral equation code used by `generate-data.py` |
| `makefile.summary` | Makefile for generating the summary PDF |
| `out` | Holds generated figures and tables |
| `params` | Holds parameters for experiments |
| `plane-mesh.pkl.gz` | Saved mesh file for the plane geometry in Section 4.3 |
| `plane-workflow.pvsm` | Paraview state for generating Figure 10 |
| `raw-data-bvp` | Holds data generated by the plane BVP experiment in Section 4.3 |
| `raw-data` | Holds data generated by experiments |
| `run.sh` | Script for re-running all experiments and generating all outputs |
| `src` | Pip install directory |
| `summary.tex` | Source code for summary PDF |
| `summary` | Holds generated summary PDF and auxiliary files |
| `utils.py` | Extra code used by `generate-data.py` |

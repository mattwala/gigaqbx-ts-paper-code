# Numerical experiments: A fast algorithm with error bounds for Quadrature by Expansion

This repository contains the code for numerical experiments in the paper 'A fast
algorithm with error bounds for Quadrature by Expansion,' available at
[doi:10.1016/j.jcp.2018.05.006](https://doi.org/10.1016/j.jcp.2018.05.006) or on
[arXiv](https://arxiv.org/abs/1801.04070).

The code that is included reproduces the experiments in Section 4 of
the paper, including:

* Data presented in Sections 4.2.2 and 4.2.3
* Figures 6, 7, 8, 9, 10, 11

## Running Everything

**Important**: A machine with a large amount of memory is required for
some experiments. This issue is currently being
[investigated](https://gitlab.tiker.net/inducer/pytential/issues/137). We
have tested these successfully on a 20-core 2.30 GHz Intel Xeon
E5-2650 v3 machine with 256 GB of RAM.

Install the [Docker image](#docker-image), and from a shell running in a
container, go to the code directory and type:
```
./run.sh
```
This script re-runs (almost all) experiments, and generates an output file
`summary/summary.pdf` containing (almost all) generated figures, tables, and
data.

This script doesn't run the plane BVP experiment from Section 4.3,
which takes quite a bit of time to run. In order to run that along
with the other experiments, set the environment variable `RUN_PLANE`
to `1` first:
```
RUN_PLANE=1 ./run.sh
```
Data from a run of this experiment is already saved in the
`raw-data-bvp` directory. Setting RUN_PLANE=1 overwrites the files
that are already there.

Processing the output of the plane BVP experiment requires Paraview
(not installed). The file `plane-workflow.pvsm` is the saved Paraview
state to generate Figure 10. This was tested with Paraview X.X.X.

It's also possible to have more selective control over what gets run. See
[below](#running-experiments).

## Installation Hints

Two options are available for installation.

### Docker Image

The simplest way to install is to use the
[Docker image](http://dx.doi.org/10.5281/zenodo.3483367). The code
and software are installed in the image directory
`/home/inteq/gigaqbx-ts-results`.

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

To ensure that everything works, you can run a short test:
```
py.test --disable-warnings utils.py
```

A more extensive set of tests can be found in the Pytential test suite (included
in `src/pytential/test`), which should also pass.

## Running Experiments

The scripts `generate-data.py` and `generate-figures-and-tables.py` can be used
to run individual experiments or groups of experiments, and postprocess
experimental outputs, respectively. Pass the `--help` option for more
documentation and the list of available experiments.

The `raw-data` directory is written to by `generate-data.py` and holds
experimental outputs. The `out` directory contains generated figures and tables
and is written to by `generate-figures-and-tables.py`.

To regenerate all outputs from the data that is already in the `raw-data`
directory, run

```
./generate-figures-and-tables.py --all
make -f makefile.summary
```

To run an individual experiment or to regenerate the data for a single
experiments, supply the command line option `-x experiment-name`. For instance, to
regenerate the results for the `bvp` experiment, run

```
./generate-data.py -x bvp
./generate-figures-and-tables.py -x bvp
```

## Contents

The following files and directories in this repository are included and/or
generated:

| Name | Description |
|----------------------------------|------------------------------------------------------------------------------------------------------------|
| `Dockerfile` | Used for generating the Docker image |
| `README.md` | This file |
| `install.sh` | Installation script |
| `generate-data.py` | Script for running experiments. Puts output in the `raw-data` directory |
| `generate-figures-and-tables.py` | Script for postprocessing experiments and producing figures and tables. Puts output in the `out` directory |
| `makefile.summary` | Makefile for generating the summary PDF |
| `utils.py` | Extra code used by `generate-data.py` |
| `summary.tex` | Source code for summary PDF |
| `run.sh` | Script for re-running all experiments and generating all outputs |
| `env` | Files used by the installer to set up the Conda and pip environments |
| `.miniconda3` | Conda install directory |
| `src` | Pip install directory |
| `raw-data` | Holds data generated by experiments |
| `out` | Holds generated figures and tables |
| `summary` | Holds generated summary PDF and auxiliary files |

## Citations

```
@article{gigaqbxts,
  title = "A fast algorithm with error bounds for {Quadrature} by {Expansion}",
  journal = "Journal of Computational Physics",
  volume = "374",
  pages = "135 - 162",
  year = "2018",
  doi = "10.1016/j.jcp.2018.05.006",
  author = "Matt Wala and Andreas Klöckner",
}
```

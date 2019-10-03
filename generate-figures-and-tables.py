#!/usr/bin/env python3
"""Generate figures from experimental data"""
import matplotlib
import numpy as np  # noqa

import logging
import pickle
import pandas as pd

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# Whether to generate a PDF file. If False, will generate pgf.
GENERATE_PDF = 0


matplotlib.use("agg" if GENERATE_PDF else "pgf")


import matplotlib.pyplot as plt  # noqa


SMALLFONTSIZE = 8
FONTSIZE = 10
LINEWIDTH = 0.5


def initialize_matplotlib():
    pass


initialize_matplotlib()


DATA_DIR = "raw-data"
OUTPUT_DIR = "out"


def open_data_file(filename, **kwargs):
    return open(os.path.join(DATA_DIR, filename), "r", **kwargs)


def open_output_file(filename, **kwargs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return open(os.path.join(OUTPUT_DIR, filename), "w", **kwargs)


EXPERIMENTS = (
        "urchin-time-prediction",
    )


def generate_urchin_time_prediction_table():
    with open_data_file("urchin-time-prediction-modeled-costs.json") as infile:
        modeled_costs = read_data(infile)
        
    with open_data_file("urchin-time-prediction-actual-costs.json") as infile:
        actual_costs = read_data(infile)


def gen_figures_and_tables(experiments):
    if "urchin-time-prediction" in experiments:
        generate_urchin_time_prediction_table()
            

def main():
    names = ["'%s'" % name for name in EXPERIMENTS]
    names[-1] = "and " + names[-1]

    description = (
            "This script postprocesses results for one or more experiments. "
            " The names of the experiments are: " + ", ".join(names)
            + ".")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
            "-x",
            metavar="experiment-name",
            action="append",
            dest="experiments",
            default=[],
            help="Postprocess results for an experiment "
                 "(may be specified multiple times)")

    parser.add_argument(
            "--all",
            action="store_true",
            dest="run_all",
            help="Postprocess results for all available experiments")

    parser.add_argument(
            "--except",
            action="append",
            metavar="experiment-name",
            dest="run_except",
            default=[],
            help="Do not postprocess results for an experiment "
                 "(may be specified multiple times)")

    result = parser.parse_args()

    experiments = set()

    if result.run_all:
        experiments = set(EXPERIMENTS)
    experiments |= set(result.experiments)
    experiments -= set(result.run_except)

    gen_figures_and_tables(experiments)


if __name__ == "__main__":
    main()

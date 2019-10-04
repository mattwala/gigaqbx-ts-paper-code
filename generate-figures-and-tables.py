#!/usr/bin/env python3
"""Generate figures from experimental data"""
import matplotlib
import numpy as np  # noqa

import os
import argparse
import json
import utils
import logging

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


def read_data(infile):
    return json.load(infile, cls=utils.CostResultDecoder)


def print_table(table, headers, outf_name, column_formats=None):
    with open_output_file(outf_name) as outfile:
        def my_print(s):
            print(s, file=outfile)
        my_print(r"\begin{tabular}{%s}" % column_formats)
        my_print(r"\toprule")
        if isinstance(headers[0], (list, tuple)):
            for header_row in headers:
                my_print(" & ".join(header_row) + r"\\")
        else:
            my_print(" & ".join(headers) + r"\\")
        my_print(r"\midrule")
        for row in table:
            my_print(" & ".join(row) + r"\\")
        my_print(r"\bottomrule")
        my_print(r"\end{tabular}")
    logger.info("Wrote %s", os.path.join(OUTPUT_DIR, outf_name))


EXPERIMENTS = (
        "urchin-time-prediction",
    )


def generate_urchin_time_prediction_table():
    with open_data_file("urchin-time-prediction-modeled-costs.json") as infile:
        modeled_costs_raw = read_data(infile)

    with open_data_file("urchin-time-prediction-actual-costs.json") as infile:
        actual_costs_raw = read_data(infile)

    #assert len(modeled_costs_raw) == len(actual_costs_raw)
    nresults = len(modeled_costs_raw)
    urchin_ks = sorted(cost["k"] for cost in modeled_costs_raw)

    from pytential.qbx.performance import estimate_calibration_params

    modeled_costs = []

    for k in urchin_ks:
        for field in modeled_costs_raw:
            if field["k"] == k:
                modeled_costs.append(field["cost"])
                break
        else:
            raise ValueError("not found: %d" % k)

    actual_costs = []

    for k in urchin_ks:
        for field in actual_costs_raw:
            if field["k"] == k:
                actual_costs.append(field["cost"])
                break
        else:
            raise ValueError("not found: %d" % k)

    headers = (
            (
                r"\multirow{2}{*}{Kind}",
                r"\multicolumn{%d}{c}{Process Time (s)}" % nresults,
                ),
            (
                r"\cmidrule(lr){2-%d}" % (1 + nresults),
                ) + tuple(r"\cellcenter{$\gamma_{%d}$}" % k for k in urchin_ks))

    rows = []

    # {{{ actual costs row

    row = ["Actual"]

    for cost in actual_costs:
        time = sum(
                timing_result["process_elapsed"]
                for timing_result in cost.values())
        row.append("%.2f" % time)

    rows.append(row)

    # }}}

    # {{{ modeled costs row

    row = ["Model"]

    for cost in modeled_costs:
        time = sum(cost.get_predicted_times(True).values())
        row.append("%.2f" % time)

    rows.append(row)

    # }}}

    col_formats = "c" + nresults * "r"
    print_table(rows, headers, "urchin-time-prediction.tex", col_formats)


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

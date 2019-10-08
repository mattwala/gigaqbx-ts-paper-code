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
    plt.rc("font", family="serif")
    plt.rc("text", usetex=True)
    plt.rc("pgf", preamble=[
            r"\usepackage{amsmath}",
            r"\providecommand{\pqbx}{p_{\text{QBX}}}",
            r"\providecommand{\pfmm}{p_{\text{FMM}}}",
            r"\providecommand{\nmax}{n_{\text{max}}}",
            r"\providecommand{\nmpole}{n_{\text{mpole}}}"])
    # https://stackoverflow.com/questions/40424249/vertical-alignment-of-matplotlib-legend-labels-with-latex-math
    plt.rc(("text.latex",), preview=True)
    plt.rc("xtick", labelsize=FONTSIZE)
    plt.rc("ytick", labelsize=FONTSIZE)
    plt.rc("axes", labelsize=1)
    plt.rc("axes", titlesize=FONTSIZE)
    plt.rc("axes", linewidth=LINEWIDTH)
    plt.rc("pgf", rcfonts=False)
    plt.rc("lines", linewidth=LINEWIDTH)
    plt.rc("patch", linewidth=LINEWIDTH)
    plt.rc("legend", fancybox=False)
    plt.rc("legend", framealpha=1)
    plt.rc("legend", frameon=False)
    plt.rc("savefig", dpi=150)


initialize_matplotlib()

# {{{ file utils

PARAMS_DIR = "params"
DATA_DIR = "raw-data"
OUTPUT_DIR = "out"


def open_data_file(filename, **kwargs):
    return open(os.path.join(DATA_DIR, filename), "r", **kwargs)


def open_output_file(filename, **kwargs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return open(os.path.join(OUTPUT_DIR, filename), "w", **kwargs)


def load_params(filename, **flags):
    with open(os.path.join(PARAMS_DIR, filename), "r", **flags) as outf:
        return json.load(outf, cls=utils.CostResultDecoder)


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

# }}}


# {{{ labeling

import matplotlib.cm
_colors = plt.cm.Paired.colors  # pylint:disable=no-member


class Colors(object):
    LIGHT_BLUE = _colors[0]
    BLUE = _colors[1]
    LIGHT_GREEN = _colors[2]
    GREEN = _colors[3]
    RED = _colors[5]
    ORANGE = _colors[7]
    PURPLE = _colors[9]


class QBXPerfLabelingBase:
    perf_line_styles = ("x-", "+-", ".-", "*-", "s-", "d-")

    summary_line_style = ".--"

    summary_label = "all"

    summary_color = Colors.RED

    perf_labels = (
            r"$U_b$",
            r"$V_b$",
            r"$W_b^\mathrm{close}$",
            r"$W_b^\mathrm{far}$",
            r"$X_b^\mathrm{close}$",
            r"$X_b^\mathrm{far}$")

    silent_summed_features = (
            "form_multipoles",
            "coarsen_multipoles",
            "eval_direct",
            "eval_multipoles",
            "refine_locals",
            "eval_locals",
            "translate_box_local_to_qbx_local",
            "eval_qbx_expansions",
            )

    perf_colors = (
            Colors.ORANGE,
            Colors.PURPLE,
            Colors.LIGHT_BLUE,
            Colors.BLUE,
            Colors.LIGHT_GREEN,
            Colors.GREEN)


class GigaQBXPerfLabeling(QBXPerfLabelingBase):
    perf_features = (
            "form_global_qbx_locals_list1",
            "multipole_to_local",
            "form_global_qbx_locals_list3",
            "translate_box_multipoles_to_qbx_local",
            "form_global_qbx_locals_list4",
            "form_locals")


class GigaQBXTSPerfLabeling(QBXPerfLabelingBase):
    perf_features = (
            "eval_target_specific_qbx_locals_list1",
            "multipole_to_local",
            "eval_target_specific_qbx_locals_list3",
            "translate_box_multipoles_to_qbx_local",
            "eval_target_specific_qbx_locals_list4",
            "form_locals")


class List3FirstGigaQBXPerfLabeling(QBXPerfLabelingBase):
    perf_labels = (
            r"$W_b^\mathrm{close}$",
            r"$W_b^\mathrm{far}$",
            r"$U_b^{\vphantom{\mathrm{close}}}$",
            r"$V_b^{\vphantom{\mathrm{close}}}$",
            r"$X_b^\mathrm{close}$",
            r"$X_b^\mathrm{far}$")

    perf_features = (
            "form_global_qbx_locals_list3",
            "translate_box_multipoles_to_qbx_local",
            "form_global_qbx_locals_list1",
            "multipole_to_local",
            "form_global_qbx_locals_list4",
            "form_locals")

    perf_colors = (
            Colors.LIGHT_BLUE,
            Colors.BLUE,
            Colors.ORANGE,
            Colors.PURPLE,
            Colors.LIGHT_GREEN,
            Colors.GREEN)


class List3FirstGigaQBXTSPerfLabeling(QBXPerfLabelingBase):
    perf_colors = (
            Colors.LIGHT_BLUE,
            Colors.BLUE,
            Colors.ORANGE,
            Colors.PURPLE,
            Colors.LIGHT_GREEN,
            Colors.GREEN)

    perf_labels = (
            r"$W_b^\mathrm{close}$",
            r"$W_b^\mathrm{far}$",
            r"$U_b^{\vphantom{\mathrm{close}}}$",
            r"$V_b^{\vphantom{\mathrm{close}}}$",
            r"$X_b^\mathrm{close}$",
            r"$X_b^\mathrm{far}$")

    perf_features = (
            "eval_target_specific_qbx_locals_list3",
            "translate_box_multipoles_to_qbx_local",
            "eval_target_specific_qbx_locals_list1",
            "multipole_to_local",
            "eval_target_specific_qbx_locals_list4",
            "form_locals")

# }}}


# {{{ timing bar chart

def generate_timing_bar_chart(
        labels, before, before_labeling, afters, after_labeling,
        name, title=None):
    """Generate a stacked bar chart showing the effect of optimizations on cost.

    Params:

        labels: Names of each test geometry
        before: For each test geometry, the ParameterizedCosts for the baseline
            cost
        before_labeling: PerfLabeling for *before*
        afters: For each test geometry, the (list of) ParameterizedCosts for
            the sequence of optimizations
        after_labeling: PerfLabeling for *afters*
        name: Output name
        title: Chart title

    """
    fig, ax = plt.subplots(1, 1)

    n = len(labels)

    fig.set_size_inches(5, max(n, 2))

    labels = list(reversed(labels))

    # Magic numbers to ensure that the widths of bars in pixels across different
    # graphs are identical.
    if n == 5:
        bar_width = 0.07
        bar_sep = 0.03
    elif n == 1:
        ax.set_ylim((-0.20350000000000001, 0.20350000000000001))
        f = 130.4181051 / 186.73218673
        bar_width = 0.07 * f
        bar_sep = 0.03 * f

    index = np.arange(n) * 0.5

    assert len(afters) == 3

    before_index = index + 3 * bar_width / 2 + 3 * bar_sep / 2

    after_indices = [
            index + bar_width / 2 + bar_sep / 2,
            index - bar_width / 2 - bar_sep/2,
            index - 3 * bar_width / 2 - 3 * bar_sep / 2]

    #ax.spines['bottom'].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ax.set_title(title)

    ax.yaxis.set_ticks_position("none")

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    ax.set_xticklabels([r"0\%", r"20\%", r"40\%", r"60\%", r"80\%", r"100\%"])

    ax.set_xlim([0, 1])

    ax.set_yticks(index)
    ax.set_yticklabels(labels)

    ax.set_yticks((index[1:] + index[:-1]) / 2, minor=True)
    ax.grid(lw=LINEWIDTH/2, which="minor", axis="y")

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_x(-0.1)

    perf_labels = before_labeling.perf_labels
    perf_colors = before_labeling.perf_colors

    cumul_before = np.zeros(n)
    cumul_afters = [np.zeros(n) for i in range(len(afters))]

    before = [b.get_predicted_times(0) for b in before]
    afters = [[a.get_predicted_times(0) for a in after] for after in afters]

    for ilabel, label in enumerate(perf_labels):
        before_vals = []
        afters_vals = [[] for i in range(len(afters))]

        for i in range(n):
            before_vals.append(before[i][before_labeling.perf_features[ilabel]]
                               / sum(before[i].values()))
            for iafter, after in enumerate(afters):
                afters_vals[iafter].append(
                        after[i][after_labeling.perf_features[ilabel]]
                        / sum(before[i].values()))

        ax.barh(before_index, before_vals, bar_width, cumul_before, label=label,
                color=perf_colors[ilabel])
        cumul_before += before_vals

        for after_vals, cumul_after, after_index, after in (
                zip(afters_vals, cumul_afters, after_indices, afters)):
            ax.barh(after_index, after_vals, bar_width, cumul_after,
                    color=perf_colors[ilabel])
            cumul_after[:] += after_vals

    before_vals = []
    afters_vals = []
    for i in range(n):
        before_vals.append(1 - cumul_before[i])
        for iafter, (after, cumul_after) in enumerate(zip(afters, cumul_afters)):
            afters_vals.append(sum(after[i].values()) / sum(before[i].values())
                               - cumul_after[i])

    ax.barh(before_index, before_vals, bar_width, cumul_before,
            label=r"\vphantom{$X_b^\mathrm{far}$}(other)",
            color=Colors.RED)
    cumul_before += before_vals

    for after_vals, cumul_after, after_index, after in (
            zip(afters_vals, cumul_afters, after_indices, afters)):
        ax.barh(after_index, after_vals, bar_width, cumul_after,
                color=Colors.RED)
        cumul_after[:] += after_vals

    for i in range(n):
        ax.text(-0.01, before_index[i], "base", ha="right",
                va='center', fontsize=SMALLFONTSIZE)

        for label, cumul_after, after_index in (
                zip((r"ts", r"$n_\mathrm{max}$", r"$n_\mathrm{mpole}$"),
                        cumul_afters, after_indices)):
            ax.text(-0.01, after_index[i], label, ha="right",
                    va='center', fontsize=SMALLFONTSIZE)

    leg_handles, leg_labels = ax.get_legend_handles_labels()

    def flip(items, ncol):
        # Flips labels of legend to go left-to-right as opposed to
        # top-to-bottom.
        import itertools
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    leg_handles = flip(leg_handles, 5)
    leg_labels = flip(leg_labels, 5)

    leg = fig.legend(leg_handles, leg_labels, ncol=5, loc="lower center",
                     fontsize=SMALLFONTSIZE)

    if n == 5:
        fig.subplots_adjust(bottom=0.2)
    else:
        fig.subplots_adjust(bottom=0.5)

    if 0:
        # Used to obtain pixel widths of bars.
        print("pixels per unit",
              ax.transData.transform([(0,1),(1,0)])-ax.transData.transform((0,0)))
        print("ylim", ax.get_ylim())

    ax.set_xlabel("Percentage of Baseline", fontdict={"size": FONTSIZE})

    suffix = "pdf" if GENERATE_PDF else "pgf"

    outfile = f"{OUTPUT_DIR}/timing-{name}.{suffix}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")

    logger.info(f"Wrote {outfile}")

# }}}


# {{{ scaling and balancing graphs

def initialize_axes(ax, title, xlabel=None, ylabel=None, grid_axes=None):
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontdict={"size": FONTSIZE})
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontdict={"size": FONTSIZE})
    if grid_axes is None:
        grid_axes = "both"
    ax.grid(lw=LINEWIDTH / 2, which="major", axis=grid_axes)


def generate_complexity_figure(
        subtitles, x_values, y_values, labeling, ylabel,
        xlabel, name, ylimits=None, size_inches=None,
        subplots_adjust=None, plot_title=None,
        plot_kind="loglog", postproc_func=None,
        summary_labels=None, grid_axes=None):
    """Generate a figure with *n* subplots

    Parameters:
        subtitles: List of *n* subtitles
        x_values: List of *n* lists of x values
        y_values: List of *n* lists of results, which are dictionaries mapping
            stage names to costs
        labeling: Subclass of *QBXPerfLabelingBase*
        ylabel: Label for y axis
        xlabel: Label for x axis
        ylimits: List of *n* y limits, or *None*
        name: Output file name
        size_inches: Passed to Figure.set_size_inches()
        subplots_adjust: Passed to Figure.subplots_adjust()
        plot_title: Plot tile

    """
    fig, axes = plt.subplots(1, len(subtitles))

    if size_inches:
        fig.set_size_inches(*size_inches)

    if len(subtitles) == 1:
        axes = [axes]

    if ylimits is None:
        ylimits = (None,) * len(subtitles)

    plot_options = dict(linewidth=LINEWIDTH, markersize=3)

    for iax, (ax, axtitle) in enumerate(zip(axes, subtitles)):
        if iax > 0:
            ylabel = None
        initialize_axes(ax, axtitle, xlabel=xlabel, ylabel=ylabel, grid_axes=grid_axes)

    # Generate results.
    for xs, ys, lim, ax in zip(x_values, y_values, ylimits, axes):
        if lim:
            ax.set_ylim(*lim)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if plot_kind == "loglog":
            plotter = ax.loglog
        elif plot_kind == "semilogy":
            plotter = ax.semilogy
        else:
            raise ValueError("unknown plot kind")

        # features
        labels = []
        for feature, label, style, color in zip(
                labeling.perf_features,
                labeling.perf_labels,
                labeling.perf_line_styles,
                labeling.perf_colors):
            ylist = [y[feature] for y in ys]
            l, = plotter(xs, ylist, style, color=color, label=label, **plot_options)
            labels.append(l)

        summary_values = [sum(y.values()) for y in ys]

        # summary
        l, = plotter(xs,
                summary_values,
                labeling.summary_line_style,
                label=labeling.summary_label,
                color=labeling.summary_color,
                **plot_options)

        labels.append(l)

        if summary_labels:
            # label
            for x, y, l in zip(xs, summary_values, summary_labels):
                ax.text(
                        x, y * 1.3, l, ha="center", va="bottom",
                        fontsize=SMALLFONTSIZE)
            miny, maxy = ax.get_ylim()
            ax.set_ylim([miny, maxy * 1.7])

    if postproc_func:
        postproc_func(fig)

    fig.legend(labels, labeling.perf_labels + ("all",), loc="center right",
               fontsize=SMALLFONTSIZE)

    suffix = "pdf" if GENERATE_PDF else "pgf"

    if plot_title:
        fig.suptitle(plot_title, fontsize=FONTSIZE)

    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)

    outfile = f"{OUTPUT_DIR}/complexity-{name}.{suffix}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    logger.info(f"Wrote {outfile}")


def _generate_complexity_figure_from_results(
        xs, ys, output_filename, labeling, xlabel, extra_kwargs):

    subtitles = (None,)

    generate_complexity_figure(
            subtitles,
            (xs,),
            (ys,),
            labeling,
            ylabel="Modeled Process Time (s)",
            xlabel=xlabel,
            name=output_filename,
            **extra_kwargs)


def scaling_graph(results, output_filename, labeling, summary_labels, title=None):
    kwargs = {
            "plot_kind": "loglog",
            "subplots_adjust": dict(right=0.7),
            "size_inches": (4, 2.5),  # was (3, 1.9)
            "summary_labels": summary_labels,
            "plot_title": title,
    }

    xs = []
    ys = []

    for result in results:
        nparticles = result.params["nsources"] + result.params["ntargets"]
        times = result.get_predicted_times(merge_close_lists=False)
        xs.append(nparticles)
        ys.append(times)

    _generate_complexity_figure_from_results(
            xs, ys, output_filename, labeling, xlabel="Number of Particles",
            extra_kwargs=kwargs)


def balancing_graph(results, xlabel, name, labeling, title=None):
    kwargs = {
            "plot_kind": "semilogy",
            "subplots_adjust": dict(right=0.7),
            "size_inches": (3, 1.9),
            "ylimits": ((10**2, 2 * 10**4),),
            "grid_axes": "y",
            "plot_title": title,
    }

    xs = []
    ys = []

    for result in results:
        times = result["cost"].get_predicted_times(merge_close_lists=False)
        xs.append(result["param_value"])
        ys.append(times)

    _generate_complexity_figure_from_results(
            xs, ys, output_filename=name,
            labeling=labeling, xlabel="Number of Particles",
            extra_kwargs=kwargs)

# }}}


EXPERIMENTS = (
        "urchin-time-prediction",
        "urchin-tuning-study",
        "urchin-optimization-study",
        "urchin-green-error",

        "donut-tuning-study",
        "donut-optimization-study",
        "donut-green-error",

        "plane-tuning-study",
        "plane-optimization-study",
        "plane-bvp-error",
)


# {{{ tuning study table

def generate_tuning_study_table(tuning_params, label):
    headers = ("Parameter", "Value")

    rows = (
            (r"$\nmax$, baseline", str(tuning_params["baseline_nmax"])),
            (r"$\nmpole$, baseline", str(tuning_params["baseline_nmpole"])),
            (r"$\nmax$, with TSQBX", str(tuning_params["tsqbx_nmax"])),
            (r"$\nmpole$, with TSQBX", str(tuning_params["tsqbx_nmpole"])),
    )

    print_table(rows, headers, f"tuning-params-{label}.tex", "lr")

# }}}


# {{{ green error table

def generate_green_error_table(labels, errors, name):
    headers = ("Geometry", "{Error}")
    rows = []

    total_error = 0

    for label, error in zip(labels, errors):
        row = []
        rows.append(row)
        row.append(label)
        error = error["err_linf"]
        row.append("%.17e" % error)
        total_error += error

    if len(rows) > 1:
        mean_row = ["\cmidrule{1-2}(avg.)"]
        rows.append(mean_row)
        mean_row.append("%.17e" % (total_error / len(rows)))

    print_table(rows, headers, f"green-error-{name}.tex", "lS")

# }}}


# {{{ optimization summary table

def generate_optimization_summary_table(labels, optimizations, name):
    headers = (
            (
                r"\cellcenter{\multirow{2}{*}{Geometry}}",
                r"\cellcenter{\multirow{2}{*}{Baseline Cost (s)}}",
                r"\multicolumn{3}{c}{Cost Reduction}"
                ),
            (
                r"\cmidrule(lr){3-5}",
                r"",
                r"\cellcenter{ts}",
                r"\cellcenter{$\nmax$}",
                r"\cellcenter{$\nmpole$}",
                )
    )

    rows = []
    cost_reductions_by_row = []

    for i, label in enumerate(labels):
        opts = [opt[i] for opt in optimizations]
        row = [label]
        rows.append(row)

        baseline_time = sum(opts[0].get_predicted_times().values())
        row.append("%.2f" % baseline_time)

        cost_reductions = []
        cost_reductions_by_row.append(cost_reductions)

        for i in range(1, 4):
            time = sum(opts[i].get_predicted_times().values())
            cost_reductions.append(time / baseline_time)
            row.append("%.2f" % (time / baseline_time))

    if len(cost_reductions_by_row) > 1:
        mean_row = [r"\cmidrule{1-5}(avg.)", "---"]
        rows.append(mean_row)
        for i in range(3):
            mean_row.append(
                    "%.2f" %
                    np.mean(
                        [
                            reductions[i]
                            for reductions in cost_reductions_by_row]))

    print_table(rows, headers, f"optimization-summary-{name}.tex", "lrrrr")

# }}}


# {{{ urchin time prediction

def generate_urchin_time_prediction_table():
    with open_data_file("time-prediction-urchin-modeled-costs.json") as infile:
        modeled_costs_raw = read_data(infile)

    with open_data_file("time-prediction-urchin-actual-costs.json") as infile:
        actual_costs_raw = read_data(infile)

    #assert len(modeled_costs_raw) == len(actual_costs_raw)
    nresults = len(modeled_costs_raw)
    urchin_ks = sorted(cost["geometry"] for cost in modeled_costs_raw)

    from pytential.qbx.performance import estimate_calibration_params

    modeled_costs = []

    for k in urchin_ks:
        for field in modeled_costs_raw:
            if field["geometry"] == k:
                modeled_costs.append(field["cost"])
                break
        else:
            raise ValueError("not found: %d" % k)

    actual_costs = []

    for k in urchin_ks:
        for field in actual_costs_raw:
            if field["geometry"] == k:
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

# }}}


# {{{ urchin tuning study

def generate_urchin_tuning_study_table():
    tuning_params = load_params("tuning-params-urchin.json")
    generate_tuning_study_table(tuning_params, "urchin")

# }}}


# {{{ urchin optimization study

def generate_urchin_optimization_study_outputs():
    with open_data_file("optimization-study-urchin-opt0.json") as inf:
        o0 = [item["cost"] for item in read_data(inf)]

    with open_data_file("optimization-study-urchin-opt1.json") as inf:
        o1 = [item["cost"] for item in read_data(inf)]

    with open_data_file("optimization-study-urchin-opt2.json") as inf:
        o2 = [item["cost"] for item in read_data(inf)]

    with open_data_file("optimization-study-urchin-opt3.json") as inf:
        o3 = [item["cost"] for item in read_data(inf)]

    labels = [fr"$\gamma_{{{i}}}$" for i in range(2, 11, 2)]

    # Stacked bar char showing cost reduction
    #
    # Use a List 3 first labeling to move List 3 to the baseline of the
    # chart, to enable better visual comparison of the most expensive part
    # of the algorithm.
    generate_timing_bar_chart(
            labels,
            o0,
            List3FirstGigaQBXPerfLabeling,
            [o1, o2, o3],
            List3FirstGigaQBXTSPerfLabeling,
            name="urchin",
            title=
                r"Optimization of GIGAQBX Running Time "
                r"on `Urchin' Geometries $\gamma_2, \gamma_4, \ldots, "
                r"\gamma_{10}$")

    # Optimization summary
    generate_optimization_summary_table(labels, [o0, o1, o2, o3], name="urchin")

    # Scaling graph
    scaling_graph(
            o0,
            "scaling-urchin-baseline",
            GigaQBXPerfLabeling,
            labels,
            title= r"Scaling of Unmodified GIGAQBX (No TSQBX)")

    # Balancing graph
    with open_data_file("tuning-study-urchin-tsqbx-nmax.json") as inf:
        tuning_results = read_data(inf)

    balancing_graph(
            tuning_results,
            r"$\nmax$",
            "nmax-urchin-study",
            GigaQBXTSPerfLabeling,
            title= r"Impact of $n_\mathrm{max}$ on Modeled Process Time")

# }}}


# {{{ urchin green error

def generate_urchin_green_error_table():
    labels = [fr"$\gamma_{{{i}}}$" for i in range(2, 11, 2)]
    
    with open_data_file("green-error-urchin.json") as inf:
        errors = [item["error"] for item in read_data(inf)]

    generate_green_error_table(labels, errors, "urchin")

# }}}


# {{{ donut tuning study

def generate_donut_tuning_study_table():
    tuning_params = load_params("tuning-params-donut.json")
    generate_tuning_study_table(tuning_params, "donut")

# }}}


# {{{ donut optimization study

def generate_donut_optimization_study_outputs():
    with open_data_file("optimization-study-donut-opt0.json") as inf:
        o0 = [item["cost"] for item in read_data(inf)]

    with open_data_file("optimization-study-donut-opt1.json") as inf:
        o1 = [item["cost"] for item in read_data(inf)]

    with open_data_file("optimization-study-donut-opt2.json") as inf:
        o2 = [item["cost"] for item in read_data(inf)]

    with open_data_file("optimization-study-donut-opt3.json") as inf:
        o3 = [item["cost"] for item in read_data(inf)]

    labels = [r"$\tau_{10}$"]

    # Stacked bar char showing cost reduction
    #
    # Use a List 3 first labeling to move List 3 to the baseline of the
    # chart, to enable better visual comparison of the most expensive part
    # of the algorithm.
    generate_timing_bar_chart(
            labels,
            o0,
            List3FirstGigaQBXPerfLabeling,
            [o1, o2, o3],
            List3FirstGigaQBXTSPerfLabeling,
            name="donut",
            title=
                r"Optimization of GIGAQBX Running Time "
                r"on `Torus Grid' Geometry $\tau_{10}$")

    # Optimization summary
    generate_optimization_summary_table(labels, [o0, o1, o2, o3], name="donut")

# }}}


# {{{ donut green error

def generate_donut_green_error_table():
    labels = [r"$\tau_{10}$"]

    with open_data_file("green-error-donut.json") as inf:
        errors = [item["error"] for item in read_data(inf)]

    generate_green_error_table(labels, errors, "donut")

# }}}


# {{{ plane tuning study

def generate_plane_tuning_study_table():
    tuning_params = load_params("tuning-params-plane.json")
    generate_tuning_study_table(tuning_params, "plane")

# }}}


# {{{ plane optimization study

def generate_plane_optimization_study_outputs():
    with open_data_file("optimization-study-plane-opt0.json") as inf:
        o0 = [item["cost"] for item in read_data(inf)]

    with open_data_file("optimization-study-plane-opt1.json") as inf:
        o1 = [item["cost"] for item in read_data(inf)]

    with open_data_file("optimization-study-plane-opt2.json") as inf:
        o2 = [item["cost"] for item in read_data(inf)]

    with open_data_file("optimization-study-plane-opt3.json") as inf:
        o3 = [item["cost"] for item in read_data(inf)]

    labels = [r"Plane"]

    # Stacked bar char showing cost reduction
    #
    # Use a List 3 first labeling to move List 3 to the baseline of the
    # chart, to enable better visual comparison of the most expensive part
    # of the algorithm.
    generate_timing_bar_chart(
            labels,
            o0,
            List3FirstGigaQBXPerfLabeling,
            [o1, o2, o3],
            List3FirstGigaQBXTSPerfLabeling,
            name="plane",
            title=
                r"Optimization of GIGAQBX Running Time on `Plane' Geometry")

    # Optimization summary
    generate_optimization_summary_table(labels, [o0, o1, o2, o3], name="plane")

# }}}


def gen_figures_and_tables(experiments):
    if "urchin-time-prediction" in experiments:
        generate_urchin_time_prediction_table()

    if "urchin-tuning-study" in experiments:
        generate_urchin_tuning_study_table()

    if "urchin-optimization-study" in experiments:
        generate_urchin_optimization_study_outputs()

    if "urchin-green-error" in experiments:
        generate_urchin_green_error_table()

    if "donut-tuning-study" in experiments:
        generate_donut_tuning_study_table()

    if "donut-optimization-study" in experiments:
        generate_donut_optimization_study_outputs()

    if "donut-green-error" in experiments:
        generate_donut_green_error_table()

    if "plane-tuning-study" in experiments:
        generate_plane_tuning_study_table()

    if "plane-optimization-study" in experiments:
        generate_plane_optimization_study_outputs()


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

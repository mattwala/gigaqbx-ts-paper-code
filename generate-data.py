#!/usr/bin/env python3
"""Gather experimental data"""
import collections
import numpy as np  # noqa
import numpy.linalg as la  # noqa
import pyopencl as cl  # noqa
import pyopencl.clmath  # noqa
import json
import utils
import os
import argparse

from functools import partial  # noqa: F401
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, NArmedStarfish, drop, n_gon, qbx_peanut,
        WobblyCircle, make_curve_mesh, starfish)
from pytential import bind, sym, norm  # noqa
from pytential.qbx.performance import PerformanceModel
from pytools import one

import logging
import multiprocessing

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


TARGET_ORDER = 8
OVSMP_FACTOR = 5
TCF = 0.9
QBX_ORDER = 5
FMM_ORDER = 15
MESH_TOL = 1e-10
FORCE_STAGE2_UNIFORM_REFINEMENT_ROUNDS = 1
SCALED_MAX_CURVATURE_THRESHOLD = 0.8
MAX_LEAF_REFINE_WEIGHT = 128
RUNS = 1
POOL_WORKERS = min(5, multiprocessing.cpu_count())

# URCHIN_PARAMS = list(range(2, 12, 2))
# URCHIN_PARAMS = list(range(2, 7, 2))
URCHIN_PARAMS = (1,)
DONUT_PARAMS = list(range(1, 6))


nan = float("nan")


CALIBRATION_PARAMS_QBX5_FMM15_LAPLACE3D = {
        'c_l2l': 5.89e-09,
        'c_l2p': nan,
        'c_l2qbxl': 2.57e-08,
        'c_m2l': 3.24e-09,
        'c_m2m': 5.43e-09,
        'c_m2p': nan,
        'c_m2qbxl': 3.36e-09,
        'c_p2l': 1.15e-08,
        'c_p2m': 1.20e-08,
        'c_p2p': nan,
        'c_p2p_tsqbx': 9.51e-09,
        'c_p2qbxl': 1.43e-08,
        'c_qbxl2p': 6.66e-07}


DEFAULT_LPOT_KWARGS = dict(
        fmm_backend="fmmlib",
        target_association_tolerance=1e-3,
        fmm_order=FMM_ORDER, qbx_order=QBX_ORDER,
        _box_extent_norm="l2",
        _from_sep_smaller_crit="static_l2",
        _well_sep_is_n_away=2,
        _expansions_in_tree_have_extent=True,
        _expansion_stick_out_factor=TCF,
        _max_leaf_refine_weight=MAX_LEAF_REFINE_WEIGHT,
        _from_sep_smaller_min_nsources_cumul=0,
        _use_target_specific_qbx=False,
        performance_model=PerformanceModel(
            calibration_params=CALIBRATION_PARAMS_QBX5_FMM15_LAPLACE3D),
        )


# {{{ general utils

def lpot_source_from_mesh(queue, mesh, lpot_kwargs=None):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = TARGET_ORDER

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    refiner_extra_kwargs = {
            "_force_stage2_uniform_refinement_rounds": (
                 FORCE_STAGE2_UNIFORM_REFINEMENT_ROUNDS),
            "_scaled_max_curvature_threshold": (
                 SCALED_MAX_CURVATURE_THRESHOLD),
            }

    if lpot_kwargs is None:
        lpot_kwargs = DEFAULT_LPOT_KWARGS

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, OVSMP_FACTOR*target_order,
            **lpot_kwargs,)

    lpot_source, _ = lpot_source.with_refinement(**refiner_extra_kwargs)

    return lpot_source


def _urchin_lpot_source(k, queue, lpot_kwargs):
    sph_m = k // 2
    sph_n = k

    from meshmode.mesh.generation import generate_urchin
    mesh = generate_urchin(
            order=TARGET_ORDER, m=sph_m, n=sph_n,
            est_rel_interp_tolerance=MESH_TOL)

    return lpot_source_from_mesh(queue, mesh, lpot_kwargs)


def urchin_geometry_getter(k):
    return partial(_urchin_lpot_source, k)


def replicate_along_axes(mesh, shape, sep_ratio):
    from meshmode.mesh.processing import (
            find_bounding_box, affine_map, merge_disjoint_meshes)

    bbox = find_bounding_box(mesh)
    sizes = bbox[1] - bbox[0]

    meshes = [mesh]

    for i in range(mesh.ambient_dim):
        for j in range(1, shape[i]):
            vec = np.zeros(mesh.ambient_dim)
            vec[i] = j * sizes[i] * (1 + sep_ratio)
            meshes.append(affine_map(mesh, A=None, b=vec))

        # FIXME: https://gitlab.tiker.net/inducer/pytential/issues/6
        mesh = merge_disjoint_meshes(meshes, single_group=True)
        meshes = [mesh]

    return mesh


def _torus_lpot_source(r_outer, r_inner, n_outer, n_inner, replicant_shape,
                       sep_ratio, queue, lpot_kwargs):
    from meshmode.mesh.generation import generate_torus
    mesh = generate_torus(r_outer, r_inner, n_outer, n_inner, order=TARGET_ORDER)
    mesh = replicate_along_axes(mesh, replicant_shape, sep_ratio)
    return lpot_source_from_mesh(queue, mesh, lpot_kwargs)


def torus_geometry_getter(r_outer, r_inner, n_outer, n_inner):
    return partial(
            _torus_lpot_source, r_outer, r_inner, n_outer, n_inner, (1, 1, 1), 0)


def donut_geometry_getter(nrows):
    return partial(_torus_lpot_source, 2, 1, 40, 20, (2, nrows, 1), 0.1)


OUTPUT_DIR = "raw-data"


def make_output_file(filename, **flags):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return open(os.path.join(OUTPUT_DIR, filename), "w", **flags)


def output_data(obj, outfile):
    json.dump(obj, outfile, cls=utils.CostResultEncoder)


# }}}


# {{{ cost getter

def get_lpot_cost(which, helmholtz_k, geometry_getter, lpot_kwargs, kind):
    """
    Parameters:

        which: "D" or "S"
        kind: "actual" or "model"
    """
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)

    lpot_source = geometry_getter(queue, lpot_kwargs)

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    sigma_sym = sym.var("sigma")
    if helmholtz_k == 0:
        k_sym = LaplaceKernel(lpot_source.ambient_dim)
        kernel_kwargs = {}
    else:
        k_sym = HelmholtzKernel(lpot_source.ambient_dim, "k")
        kernel_kwargs = {"k": helmholtz_k}

    if which == "S":
        op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1, **kernel_kwargs)
    elif which == "D":
        op = sym.D(k_sym, sigma_sym, qbx_forced_limit="avg", **kernel_kwargs)
    else:
        raise ValueError("unknown lpot symbol: '%s'" % which)

    bound_op = bind(lpot_source, op)

    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    if kind == "actual":
        timing_data = {}
        result = bound_op.eval(queue, {"sigma": sigma}, timing_data=timing_data)
        assert not np.isnan(result.get(queue)).any()
        result = one(timing_data.values())

    elif kind == "model":
        perf_results = bound_op.get_modeled_performance(queue, sigma=sigma)
        result = one(perf_results.values())

    return result

# }}}


# {{{ run the perf model on a set of geometries

def run_perf_model(
        geometry_getters, perf_model, lpot_kwargs=None, which_op="S",
        helmholtz_k=0):
    """Run the performance model on a set of geometries, in parallel.
    
    Params:

        geometry_getters: List of callables returning geometries
        perf_model: The performance model to test
        lpot_kwargs: Overrides DEFAULT_LPOT_KWARGS
        which_op: "S" or "D"
        helmholtz_k: Helmholtz parameter

    Returns:

        A list of performance model results
    """

    if lpot_kwargs is None:
        lpot_kwargs = DEFAULT_LPOT_KWARGS

    lpot_kwargs = lpot_kwargs.copy()
    lpot_kwargs["performance_model"] = perf_model

    runner = partial(
            get_lpot_cost,
            which_op, helmholtz_k, lpot_kwargs=lpot_kwargs, kind="model")

    with multiprocessing.Pool(POOL_WORKERS) as pool:
        return pool.map(runner, geometry_getters)

# }}}


def run_urchin_time_prediction_experiment():
    perf_model = PerformanceModel()
    urchins = [urchin_geometry_getter(k) for k in URCHIN_PARAMS]
    perf_results = run_perf_model(urchins, perf_model)
    results = [{"k": k, "cost": result}
            for k, result in zip(URCHIN_PARAMS, perf_results)]
    with make_output_file("urchin-time-prediction-modeled-costs.json")\
            as outfile:
        output_data(results, outfile)


def run_experiments(experiments):
    if "urchin-time-prediction" in experiments:
        run_urchin_time_prediction_experiment()


EXPERIMENTS = (
        "urchin-time-prediction",
    )


def main():
    names = ["'%s'" % name for name in EXPERIMENTS]
    names[-1] = "and " + names[-1]

    description = (
            "This script collects data from one or more experiments. "
            " The names of the experiments are: " + ", ".join(names)
            + ".")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
            "-x",
            metavar="experiment-name",
            action="append",
            dest="experiments",
            default=[],
            help="Run an experiment (may be specified multiple times)")

    parser.add_argument(
            "--all",
            action="store_true",
            dest="run_all",
            help="Run all available experiments")

    parser.add_argument(
            "--except",
            action="append",
            metavar="experiment-name",
            dest="run_except",
            default=[],
            help="Do not run an experiment (may be specified multiple times)")

    result = parser.parse_args()

    experiments = set()

    if result.run_all:
        experiments = set(EXPERIMENTS)
    experiments |= set(result.experiments)
    experiments -= set(result.run_except)

    run_experiments(experiments)


if __name__ == "__main__":
    # Avoid issues with fork()-based multiprocessing and pyopencl - see
    # https://github.com/inducer/pyopencl/issues/156
    multiprocessing.set_start_method("spawn")
    main()


# vim: foldmethod=marker
